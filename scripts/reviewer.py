"""Stage 2：本地小模型语义复核器（OpenVINO GPU/NPU，INT4 权重）。

职责：对规则引擎产出的候选违规（关键词/正则命中 + absent 缺失 +
语义类规则任务）做二次确认，降低误报，产出可解释的判定理由。

分级调度中的第二级——只处理第一级筛出的候选（约 15% 片段），
全部推理在本机完成（默认 GPU，可切 NPU/CPU），token 开销远低于
"全量逐字稿直接喂云端大模型"的方案。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

SYSTEM_PROMPT = (
    "你是电销通话合规质检复核员。你会收到若干疑似违规候选，"
    "判断每条违规是否真实成立。/no_think\n\n"
    "判定原则：\n"
    "- 说话人确实说了违规话术 → confirmed: true\n"
    "- 话术合法合规、仅引用法规说明 → confirmed: false\n"
    "- 'absent' 类：坐席确实未做到应做事项 → confirmed: true\n"
    "- 上下文无明确证据时 → confirmed: false（不能仅凭猜测定违规）\n\n"
    "只输出JSON数组，每项形如 "
    '{"index": 序号, "confirmed": true或false, "reason": "一句话理由"}。'
    "不要输出任何其他内容。"
)


@dataclass
class ReviewCandidate:
    index: int
    rule_id: str
    rule_name: str
    semantic_hint: str  # 规则语义说明
    excerpt: str  # 命中摘录或相关上下文
    context: str = ""  # 前后文
    advice: str = ""


@dataclass
class ReviewVerdict:
    index: int
    rule_id: str
    confirmed: bool
    reason: str


class LocalReviewer:
    """OpenVINO 本地模型复核器。

    推理后端：openvino-genai LLMPipeline（官方路线，原生支持 INT4/WOQ，
    GPU/NPU 均可；不依赖 transformers.generate，规避版本冲突）。
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "GPU",
        max_new_tokens: int = 200,
        batch_size: int = 10,
    ):
        self.model_dir = model_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.pipe = None
        self.load_elapsed_s = 0.0
        self.last_infer_s = 0.0
        self.last_output_tokens = 0

    def load(self):
        t0 = time.perf_counter()
        import openvino_genai

        self._genai = openvino_genai
        self.pipe = openvino_genai.LLMPipeline(self.model_dir, self.device)
        self.load_elapsed_s = round(time.perf_counter() - t0, 2)
        return self

    def _build_messages(self, candidates: List[ReviewCandidate]) -> list:
        items = []
        for c in candidates:
            items.append(
                {
                    "index": c.index,
                    "rule_id": c.rule_id,
                    "rule_name": c.rule_name,
                    "rule_desc": c.semantic_hint or c.advice,
                    "excerpt": c.excerpt[:400],
                    "context": c.context[:800],
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"待复核候选": items}, ensure_ascii=False),
            },
        ]

    @staticmethod
    def _parse_verdicts(
        text: str, candidates: List[ReviewCandidate]
    ) -> List[ReviewVerdict]:
        # Strip Qwen3 thinking section safely (chr(60)='<' chr(62)='>')
        _open = chr(60) + "think" + chr(62)
        _close = chr(60) + "/think" + chr(62)
        start = text.find(_open)
        if start != -1:
            end = text.find(_close, start)
            if end != -1:
                text = text[end + len(_close) :]
            else:
                # Unclosed think: find first { after think-open as JSON start
                brace = text.find("{", start)
                if brace != -1:
                    text = text[brace:]
        text = text.strip()
        # Parse JSON: try array first, then individual objects
        parsed = []
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                v = json.loads(m.group(0))
                parsed = v if isinstance(v, list) else [v]
            except json.JSONDecodeError:
                parsed = []
        if not parsed:
            for obj_m in re.finditer(r'\{[^{}]*"index"[^{}]*\}', text):
                try:
                    parsed.append(json.loads(obj_m.group(0)))
                except json.JSONDecodeError:
                    continue
        by_index = {}
        for p in parsed:
            if isinstance(p, dict) and "index" in p:
                try:
                    by_index[int(p["index"])] = p
                except (TypeError, ValueError):
                    continue
        verdicts = []
        for c in candidates:
            p = by_index.get(c.index, {})
            confirmed = bool(p.get("confirmed", False))
            reason = str(p.get("reason", "")).strip()[:200]
            if not reason:
                reason = "未发现明确违规话术" if not confirmed else "说话人存在违规话术"
            verdicts.append(
                ReviewVerdict(
                    index=c.index,
                    rule_id=c.rule_id,
                    confirmed=confirmed,
                    reason=reason,
                )
            )
        return verdicts

    def review(self, candidates: List[ReviewCandidate]) -> List[ReviewVerdict]:
        """分批复核：候选按 batch_size 分组，每组独立推理后合并结果。

        batch_size=10 (D2优化)：将所有候选一次性送入模型（/no_think 抑制思考链），
        prompt 处理开销从 N 次降为 1 次，total_tokens 显著降低。
        max_new_tokens 按 batch 大小动态缩放（每候选 ~50 tokens）。
        """
        if not candidates:
            return []
        if self.pipe is None:
            self.load()
        all_verdicts: List[ReviewVerdict] = []
        total_tokens = 0
        total_elapsed = 0.0
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]
            messages = self._build_messages(batch)
            # Dynamic max_new_tokens: ~50 tokens per candidate + 20 overhead
            dynamic_max = min(500, max(self.max_new_tokens, len(batch) * 50 + 20))
            cfg = self._genai.GenerationConfig(max_new_tokens=dynamic_max)
            cfg.do_sample = False
            history = self._genai.ChatHistory()
            history.set_messages(messages)
            t0 = time.perf_counter()
            result = self.pipe(history, cfg)
            text = result.texts[0] if hasattr(result, "texts") else str(result)
            elapsed = time.perf_counter() - t0
            try:
                metrics = result.perf_metrics
                total_tokens += int(metrics.get_num_generated_tokens())
            except Exception:
                total_tokens += max(50, len(text) // 2)
            total_elapsed += elapsed
            all_verdicts.extend(self._parse_verdicts(text, batch))
        self.last_infer_s = round(total_elapsed, 2)
        self.last_output_tokens = total_tokens
        return all_verdicts


def candidates_from_engine(engine_result, segments_text: str) -> List[ReviewCandidate]:
    """把规则引擎结果转为复核候选：确定性命中 + 语义任务统一编号。"""
    cands: List[ReviewCandidate] = []
    idx = 0
    for h in engine_result.hits:
        match_desc = {
            "keyword": "关键词命中",
            "regex": "正则命中",
            "absent": "应做未做（缺失检测）",
        }.get(h.matched_by, h.matched_by)
        cands.append(
            ReviewCandidate(
                index=idx,
                rule_id=h.rule_id,
                rule_name=h.rule_name,
                semantic_hint=f"[{h.level}] {match_desc}。{h.basis}",
                excerpt=h.excerpt,
                context=segments_text,
                advice=h.advice,
            )
        )
        idx += 1
    for t in engine_result.semantic_tasks:
        cands.append(
            ReviewCandidate(
                index=idx,
                rule_id=t.rule_id,
                rule_name=t.rule_name,
                semantic_hint=f"[{t.level}] 语义检测。{t.semantic_hint}",
                excerpt="",
                context=t.context_text or segments_text,
            )
        )
        idx += 1
    return cands


class OllamaReviewer:
    """Ollama 本地模型复核器（HTTP API，支持 qwen2.5 等非思考模型）。

    优势：qwen2.5 无思考模式，直接输出 JSON，解析可靠；
    GPU 加速：启用 OLLAMA_IGPU_ENABLE=1 后通过 Vulkan 后端使用 Intel Arc GPU，
    生成速度 ~19 tok/s（vs CPU 2.7 tok/s，7x 加速），prompt 处理 ~2385 tok/s。
    适用：OpenVINO 不可用时的自动降级，或需要更高精度（4B vs 1.7B）时手动指定。
    """

    # 去掉 /no_think 标签（qwen2.5 不需要）
    _SYSTEM_PROMPT = SYSTEM_PROMPT.replace("/no_think\n\n", "\n\n")

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        host: str = "localhost:11434",
        max_new_tokens: int = 200,
        batch_size: int = 10,
    ):
        self.model_name = model_name
        self.host = host
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.load_elapsed_s = 0.0
        self.last_infer_s = 0.0
        self.last_output_tokens = 0

    def load(self):
        import requests

        t0 = time.perf_counter()
        try:
            r = requests.get(f"http://{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Ollama 不可达 ({self.host}): {e}. 请先运行 `ollama serve`。"
            )
        # 预加载模型（首次推理会慢，提前 warm-up）
        try:
            r = requests.post(
                f"http://{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=120,
            )
            r.raise_for_status()
        except Exception:
            pass  # warm-up 失败不阻塞，首次 review 时会自动加载
        self.load_elapsed_s = round(time.perf_counter() - t0, 2)
        return self

    def _build_messages(self, candidates: List[ReviewCandidate]) -> list:
        items = []
        for c in candidates:
            items.append(
                {
                    "index": c.index,
                    "rule_id": c.rule_id,
                    "rule_name": c.rule_name,
                    "rule_desc": c.semantic_hint or c.advice,
                    "excerpt": c.excerpt[:200],
                    "context": c.context[:300],
                }
            )
        return [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"candidates": items}, ensure_ascii=False),
            },
        ]

    def review(self, candidates: List[ReviewCandidate]) -> List[ReviewVerdict]:
        """分批复核：调用 Ollama chat API，每组独立推理。"""
        import requests

        if not candidates:
            return []
        all_verdicts: List[ReviewVerdict] = []
        total_tokens = 0
        total_elapsed = 0.0
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]
            messages = self._build_messages(batch)
            dynamic_max = min(500, max(self.max_new_tokens, len(batch) * 50 + 20))
            t0 = time.perf_counter()
            resp = requests.post(
                f"http://{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": dynamic_max,
                        "num_ctx": 8192,
                    },
                },
                timeout=300,
            )
            resp.raise_for_status()
            elapsed = time.perf_counter() - t0
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            total_tokens += int(data.get("eval_count", 0))
            total_elapsed += elapsed
            all_verdicts.extend(LocalReviewer._parse_verdicts(text, batch))
        self.last_infer_s = round(total_elapsed, 2)
        self.last_output_tokens = total_tokens
        return all_verdicts


def create_reviewer(
    backend: str = "auto",
    model_dir: Optional[str] = None,
    device: str = "GPU",
    ollama_model: str = "qwen2.5:3b",
    ollama_host: str = "localhost:11434",
    max_new_tokens: int = 200,
    batch_size: int = 10,
):
    """工厂函数：按 backend 选择复核后端。

    backend:
        "openvino" → LocalReviewer（OpenVINO INT4，GPU/NPU/CPU）
        "ollama"   → OllamaReviewer（Ollama HTTP API，GPU Vulkan 或 CPU）
        "auto"     → 优先 OpenVINO，不可用则降级 Ollama
    """
    if backend == "ollama":
        return OllamaReviewer(
            model_name=ollama_model,
            host=ollama_host,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    if (
        backend == "auto"
        and model_dir
        and os.path.isfile(os.path.join(model_dir, "openvino_model.xml"))
    ):
        try:
            return LocalReviewer(
                model_dir,
                device=device,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            ).load()
        except Exception as e:
            print(f"WARN: OpenVINO 加载失败，尝试 Ollama 降级: {e}", file=sys.stderr)

    if backend in ("auto", "ollama"):
        return OllamaReviewer(
            model_name=ollama_model,
            host=ollama_host,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    if model_dir:
        return LocalReviewer(
            model_dir,
            device=device,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    raise ValueError(f"无法创建复核器: backend={backend}, model_dir={model_dir}")


if __name__ == "__main__":
    # 自测：构造候选 → 加载模型 → 复核（需先完成 OpenVINO 转换）
    import sys

    backend_type = sys.argv[1] if len(sys.argv) > 1 else "openvino"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else r"D:\vg_ov\Qwen3-1.7B-ov-int4"
    device = sys.argv[3] if len(sys.argv) > 3 else "GPU"

    cands = [
        ReviewCandidate(
            index=0,
            rule_id="FIN-001",
            rule_name="承诺保本保收益",
            semantic_hint="坐席对产品作出保本保收益承诺",
            excerpt="这个产品保本保息，年化收益30%起步，稳赚不赔的",
            context="坐席：这个产品保本保息，年化收益30%起步，稳赚不赔的。客户：真的没有风险吗？坐席：绝对安全。",
        ),
        ReviewCandidate(
            index=1,
            rule_id="FIN-001",
            rule_name="承诺保本保收益",
            semantic_hint="坐席对产品作出保本保收益承诺",
            excerpt="我们的存款产品利率是2%，受存款保险保障",
            context="坐席：我们的存款产品利率是2%，受存款保险保障。客户：好的我了解一下。",
        ),
    ]

    if backend_type == "ollama":
        rev = OllamaReviewer(model_name=model_arg or "qwen2.5:3b").load()
        print("backend: Ollama, model:", rev.model_name)
    else:
        rev = LocalReviewer(model_arg, device=device).load()
        print("backend: OpenVINO, device:", device, "load_s:", rev.load_elapsed_s)

    vs = rev.review(cands)
    print(
        json.dumps(
            [
                {
                    "index": v.index,
                    "rule": v.rule_id,
                    "confirmed": v.confirmed,
                    "reason": v.reason,
                }
                for v in vs
            ],
            ensure_ascii=False,
            indent=1,
        )
    )
    print("infer_s:", rev.last_infer_s, "output_tokens:", rev.last_output_tokens)
