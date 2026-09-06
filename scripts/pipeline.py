"""VoiceGuard 主管线（分级调度编排）。

Stage 1：音频 → ASR 转写 → 规则引擎零 token 初筛 → 候选清单
Stage 2：候选片段 → 本地小模型语义复核（OpenVINO GPU/NPU）
Stage 3：结构化质检报告（评分/违规清单/漏斗/元数据）
Stage 4（可选）：云端脱敏增强 → 辅导建议（弱网自动跳过）

D2 里程碑：run_full_pipeline() 实现 Stage1→Stage2→Stage3→Stage4 全链路一键编排。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules_engine import EngineResult, Segment, load_rule_packs, run_engine  # noqa: E402
from transcribe import transcribe  # noqa: E402

if TYPE_CHECKING:
    from report_gen import QAReport

DEFAULT_RULES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BASE = os.environ.get("VOICEGUARD_MODEL_DIR", os.path.join(ROOT, "models"))


@dataclass
class Stage1Result:
    audio: str
    duration_ms: int
    lang: str
    segment_count: int
    engine: EngineResult
    asr_elapsed_s: float
    total_elapsed_s: float
    segments_ref: List[Segment] = field(default_factory=list)
    asr_device: str = "cpu"
    asr_stats: dict = field(default_factory=dict)


def run_stage1(
    audio_path: str,
    rules_dir: str = DEFAULT_RULES_DIR,
    packs: Optional[List[str]] = None,
    asr_model_dir: Optional[str] = None,
    asr_device: str = "cpu",
) -> Stage1Result:
    """Stage 1：音频 → 逐字稿 → 规则引擎候选清单。"""
    t0 = time.perf_counter()

    kwargs = {"device": asr_device}
    if asr_model_dir:
        kwargs["model_dir"] = asr_model_dir
    tr = transcribe(audio_path, **kwargs)
    t_asr = time.perf_counter() - t0

    segments: List[Segment] = [
        Segment(
            segment_id=s.segment_id,
            text=s.text,
            start_ms=s.start_ms,
            end_ms=s.end_ms,
            speaker=s.speaker,
        )
        for s in tr["segments"]
    ]

    loaded = load_rule_packs(rules_dir, pack_ids=packs)
    engine = run_engine(segments, loaded)

    return Stage1Result(
        audio=os.path.basename(audio_path),
        duration_ms=tr.get("duration_ms", 0),
        lang=tr.get("lang", ""),
        segment_count=len(segments),
        engine=engine,
        asr_elapsed_s=round(t_asr, 2),
        total_elapsed_s=round(time.perf_counter() - t0, 2),
        segments_ref=segments,
        asr_device=tr.get("device", asr_device),
        asr_stats=tr.get("asr_stats", {}),
    )


def stage1_report(r: Stage1Result) -> dict:
    """Stage 1 结果序列化为可展示摘要。"""
    return {
        "audio": r.audio,
        "duration_ms": r.duration_ms,
        "lang": r.lang,
        "segments": r.segment_count,
        "asr_elapsed_s": r.asr_elapsed_s,
        "total_elapsed_s": r.total_elapsed_s,
        "funnel": r.engine.funnel,
        "hits": [
            {
                "rule_id": h.rule_id,
                "name": h.rule_name,
                "level": h.level,
                "matched_by": h.matched_by,
                "excerpt": h.excerpt,
            }
            for h in r.engine.hits
        ],
        "semantic_tasks": [
            {"rule_id": t.rule_id, "name": t.rule_name} for t in r.engine.semantic_tasks
        ],
        "packs": r.engine.packs_loaded,
    }


@dataclass
class PipelineResult:
    """D2 全链路结果（Stage1 + Stage2 + Stage3 + 可选 Stage4 云增强）。"""

    stage1: Stage1Result
    report: "QAReport"  # QAReport
    verdicts: Optional[list] = None  # List[ReviewVerdict]
    reviewer_backend: str = "未启用"
    review_s: float = 0.0
    review_prompt_tokens: int = 0
    cloud_enhance: dict = field(default_factory=dict)


def _find_review_model() -> Optional[str]:
    """搜索 OpenVINO INT4 复核模型目录（SKILL.md 契约）。"""
    for name in ("Qwen3-4B-ov-int4", "Qwen3-1.7B-ov-int4"):
        for base in (MODEL_BASE, r"D:\vg_ov"):
            p = os.path.join(base, name)
            if os.path.isfile(os.path.join(p, "openvino_model.xml")):
                return p
    return None


def run_full_pipeline(
    audio_path: str,
    packs: Optional[List[str]] = None,
    device: str = "GPU",
    no_model: bool = False,
    cloud_enhance: bool = False,
    asr_model_dir: Optional[str] = None,
    rules_dir: Optional[str] = None,
    reviewer_backend: str = "auto",
    ollama_model: str = "qwen2.5:3b",
    ollama_host: str = "localhost:11434",
) -> PipelineResult:
    """全链路一键编排：Stage1(ASR→规则) → Stage2(语义复核) → Stage3(报告)。

    参数:
        audio_path: 音频文件路径
        packs: 规则包列表（None=全部）
        device: ASR 推理设备 GPU/NPU/CPU
        no_model: True=跳过语义复核（仅规则引擎，最快）
        cloud_enhance: True=启用云端脱敏增强（默认关闭）
        asr_model_dir: ASR 模型目录（None=默认）
        rules_dir: 规则目录（None=默认）
        reviewer_backend: 复核后端 auto/openvino/ollama
        ollama_model: Ollama 模型名（默认 qwen2.5:3b）
        ollama_host: Ollama 服务地址

    返回:
        PipelineResult（含 stage1, report, verdicts 等）
    """
    from report_gen import build_report, estimate_tokens_zh

    _rules_dir = rules_dir or DEFAULT_RULES_DIR

    # ---- Stage 1：ASR 转写 + 规则引擎零 token 初筛 ----
    s1 = run_stage1(
        audio_path,
        rules_dir=_rules_dir,
        packs=packs,
        asr_model_dir=asr_model_dir,
        asr_device=device,
    )

    # ---- Stage 2：本地小模型语义复核（分级调度第二级） ----
    verdicts = None
    backend_label = "未启用"
    review_s = 0.0
    prompt_tokens = 0

    if not no_model:
        model_dir = _find_review_model()
        cands = []
        create_reviewer = None  # type: ignore
        if model_dir or reviewer_backend in ("auto", "ollama"):
            try:
                from reviewer import (  # type: ignore
                    candidates_from_engine,
                    create_reviewer,
                )

                seg_text = "\n".join(
                    f"[{s.speaker or '说话人'}] {s.text}" for s in s1.segments_ref
                )
                cands = candidates_from_engine(s1.engine, seg_text)
            except Exception as e:
                print(f"WARN: 候选构建失败: {e}", file=sys.stderr)

        if cands and create_reviewer is not None:
            try:
                rev: Any = create_reviewer(  # type: ignore
                    backend=reviewer_backend,
                    model_dir=model_dir,
                    device=device,
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                    max_new_tokens=200,
                    batch_size=10,
                )
                # OllamaReviewer 需显式 load（LocalReviewer 已在 create_reviewer 中 load）
                if hasattr(rev, "model_name") and not getattr(rev, "pipe", None):
                    rev.load()
                verdicts = rev.review(cands)
                review_s = rev.last_infer_s
                if hasattr(rev, "model_name"):
                    # Detect GPU acceleration via Ollama /api/ps size_vram field
                    gpu_label = "GPU"
                    try:
                        import requests as _req

                        _ps = _req.get(f"http://{rev.host}/api/ps", timeout=5).json()
                        for _m in _ps.get("models", []):
                            if _m.get("name") == rev.model_name:
                                _vram = _m.get("size_vram", 0)
                                if _vram and _vram > 0:
                                    gpu_label = "GPU (Vulkan/Intel Arc)"
                                else:
                                    gpu_label = "CPU"
                                break
                    except Exception:
                        gpu_label = "GPU?"  # 无法确认，标记为可能 GPU
                    backend_label = f"Ollama {gpu_label} ({rev.model_name})"
                else:
                    backend_label = (
                        f"OpenVINO {device} INT4 "
                        f"({os.path.basename(model_dir) if model_dir else 'auto'})"
                    )
                prompt_tokens = (
                    sum(estimate_tokens_zh(c.excerpt + c.context[:200]) for c in cands)
                    + 200
                )
            except Exception as e:
                backend_label = f"复核失败降级: {type(e).__name__}"
                print(
                    f"WARN: 语义复核失败，降级为纯规则引擎: {e}",
                    file=sys.stderr,
                )
        elif model_dir:
            backend_label = "未找到模型（仅规则引擎）"
        elif reviewer_backend == "ollama":
            backend_label = "Ollama GPU 未启用（无候选或连接失败）"

    # ---- Stage 3：结构化质检报告 ----
    asr_stats = s1.asr_stats
    if asr_stats and asr_stats.get("ctc_device"):
        # Complete ASR on NPU: both encoder and CTC head on same device
        asr_backend_label = f"funasr+OpenVINO({s1.asr_device}, Encoder+CTC 双NPU组件)"
    else:
        asr_backend_label = f"funasr+OpenVINO({s1.asr_device})"
    report = build_report(
        s1,
        verdicts=verdicts,
        reviewer_backend=backend_label,
        asr_backend=asr_backend_label,
        review_prompt_tokens=prompt_tokens,
        review_s=review_s,
    )

    # ---- Stage 4 (optional): Cloud enhancement with desensitization ----
    cloud_result = {}
    if cloud_enhance:
        try:
            from cloud_enhance import run_cloud_enhance

            cloud_result = run_cloud_enhance(report, enabled=True)
            if cloud_result.get("status") == "success":
                # Append cloud coaching advice to the report's advisory section
                advice = cloud_result.get("advice", [])
                if advice:
                    cloud_advice_text = "\n".join(
                        f"- [Cloud Coaching] {item}" for item in advice
                    )
                    # Store cloud advice in report for rendering
                    report.cloud_advice = cloud_advice_text
                    report.cloud_enhance_meta = cloud_result
        except Exception as e:
            cloud_result = {
                "enabled": True,
                "status": "degraded",
                "error": f"Cloud enhancement error: {type(e).__name__}: {e}",
            }

    return PipelineResult(
        stage1=s1,
        report=report,
        verdicts=verdicts,
        reviewer_backend=backend_label,
        review_s=review_s,
        review_prompt_tokens=prompt_tokens,
        cloud_enhance=cloud_result,
    )


if __name__ == "__main__":
    import json

    audio = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "demo",
            "samples",
            "demo_finance_violation.wav",
        )
    )
    result = run_stage1(audio)
    print(json.dumps(stage1_report(result), ensure_ascii=False, indent=1))
