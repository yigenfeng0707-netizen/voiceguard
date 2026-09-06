"""Ollama vs OpenVINO 复核后端对比基准。

对比维度：
1. 推理延迟（infer_s）
2. 输出 token 数量
3. 吞吐量（tokens/s）
4. JSON 解析成功率
5. 判定准确率（与标注对比）

用法：
    D:/miniconda3/python.exe scripts/benchmark_ollama.py [--no-openvino]
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reviewer import (
    ReviewCandidate,
    ReviewVerdict,
    LocalReviewer,
    OllamaReviewer,
)

# 标注真值（来自 5 场景 E2E 验证基线）
LABELED_CANDIDATES = [
    # 金融违规场景
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
    ReviewCandidate(
        index=2,
        rule_id="FIN-002",
        rule_name="夸大宣传",
        semantic_hint="使用极限词或夸大产品效果",
        excerpt="这是市面上最好的理财产品，没有任何一款能比得上",
        context="坐席：这是市面上最好的理财产品，没有任何一款能比得上。",
    ),
    ReviewCandidate(
        index=3,
        rule_id="FIN-003",
        rule_name="诱导隐瞒风险",
        semantic_hint="未充分提示投资风险",
        excerpt="放心买就行，不用管那些风险提示书",
        context="坐席：放心买就行，不用管那些风险提示书。客户：那风险提示书不用签吗？坐席：不用不用，签了反而麻烦。",
    ),
    ReviewCandidate(
        index=4,
        rule_id="FIN-004",
        rule_name="违规推荐",
        semantic_hint="向风险承受能力不匹配的客户推荐产品",
        excerpt="虽然您是保守型投资者，但这款激进型基金收益高，特别适合您",
        context="坐席：虽然您是保守型投资者，但这款激进型基金收益高，特别适合您。",
    ),
    ReviewCandidate(
        index=5,
        rule_id="FIN-001",
        rule_name="承诺保本保收益",
        semantic_hint="坐席对产品作出保本保收益承诺",
        excerpt="根据历史数据，该产品过去五年平均年化收益率为6%",
        context="坐席：根据历史数据，该产品过去五年平均年化收益率为6%。客户：那未来呢？坐席：历史业绩不代表未来表现。",
    ),
    ReviewCandidate(
        index=6,
        rule_id="FIN-005",
        rule_name="误导性陈述",
        semantic_hint="歪曲产品性质或运作方式",
        excerpt="这个基金跟银行存款一样安全，随时可以取",
        context="坐席：这个基金跟银行存款一样安全，随时可以取。",
    ),
    ReviewCandidate(
        index=7,
        rule_id="FIN-006",
        rule_name="适当性缺失",
        semantic_hint="未进行风险承受能力评估",
        excerpt="风险测评不用做，我们直接帮您买了",
        context="坐席：风险测评不用做，我们直接帮您买了。客户：不需要评估吗？坐席：不需要，我帮您搞定。",
    ),
]

# 真值标注
GROUND_TRUTH = {
    0: True,  # 保本保息+30% → 违规
    1: False,  # 存款利率2%+存款保险 → 合规
    2: True,  # "最好的"极限词 → 违规
    3: True,  # 不签风险提示书 → 违规
    4: True,  # 保守型推荐激进型 → 违规
    5: False,  # 历史数据+风险提示 → 合规
    6: True,  # 基金跟存款一样安全 → 违规
    7: True,  # 不做风险测评 → 违规
}


def run_openvino_benchmark(device: str = "GPU") -> dict:
    """OpenVINO Qwen3-1.7B INT4 GPU 基准。"""
    model_dir = r"D:\vg_ov\Qwen3-1.7B-ov-int4"
    if not os.path.isfile(os.path.join(model_dir, "openvino_model.xml")):
        return {"error": f"OpenVINO model not found: {model_dir}"}

    print(f"[OpenVINO] Loading model from {model_dir} ({device})...")
    t0 = time.perf_counter()
    try:
        rev = LocalReviewer(model_dir, device=device, batch_size=10).load()
    except Exception as e:
        return {"error": f"Load failed: {e}"}
    load_s = round(time.perf_counter() - t0, 2)
    print(f"[OpenVINO] Model loaded in {load_s}s")

    print(f"[OpenVINO] Reviewing {len(LABELED_CANDIDATES)} candidates...")
    t0 = time.perf_counter()
    verdicts = rev.review(LABELED_CANDIDATES)
    infer_s = round(time.perf_counter() - t0, 2)
    print(f"[OpenVINO] Inference done in {infer_s}s, {rev.last_output_tokens} tokens")

    correct = 0
    for v in verdicts:
        truth = GROUND_TRUTH.get(v.index, None)
        if truth is not None and v.confirmed == truth:
            correct += 1

    return {
        "backend": f"OpenVINO {device} INT4 (Qwen3-1.7B)",
        "load_s": load_s,
        "infer_s": rev.last_infer_s,
        "total_s": infer_s,
        "output_tokens": rev.last_output_tokens,
        "tokens_per_s": round(rev.last_output_tokens / max(rev.last_infer_s, 0.01), 1),
        "correct": correct,
        "total": len(LABELED_CANDIDATES),
        "accuracy": round(correct / len(LABELED_CANDIDATES) * 100, 1),
        "verdicts": [
            {"index": v.index, "confirmed": v.confirmed, "reason": v.reason}
            for v in verdicts
        ],
    }


def run_ollama_benchmark(model_name: str = "qwen2.5:3b") -> dict:
    """Ollama qwen2.5:3b GPU(Vulkan/Intel Arc) 或 CPU 基准。"""
    import requests

    print(f"[Ollama] Connecting to model {model_name}...")
    t0 = time.perf_counter()
    try:
        rev = OllamaReviewer(model_name=model_name, batch_size=10).load()
    except Exception as e:
        return {"error": f"Load failed: {e}"}
    load_s = round(time.perf_counter() - t0, 2)
    print(f"[Ollama] Model ready in {load_s}s")

    # Detect GPU acceleration
    gpu_label = "CPU"
    vram_mb = 0
    try:
        ps = requests.get(f"http://{rev.host}/api/ps", timeout=5).json()
        for m in ps.get("models", []):
            if m.get("name") == model_name:
                vram_mb = round(m.get("size_vram", 0) / 1024**2, 1)
                if vram_mb and vram_mb > 0:
                    gpu_label = "GPU (Vulkan/Intel Arc)"
                break
    except Exception:
        pass
    print(f"[Ollama] Backend: {gpu_label}, VRAM: {vram_mb} MB")

    print(f"[Ollama] Reviewing {len(LABELED_CANDIDATES)} candidates...")
    t0 = time.perf_counter()
    verdicts = rev.review(LABELED_CANDIDATES)
    infer_s = round(time.perf_counter() - t0, 2)
    print(f"[Ollama] Inference done in {infer_s}s, {rev.last_output_tokens} tokens")

    correct = 0
    for v in verdicts:
        truth = GROUND_TRUTH.get(v.index, None)
        if truth is not None and v.confirmed == truth:
            correct += 1

    tok_s = round(rev.last_output_tokens / max(rev.last_infer_s, 0.01), 1)
    return {
        "backend": f"Ollama {gpu_label} ({model_name})",
        "gpu_enabled": gpu_label != "CPU",
        "vram_mb": vram_mb,
        "load_s": load_s,
        "infer_s": rev.last_infer_s,
        "total_s": infer_s,
        "output_tokens": rev.last_output_tokens,
        "tokens_per_s": tok_s,
        "correct": correct,
        "total": len(LABELED_CANDIDATES),
        "accuracy": round(correct / len(LABELED_CANDIDATES) * 100, 1),
        "verdicts": [
            {"index": v.index, "confirmed": v.confirmed, "reason": v.reason}
            for v in verdicts
        ],
    }


def main():
    no_openvino = "--no-openvino" in sys.argv
    results = {}

    if not no_openvino:
        print("=" * 60)
        print("Benchmark 1/2: OpenVINO Qwen3-1.7B (GPU INT4)")
        print("=" * 60)
        results["openvino"] = run_openvino_benchmark("GPU")

    print()
    print("=" * 60)
    print("Benchmark 2/2: Ollama qwen2.5:3b (GPU Vulkan/Intel Arc)")
    print("=" * 60)
    results["ollama_gpu"] = run_ollama_benchmark("qwen2.5:3b")

    # 汇总
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for key, r in results.items():
        if "error" in r:
            print(f"  {key}: ERROR - {r['error']}")
            continue
        print(f"  {r['backend']}:")
        print(f"    Accuracy: {r['correct']}/{r['total']} ({r['accuracy']}%)")
        print(
            f"    Infer: {r['infer_s']}s, {r['output_tokens']} tokens, {r['tokens_per_s']} tok/s"
        )

    # 保存结果
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ollama_gpu_benchmark_full.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
