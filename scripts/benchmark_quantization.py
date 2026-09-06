"""量化对比 Benchmark - FP32/FP16/INT8 性能对比。"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class QuantResult:
    quantization: str
    duration: float = 0.0
    memory_mb: float = 0.0
    success: bool = False


def _run_with_quant(
    sample_audio: str, quant: str, backend: str, output_dir: str
) -> QuantResult:
    env = os.environ.copy()
    env["BACKEND_PREFERENCE"] = backend
    env["OUTPUT_DIR"] = output_dir
    env["ASR_QUANTIZATION"] = quant

    cmd = [sys.executable, "scripts/run.py", sample_audio, "--format", "json"]
    t0 = time.time()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        elapsed = time.time() - t0
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0
        return QuantResult(
            quant, elapsed, max(mem_before, mem_after), result.returncode == 0
        )
    except Exception as exc:
        return QuantResult(quant, time.time() - t0, mem_before, False)


def main(
    sample_audio: str = "demo/samples/demo_finance_violation.wav",
    backend: str = "cpu",
) -> List[QuantResult]:
    results: List[QuantResult] = []
    for quant in ["fp32", "fp16", "int8"]:
        print(f"[BENCHMARK] {quant} on {backend}")
        output_dir = f"reports/quant_output/{quant}"
        os.makedirs(output_dir, exist_ok=True)
        r = _run_with_quant(sample_audio, quant, backend, output_dir)
        results.append(r)
        print(
            f"  Duration: {r.duration:.2f}s, Memory: {r.memory_mb:.1f}MB, Success: {r.success}"
        )

    os.makedirs("reports", exist_ok=True)
    lines = [
        "# 量化对比 Benchmark 报告",
        "",
        f"**样例音频**: `{sample_audio}`",
        f"**后端**: {backend}",
        f"**操作系统**: {platform.system()} {platform.release()}",
        "",
        "## 对比数据",
        "",
        "| 量化档位 | 耗时(秒) | 内存峰值(MB) | 成功 |",
        "|---------|---------|-------------|------|",
    ]
    for r in results:
        lines.append(
            f"| {r.quantization} | {r.duration:.2f} | {r.memory_mb:.1f} | {'是' if r.success else '否'} |"
        )

    lines.extend(["", "## 结论", ""])
    successful = [r for r in results if r.success]
    if len(successful) >= 2:
        fastest = min(successful, key=lambda r: r.duration)
        lines.append(f"最快量化档位: {fastest.quantization} ({fastest.duration:.2f}s)")
    lines.append(
        "注: INT8 耗时通常 ≤ FP16 ≤ FP32，若反常可能与模型结构或硬件支持有关。"
    )

    with open("reports/benchmark_quantization.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to reports/benchmark_quantization.md")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Benchmark")
    parser.add_argument(
        "--sample-audio", default="demo/samples/demo_finance_violation.wav"
    )
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    main(args.sample_audio, args.backend)
