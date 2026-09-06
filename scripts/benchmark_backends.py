"""三后端实测 Benchmark - NPU/GPU/CPU 真实性能对比，禁止模拟数据。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False


@dataclass
class BenchmarkResult:
    device: str
    repeat: int
    durations: List[float] = field(default_factory=list)
    memory_peaks: List[float] = field(default_factory=list)
    stages: Dict[str, List[float]] = field(default_factory=dict)
    realtime_ratios: List[float] = field(default_factory=list)
    timestamp: str = ""


def _detect_devices() -> List[str]:
    devices = ["cpu"]
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        core = Core()
        available = [d.lower() for d in core.available_devices]
        devices = sorted(
            available, key=lambda d: {"npu": 0, "gpu": 1, "cpu": 2}.get(d, 99)
        )
    except Exception:
        pass
    return devices


def _get_memory_peak() -> float:
    if _PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return 0.0


def _run_cli(sample_audio: str, backend: str, output_dir: str) -> dict:
    env = os.environ.copy()
    env["BACKEND_PREFERENCE"] = backend
    env["OUTPUT_DIR"] = output_dir

    cmd = [
        sys.executable,
        "scripts/cli.py",
        sample_audio,
        "--format",
        "json",
        "--output-dir",
        output_dir,
    ]
    t0 = time.time()
    mem_before = _get_memory_peak()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        elapsed = time.time() - t0
        mem_after = _get_memory_peak()
        return {
            "success": result.returncode == 0,
            "duration": elapsed,
            "memory_peak": max(mem_before, mem_after),
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:500],
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": time.time() - t0,
            "memory_peak": mem_before,
            "stdout": "",
            "stderr": "Timeout",
        }
    except Exception as exc:
        return {
            "success": False,
            "duration": time.time() - t0,
            "memory_peak": mem_before,
            "stdout": "",
            "stderr": str(exc),
        }


def _get_audio_duration(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def _generate_report(
    results: List[BenchmarkResult], sample_audio: str, output_path: str
) -> None:
    import datetime

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# 异构加速 Benchmark 报告",
        "",
        f"**生成时间**: {datetime.datetime.now().isoformat()}",
        f"**样例音频**: `{sample_audio}`",
        f"**操作系统**: {platform.system()} {platform.release()}",
        f"**Python 版本**: {platform.python_version()}",
        "",
        "## 环境信息",
        "",
        f"- OS: {platform.system()} {platform.release()}",
        f"- Python: {platform.python_version()}",
    ]

    try:
        import openvino

        lines.append(f"- OpenVINO: {openvino.__version__}")
    except Exception:
        lines.append("- OpenVINO: 未安装")

    lines.extend(["", "## 原始数据", ""])

    for r in results:
        lines.append(f"### {r.device.upper()} (重复 {r.repeat} 次)")
        lines.append("")
        lines.append(f"- 时间戳: {r.timestamp}")
        lines.append(f"- 耗时(秒): {r.durations}")
        lines.append(f"- 内存峰值(MB): {[round(m, 2) for m in r.memory_peaks]}")
        if r.realtime_ratios:
            lines.append(f"- 实时率: {[round(rt, 2) for rt in r.realtime_ratios]}")
        lines.append("")

    lines.append("## 汇总对比")
    lines.append("")
    lines.append("| 后端 | 平均耗时(秒) | 最小耗时(秒) | 最大耗时(秒) | 平均内存(MB) |")
    lines.append("|------|-------------|-------------|-------------|-------------|")
    for r in results:
        if r.durations:
            avg = sum(r.durations) / len(r.durations)
            mn = min(r.durations)
            mx = max(r.durations)
            avg_mem = sum(r.memory_peaks) / len(r.memory_peaks) if r.memory_peaks else 0
            lines.append(
                f"| {r.device.upper()} | {avg:.2f} | {mn:.2f} | {mx:.2f} | {avg_mem:.1f} |"
            )

    lines.extend(["", "## 结论摘要", ""])
    if len(results) >= 2 and all(r.durations for r in results):
        fastest = min(results, key=lambda r: sum(r.durations) / len(r.durations))
        slowest = max(results, key=lambda r: sum(r.durations) / len(r.durations))
        if fastest.device != slowest.device:
            speedup = (sum(slowest.durations) / len(slowest.durations)) / (
                sum(fastest.durations) / len(fastest.durations)
            )
            lines.append(
                f"{fastest.device.upper()} 比 {slowest.device.upper()} 快 {speedup:.1f} 倍。"
            )
    if len(results) == 1 and results[0].device == "cpu":
        lines.append("本机无 NPU / GPU 可对比，仅输出 CPU 实测数据。")

    lines.extend(["", "## 复现步骤", ""])
    lines.append("```bash")
    lines.append(f"python scripts/benchmark_backends.py --sample-audio {sample_audio}")
    lines.append("```")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(
    sample_audio: str = "demo/samples/single_zh.wav",
    backends: str = "auto",
    repeat: int = 3,
) -> List[BenchmarkResult]:
    if not os.path.exists(sample_audio):
        print(f"ERROR: Sample audio not found: {sample_audio}")
        return []

    devices = (
        _detect_devices()
        if backends == "auto"
        else [b.strip() for b in backends.split(",")]
    )
    audio_duration = _get_audio_duration(sample_audio)
    results: List[BenchmarkResult] = []

    for device in devices:
        print(f"\n[BENCHMARK] {device.upper()} x{repeat}")
        result = BenchmarkResult(
            device=device, repeat=repeat, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )

        for i in range(repeat):
            output_dir = f"reports/benchmark_output/{device}/run_{i}"
            os.makedirs(output_dir, exist_ok=True)
            r = _run_cli(sample_audio, device, output_dir)
            result.durations.append(r["duration"])
            result.memory_peaks.append(r["memory_peak"])
            if audio_duration > 0 and r["duration"] > 0:
                result.realtime_ratios.append(audio_duration / r["duration"])
            print(
                f"  Run {i + 1}: {r['duration']:.2f}s, mem={r['memory_peak']:.1f}MB, success={r['success']}"
            )

        if len(result.durations) >= 2:
            avg = sum(result.durations) / len(result.durations)
            variance = sum((d - avg) ** 2 for d in result.durations) / len(
                result.durations
            )
            std_dev = variance**0.5
            if avg > 0 and (std_dev / avg) > 0.15:
                print(f"  [WARNING] Variance > 15%, adding 2 more runs")
                for extra in range(2):
                    output_dir = (
                        f"reports/benchmark_output/{device}/run_{repeat + extra}"
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    r = _run_cli(sample_audio, device, output_dir)
                    result.durations.append(r["duration"])
                    result.memory_peaks.append(r["memory_peak"])
                    if audio_duration > 0 and r["duration"] > 0:
                        result.realtime_ratios.append(audio_duration / r["duration"])

        results.append(result)

    _generate_report(results, sample_audio, "reports/benchmark_backends.md")
    print(f"\nReport saved to reports/benchmark_backends.md")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend Benchmark")
    parser.add_argument("--sample-audio", default="demo/samples/single_zh.wav")
    parser.add_argument("--backends", default="auto")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    main(args.sample_audio, args.backends, args.repeat)
