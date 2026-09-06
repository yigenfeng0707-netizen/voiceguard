"""模型一键下载与量化脚本 - 调用 optimum-cli 导出 OpenVINO 版本并量化。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SetupReport:
    model_paths: Dict[str, str] = field(default_factory=dict)
    step_durations: Dict[str, float] = field(default_factory=dict)
    validations: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


_MODELS = {
    "asr": {
        "source": "iic/SenseVoiceSmall",
        "task": "automatic-speech-recognition",
        "dir": "SenseVoiceSmall",
    },
    "nlp": {
        "source": "Qwen/Qwen3-1.7B",
        "task": "text-generation",
        "dir": "Qwen3-1.7B",
    },
    "nlp_4b": {
        "source": "Qwen/Qwen3-4B",
        "task": "text-generation",
        "dir": "Qwen3-4B",
    },
}


def _check_openvino() -> bool:
    try:
        import openvino

        return True
    except ImportError:
        return False


def _run_cmd(cmd: List[str], timeout: int = 600) -> tuple:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as exc:
        return -1, "", str(exc)


def _export_model(source: str, task: str, output_path: str, quant: str = "") -> bool:
    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        source,
        "--task",
        task,
        "--trust-remote-code",
        "--output",
        output_path,
    ]
    if quant in ("int8", "int4"):
        cmd.extend(["--weight-format", quant])
    elif quant == "fp16":
        cmd.extend(["--weight-format", "fp16"])
    code, out, err = _run_cmd(cmd)
    return code == 0


def _validate_model(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        core = Core()
        model = core.read_model(os.path.join(path, "openvino_model.xml"))
        return model is not None
    except Exception:
        return os.path.exists(path)


def main(
    models: str = "asr,nlp",
    quant: str = "int4",
    output_dir: str = "models",
    force: bool = False,
) -> SetupReport:
    report = SetupReport()

    if not _check_openvino():
        report.errors.append(
            "ERR_OPENVINO_MISSING: OpenVINO not installed. Run: pip install openvino"
        )
        print("ERROR: OpenVINO not installed. Run: pip install openvino")
        return report

    os.makedirs(output_dir, exist_ok=True)
    model_list = [m.strip() for m in models.split(",")]
    quant_list = [q.strip() for q in quant.split(",")]

    for model_key in model_list:
        if model_key not in _MODELS:
            report.warnings.append(f"Unknown model: {model_key}")
            continue

        info = _MODELS[model_key]
        for q in quant_list:
            tag = f"{model_key}_{q}"
            output_path = os.path.join(output_dir, info["dir"])

            if os.path.exists(output_path) and not force:
                report.model_paths[tag] = output_path
                report.validations[tag] = _validate_model(output_path)
                print(f"[SKIP] {tag} already exists at {output_path}")
                continue

            print(f"[EXPORT] {tag}: {info['source']} -> {output_path}")
            t0 = time.time()
            success = _export_model(info["source"], info["task"], output_path, q)
            elapsed = time.time() - t0
            report.step_durations[tag] = elapsed

            if success:
                report.model_paths[tag] = output_path
                report.validations[tag] = _validate_model(output_path)
                print(
                    f"[OK] {tag} exported in {elapsed:.1f}s, validation: {report.validations[tag]}"
                )
            else:
                if q == "int8":
                    report.warnings.append(
                        f"INT8 quantization failed for {model_key}, trying FP16"
                    )
                    success = _export_model(
                        info["source"], info["task"], output_path, "fp16"
                    )
                    if success:
                        report.model_paths[tag] = output_path
                        report.validations[tag] = _validate_model(output_path)
                        print(f"[FALLBACK] {tag} exported as FP16")
                    else:
                        report.errors.append(f"ERR_QUANT_FAILED: {tag} export failed")
                else:
                    report.errors.append(f"ERR_QUANT_FAILED: {tag} export failed")

    print(
        f"\nSetup complete: {len(report.model_paths)} models, {len(report.errors)} errors"
    )
    if report.warnings:
        print(f"Warnings: {report.warnings}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model download and quantization")
    parser.add_argument(
        "--models",
        default="asr,nlp",
        help="Comma-separated model keys: asr, nlp, nlp_4b",
    )
    parser.add_argument(
        "--quant", default="int4", help="Quantization level: int4, int8, or fp16"
    )
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force re-export")
    args = parser.parse_args()
    main(args.models, args.quant, args.output_dir, args.force)
