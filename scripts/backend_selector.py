"""异构推理后端选择器 - OpenVINO(NPU>GPU>CPU) → ONNXRuntime → PyTorch CPU。

D2 升级：新增 ASR encoder 的 NPU/GPU OpenVINO 加速支持。
    SenseVoiceSmall encoder 已转换为 OpenVINO IR，NPU 使用 100 帧静态 shape
    + 分块重叠推理（SANM 局部注意力 kernel=11）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Backend:
    device: str = "cpu"
    quantization: str = "int8"
    runtime: str = "torch"
    model_handle: object = None
    infer_fn: object = None
    degraded: bool = False
    warnings: List[str] = field(default_factory=list)


_DEVICE_PRIORITY = {"npu": 0, "gpu": 1, "cpu": 2}
_QUANT_DEFAULT = {"npu": "int8", "gpu": "fp16", "cpu": "int8"}

# ASR encoder IR path (SenseVoiceSmall converted to OpenVINO)
_ASR_ENCODER_IR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "SenseVoiceSmall_ov",
    "encoder.xml",
)


def _detect_openvino_devices() -> List[str]:
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        return [d.lower() for d in devices]
    except Exception:
        return []


def _try_load_openvino(
    model_path: str, model_kind: str, device: str, quantization: str
) -> Optional[object]:
    try:
        from optimum.intel import (
            OVModelForSpeechSeq2Seq,
            OVModelForCTC,
            OVModelForMaskedLM,
        )

        loaders = {
            "asr_seq2seq": OVModelForSpeechSeq2Seq,
            "asr_ctc": OVModelForCTC,
            "nlp": OVModelForMaskedLM,
        }
        loader = loaders.get(model_kind, OVModelForMaskedLM)
        return loader.from_pretrained(model_path, export=True, device=device)
    except Exception:
        return None


def _try_load_onnxruntime(model_path: str) -> Optional[object]:
    try:
        import onnxruntime as ort

        return ort.InferenceSession(model_path)
    except Exception:
        return None


def _try_load_torch(model_path: str, model_kind: str) -> Optional[object]:
    try:
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoModelForCTC,
            AutoModelForMaskedLM,
        )

        loaders = {
            "asr_seq2seq": AutoModelForSpeechSeq2Seq,
            "asr_ctc": AutoModelForCTC,
            "nlp": AutoModelForMaskedLM,
        }
        loader = loaders.get(model_kind, AutoModelForMaskedLM)
        return loader.from_pretrained(model_path)
    except Exception:
        return None


def select(
    model_path: str = "",
    model_kind: str = "nlp",
    device: str = "auto",
    quantization: str = "auto",
) -> Backend:
    backend = Backend()

    ov_devices = _detect_openvino_devices()
    if device == "auto":
        candidates = sorted(ov_devices, key=lambda d: _DEVICE_PRIORITY.get(d, 99))
        if not candidates:
            candidates = ["cpu"]
    else:
        if device not in ov_devices and device != "cpu":
            backend.warnings.append(
                f"ERR_BACKEND_UNAVAILABLE: {device} not available, available: {ov_devices}"
            )
            candidates = ["cpu"]
        else:
            candidates = [device]

    for dev in candidates:
        quant = (
            quantization if quantization != "auto" else _QUANT_DEFAULT.get(dev, "int8")
        )
        handle = _try_load_openvino(model_path, model_kind, dev, quant)
        if handle is not None:
            backend.device = dev
            backend.quantization = quant
            backend.runtime = "openvino"
            backend.model_handle = handle
            return backend

    handle = _try_load_onnxruntime(model_path)
    if handle is not None:
        backend.runtime = "onnxruntime"
        backend.degraded = True
        backend.warnings.append("OpenVINO unavailable, fallback to ONNXRuntime")
        backend.model_handle = handle
        return backend

    handle = _try_load_torch(model_path, model_kind)
    if handle is not None:
        backend.runtime = "torch"
        backend.degraded = True
        backend.warnings.append(
            "OpenVINO and ONNXRuntime unavailable, fallback to PyTorch CPU"
        )
        backend.model_handle = handle
        return backend

    backend.warnings.append("ERR_BACKEND_UNAVAILABLE: No backend available")
    return backend


def select_asr_encoder(
    funasr_model,
    device: str = "auto",
    ir_path: str = "",
) -> tuple:
    """Select and patch ASR encoder backend for SenseVoiceSmall.

    Args:
        funasr_model: funasr.AutoModel instance
        device: "auto"|"npu"|"gpu"|"cpu"
        ir_path: encoder.xml path (default: models/SenseVoiceSmall_ov/encoder.xml)

    Returns:
        (device_str, encoder_obj_or_none, stats_dict)
    """
    ir = ir_path or _ASR_ENCODER_IR

    if not os.path.isfile(ir):
        return "cpu", None, {"warning": f"Encoder IR not found: {ir}"}

    ov_devices = _detect_openvino_devices()

    if device.lower() == "auto":
        for dev in ["npu", "gpu", "cpu"]:
            if dev in ov_devices:
                target = dev
                break
        else:
            target = "cpu"
    else:
        target = device.lower()
        if target not in ov_devices and target != "cpu":
            return "cpu", None, {"warning": f"{device} unavailable, using CPU"}

    if target == "cpu":
        return "cpu", None, {}

    try:
        from npu_asr import patch_encoder

        encoder = patch_encoder(funasr_model, target, ir)
        return (
            target,
            encoder,
            {
                "device": target,
                "compile_time_s": round(encoder.compile_time, 2),
            },
        )
    except Exception as e:
        return "cpu", None, {"warning": f"OpenVINO encoder failed ({e}), using CPU"}
