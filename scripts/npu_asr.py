"""NPU ASR Encoder: OpenVINO IR on Intel AI Boost NPU.

SenseVoiceSmall encoder converted to OpenVINO IR, compiled on NPU with
static shape (100 frames). Variable-length audio handled via chunked
inference with 11-frame overlap (SANM attention window).

Device priority: NPU > GPU > CPU
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Static shape constants for NPU compilation
_CHUNK_FRAMES = 100  # NPU-verified static shape
_MEL_BINS = 560  # SenseVoiceSmall mel bins
_OVERLAP = 11  # SANM attention kernel size
_STRIDE = _CHUNK_FRAMES - 2 * _OVERLAP  # 78 frames per stride

_ENCODER_IR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "SenseVoiceSmall_ov",
    "encoder.xml",
)


@dataclass
class NpuStats:
    """Inference statistics for NPU/GPU/CPU comparison."""

    device: str = ""
    compile_time_s: float = 0.0
    total_infer_s: float = 0.0
    chunk_count: int = 0
    frames_processed: int = 0
    avg_chunk_ms: float = 0.0
    rtf: float = 0.0  # Real-Time Factor
    audio_duration_s: float = 0.0
    warnings: List[str] = field(default_factory=list)


class OpenVINOEncoder:
    """OpenVINO-compiled SenseVoiceSmall encoder for NPU/GPU/CPU.

    Replaces funasr SenseVoiceEncoderSmall.forward via monkey-patch.
    Handles variable-length input by chunking with overlap.
    """

    def __init__(self, ir_path: str = _ENCODER_IR, device: str = "NPU"):
        import openvino
        from openvino import PartialShape

        self.device = device.upper()
        self.core = openvino.Core()
        self._ir_path = ir_path

        # Read and reshape to static
        model = self.core.read_model(ir_path)
        model.reshape(
            {
                0: PartialShape([1, _CHUNK_FRAMES, _MEL_BINS]),
                1: PartialShape([1]),
            }
        )

        t0 = time.perf_counter()
        self._compiled = self.core.compile_model(model, self.device)
        self._compile_time = time.perf_counter() - t0

        # Stats tracking
        self._infer_time = 0.0
        self._chunk_count = 0
        self._total_frames = 0

    @property
    def compile_time(self) -> float:
        return self._compile_time

    def _run_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Run one 100-frame chunk through the compiled model."""
        ilens = np.array([_CHUNK_FRAMES], dtype=np.int64)
        t0 = time.perf_counter()
        result = self._compiled({0: chunk, 1: ilens})
        self._infer_time += time.perf_counter() - t0
        self._chunk_count += 1
        return result[self._compiled.output(0)]

    def __call__(self, xs_pad, ilens):
        """Encoder forward replacement.

        Args:
            xs_pad: torch.Tensor [batch=1, frames, mel_bins=560]
            ilens: torch.Tensor [batch=1]

        Returns:
            (encoder_out, olens) as torch.Tensor, matching original interface.
        """
        import torch

        frames = xs_pad.shape[1]
        self._total_frames = frames
        self._infer_time = 0.0
        self._chunk_count = 0

        xs_np = xs_pad.detach().cpu().numpy().astype(np.float32)

        if frames <= _CHUNK_FRAMES:
            # Single chunk: pad to _CHUNK_FRAMES
            padded = np.zeros((1, _CHUNK_FRAMES, _MEL_BINS), dtype=np.float32)
            padded[:, :frames, :] = xs_np[:, :frames, :]
            out = self._run_chunk(padded)
            encoder_out = out[:, :frames, :]
        else:
            # Chunked inference with overlap
            outputs: List[np.ndarray] = []
            pos = 0
            while pos < frames:
                end = min(pos + _CHUNK_FRAMES, frames)
                actual = end - pos

                chunk = np.zeros((1, _CHUNK_FRAMES, _MEL_BINS), dtype=np.float32)
                chunk[:, :actual, :] = xs_np[:, pos:end, :]

                out = self._run_chunk(chunk)

                # Determine slice boundaries (remove overlap)
                slice_start = _OVERLAP if pos > 0 else 0
                if end < frames:
                    slice_end = actual - _OVERLAP
                else:
                    slice_end = actual

                outputs.append(out[:, slice_start:slice_end, :])
                pos += _STRIDE if end < frames else _STRIDE

            encoder_out = np.concatenate(outputs, axis=1)

        # Construct olens from ilens (proportional adjustment)
        ratio = encoder_out.shape[1] / max(frames, 1)
        olens_np = (ilens.detach().cpu().numpy().astype(np.float32) * ratio).astype(
            np.int32
        )
        olens = torch.from_numpy(olens_np)

        encoder_out_t = torch.from_numpy(encoder_out).to(xs_pad.dtype)
        return encoder_out_t, olens

    def get_stats(self, audio_duration_s: float = 0.0) -> NpuStats:
        rtf = self._infer_time / audio_duration_s if audio_duration_s > 0 else 0.0
        avg_chunk = (
            (self._infer_time / self._chunk_count * 1000) if self._chunk_count else 0
        )
        return NpuStats(
            device=self.device,
            compile_time_s=self._compile_time,
            total_infer_s=self._infer_time,
            chunk_count=self._chunk_count,
            frames_processed=self._total_frames,
            avg_chunk_ms=avg_chunk,
            rtf=rtf,
            audio_duration_s=audio_duration_s,
        )


def patch_encoder(funasr_model, device: str = "NPU", ir_path: str = _ENCODER_IR):
    """Monkey-patch funasr model's encoder to use OpenVINO on specified device.

    Args:
        funasr_model: funasr.AutoModel instance
        device: "NPU", "GPU", or "CPU"
        ir_path: Path to encoder.xml

    Returns:
        OpenVINOEncoder instance with stats tracking
    """
    encoder = OpenVINOEncoder(ir_path=ir_path, device=device)

    # Save original forward for fallback
    original_forward = funasr_model.model.encoder.forward

    def patched_forward(self_inner, xs_pad, ilens):
        return encoder(xs_pad, ilens)

    funasr_model.model.encoder.forward = types.MethodType(
        patched_forward, funasr_model.model.encoder
    )
    funasr_model.model.encoder._original_forward = original_forward
    funasr_model.model.encoder._ov_encoder = encoder

    return encoder


def unpatch_encoder(funasr_model):
    """Restore original encoder forward."""
    enc = funasr_model.model.encoder
    if hasattr(enc, "_original_forward"):
        enc.forward = enc._original_forward
        delattr(enc, "_original_forward")
        delattr(enc, "_ov_encoder")


def select_device(preference: str = "auto") -> str:
    """Select best available OpenVINO device for ASR encoder.

    Priority: NPU > GPU > CPU
    """
    try:
        import openvino

        core = openvino.Core()
        devices = [d.upper() for d in core.available_devices]
    except Exception:
        return "CPU"

    if preference.upper() != "AUTO":
        pref = preference.upper()
        if pref in devices:
            return pref
        # Fall through to auto

    for dev in ["NPU", "GPU", "CPU"]:
        if dev in devices:
            return dev
    return "CPU"
