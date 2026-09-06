"""NPU CTC Head: OpenVINO IR on Intel AI Boost NPU.

SenseVoiceSmall CTC decoder head (ctc.ctc_lo: Linear 512->25055, 12.8M params)
converted to OpenVINO IR (FP32 precision), compiled on NPU with static shape.

Unlike the encoder (which needs chunked inference for variable-length audio),
the CTC head is a simple linear projection -- the input shape is determined
by the encoder output length. We compile with a fixed shape matching the
encoder chunk output and pad/truncate as needed.

Note: FP32 precision is required -- FP16 (compress_to_fp16=True) causes
unacceptable numerical drift (max diff ~14) for the 25055-dim projection.
FP32 achieves max diff <0.01 vs PyTorch reference.

Architecture: ASR encoder (NPU) -> CTC decoder (NPU) -> tokens (CPU)
Both ASR model components now run on NPU, forming a complete "ASR on NPU" story.
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Static shape constants for NPU compilation
# CTC head input: (batch=1, frames, hidden_dim=512)
# Encoder outputs ~100 frames per 100-frame mel chunk, so we use 100
# FP32 IR required (FP16 causes max diff ~14 for 25055-dim projection)
_CTC_MAX_FRAMES = 100
_CTC_HIDDEN_DIM = 512
_CTC_VOCAB_SIZE = 25055

_CTC_IR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "SenseVoiceSmall_ov",
    "ctc_head_fp32.xml",
)


@dataclass
class CtcNpuStats:
    """Statistics for CTC head NPU inference."""

    device: str = ""
    compile_time_s: float = 0.0
    total_infer_s: float = 0.0
    call_count: int = 0
    avg_latency_ms: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class OpenVINOCTCHead:
    """OpenVINO-compiled CTC linear head for NPU/GPU/CPU.

    Replaces funasr CTC ctc_lo.forward via monkey-patch.
    Handles variable-length encoder output by padding to static max_frames.
    """

    def __init__(self, ir_path: str = _CTC_IR, device: str = "NPU"):
        import openvino
        from openvino import PartialShape

        self.device = device.upper()
        self.core = openvino.Core()
        self._ir_path = ir_path
        self._max_frames = _CTC_MAX_FRAMES

        # Read and reshape to static
        model = self.core.read_model(ir_path)
        model.reshape(
            {
                0: PartialShape([1, _CTC_MAX_FRAMES, _CTC_HIDDEN_DIM]),
            }
        )

        t0 = time.perf_counter()
        self._compiled = self.core.compile_model(model, self.device)
        self._compile_time = time.perf_counter() - t0

        # Stats tracking
        self._infer_time = 0.0
        self._call_count = 0
        self._warnings = []

    @property
    def compile_time(self) -> float:
        return self._compile_time

    def _run(self, x: np.ndarray) -> np.ndarray:
        """Run CTC head on one (1, frames, 512) input, padding to max_frames."""
        frames = x.shape[1]

        if frames > self._max_frames:
            # Chunk if exceeds max (rare, but handle gracefully)
            outputs = []
            pos = 0
            while pos < frames:
                end = min(pos + self._max_frames, frames)
                chunk = x[:, pos:end, :]
                actual = end - pos
                padded = np.zeros(
                    (1, self._max_frames, _CTC_HIDDEN_DIM), dtype=np.float32
                )
                padded[:, :actual, :] = chunk
                out = self._run_one(padded)
                outputs.append(out[:, :actual, :])
                pos = end
            return np.concatenate(outputs, axis=1)

        # Pad to max_frames
        padded = np.zeros((1, self._max_frames, _CTC_HIDDEN_DIM), dtype=np.float32)
        padded[:, :frames, :] = x[:, :frames, :]
        out = self._run_one(padded)
        return out[:, :frames, :]

    def _run_one(self, padded: np.ndarray) -> np.ndarray:
        """Run one padded chunk through the compiled model."""
        t0 = time.perf_counter()
        result = self._compiled({0: padded})
        self._infer_time += time.perf_counter() - t0
        self._call_count += 1
        return result[self._compiled.output(0)]

    def __call__(self, x):
        """CTC head forward replacement.

        Args:
            x: torch.Tensor [batch=1, frames, hidden_dim=512]

        Returns:
            torch.Tensor [batch=1, frames, vocab_size=25055]
        """
        import torch

        x_np = x.detach().cpu().numpy().astype(np.float32)
        out_np = self._run(x_np)
        return torch.from_numpy(out_np).to(x.dtype)

    def get_stats(self) -> CtcNpuStats:
        avg = (self._infer_time / self._call_count * 1000) if self._call_count else 0
        return CtcNpuStats(
            device=self.device,
            compile_time_s=self._compile_time,
            total_infer_s=self._infer_time,
            call_count=self._call_count,
            avg_latency_ms=avg,
            warnings=list(self._warnings),
        )


def patch_ctc_head(funasr_model, device: str = "NPU", ir_path: str = _CTC_IR):
    """Monkey-patch funasr model's CTC head to use OpenVINO on specified device.

    Args:
        funasr_model: funasr.AutoModel instance
        device: "NPU", "GPU", or "CPU"
        ir_path: Path to ctc_head.xml

    Returns:
        OpenVINOCTCHead instance with stats tracking, or None if patch failed
    """
    if not os.path.isfile(ir_path):
        return None

    try:
        ctc_head = OpenVINOCTCHead(ir_path=ir_path, device=device)
    except Exception as e:
        return None

    # Save original forward
    ctc_lo = funasr_model.model.ctc.ctc_lo
    original_forward = ctc_lo.forward

    def patched_forward(self_inner, x):
        return ctc_head(x)

    ctc_lo.forward = types.MethodType(patched_forward, ctc_lo)
    ctc_lo._original_forward = original_forward
    ctc_lo._ov_ctc_head = ctc_head

    return ctc_head


def unpatch_ctc_head(funasr_model):
    """Restore original CTC head forward."""
    ctc_lo = funasr_model.model.ctc.ctc_lo
    if hasattr(ctc_lo, "_original_forward"):
        ctc_lo.forward = ctc_lo._original_forward
        delattr(ctc_lo, "_original_forward")
        delattr(ctc_lo, "_ov_ctc_head")
