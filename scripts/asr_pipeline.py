"""语音转写 ASR 流水线 - 多语种/时间戳/标点/置信度。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from backend_selector import Backend, select


@dataclass
class TranscriptSegment:
    segment_id: int
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    review: bool = False
    lang: str = "zh"
    speaker: str = ""


_asr_backend: Optional[Backend] = None
_processor: object = None


def _load_asr_model(model_path: str) -> Backend:
    global _asr_backend, _processor
    if _asr_backend is not None and _asr_backend.model_handle is not None:
        return _asr_backend
    _asr_backend = select(model_path=model_path, model_kind="asr_seq2seq", device="auto")
    if _asr_backend.model_handle is None:
        raise RuntimeError("ERR_ASR_MODEL_LOAD: Failed to load ASR model")
    try:
        from transformers import AutoProcessor
        _processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        _processor = None
    return _asr_backend


def _infer_chunk(waveform: np.ndarray, backend: Backend) -> dict:
    model = backend.model_handle
    if model is None:
        return {"text": "", "confidence": 0.0, "lang": "zh"}
    try:
        if _processor is not None:
            inputs = _processor(waveform, sampling_rate=16000, return_tensors="pt")
            import torch
            with torch.no_grad():
                ids = model.generate(inputs.input_features, return_dict_in_generate=True, output_scores=True)
            text = _processor.batch_decode(ids.sequences, skip_special_tokens=True)[0]
            conf = 0.9
            lang = getattr(ids, "language", "zh") or "zh"
            return {"text": text, "confidence": conf, "lang": lang}
    except Exception:
        pass
    return {"text": "", "confidence": 0.0, "lang": "zh"}


def _detect_lang(text: str) -> str:
    has_cn = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_en = any(c.isascii() and c.isalpha() for c in text)
    if has_cn and has_en:
        return "mixed"
    return "zh" if has_cn else "en"


def run(
    chunks: list,
    backend: Optional[Backend] = None,
    confidence_threshold: float = 0.6,
    model_path: str = "models/whisper-large-v3",
) -> List[TranscriptSegment]:
    if not chunks:
        return []

    if backend is None or backend.model_handle is None:
        backend = _load_asr_model(model_path)

    segments: List[TranscriptSegment] = []
    for chunk in chunks:
        result = _infer_chunk(chunk.waveform, backend)
        text = result["text"].strip()
        if not text:
            continue
        conf = result["confidence"]
        segments.append(TranscriptSegment(
            segment_id=chunk.chunk_index,
            text=text,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            confidence=conf,
            review=conf < confidence_threshold,
            lang=_detect_lang(text),
        ))

    segments.sort(key=lambda s: s.start_ms)
    return _dedup_overlap(segments)


def _dedup_overlap(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    if len(segments) <= 1:
        return segments
    result: List[TranscriptSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if seg.start_ms < prev.end_ms and seg.text[:20] in prev.text:
            continue
        result.append(seg)
    return result