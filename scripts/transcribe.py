"""ASR 转写前端：funasr + SenseVoiceSmall → Segment 列表（供 rules_engine 消费）。

D1: funasr CPU 基线已验证（真实语音 3223 字转写成功）。
D2: SenseVoice encoder 导出 OpenVINO IR → NPU/GPU/CPU 异构加速。
    NPU 使用 100 帧静态 shape + 分块重叠推理（SANM 局部注意力 kernel=11）。
"""  # noqa: E501
# LSP stubs in D:\dumateStoreData\lightsandbox\python\Lib\site-packages\{numpy,scipy,...}/__init__.pyi

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

DEFAULT_MODEL_DIR = (
    r"D:/APPs/OpenVINO/demo/local-meeting-minutes/models/SenseVoiceSmall"
)

# SenseVoice 输出的特殊标记（语言/情感/事件/ITN）
_TAG_RE = re.compile(r"<\|[a-zA-Z_]+\|>")
# 句边界：中文句读标点，或英文句点后跟空白/结尾
_BOUNDARY_RE = re.compile(r"[。！？!?；;]|[.!?](?=\s+|$)")
_MIN_SEG_CHARS = 8
_MAX_SEG_CHARS = 120


@dataclass
class TranscriptSegment:
    segment_id: int
    text: str
    start_ms: int
    end_ms: int
    speaker: str = ""
    lang: str = ""


def _load_waveform(audio_path: str) -> tuple:
    """读取音频为 (float32 mono, sr)，统一重采样到 16k。"""
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, 16000)
        data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    return data, sr


_model_cache: dict = {}


def _get_model(model_dir: str):
    if model_dir not in _model_cache:
        from funasr import AutoModel

        _model_cache[model_dir] = AutoModel(
            model=model_dir, device="cpu", disable_update=True
        )
    return _model_cache[model_dir]


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _split_to_segments(text: str, total_ms: int) -> List[TranscriptSegment]:
    """无词级时间戳时的兜底：按句边界切分，按字数比例分配时间轴。

    - 过短片段（<_MIN_SEG_CHARS）并入上一片段；
    - 过长片段（>_MAX_SEG_CHARS）按逗号/空格硬切，保证规则引擎 span 粒度。
    """
    # 1) 按句边界切分
    parts, prev = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        seg = text[prev : m.end()].strip()
        if seg:
            parts.append(seg)
        prev = m.end()
    tail = text[prev:].strip()
    if tail:
        parts.append(tail)

    # 2) 过短并入上一段
    merged: List[str] = []
    for p in parts:
        if merged and len(p) < _MIN_SEG_CHARS:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    parts = merged or ([text] if text else [])

    # 3) 过长硬切（逗号优先，其次按长度）
    final_parts: List[str] = []
    for p in parts:
        if len(p) <= _MAX_SEG_CHARS:
            final_parts.append(p)
            continue
        buf = ""
        for piece in re.split(r"(?<=[，,、：:])", p):
            if len(buf) + len(piece) > _MAX_SEG_CHARS and buf:
                final_parts.append(buf)
                buf = piece
            else:
                buf += piece
            while len(buf) > _MAX_SEG_CHARS:  # 无标点的超长串直接截断
                final_parts.append(buf[:_MAX_SEG_CHARS])
                buf = buf[_MAX_SEG_CHARS:]
        if buf:
            final_parts.append(buf)
    parts = [p for p in final_parts if p.strip()]

    # 4) 按字数比例分配时间轴
    total_chars = sum(len(p) for p in parts) or 1
    segments, cursor = [], 0
    for i, p in enumerate(parts):
        dur = int(total_ms * len(p) / total_chars)
        segments.append(
            TranscriptSegment(
                segment_id=i,
                text=p,
                start_ms=cursor,
                end_ms=cursor + dur,
            )
        )
        cursor += dur
    return segments


def transcribe(
    audio_path: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    language: str = "auto",
    device: str = "cpu",
) -> dict:
    """端到端转写：音频 → 去标记文本 → Segment 列表。

    Args:
        audio_path: 音频文件路径
        model_dir: SenseVoiceSmall 模型目录
        language: 语言代码 ("auto"|"zh"|"en"|"ja"|"ko")
        device: ASR encoder 运行设备 ("cpu"|"npu"|"gpu"|"auto")

    返回 {"segments": [...], "lang": str, "text": str, "raw": str, "device": str, "asr_stats": dict}
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    waveform, sr = _load_waveform(audio_path)
    total_ms = int(len(waveform) / sr * 1000)

    model = _get_model(model_dir)

    # NPU/GPU encoder acceleration via OpenVINO monkey-patch
    asr_stats = {}
    ov_encoder = None
    actual_device = "cpu"

    if device.lower() in ("npu", "gpu", "auto"):
        try:
            from npu_asr import patch_encoder, select_device

            target = device.upper() if device.lower() != "auto" else "auto"
            actual_device = select_device(target)
            ov_encoder = patch_encoder(model, actual_device)
            asr_stats["device"] = actual_device
            asr_stats["compile_time_s"] = round(ov_encoder.compile_time, 3)

            # Also patch CTC head on same device (complete ASR on NPU)
            try:
                from npu_ctc import patch_ctc_head, _CTC_IR
                import os as _os

                if _os.path.isfile(_CTC_IR):
                    ov_ctc = patch_ctc_head(model, actual_device)
                    if ov_ctc is not None:
                        asr_stats["ctc_device"] = actual_device
                        asr_stats["ctc_compile_time_s"] = round(ov_ctc.compile_time, 3)
            except Exception as e:
                asr_stats["ctc_warning"] = f"CTC head NPU patch failed: {e}"
        except Exception as e:
            asr_stats["warning"] = f"OpenVINO encoder unavailable ({e}), using CPU"
            actual_device = "cpu"

    audio_duration_s = total_ms / 1000.0
    res = model.generate(
        input=audio_path, language=language, use_itn=True, batch_size_s=300
    )

    # Collect NPU stats after inference
    if ov_encoder is not None:
        stats = ov_encoder.get_stats(audio_duration_s)
        asr_stats.update(
            {
                "infer_time_s": round(stats.total_infer_s, 4),
                "chunk_count": stats.chunk_count,
                "frames_processed": stats.frames_processed,
                "avg_chunk_ms": round(stats.avg_chunk_ms, 2),
                "rtf": round(stats.rtf, 4),
                "audio_duration_s": audio_duration_s,
            }
        )
        # Collect CTC head stats if patched
        ctc_lo = getattr(model.model.ctc.ctc_lo, "_ov_ctc_head", None)
        if ctc_lo is not None:
            ctc_stats = ctc_lo.get_stats()
            asr_stats["ctc_infer_s"] = round(ctc_stats.total_infer_s, 4)
            asr_stats["ctc_call_count"] = ctc_stats.call_count
            asr_stats["ctc_avg_latency_ms"] = round(ctc_stats.avg_latency_ms, 3)
        from npu_asr import unpatch_encoder
        from npu_ctc import unpatch_ctc_head

        unpatch_ctc_head(model)
        unpatch_encoder(model)
    raw = res[0].get("text", "") if res else ""
    text = _strip_tags(raw)

    lang_m = re.search(r"<\|([a-zA-Z]{2})\|>", raw or "")
    lang = lang_m.group(1) if lang_m else ""

    if not text:
        return {
            "segments": [],
            "lang": lang,
            "text": "",
            "raw": raw,
            "duration_ms": total_ms,
            "device": actual_device,
            "asr_stats": asr_stats,
        }

    # SenseVoice 词级时间戳（若可用）暂统一走按句分配；D2 接入精确时间戳
    segments = _split_to_segments(text, total_ms)
    return {
        "segments": segments,
        "lang": lang,
        "text": text,
        "raw": raw,
        "duration_ms": total_ms,
        "device": actual_device,
        "asr_stats": asr_stats,
    }


if __name__ == "__main__":
    import json
    import sys

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
    dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    out = transcribe(audio, device=dev)
    print(
        json.dumps(
            {
                "lang": out["lang"],
                "duration_ms": out["duration_ms"],
                "device": out.get("device", "cpu"),
                "asr_stats": out.get("asr_stats", {}),
                "segment_count": len(out["segments"]),
                "first_segments": [
                    {
                        "id": s.segment_id,
                        "text": s.text[:60],
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                    }
                    for s in out["segments"][:5]
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
