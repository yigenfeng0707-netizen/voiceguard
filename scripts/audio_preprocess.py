"""音频接入与预处理流水线 - 格式校验/音轨提取/重采样/分片。"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

try:
    import imageio_ffmpeg
    _FFMPEG_AVAILABLE = True
except ImportError:
    _FFMPEG_AVAILABLE = False

try:
    from scipy.signal import resample_poly
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass
class AudioChunk:
    chunk_index: int
    waveform: np.ndarray
    sample_rate: int
    start_ms: int
    end_ms: int
    overlap_with_previous_ms: float = 0.0


def read_audio(file_path: str) -> tuple:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".wav", ".flac") and _SF_AVAILABLE:
        waveform, sr = sf.read(file_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, sr
    if ext in (".mp3", ".m4a") and _FFMPEG_AVAILABLE:
        wav_path = tempfile.mktemp(suffix=".wav")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-i", file_path, "-f", "wav", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", wav_path, "-y"],
            capture_output=True,
        )
        if _SF_AVAILABLE:
            waveform, sr = sf.read(wav_path, dtype="float32")
            os.unlink(wav_path)
            return waveform, sr
    if ext == ".mp4" and _FFMPEG_AVAILABLE:
        return extract_audio_track(file_path)
    raise ValueError(f"ERR_FORMAT_UNSUPPORTED: Cannot read {ext}")


def extract_audio_track(video_path: str) -> tuple:
    wav_path = tempfile.mktemp(suffix=".wav")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe() if _FFMPEG_AVAILABLE else "ffmpeg"
    result = subprocess.run(
        [ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", wav_path, "-y"],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(wav_path):
        raise ValueError("ERR_NO_AUDIO_TRACK: No audio track found in video")
    if _SF_AVAILABLE:
        waveform, sr = sf.read(wav_path, dtype="float32")
        os.unlink(wav_path)
        return waveform, sr
    raise ValueError("soundfile not available")


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    if _SCIPY_AVAILABLE:
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(waveform, target_sr // g, orig_sr // g).astype(np.float32)
    ratio = target_sr / orig_sr
    n_out = int(len(waveform) * ratio)
    indices = np.linspace(0, len(waveform) - 1, n_out)
    return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)


def normalize_volume(waveform: np.ndarray, target_peak_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(waveform))
    if peak < 1e-8:
        return waveform
    target_peak = 10 ** (target_peak_db / 20.0)
    return (waveform / peak * target_peak).astype(np.float32)


def split_by_silence(
    waveform: np.ndarray,
    sample_rate: int,
    max_chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.2,
) -> List[tuple]:
    chunk_len = int(max_chunk_seconds * sample_rate)
    overlap_len = int(overlap_seconds * sample_rate)
    chunks: List[tuple] = []
    pos = 0
    idx = 0
    while pos < len(waveform):
        end = min(pos + chunk_len, len(waveform))
        chunk = waveform[pos:end]
        start_ms = int(pos * 1000 / sample_rate)
        end_ms = int(end * 1000 / sample_rate)
        overlap_ms = int(overlap_len * 1000 / sample_rate) if pos > 0 else 0
        chunks.append((idx, chunk, start_ms, end_ms, overlap_ms))
        idx += 1
        if end >= len(waveform):
            break
        pos = end - overlap_len
    return chunks


def run(
    file_path: str,
    target_sr: int = 16000,
    max_chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.2,
    max_duration_minutes: int = 240,
) -> List[AudioChunk]:
    waveform, sr = read_audio(file_path)
    duration_minutes = len(waveform) / sr / 60
    if duration_minutes > max_duration_minutes:
        raise ValueError(f"ERR_DURATION_EXCEED: {duration_minutes:.1f}min > {max_duration_minutes}min")

    waveform = resample(waveform, sr, target_sr)
    waveform = normalize_volume(waveform)

    if np.max(np.abs(waveform)) < 0.01:
        raise ValueError("ERR_NO_VALID_SPEECH: Audio appears to be silence")

    raw_chunks = split_by_silence(waveform, target_sr, max_chunk_seconds, overlap_seconds)
    return [
        AudioChunk(idx, chunk, target_sr, start_ms, end_ms, overlap_ms)
        for idx, chunk, start_ms, end_ms, overlap_ms in raw_chunks
    ]