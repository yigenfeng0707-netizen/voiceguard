"""NPU ASR Encoder Benchmark: NPU vs GPU vs CPU.

Loads funasr SenseVoiceSmall model, extracts mel features from real audio,
runs encoder on each device, and compares latency + RTF.

Usage:
    python scripts/benchmark_npu_asr.py [audio_path]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add scripts dir to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

DEFAULT_AUDIO = os.path.join(
    _PROJECT_DIR, "demo", "samples", "demo_finance_violation.wav"
)
ENCODER_IR = os.path.join(_PROJECT_DIR, "models", "SenseVoiceSmall_ov", "encoder.xml")
MODEL_DIR = r"D:/APPs/OpenVINO/demo/local-meeting-minutes/models/SenseVoiceSmall"


def extract_mel_features(audio_path: str, model_dir: str = MODEL_DIR):
    """Extract mel-spectrogram features using funasr frontend.

    Returns (mel_features, audio_duration_s)
    """
    import torch
    import numpy as np
    from funasr.utils.load_utils import extract_fbank
    from funasr.utils.load_utils import load_audio_text_image_video

    # Load audio
    waveform = load_audio_text_image_video(audio_path)
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    audio_duration_s = len(waveform) / 16000.0

    # Extract fbank features (mel-spectrogram)
    # SenseVoiceSmall uses: 560 mel bins, 25ms frame, 10ms shift
    mel, _ = extract_fbank(
        waveform,
        fs=16000,
        n_mels=560,
        window_length=25,
        frame_length=10,
    )

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # [1, frames, 560]

    return mel, audio_duration_s


def benchmark_device(
    mel: "torch.Tensor",
    device: str,
    ir_path: str = ENCODER_IR,
):
    """Benchmark encoder on a specific device."""
    import torch
    import numpy as np

    frames = mel.shape[1]
    print(f"  [{device}] Input shape: {mel.shape}, frames={frames}")

    if device == "cpu":
        # Run PyTorch encoder on CPU
        from funasr import AutoModel

        model = AutoModel(model=MODEL_DIR, device="cpu", disable_update=True)
        encoder = model.model.encoder
        ilens = torch.tensor([frames], dtype=torch.long)

        # Warm up
        with torch.no_grad():
            _ = encoder(mel, ilens)

        # Benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            out, olens = encoder(mel, ilens)
        infer_time = time.perf_counter() - t0

        return {
            "device": "CPU (PyTorch)",
            "frames": frames,
            "infer_time_s": round(infer_time, 4),
            "output_shape": list(out.shape),
            "compile_time_s": 0,
        }
    else:
        # OpenVINO on NPU/GPU
        import openvino
        from openvino import PartialShape

        CHUNK = 100
        MEL_BINS = 560
        OVERLAP = 11
        STRIDE = CHUNK - 2 * OVERLAP

        core = openvino.Core()
        model = core.read_model(ir_path)
        model.reshape(
            {
                0: PartialShape([1, CHUNK, MEL_BINS]),
                1: PartialShape([1]),
            }
        )

        t0 = time.perf_counter()
        compiled = core.compile_model(model, device.upper())
        compile_time = time.perf_counter() - t0

        mel_np = mel.detach().cpu().numpy().astype(np.float32)

        # Process in chunks
        total_infer = 0.0
        chunk_count = 0
        outputs = []

        if frames <= CHUNK:
            padded = np.zeros((1, CHUNK, MEL_BINS), dtype=np.float32)
            padded[:, :frames, :] = mel_np[:, :frames, :]
            t1 = time.perf_counter()
            result = compiled({0: padded, 1: np.array([CHUNK], dtype=np.int64)})
            total_infer += time.perf_counter() - t1
            chunk_count = 1
            out = result[compiled.output(0)][:, :frames, :]
        else:
            pos = 0
            while pos < frames:
                end = min(pos + CHUNK, frames)
                actual = end - pos
                chunk = np.zeros((1, CHUNK, MEL_BINS), dtype=np.float32)
                chunk[:, :actual, :] = mel_np[:, pos:end, :]

                t1 = time.perf_counter()
                result = compiled({0: chunk, 1: np.array([CHUNK], dtype=np.int64)})
                total_infer += time.perf_counter() - t1
                chunk_count += 1

                slice_start = OVERLAP if pos > 0 else 0
                slice_end = actual - OVERLAP if end < frames else actual
                outputs.append(result[compiled.output(0)][:, slice_start:slice_end, :])
                pos += STRIDE

            out = np.concatenate(outputs, axis=1)

        return {
            "device": f"{device.upper()} (OpenVINO)",
            "frames": frames,
            "infer_time_s": round(total_infer, 4),
            "output_shape": list(out.shape),
            "compile_time_s": round(compile_time, 2),
            "chunk_count": chunk_count,
            "avg_chunk_ms": round(total_infer / max(chunk_count, 1) * 1000, 2),
        }


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO

    print("=" * 60)
    print("VoiceGuard NPU ASR Encoder Benchmark")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Encoder IR: {ENCODER_IR}")
    print()

    # Extract mel features
    print("Extracting mel features...")
    mel, audio_dur = extract_mel_features(audio_path)
    print(f"  Audio duration: {audio_dur:.2f}s")
    print(f"  Mel shape: {mel.shape}")
    print()

    # Detect available devices
    import openvino

    core = openvino.Core()
    devices = [d.upper() for d in core.available_devices]
    print(f"Available OpenVINO devices: {devices}")
    print()

    results = []

    # Benchmark CPU (PyTorch baseline)
    print("Benchmarking CPU (PyTorch)...")
    try:
        r = benchmark_device(mel, "cpu")
        r["rtf"] = round(r["infer_time_s"] / audio_dur, 4) if audio_dur > 0 else 0
        results.append(r)
        print(f"  -> {r['infer_time_s']}s, RTF={r['rtf']}")
    except Exception as e:
        print(f"  CPU FAILED: {e}")
        results.append({"device": "CPU", "error": str(e)[:200]})
    print()

    # Benchmark OpenVINO devices
    for dev in ["NPU", "GPU", "CPU"]:
        ov_dev = f"{dev} (OpenVINO)"
        if dev not in devices:
            print(f"{ov_dev}: device not available, skipping")
            continue

        print(f"Benchmarking {ov_dev}...")
        try:
            r = benchmark_device(mel, dev)
            r["rtf"] = round(r["infer_time_s"] / audio_dur, 4) if audio_dur > 0 else 0
            results.append(r)
            print(
                f"  -> compile={r['compile_time_s']}s, infer={r['infer_time_s']}s, RTF={r['rtf']}"
            )
        except Exception as e:
            print(f"  {ov_dev} FAILED: {e}")
            results.append({"device": ov_dev, "error": str(e)[:200]})
        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Device':<25} {'Compile(s)':>10} {'Infer(s)':>10} {'RTF':>8} {'Chunks':>8}"
    )
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"{r['device']:<25} {'FAILED':>10}")
        else:
            print(
                f"{r['device']:<25} "
                f"{r.get('compile_time_s', 0):>10} "
                f"{r['infer_time_s']:>10} "
                f"{r.get('rtf', 0):>8} "
                f"{r.get('chunk_count', '-'):>8}"
            )
    print()

    # Save results
    out_path = os.path.join(_PROJECT_DIR, "output", "npu_asr_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": audio_path,
                "audio_duration_s": round(audio_dur, 2),
                "mel_shape": list(mel.shape),
                "devices_tested": devices,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
