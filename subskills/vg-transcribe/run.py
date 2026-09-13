# -*- coding: utf-8 -*-
"""vg-transcribe 子技能独立 CLI 入口。

把通话录音留在本机完成 ASR 转写（SenseVoiceSmall + OpenVINO 异构加速），
输出带时间戳的逐字稿 JSON（segments 契约），供 vg-rules-check / 上层技能消费。

用法：
  python subskills/vg-transcribe/run.py <音频文件> [--device auto|cpu|gpu|npu] [--language auto] [--output seg.json]

输出契约（segments JSON）：
  {
    "skill": "vg-transcribe",
    "audio": "xxx.wav",
    "duration_ms": 30000,
    "lang": "zh",
    "device": "npu",
    "asr_stats": {...},
    "segments": [{"segment_id": 0, "text": "...", "start_ms": 0, "end_ms": 3000, "speaker": ""}]
  }
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Windows GBK 控制台保护
if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 定位主仓库 scripts/（vg-transcribe 的推理实现）
VG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VG_ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="vg-transcribe · VoiceGuard 端侧转写子技能（可独立调用）"
    )
    ap.add_argument("audio", help="音频文件路径（wav/mp3/m4a/flac/mp4）")
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "gpu", "npu"],
        help="ASR 推理设备（默认 auto：NPU>GPU>CPU 自动降级）",
    )
    ap.add_argument("--language", default="auto", help="auto / zh / en / ja / ko")
    ap.add_argument("--output", default=None, help="逐字稿 JSON 输出路径（默认 stdout）")
    args = ap.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERR: 音频文件不存在: {args.audio}", file=sys.stderr)
        return 2

    from transcribe import transcribe  # noqa: E402

    try:
        tr = transcribe(args.audio, language=args.language, device=args.device)
    except Exception as e:
        print(f"ERR: 转写失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    payload = {
        "skill": "vg-transcribe",
        "skill_version": "1.0",
        "audio": os.path.basename(args.audio),
        "audio_path": str(Path(args.audio).resolve()),
        "duration_ms": tr.get("duration_ms", 0),
        "lang": tr.get("lang", ""),
        "text": tr.get("text", ""),
        "device": tr.get("device", args.device),
        "asr_stats": tr.get("asr_stats", {}),
        "segments": [asdict(s) for s in tr["segments"]],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"逐字稿已写入: {args.output} ({len(payload['segments'])} segments, device={payload['device']})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    rc = main()
    # 硬退出：funasr/OpenVINO 的非守护后台线程在解释器清理阶段会
    # 挂起或崩溃（rc!=0），导致 skill 调用方等待/误判失败。
    # 产物已落盘并 flush，直接 os._exit 绕过清理。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
