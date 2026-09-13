# -*- coding: utf-8 -*-
"""vg-rules-check 子技能独立 CLI 入口。

对逐字稿执行零 token 规则引擎初筛（keyword/regex/absent/semantic 四策略，
金融+电销 72 条规则），输出候选违规命中 JSON（hits 契约）。

两种输入模式（体现 Skill 互调）：
  A. --segments seg.json   直接消费 vg-transcribe 的输出（上游 skill 已跑过）
  B. --audio xxx.wav       先以子进程调用 vg-transcribe skill 完成转写，再跑规则引擎
                            （本 skill 调用其它 skill 的实证，记录在 called_skills 字段）

用法：
  python subskills/vg-rules-check/run.py --segments seg.json [--packs all|finance,telesales]
  python subskills/vg-rules-check/run.py --audio xxx.wav [--packs finance] [--device auto]
  python subskills/vg-rules-check/run.py --text "保本保息年化30%"   # 免模型快速自测

输出契约（hits JSON）：
  {
    "skill": "vg-rules-check",
    "called_skills": ["vg-transcribe"],   # 本次调用链上被本技能调用的其它 skills
    "audio": "...", "duration_ms": 0, "lang": "zh",
    "packs_loaded": ["finance", "telesales"],
    "hits": [...], "semantic_tasks": [...], "funnel": {...}, "elapsed_ms": 3,
    "segments": [...]   # 透传逐字稿，供 vg-report-gen 消费
  }
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

VG_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = VG_ROOT / "scripts"
RULES_DIR = VG_ROOT / "rules"
VG_TRANSCRIBE_CLI = Path(__file__).resolve().parent.parent / "vg-transcribe" / "run.py"

sys.path.insert(0, str(SCRIPTS))


def _call_vg_transcribe(audio: str, device: str) -> dict:
    """Skill 级调用 vg-transcribe（子进程），返回其 segments JSON。

    注意：stdout/stderr 重定向到临时文件而非管道（capture_output）——
    funasr/OpenVINO 可能留下持有管道句柄的后台线程，管道 EOF 死锁会让
    父进程永久挂起；文件重定向 + 超时兜底彻底规避该问题。
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="vg_rules_"))
    seg_file = tmp / "segments.json"
    cmd = [sys.executable, str(VG_TRANSCRIBE_CLI), audio, "--device", device, "--output", str(seg_file)]
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                cmd, stdout=out_f, stderr=err_f,
                timeout=900, cwd=str(VG_ROOT),
            )
            elapsed = round(time.perf_counter() - t0, 2)
        except subprocess.TimeoutExpired:
            raise RuntimeError("vg-transcribe 调用超时（>900s）")
        err_f.seek(0)
        # 子进程 stderr 可能混入 GBK 字节（funasr/OpenVINO 警告），容错解码
        err_text = err_f.read().decode("utf-8", errors="replace").strip()
    # 以产物文件为准：funasr/OpenVINO 在写完输出后的解释器清理阶段可能
    # 非零退出（不影响已落盘的 segments.json）。文件存在且可解析即成功。
    if seg_file.is_file():
        try:
            payload = json.loads(seg_file.read_text(encoding="utf-8"))
            warn = "" if r.returncode == 0 else f" (rc={r.returncode}, 清理阶段非零退出, 产物有效)"
            print(f"[skill-call] vg-rules-check -> vg-transcribe: rc={r.returncode}, {elapsed}s, out=segments.json{warn}", file=sys.stderr)
            return payload
        except (json.JSONDecodeError, OSError):
            pass
    raise RuntimeError(
        f"vg-transcribe 调用失败 (rc={r.returncode}, {elapsed}s): {err_text[:500]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="vg-rules-check · VoiceGuard 规则引擎零 token 初筛子技能（可独立调用）"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--segments", help="vg-transcribe 输出的逐字稿 JSON 路径")
    g.add_argument("--audio", help="音频文件路径（将自动调用 vg-transcribe skill 转写）")
    g.add_argument("--text", help="纯文本快速检测（免 ASR，用于规则自测）")
    ap.add_argument("--packs", default="all", help="规则包：all / finance / telesales / insurance（逗号分隔）")
    ap.add_argument("--device", default="auto", help="--audio 模式下传给 vg-transcribe 的设备")
    ap.add_argument("--output", default=None, help="命中结果 JSON 输出路径（默认 stdout）")
    args = ap.parse_args()

    from rules_engine import Segment, load_rule_packs, run_engine  # noqa: E402

    called_skills: list = []
    meta = {"audio": "-", "duration_ms": 0, "lang": "", "asr": None}
    segments = []

    if args.text:
        segments = [Segment(segment_id=0, text=args.text, start_ms=0, end_ms=0)]
        meta["audio"] = "<inline-text>"
    elif args.segments:
        data = json.loads(Path(args.segments).read_text(encoding="utf-8"))
        meta["audio"] = data.get("audio", Path(args.segments).name)
        meta["duration_ms"] = data.get("duration_ms", 0)
        meta["lang"] = data.get("lang", "")
        meta["asr"] = {"device": data.get("device"), "skill": data.get("skill")}
        segments = [Segment(**s) for s in data.get("segments", [])]
    elif args.audio:
        if not os.path.isfile(args.audio):
            print(f"ERR: 音频文件不存在: {args.audio}", file=sys.stderr)
            return 2
        # Skill-to-skill 调用：vg-rules-check -> vg-transcribe
        tr = _call_vg_transcribe(args.audio, args.device)
        called_skills.append("vg-transcribe")
        meta["audio"] = tr.get("audio", Path(args.audio).name)
        meta["duration_ms"] = tr.get("duration_ms", 0)
        meta["lang"] = tr.get("lang", "")
        meta["asr"] = {"device": tr.get("device"), "skill": "vg-transcribe"}
        segments = [Segment(**s) for s in tr.get("segments", [])]

    packs = None if args.packs == "all" else [p.strip() for p in args.packs.split(",")]
    try:
        loaded = load_rule_packs(str(RULES_DIR), pack_ids=packs)
        engine = run_engine(segments, loaded)
    except FileNotFoundError as e:
        print(f"ERR: {e}", file=sys.stderr)
        return 2

    payload = {
        "skill": "vg-rules-check",
        "skill_version": "1.0",
        "called_skills": called_skills,
        **meta,
        "packs_loaded": [p.get("pack_id", "?") for p in loaded],
        "hits": [asdict(h) for h in engine.hits],
        "semantic_tasks": [asdict(t) for t in engine.semantic_tasks],
        "funnel": engine.funnel,
        "elapsed_ms": engine.elapsed_ms,
        "segments": [asdict(s) for s in segments],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(
            f"规则初筛完成: {len(engine.hits)} 命中 / {len(segments)} 片段, "
            f"{engine.elapsed_ms}ms, 零 token -> {args.output}"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
