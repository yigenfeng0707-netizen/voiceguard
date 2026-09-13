# -*- coding: utf-8 -*-
"""vg-report-gen 子技能独立 CLI 入口。

消费 vg-rules-check 输出的 hits JSON（以及可选的语义复核 verdicts JSON），
生成结构化质检报告（评分 / 违规清单 / 调度漏斗 / token 经济性），支持
Markdown / JSON / HTML 三种格式。可独立调用，也可被上层技能（如 qa-ops-daily）调用。

用法：
  python subskills/vg-report-gen/run.py --hits hits.json [--format html] [--output report.html]
  python subskills/vg-report-gen/run.py --hits hits.json --verdicts verdicts.json

输入契约（hits JSON = vg-rules-check 输出）：
  hits / semantic_tasks / funnel / segments / audio / duration_ms / lang / elapsed_ms ...
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

VG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VG_ROOT / "scripts"))


def _rebuild_stage1(data: dict):
    """从 hits JSON 重建 build_report 所需的 stage1 形状（鸭子类型）。"""
    from rules_engine import RuleHit, Segment, SemanticTask, EngineResult  # noqa: E402

    engine = EngineResult(
        hits=[RuleHit(**h) for h in data.get("hits", [])],
        semantic_tasks=[SemanticTask(**t) for t in data.get("semantic_tasks", [])],
        funnel=data.get("funnel", {}),
        elapsed_ms=data.get("elapsed_ms", 0),
        packs_loaded=data.get("packs_loaded", []),
    )
    asr = data.get("asr") or {}
    return SimpleNamespace(
        audio=data.get("audio", "-"),
        duration_ms=data.get("duration_ms", 0),
        lang=data.get("lang", ""),
        engine=engine,
        asr_elapsed_s=0.0,
        total_elapsed_s=0.0,
        segments_ref=[Segment(**s) for s in data.get("segments", [])],
        asr_device=asr.get("device", "-"),
    )


def _load_verdicts(path: str):
    """加载语义复核结论（reviewer 输出），重建 verdict 对象。"""
    from reviewer import ReviewVerdict  # noqa: E402  (若 reviewer 不可用则跳过)

    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("verdicts", [])
    return [ReviewVerdict(**v) for v in items]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="vg-report-gen · VoiceGuard 质检报告生成子技能（可独立调用）"
    )
    ap.add_argument("--hits", required=True, help="vg-rules-check 输出的 hits JSON 路径")
    ap.add_argument("--verdicts", default=None, help="语义复核结论 JSON（可选，无则仅规则模式）")
    ap.add_argument("--format", default="html", choices=["markdown", "json", "html"])
    ap.add_argument("--reviewer-backend", default="-", help="复核后端标签（如 OpenVINO GPU INT4）")
    ap.add_argument("--output", default=None, help="报告输出路径（默认 stdout）")
    args = ap.parse_args()

    if not Path(args.hits).is_file():
        print(f"ERR: hits 文件不存在: {args.hits}", file=sys.stderr)
        return 2

    from report_gen import build_report, render_html, render_json, render_markdown  # noqa: E402

    data = json.loads(Path(args.hits).read_text(encoding="utf-8"))
    stage1 = _rebuild_stage1(data)

    verdicts = None
    if args.verdicts:
        verdicts = _load_verdicts(args.verdicts)

    asr_label = "vg-transcribe"
    asr_meta = data.get("asr") or {}
    if asr_meta.get("device"):
        asr_label = f"vg-transcribe ({asr_meta['device']})"

    report = build_report(
        stage1,
        verdicts=verdicts,
        reviewer_backend=args.reviewer_backend,
        asr_backend=asr_label,
    )

    if args.format == "json":
        text = render_json(report)
    elif args.format == "markdown":
        text = render_markdown(report)
    else:
        text = render_html(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"质检报告已生成: {args.output} (得分 {report.score}/100 · {report.grade})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
