"""VoiceGuard QA Ops Daily: batch quality audit + daily operations report.

This script demonstrates the Skill composability of VoiceGuard. It is an
upper-layer skill that calls three sub-skills in sequence:
  1. vg-transcribe (ASR)
  2. vg-rules-check (rules engine)
  3. vg-report-gen (report generation)

Usage:
    python qa_ops_daily.py <audio_dir> [--packs all] [--device GPU] [--format html]
    python qa_ops_daily.py demo/samples/ --packs all --device NPU --format html

Output:
    output/qa_daily_report.html (or .md) -- aggregated daily report
    output/qa_daily_summary.json    -- machine-readable summary
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

# UTF-8 stdout (Windows GBK protection)
if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in (
    "utf-8",
    "utf8",
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_full_pipeline  # noqa: E402
from report_gen import QAReport  # noqa: E402

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".mp4")

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
)


@dataclass
class FileResult:
    """Result of a single file's QA check."""

    filename: str
    score: int = 0
    grade: str = ""
    duration_ms: int = 0
    total_violations: int = 0
    confirmed_violations: int = 0
    redline_count: int = 0
    warning_count: int = 0
    notice_count: int = 0
    asr_device: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    report: QAReport = None


@dataclass
class DailySummary:
    """Aggregated daily operations summary."""

    date: str = ""
    total_files: int = 0
    total_duration_s: float = 0.0
    total_elapsed_s: float = 0.0
    avg_score: float = 0.0
    pass_count: int = 0  # score >= 75
    fail_count: int = 0  # score < 60
    total_violations: int = 0
    confirmed_violations: int = 0
    redline_total: int = 0
    warning_total: int = 0
    notice_total: int = 0
    top_violation_rules: List[dict] = field(default_factory=list)
    files: List[FileResult] = field(default_factory=list)
    asr_devices_used: List[str] = field(default_factory=list)

    def compute(self):
        """Compute aggregate metrics from file results."""
        valid = [f for f in self.files if not f.error]
        if not valid:
            return
        self.total_files = len(valid)
        scores = [f.score for f in valid]
        self.avg_score = round(sum(scores) / len(scores), 1)
        self.pass_count = sum(1 for s in scores if s >= 75)
        self.fail_count = sum(1 for s in scores if s < 60)
        self.total_violations = sum(f.total_violations for f in valid)
        self.confirmed_violations = sum(f.confirmed_violations for f in valid)
        self.redline_total = sum(f.redline_count for f in valid)
        self.warning_total = sum(f.warning_count for f in valid)
        self.notice_total = sum(f.notice_count for f in valid)
        self.total_duration_s = round(sum(f.duration_ms for f in valid) / 1000, 1)
        self.total_elapsed_s = round(sum(f.elapsed_s for f in valid), 1)
        self.asr_devices_used = list(set(f.asr_device for f in valid if f.asr_device))

        # Top violation rules
        rule_counts = {}
        for f in valid:
            if f.report:
                for v in f.report.violations:
                    if v.confirmed:
                        key = f"{v.rule_id} {v.rule_name}"
                        rule_counts[key] = rule_counts.get(key, 0) + 1
        self.top_violation_rules = sorted(
            [{"rule": k, "count": c} for k, c in rule_counts.items()],
            key=lambda x: -x["count"],
        )[:10]


def scan_audio_files(directory: str) -> List[str]:
    """Scan directory for supported audio files."""
    files = []
    if not os.path.isdir(directory):
        print(f"ERR: directory not found: {directory}", file=sys.stderr)
        return files
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(AUDIO_EXTS):
            files.append(os.path.join(directory, name))
    return files


def audit_one_file(
    audio_path: str,
    packs=None,
    device: str = "GPU",
    no_model: bool = False,
    reviewer_backend: str = "auto",
) -> FileResult:
    """Run full pipeline on one audio file, return FileResult."""
    fr = FileResult(filename=os.path.basename(audio_path))
    t0 = time.perf_counter()
    try:
        result = run_full_pipeline(
            audio_path,
            packs=packs,
            device=device,
            no_model=no_model,
            reviewer_backend=reviewer_backend,
        )
        report = result.report
        fr.score = report.score
        fr.grade = report.grade
        fr.duration_ms = report.duration_ms
        fr.total_violations = len(report.violations)
        fr.confirmed_violations = sum(1 for v in report.violations if v.confirmed)
        fr.redline_count = sum(
            1 for v in report.violations if v.confirmed and v.level == "redline"
        )
        fr.warning_count = sum(
            1 for v in report.violations if v.confirmed and v.level == "warning"
        )
        fr.notice_count = sum(
            1 for v in report.violations if v.confirmed and v.level == "notice"
        )
        fr.asr_device = result.stage1.asr_device
        fr.report = report
    except Exception as e:
        fr.error = f"{type(e).__name__}: {e}"
    fr.elapsed_s = round(time.perf_counter() - t0, 1)
    return fr


def render_daily_html(summary: DailySummary, output_path: str) -> str:
    """Render daily operations report as HTML."""
    rows = ""
    for f in summary.files:
        status = (
            '<span style="color:#dc2626">FAIL</span>'
            if f.score < 60
            else '<span style="color:#1a8a3b">PASS</span>'
            if f.score >= 75
            else '<span style="color:#d97706">WARN</span>'
        )
        err = (
            f'<br><span style="color:#999;font-size:0.8em">{f.error}</span>'
            if f.error
            else ""
        )
        rows += (
            f"<tr><td>{f.filename}</td><td>{f.score}</td><td>{f.grade}</td>"
            f"<td>{round(f.duration_ms / 1000, 1)}s</td>"
            f"<td>{f.confirmed_violations}</td>"
            f"<td>{f.redline_count}</td>"
            f"<td>{f.elapsed_s}s</td><td>{status}{err}</td></tr>"
        )

    top_rules = ""
    for r in summary.top_violation_rules[:5]:
        top_rules += f"<li>{r['rule']} ({r['count']} times)</li>"

    pass_rate = (
        round(summary.pass_count / summary.total_files * 100, 1)
        if summary.total_files
        else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>VoiceGuard Daily QA Report - {summary.date}</title>
<style>
body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #333; }}
h1 {{ color: #1a56c4; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1em 0; }}
.card {{ text-align: center; padding: 16px; border-radius: 8px; background: #f8f9fa; }}
.card .num {{ font-size: 2em; font-weight: bold; }}
.card .label {{ font-size: 0.8em; color: #666; margin-top: 4px; }}
.card-score {{ background: #1a56c4; color: #fff; }}
.card-score .num {{ color: #fff; }}
.card-red {{ background: #fef2f2; }}
.card-warn {{ background: #fffbeb; }}
.card-ok {{ background: #f0fdf4; }}
table {{ width: 100%; border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
th {{ background: #f5f5f5; }}
</style></head>
<body>
<h1>VoiceGuard Daily QA Operations Report</h1>
<p>Date: {summary.date} | Files: {summary.total_files} | Total Audio: {summary.total_duration_s}s | Processing: {summary.total_elapsed_s}s</p>

<div class="summary-grid">
<div class="card card-score"><div class="num">{summary.avg_score}</div><div class="label">Avg Score</div></div>
<div class="card card-ok"><div class="num">{summary.pass_count}/{summary.total_files}</div><div class="label">Pass Rate ({pass_rate}%)</div></div>
<div class="card card-red"><div class="num">{summary.redline_total}</div><div class="label">Redline Violations</div></div>
<div class="card card-warn"><div class="num">{summary.confirmed_violations}</div><div class="label">Total Confirmed</div></div>
</div>

<h2>File Details</h2>
<table>
<tr><th>File</th><th>Score</th><th>Grade</th><th>Duration</th><th>Confirmed</th><th>Redline</th><th>Processing</th><th>Status</th></tr>
{rows}
</table>

<h2>Top Violation Rules</h2>
<ol>
{top_rules or "<li>No violations detected.</li>"}
</ol>

<h2>ASR Devices Used</h2>
<p>{", ".join(summary.asr_devices_used) or "N/A"}</p>

<hr>
<p style="color:#999;font-size:0.8em">Generated by VoiceGuard qa_ops_daily.py | VoiceGuard Skill Composability Demo</p>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html


def render_daily_markdown(summary: DailySummary) -> str:
    """Render daily operations report as Markdown."""
    lines = [
        f"# VoiceGuard Daily QA Operations Report - {summary.date}",
        "",
        f"- Files: {summary.total_files} | Total Audio: {summary.total_duration_s}s | Processing: {summary.total_elapsed_s}s",
        f"- Avg Score: **{summary.avg_score}** | Pass: {summary.pass_count}/{summary.total_files} | Fail: {summary.fail_count}",
        f"- Confirmed Violations: {summary.confirmed_violations} (Redline: {summary.redline_total}, Warning: {summary.warning_total})",
        "",
        "## File Details",
        "",
        "| File | Score | Grade | Duration | Confirmed | Redline | Status |",
        "|------|-------|-------|----------|-----------|---------|--------|",
    ]
    for f in summary.files:
        status = "FAIL" if f.score < 60 else "PASS" if f.score >= 75 else "WARN"
        if f.error:
            status += f" ({f.error})"
        lines.append(
            f"| {f.filename} | {f.score} | {f.grade} | {round(f.duration_ms / 1000, 1)}s | "
            f"{f.confirmed_violations} | {f.redline_count} | {status} |"
        )
    lines += ["", "## Top Violation Rules", ""]
    for r in summary.top_violation_rules[:5]:
        lines.append(f"- {r['rule']} ({r['count']} times)")
    if not summary.top_violation_rules:
        lines.append("- No violations detected.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="VoiceGuard Batch QA + Daily Operations Report"
    )
    ap.add_argument("audio_dir", help="Directory containing audio files")
    ap.add_argument("--packs", default="all", help="Rule packs: all/finance/telesales")
    ap.add_argument("--device", default="GPU", choices=["GPU", "NPU", "CPU"])
    ap.add_argument(
        "--no-model",
        action="store_true",
        help="Rules-only mode (skip LLM review, faster)",
    )
    ap.add_argument(
        "--reviewer-backend",
        default="auto",
        choices=["auto", "openvino", "ollama"],
    )
    ap.add_argument("--format", default="html", choices=["html", "markdown", "json"])
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    # Scan audio files
    files = scan_audio_files(args.audio_dir)
    if not files:
        print(f"No audio files found in {args.audio_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files)} audio files in {args.audio_dir}")

    packs = None if args.packs == "all" else [p.strip() for p in args.packs.split(",")]

    # Run pipeline on each file
    summary = DailySummary(date=time.strftime("%Y-%m-%d"))
    for i, audio_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {os.path.basename(audio_path)}...")
        fr = audit_one_file(
            audio_path,
            packs=packs,
            device=args.device,
            no_model=args.no_model,
            reviewer_backend=args.reviewer_backend,
        )
        if fr.error:
            print(f"  ERROR: {fr.error}")
        else:
            print(
                f"  Score: {fr.score} ({fr.grade}), Violations: {fr.confirmed_violations}"
            )
        summary.files.append(fr)

    summary.compute()

    # Render output
    os.makedirs(args.output_dir, exist_ok=True)
    date_str = time.strftime("%Y%m%d")

    if args.format == "html":
        out_path = os.path.join(args.output_dir, f"qa_daily_report_{date_str}.html")
        render_daily_html(summary, out_path)
    elif args.format == "markdown":
        out_path = os.path.join(args.output_dir, f"qa_daily_report_{date_str}.md")
        md = render_daily_markdown(summary)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    else:  # json
        out_path = os.path.join(args.output_dir, f"qa_daily_report_{date_str}.json")
        json_data = {
            "date": summary.date,
            "total_files": summary.total_files,
            "avg_score": summary.avg_score,
            "pass_count": summary.pass_count,
            "fail_count": summary.fail_count,
            "confirmed_violations": summary.confirmed_violations,
            "redline_total": summary.redline_total,
            "files": [
                {
                    "filename": f.filename,
                    "score": f.score,
                    "grade": f.grade,
                    "confirmed_violations": f.confirmed_violations,
                    "redline_count": f.redline_count,
                    "elapsed_s": f.elapsed_s,
                    "error": f.error,
                }
                for f in summary.files
            ],
            "top_violation_rules": summary.top_violation_rules,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\nDaily report saved: {out_path}")
    print(
        f"Summary: {summary.total_files} files, avg score {summary.avg_score}, {summary.confirmed_violations} confirmed violations"
    )


if __name__ == "__main__":
    main()
