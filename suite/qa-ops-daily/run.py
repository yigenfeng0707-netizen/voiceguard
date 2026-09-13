# -*- coding: utf-8 -*-
"""qa-ops-daily · 质检运营日报技能（VoiceGuard 专家套件上层技能）。

批量扫描录音目录，逐段以 Skill 级互调方式调用 voiceguard-qa（CLI 或 HTTP API），
汇总生成质检运营日报 + 调用链日志（Skill 互调证据）。

用法：
  python run.py                          # 快速模式（ASR + 规则引擎），扫描默认 demo 目录
  python run.py --full                  # 完整模式（含本地模型语义复核）
  python run.py --audio-dir D:\\calls --pattern "*.wav"
  python run.py --via http --api-base http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows GBK 控制台保护
if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SELF_DIR = Path(__file__).resolve().parent
VG_ROOT = SELF_DIR.parent.parent                  # 主技能仓库（voiceguard/）
VG_CLI = VG_ROOT / "scripts" / "run.py"           # voiceguard-qa 的 CLI 入口

LEVEL_LABEL = {"redline": "红线", "warning": "警告", "notice": "提示"}


# ---------------------------------------------------------------------------
# Skill 级互调：调用 voiceguard-qa
# ---------------------------------------------------------------------------

def call_voiceguard_cli(audio: Path, out_dir: Path, full: bool, device: str, packs: str) -> dict:
    """以子进程调用 voiceguard-qa skill 的 CLI 入口（Skill-to-Skill 互调实证）。

    stdout/stderr 重定向到临时文件而非管道，规避 funasr/OpenVINO 后台线程
    持有管道句柄导致的 EOF 死锁；并带超时兜底。
    """
    import tempfile

    report_file = out_dir / f"{audio.stem}.report.json"
    cmd = [
        sys.executable, str(VG_CLI), str(audio),
        "--format", "json",
        "--packs", packs,
        "--device", device,
        "--output", str(report_file),
    ]
    if not full:
        cmd.append("--no-model")
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, stdout=out_f, stderr=err_f, timeout=1800, cwd=str(VG_ROOT))
            elapsed = round(time.perf_counter() - t0, 1)
        except subprocess.TimeoutExpired:
            return {
                "audio": audio.name, "via": "cli",
                "command": " ".join(cmd), "returncode": 124, "elapsed_s": 1800,
                "stderr": "timeout", "report_file": None,
            }
        err_f.seek(0)
        # 子进程 stderr 可能混入 GBK 字节（funasr/OpenVINO 警告），容错解码
        err_text = err_f.read().decode("utf-8", errors="replace").strip()[:300]
    rec = {
        "audio": audio.name,
        "via": "cli",
        "command": " ".join(cmd),
        "returncode": r.returncode,
        "elapsed_s": elapsed,
        "report_file": str(report_file) if report_file.is_file() else None,
        "stderr": err_text,
    }
    # 以产物文件为准：voiceguard-qa 清理阶段可能非零退出，但报告已落盘即有效
    if report_file.is_file():
        try:
            rec["report"] = json.loads(report_file.read_text(encoding="utf-8"))
            if r.returncode != 0:
                rec["warn"] = f"清理阶段非零退出 rc={r.returncode}（产物有效）"
        except Exception as e:
            rec["parse_error"] = f"{type(e).__name__}: {e}"
    return rec


def call_voiceguard_http(audio: Path, out_dir: Path, api_base: str, full: bool, packs: str) -> dict:
    """以 HTTP API 调用 voiceguard-qa skill（服务器需先启动 scripts/server.py）。"""
    import urllib.request

    endpoint = "/v1/qa" if full else "/v1/qa/fast"
    url = api_base.rstrip("/") + endpoint
    payload = json.dumps({
        "audio_path": str(audio.resolve()),
        "packs": packs,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            rc = 0
    except Exception as e:
        body, rc = None, 1
        err = f"{type(e).__name__}: {e}"
    elapsed = round(time.perf_counter() - t0, 1)

    rec = {
        "audio": audio.name,
        "via": "http",
        "command": f"POST {url} {{audio_path: {audio.name}}}",
        "returncode": rc,
        "elapsed_s": elapsed,
        "stderr": "" if rc == 0 else err,
    }
    if rc == 0:
        rep = body.get("report", body)
        report_file = out_dir / f"{audio.stem}.report.json"
        report_file.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        rec["report_file"] = str(report_file)
        rec["report"] = rep
    return rec


# ---------------------------------------------------------------------------
# 汇总统计与渲染
# ---------------------------------------------------------------------------

def summarize(records: list) -> dict:
    ok = [r for r in records if r.get("report")]
    scores = [r["report"].get("score", 0) for r in ok]
    level_counts = {"redline": 0, "warning": 0, "notice": 0}
    rule_counter: dict = {}
    funnel_total = {"segments": 0, "flagged": 0}
    token_saved_total = 0
    for r in ok:
        rep = r["report"]
        for v in rep.get("violations", []):
            if v.get("confirmed", True):
                level_counts[v["level"]] = level_counts.get(v["level"], 0) + 1
                key = f"{v['rule_id']} {v['rule_name']}"
                rule_counter[key] = rule_counter.get(key, 0) + 1
        f = rep.get("funnel", {})
        funnel_total["segments"] += f.get("total_segments", 0)
        funnel_total["flagged"] += f.get("segments_flagged", 0)
        token_saved_total += rep.get("token_economics", {}).get("cloud_api_est_tokens", 0)
    top_rules = sorted(rule_counter.items(), key=lambda kv: -kv[1])[:8]
    return {
        "total_calls": len(records),
        "ok_calls": len(ok),
        "failed_calls": len(records) - len(ok),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "pass_rate": round(
            100.0 * sum(1 for s in scores if s >= 75) / len(scores), 1
        ) if scores else None,
        "level_counts": level_counts,
        "top_rules": [{"rule": k, "count": c} for k, c in top_rules],
        "funnel_total": funnel_total,
        "api_token_saved_total": token_saved_total,
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in records), 1),
    }


def render_markdown(summary: dict, records: list, now: str) -> str:
    lines = [
        "# 质检运营日报",
        "",
        f"> 生成时间：{now} · 由 **qa-ops-daily** 上层技能编排，逐段调用 **voiceguard-qa** skill 完成（Skill 级互调）",
        "",
        "## 汇总",
        "",
        f"- 批量质检：**{summary['total_calls']}** 通（成功 {summary['ok_calls']} / 失败 {summary['failed_calls']}）",
        f"- 平均质检得分：**{summary['avg_score']}** / 100，合格率（≥75 分）：**{summary['pass_rate']}%**",
        f"- 确认违规：红线 **{summary['level_counts']['redline']}** · 警告 **{summary['level_counts']['warning']}** · 提示 **{summary['level_counts']['notice']}**",
        f"- 分级调度漏斗：{summary['funnel_total']['segments']} 逐字稿片段 → {summary['funnel_total']['flagged']} 规则标记（仅复核候选）",
        f"- API token 节省合计：**{summary['api_token_saved_total']}** tokens（全端侧推理）",
        f"- 批量总耗时：{summary['total_elapsed_s']}s",
        "",
        "## 逐通明细",
        "",
        "| 录音 | 得分 | 等级 | 红线 | 警告 | 提示 | 耗时(s) | 调用 |",
        "|------|-----:|------|-----:|-----:|-----:|--------:|------|",
    ]
    for r in records:
        rep = r.get("report")
        if rep:
            lv = {"redline": 0, "warning": 0, "notice": 0}
            for v in rep.get("violations", []):
                if v.get("confirmed", True):
                    lv[v["level"]] = lv.get(v["level"], 0) + 1
            lines.append(
                f"| {r['audio']} | {rep.get('score')} | {rep.get('grade')} "
                f"| {lv['redline']} | {lv['warning']} | {lv['notice']} "
                f"| {r['elapsed_s']} | {r['via']} |"
            )
        else:
            lines.append(f"| {r['audio']} | ERR | 调用失败 rc={r['returncode']} | - | - | - | {r['elapsed_s']} | {r['via']} |")

    if summary["top_rules"]:
        lines += ["", "## 高频违规规则 TOP", "", "| 规则 | 命中次数 |", "|------|---------:|"]
        lines += [f"| {t['rule']} | {t['count']} |" for t in summary["top_rules"]]

    lines += [
        "",
        "## Skill 调用链",
        "",
        "```",
        "qa-ops-daily（本技能）",
        "  └── 调用 → voiceguard-qa（主编排技能，via " + records[0]["via"] + "）",
        "        ├── vg-transcribe   （端侧 ASR）",
        "        ├── vg-rules-check  （规则引擎初筛）",
        "        └── vg-report-gen   （质检报告）",
        "```",
        "",
        f"完整调用证据见同目录 `call_chain.md`（{summary['total_calls']} 条调用记录）。",
        "",
    ]
    return "\n".join(lines)


def render_html(summary: dict, records: list, now: str) -> str:
    md = render_markdown(summary, records, now)
    body = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>质检运营日报 · qa-ops-daily × voiceguard-qa</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 32px auto; max-width: 1080px; color: #1a2333; background: #f6f7fb; }}
  .card {{ background: #fff; border-radius: 12px; padding: 20px 24px; margin: 14px 0; box-shadow: 0 1px 4px rgba(20,30,60,.08); }}
  h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; background:#fff; }}
  th, td {{ border: 1px solid #dde3ee; padding: 6px 10px; text-align: left; }}
  th {{ background: #eef2fa; }}
  code, pre {{ background: #f0f3f9; border-radius: 6px; }}
  pre {{ padding: 12px; overflow-x: auto; }}
</style></head><body>
{body}
</body></html>"""


def render_call_chain(records: list, now: str) -> str:
    lines = [
        "# Skill 互调调用链日志（证据）",
        "",
        f"生成时间：{now} · 上层技能 `qa-ops-daily` 对 `voiceguard-qa` 的全部 {len(records)} 次 Skill 级调用",
        "",
        "| # | 调用方式 | 返回码 | 耗时(s) | 命令 |",
        "|--|---------|-------:|--------:|------|",
    ]
    for i, r in enumerate(records, 1):
        cmd = r["command"].replace("|", "\\|")
        lines.append(f"| {i} | {r['via']} | {r['returncode']} | {r['elapsed_s']} | `{cmd}` |")
    lines += ["", "说明：以上每次调用均为主仓库 `voiceguard/scripts/run.py` 的独立进程执行，", "即 qa-ops-daily（上层技能）→ voiceguard-qa（主编排技能）的真实 Skill 互调。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="qa-ops-daily · 质检运营日报（VoiceGuard 上层技能）")
    ap.add_argument("--audio-dir", default=str(VG_ROOT / "demo" / "samples"), help="录音目录")
    ap.add_argument("--pattern", default="demo_*.wav", help="文件名 glob（默认 demo_*.wav）")
    ap.add_argument("--full", action="store_true", help="完整模式（含本地模型语义复核）")
    ap.add_argument("--via", default="cli", choices=["cli", "http"], help="调用方式")
    ap.add_argument("--api-base", default="http://127.0.0.1:8765", help="HTTP 模式服务地址")
    ap.add_argument("--device", default="GPU", choices=["GPU", "NPU", "CPU"], help="ASR 设备（cli 模式，传给 voiceguard-qa）")
    ap.add_argument("--packs", default="all", help="规则包")
    ap.add_argument("--output-dir", default=str(VG_ROOT / "output" / "qa_ops_daily"), help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 个文件（0=全部）")
    args = ap.parse_args()

    if not VG_CLI.is_file():
        print(f"ERR: 未找到 voiceguard-qa skill CLI: {VG_CLI}", file=sys.stderr)
        return 2

    audio_dir = Path(args.audio_dir)
    files = sorted(
        p for p in audio_dir.glob(args.pattern) if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".mp4")
    )
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"ERR: 目录 {audio_dir} 下没有匹配 {args.pattern} 的音频", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[qa-ops-daily] 扫描 {audio_dir} -> {len(files)} 段录音（模式: {'完整' if args.full else '快速'}, via {args.via}）")
    records = []
    for i, f in enumerate(files, 1):
        print(f"[qa-ops-daily] ({i}/{len(files)}) 调用 voiceguard-qa: {f.name} ...", flush=True)
        rec = (
            call_voiceguard_http(f, out_dir, args.api_base, args.full, args.packs)
            if args.via == "http"
            else call_voiceguard_cli(f, out_dir, args.full, args.device, args.packs)
        )
        status = "OK" if rec.get("report") else f"FAIL rc={rec['returncode']}"
        print(f"[qa-ops-daily]   -> {status} ({rec['elapsed_s']}s)")
        records.append(rec)

    summary = summarize(records)

    (out_dir / "qa_ops_daily_report.md").write_text(render_markdown(summary, records, now), encoding="utf-8")
    (out_dir / "qa_ops_daily_report.html").write_text(render_html(summary, records, now), encoding="utf-8")
    (out_dir / "qa_ops_daily_report.json").write_text(
        json.dumps({"generated_at": now, "summary": summary,
                    "records": [{k: v for k, v in r.items() if k != "report"} for r in records]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "call_chain.md").write_text(render_call_chain(records, now), encoding="utf-8")

    print()
    print(f"[qa-ops-daily] 日报完成: {out_dir / 'qa_ops_daily_report.html'}")
    print(f"[qa-ops-daily]   平均分 {summary['avg_score']} / 合格率 {summary['pass_rate']}% / 红线 {summary['level_counts']['redline']} 条")
    print(f"[qa-ops-daily] 调用链证据: {out_dir / 'call_chain.md'}")
    return 0 if summary["failed_calls"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
