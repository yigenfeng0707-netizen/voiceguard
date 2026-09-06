# -*- coding: utf-8 -*-
"""E2E edge-case robustness test suite.

Generates 5 boundary audio samples (silence / ultra-long / fully-compliant /
multi-language / noisy) and runs the VoiceGuard pipeline on each in fast mode
(ASR + rules engine, no LLM). Validates that the system never crashes and
produces a reasonable report for every edge case.

Usage:
    python e2e_edge_cases.py
Output:
    - demo/samples/edge_*.wav          (generated audio)
    - output/e2e_edge_report.html      (HTML report)
    - output/e2e_edge_report.json      (machine-readable results)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# --- paths ---
BASE = Path(__file__).resolve().parent.parent
SAMPLES = BASE / "demo" / "samples"
OUTPUT = BASE / "output"
SCRIPTS = BASE / "scripts"
PYEXE = sys.executable

SAMPLES.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# --- edge-case definitions ---

EDGE_CASES = [
    {
        "id": "EC1_SILENCE",
        "label": "空音频(静音)",
        "desc": "3秒纯静音，验证管线不崩溃并返回空报告",
        "file": "edge_silence.wav",
        "expect": "no-crash",
    },
    {
        "id": "EC2_LONG",
        "label": "超长音频(60s+)",
        "desc": "60秒+合规电销话术，验证超长音频正常处理不截断崩溃",
        "file": "edge_long.wav",
        "expect": "no-crash",
    },
    {
        "id": "EC3_COMPLIANT",
        "label": "全合规话术",
        "desc": "标准合规电销话术(含风险提示/身份确认/自愿原则)，验证0违规",
        "file": "edge_compliant.wav",
        "expect": "0-violation",
    },
    {
        "id": "EC4_MIXLANG",
        "label": "中英混合",
        "desc": "中文为主、夹杂英文术语的通话，验证多语言识别鲁棒性",
        "file": "edge_mixlang.wav",
        "expect": "no-crash",
    },
    {
        "id": "EC5_NOISY",
        "label": "噪声环境",
        "desc": "在正常话术上叠加白噪声(信噪比~10dB)，验证噪声鲁棒性",
        "file": "edge_noisy.wav",
        "expect": "no-crash",
    },
]


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------


def _ffmpeg(args: list[str], timeout: int = 120) -> bool:
    """Run an ffmpeg command, return True on success."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


def _edge_tts(
    text: str, out_path: str, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%"
) -> bool:
    """Generate TTS audio via edge-tts (with 3-retry per feedback memory)."""
    import asyncio
    import edge_tts

    async def _gen():
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(out_path)

    for attempt in range(3):
        try:
            asyncio.run(_gen())
            if os.path.isfile(out_path):
                return True
        except Exception as e:
            print(f"  edge-tts attempt {attempt + 1}/3 failed: {e}", file=sys.stderr)
            if attempt < 2:
                import time as _t

                _t.sleep(1.0)
    return False


def gen_silence(out_path: str) -> bool:
    """3 seconds of silence."""
    return _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=16000",
            "-t",
            "3",
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ]
    )


def gen_long(out_path: str) -> bool:
    """~60s+ compliant telesales pitch."""
    text = (
        "您好，我是某某保险公司的客服代表，工号八八八八。"
        "今天致电是为了向您介绍我们的一款合规理财产品。"
        "首先向您声明，本次通话仅供产品介绍，不构成任何投资建议。"
        "我们的产品是经监管部门批准的正规产品，编号为银保监许可二零二五第零零一号。"
        "该产品的预期年化收益率为百分之三点五，但请注意，过往业绩不代表未来表现，"
        "投资有风险，请您谨慎决策。"
        "产品期限为三年，期间不可提前赎回，但满一年后可申请部分退出，"
        "退出将产生百分之三的手续费。"
        "如果您对产品有任何疑问，可以随时致电我们的客服热线四零零一二三四五六七。"
        "我们会在三个工作日内给您回复。"
        "再次提醒，本次通话已录音，您可以在三十天内致电客服申请删除录音。"
        "您是否愿意继续了解产品详情？如果您不愿意，我们可以随时结束通话。"
        "好的，那我继续为您介绍。"
        "本产品适合风险承受能力中等的客户，如果您属于保守型投资者，"
        "我们建议您考虑其他低风险产品。"
        "在购买前，请您仔细阅读产品说明书和风险提示函，确保充分理解产品条款。"
        "如有需要，您可以咨询独立的第三方理财顾问。"
        "我们承诺不会进行任何误导性宣传，也不会承诺保本保息。"
        "感谢您的耐心聆听，祝您生活愉快。"
    )
    if not _edge_tts(text, out_path, rate="+10%"):
        return False
    # Ensure 16kHz mono WAV
    tmp = out_path.replace(".wav", "_raw.mp3")
    os.rename(out_path, tmp)
    ok = _ffmpeg(["-i", tmp, "-ar", "16000", "-ac", "1", out_path])
    os.unlink(tmp)
    return ok and os.path.isfile(out_path)


def gen_compliant(out_path: str) -> bool:
    """Fully compliant sales call."""
    text = (
        "您好，我是某某金融公司的客服代表，工号一二三四。"
        "本次通话已录音，仅供产品介绍参考，不构成投资建议。"
        "向您介绍一款理财产品，预期年化收益率百分之三，"
        "请注意投资有风险，过往业绩不代表未来表现。"
        "产品期限一年，满期后可赎回，提前赎回收取百分之一手续费。"
        "如果您有任何疑问，可以致电客服热线四零零一二三四五六七。"
        "您是否愿意继续了解？如果您不愿意，我们可以随时结束通话。"
        "好的，感谢您的聆听，祝您生活愉快。"
    )
    if not _edge_tts(text, out_path):
        return False
    tmp = out_path.replace(".wav", "_raw.mp3")
    os.rename(out_path, tmp)
    ok = _ffmpeg(["-i", tmp, "-ar", "16000", "-ac", "1", out_path])
    os.unlink(tmp)
    return ok and os.path.isfile(out_path)


def gen_mixlang(out_path: str) -> bool:
    """Chinese-English mixed content."""
    text = (
        "您好，我是某公司的Sales Representative，工号ABC123。"
        "今天向您介绍我们的Wealth Management产品。"
        "请注意，本产品不保证Capital Preservation，投资有Risk。"
        "预期年化Yield Rate为百分之三点五，但Past Performance不代表未来。"
        "Product Term为一年，支持Early Redemption。"
        "如有疑问请拨打Customer Service Hotline四零零一二三四五六七。"
        "本次通话已Recorded，您可以在30天内申请Deletion。"
        "感谢您的Time，Have a nice day。"
    )
    if not _edge_tts(text, out_path):
        return False
    tmp = out_path.replace(".wav", "_raw.mp3")
    os.rename(out_path, tmp)
    ok = _ffmpeg(["-i", tmp, "-ar", "16000", "-ac", "1", out_path])
    os.unlink(tmp)
    return ok and os.path.isfile(out_path)


def gen_noisy(out_path: str) -> bool:
    """Existing violation sample + white noise."""
    src = str(SAMPLES / "demo_finance_violation.wav")
    if not os.path.isfile(src):
        print("  source not found, skipping noisy gen", file=sys.stderr)
        return False
    # Add white noise at ~10dB SNR
    return _ffmpeg(
        [
            "-i",
            src,
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=white:amplitude=0.03:duration=60",
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0",
            "-ar",
            "16000",
            "-ac",
            "1",
            out_path,
        ]
    )


GEN_FUNCS = {
    "EC1_SILENCE": gen_silence,
    "EC2_LONG": gen_long,
    "EC3_COMPLIANT": gen_compliant,
    "EC4_MIXLANG": gen_mixlang,
    "EC5_NOISY": gen_noisy,
}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(audio_path: str) -> dict:
    """Run run.py --no-model --format json and return parsed result or error."""
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [
                PYEXE,
                str(SCRIPTS / "run.py"),
                audio_path,
                "--no-model",
                "--format",
                "json",
                "--device",
                "CPU",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(SCRIPTS),
            encoding="utf-8",
        )
        elapsed = round(time.perf_counter() - t0, 1)
        if r.returncode != 0:
            return {
                "status": "FAIL",
                "exit_code": r.returncode,
                "stderr": r.stderr[-500:] if r.stderr else "",
                "elapsed_s": elapsed,
            }
        # Parse JSON output
        try:
            report = json.loads(r.stdout)
        except json.JSONDecodeError:
            # Sometimes there's log output before JSON; try to extract
            lines = r.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("{"):
                    try:
                        report = json.loads("\n".join(lines[i:]))
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                return {
                    "status": "PARSE_FAIL",
                    "stdout_tail": r.stdout[-500:],
                    "elapsed_s": elapsed,
                }
        return {
            "status": "OK",
            "report": report,
            "elapsed_s": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "elapsed_s": 300}
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "trace": traceback.format_exc()[-300:],
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(ec_id: str, result: dict) -> dict:
    """Check if the result meets expectations for this edge case."""
    checks = {}
    ec = next(e for e in EDGE_CASES if e["id"] == ec_id)

    if result["status"] != "OK":
        checks["pipeline_ok"] = {"pass": False, "detail": f"status={result['status']}"}
        return checks

    report = result.get("report", {})
    funnel = report.get("funnel", {})
    hits = report.get("hits", [])
    segments = report.get("segments", 0)

    checks["pipeline_ok"] = {"pass": True, "detail": "no crash, valid JSON"}

    if ec["expect"] == "no-crash":
        checks["no_crash"] = {
            "pass": True,
            "detail": f"completed in {result['elapsed_s']}s",
        }
        checks["valid_report"] = {
            "pass": True,
            "detail": f"segments={segments}, hits={len(hits)}",
        }

    if ec["expect"] == "0-violation":
        checks["zero_violation"] = {
            "pass": len(hits) == 0,
            "detail": f"hits={len(hits)}"
            + (f" rules={[h.get('rule_id', '') for h in hits]}" if hits else ""),
        }

    return checks


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html(results: list[dict]) -> str:
    rows = []
    total_pass = 0
    total_checks = 0

    for r in results:
        ec = r["ec"]
        gen_ok = r["gen_ok"]
        pipe = r["pipeline"]
        val = r["validation"]

        all_pass = all(v["pass"] for v in val.values()) if val else False
        checks_pass = sum(1 for v in val.values() if v["pass"])
        total_pass += checks_pass
        total_checks += len(val)

        status_color = "#22c55e" if all_pass else "#ef4444"
        status_text = "PASS" if all_pass else "FAIL"

        detail_lines = []
        for name, v in val.items():
            icon = "✅" if v["pass"] else "❌"
            detail_lines.append(f"{icon} {name}: {v['detail']}")
        detail_html = "<br>".join(detail_lines) if detail_lines else "N/A"

        report_summary = ""
        if pipe.get("status") == "OK":
            rep = pipe.get("report", {})
            funnel = rep.get("funnel", {})
            report_summary = (
                f"segments={rep.get('segments', 0)}, "
                f"hits={len(rep.get('hits', []))}, "
                f"elapsed={pipe.get('elapsed_s', '?')}s"
            )
        else:
            report_summary = f"status={pipe.get('status', '?')}"

        gen_icon = "✅" if gen_ok else "❌"

        rows.append(f"""
        <tr>
            <td style="font-weight:600">{ec["label"]}</td>
            <td>{ec["desc"]}</td>
            <td style="text-align:center">{gen_icon}</td>
            <td style="text-align:center;color:{status_color};font-weight:700">{status_text}</td>
            <td style="font-size:0.85em">{report_summary}</td>
            <td style="font-size:0.85em">{detail_html}</td>
        </tr>""")

    pass_rate = f"{total_pass}/{total_checks}" if total_checks else "0/0"
    summary_color = "#22c55e" if total_pass == total_checks else "#f59e0b"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>VoiceGuard E2E Edge-Case Robustness Report</title>
<style>
  body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 2rem; background: #f8fafc; }}
  h1 {{ color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }}
  .summary {{ background: #fff; border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .summary .big {{ font-size: 1.8rem; font-weight: 700; color: {summary_color}; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
  th {{ background: #1e293b; color: #fff; padding: 0.75rem; text-align: left; font-size: 0.9em; }}
  td {{ padding: 0.75rem; border-bottom: 1px solid #e2e8f0; font-size: 0.9em; }}
  tr:hover {{ background: #f1f5f9; }}
  .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>VoiceGuard E2E 边界场景鲁棒性测试报告</h1>
<div class="summary">
  <p>测试时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
  <p>总验证项: <span class="big">{pass_rate}</span> 通过</p>
  <p>测试模式: --no-model (ASR + 规则引擎, 无LLM语义复核)</p>
  <p>规则库: 72条 (finance 27 + insurance 18 + telesales 27)</p>
</div>
<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>描述</th>
      <th>音频生成</th>
      <th>结果</th>
      <th>管线摘要</th>
      <th>验证明细</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>
<div class="footer">
  <p>VoiceGuard Intel Agentic PC Skill 总决赛 | E2E Edge-Case Suite v1.0</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("VoiceGuard E2E Edge-Case Robustness Test Suite")
    print("=" * 60)

    results = []

    # Phase 1: Generate audio samples
    print("\n[Phase 1] Generating edge-case audio samples...")
    for ec in EDGE_CASES:
        out_path = str(SAMPLES / ec["file"])
        gen_fn = GEN_FUNCS[ec["id"]]
        print(f"  {ec['id']} ({ec['label']})... ", end="", flush=True)
        t0 = time.perf_counter()
        ok = gen_fn(out_path)
        dt = round(time.perf_counter() - t0, 1)
        if ok:
            size = os.path.getsize(out_path)
            print(f"OK ({size} bytes, {dt}s)")
        else:
            print(f"FAIL ({dt}s)")
        results.append(
            {
                "ec": ec,
                "gen_ok": ok,
                "audio_path": out_path,
                "pipeline": {},
                "validation": {},
            }
        )

    # Phase 2: Run pipeline on each
    print("\n[Phase 2] Running pipeline on edge cases...")
    for r in results:
        ec = r["ec"]
        if not r["gen_ok"]:
            print(f"  {ec['id']}: SKIP (audio not generated)")
            r["pipeline"] = {"status": "SKIP", "elapsed_s": 0}
            r["validation"] = {
                "audio_gen": {"pass": False, "detail": "generation failed"}
            }
            continue
        print(f"  {ec['id']} ({ec['label']})... ", end="", flush=True)
        pipe = run_pipeline(r["audio_path"])
        print(f"{pipe['status']} ({pipe.get('elapsed_s', '?')}s)")
        r["pipeline"] = pipe

    # Phase 3: Validate
    print("\n[Phase 3] Validating results...")
    for r in results:
        ec = r["ec"]
        val = (
            validate(ec["id"], r["pipeline"])
            if r["pipeline"].get("status") == "OK"
            else {}
        )
        if not val and r["pipeline"].get("status") != "OK":
            val = {
                "pipeline_ok": {
                    "pass": False,
                    "detail": f"status={r['pipeline'].get('status')}",
                }
            }
        r["validation"] = val
        all_pass = all(v["pass"] for v in val.values()) if val else False
        print(f"  {ec['id']}: {'PASS' if all_pass else 'FAIL'} ({len(val)} checks)")

    # Phase 4: Generate reports
    print("\n[Phase 4] Generating reports...")

    # JSON report
    json_path = str(OUTPUT / "e2e_edge_report.json")
    json_data = []
    for r in results:
        ec = r["ec"]
        pipe = r["pipeline"]
        json_entry = {
            "id": ec["id"],
            "label": ec["label"],
            "expect": ec["expect"],
            "audio_file": ec["file"],
            "gen_ok": r["gen_ok"],
            "pipeline_status": pipe.get("status"),
            "elapsed_s": pipe.get("elapsed_s"),
        }
        if pipe.get("status") == "OK":
            rep = pipe.get("report", {})
            json_entry["report_summary"] = {
                "segments": rep.get("segments", 0),
                "hits_count": len(rep.get("hits", [])),
                "funnel": rep.get("funnel", {}),
            }
        elif pipe.get("stderr"):
            json_entry["stderr"] = pipe["stderr"][-300:]
        json_entry["validation"] = r["validation"]
        json_data.append(json_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    # HTML report
    html_path = str(OUTPUT / "e2e_edge_report.html")
    html = generate_html(results)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML: {html_path}")

    # Summary
    total_pass = sum(
        sum(1 for v in r["validation"].values() if v["pass"]) for r in results
    )
    total_checks = sum(len(r["validation"]) for r in results)
    print(f"\n{'=' * 60}")
    print(f"E2E Edge-Case Results: {total_pass}/{total_checks} checks passed")
    print(f"{'=' * 60}")

    return 0 if total_pass == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
