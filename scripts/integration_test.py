"""VoiceGuard 四平台集成验证脚本（D4）。

对每平台执行 N 次调用 /v1/qa（或 /v1/qa/fast），
统计成功率、平均耗时、错误码分布，产出 HTML 集成验证报告。

用法：
  python scripts/integration_test.py [--calls 10] [--port 8765] [--fast]

说明：
  --fast  使用 /v1/qa/fast（仅规则引擎，秒级响应），默认使用完整 /v1/qa
  --calls 每平台调用次数（默认 10）
  --port  server.py 端口（默认 8765）

前置：需先启动 scripts/server.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

# Windows GBK 控制台保护
if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in (
    "utf-8",
    "utf8",
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_AUDIO = os.path.join(ROOT, "demo", "samples", "demo_finance_violation.wav")
CONFIG_DIR = os.path.join(ROOT, "agent_integrations")
OUTPUT_DIR = os.path.join(ROOT, "output")

PLATFORMS = ["qwenwork", "workbuddy", "trae_work", "doubao"]


def load_platform_config(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, name, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_api(base_url: str, endpoint: str, audio_path: str, use_fast: bool) -> dict:
    """调用 API 一次，返回 {success, elapsed, status_code, error, response_data}。"""
    import requests

    url = base_url.rstrip("/") + endpoint
    t0 = time.perf_counter()
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
            params = {"packs": "all"}
            if not use_fast:
                params["device"] = "GPU"
            resp = requests.post(url, files=files, params=params, timeout=300)
        elapsed = round(time.perf_counter() - t0, 2)
        if resp.status_code == 200:
            resp_data = {}
            try:
                resp_data = resp.json()
            except Exception:
                pass
            return {
                "success": True,
                "elapsed": elapsed,
                "status_code": 200,
                "error": "",
                "score": resp_data.get("score"),
                "grade": resp_data.get("grade"),
                "violations": len(resp_data.get("violations", [])),
                "confirmed": sum(
                    1 for v in resp_data.get("violations", []) if v.get("confirmed")
                ),
                "asr_device": resp_data.get("timings", {}).get("asr_device", ""),
                "review_s": resp_data.get("timings", {}).get("review_s"),
                "api_elapsed_s": resp_data.get("api_elapsed_s"),
            }
        return {
            "success": False,
            "elapsed": elapsed,
            "status_code": resp.status_code,
            "error": resp.text[:200],
        }
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 2)
        return {
            "success": False,
            "elapsed": elapsed,
            "status_code": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def run_tests(num_calls: int, port: int, use_fast: bool) -> dict:
    """对每平台执行 N 次调用，收集统计。"""
    base_url = f"http://127.0.0.1:{port}"
    endpoint = "/v1/qa/fast" if use_fast else "/v1/qa"
    results = {}

    for plat in PLATFORMS:
        cfg = load_platform_config(plat)
        mode = cfg.get("mode", "http_api")
        # QwenWork 是 native_skill 模式，不走 HTTP
        if mode == "native_skill":
            results[plat] = {
                "platform": cfg["platform"],
                "mode": mode,
                "calls": 0,
                "note": "原生 Skill 模式，不走 HTTP API。验证方式：复制到 ~/.qwenworkcn/skills/ 后通过 QwenWork 触发词调用 run.py。",
                "success_rate": "N/A",
                "avg_elapsed": "N/A",
                "errors": [],
            }
            continue

        calls = []
        for i in range(num_calls):
            r = call_api(base_url, endpoint, SAMPLE_AUDIO, use_fast)
            calls.append(r)
            ok = "OK" if r["success"] else f"FAIL({r['status_code']})"
            detail = ""
            if r["success"] and not use_fast:
                detail = f" score={r.get('score')} v={r.get('violations')} c={r.get('confirmed')}"
            print(f"  [{plat}] call {i + 1}/{num_calls}: {ok} {r['elapsed']}s{detail}")

        successes = sum(1 for c in calls if c["success"])
        elapsed_list = [c["elapsed"] for c in calls if c["success"]]
        avg_elapsed = (
            round(sum(elapsed_list) / len(elapsed_list), 2) if elapsed_list else 0
        )
        error_codes = {}
        for c in calls:
            if not c["success"]:
                code = str(c["status_code"])
                error_codes[code] = error_codes.get(code, 0) + 1

        success_rate = round(successes / num_calls * 100, 1) if num_calls else 0

        # Collect full-mode metrics
        full_metrics = {}
        if not use_fast and successes > 0:
            scores = [
                c["score"] for c in calls if c["success"] and c.get("score") is not None
            ]
            violation_counts = [c["violations"] for c in calls if c["success"]]
            confirmed_counts = [c["confirmed"] for c in calls if c["success"]]
            review_times = [
                c["review_s"] for c in calls if c["success"] and c.get("review_s")
            ]
            asr_devices = [
                c["asr_device"] for c in calls if c["success"] and c.get("asr_device")
            ]
            full_metrics = {
                "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
                "avg_violations": round(
                    sum(violation_counts) / len(violation_counts), 1
                )
                if violation_counts
                else None,
                "avg_confirmed": round(sum(confirmed_counts) / len(confirmed_counts), 1)
                if confirmed_counts
                else None,
                "avg_review_s": round(sum(review_times) / len(review_times), 2)
                if review_times
                else None,
                "asr_device": asr_devices[0] if asr_devices else "",
            }

        results[plat] = {
            "platform": cfg["platform"],
            "mode": mode,
            "calls": num_calls,
            "successes": successes,
            "success_rate": success_rate,
            "avg_elapsed": avg_elapsed,
            "errors": [c for c in calls if not c["success"]],
            "error_codes": error_codes,
            "full_metrics": full_metrics,
            "call_details": calls,
        }
        print(
            f"  [{plat}] {successes}/{num_calls} success ({success_rate}%), avg={avg_elapsed}s"
        )

    return results


def generate_report(results: dict, use_fast: bool) -> str:
    """生成 HTML 集成验证报告。"""
    mode_label = (
        "快速模式（仅规则引擎）"
        if use_fast
        else "完整模式（含模型复核 + NPU/GPU 异构加速）"
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for plat, r in results.items():
        if r.get("mode") == "native_skill":
            rows.append(
                f"<tr><td>{r['platform']}</td><td>{r['mode']}</td>"
                f"<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>"
                f'<td style="color:#666;font-size:0.85em;">{r.get("note", "")}</td></tr>'
            )
        else:
            sr = r["success_rate"]
            sr_color = "#1a8a3b" if sr >= 95 else "#d97706" if sr >= 80 else "#dc2626"
            err_detail = ""
            if r.get("error_codes"):
                err_detail = "; ".join(
                    f"HTTP {k} x{v}" for k, v in r["error_codes"].items()
                )
            fm = r.get("full_metrics", {})
            fm_cells = ""
            if fm and not use_fast:
                fm_cells = (
                    f"<td>{fm.get('avg_score', '-')}</td>"
                    f"<td>{fm.get('avg_violations', '-')}/{fm.get('avg_confirmed', '-')}</td>"
                    f"<td>{fm.get('avg_review_s', '-')}s</td>"
                )
            else:
                fm_cells = "<td>-</td><td>-</td><td>-</td>"
            rows.append(
                f"<tr><td>{r['platform']}</td><td>{r['mode']}</td>"
                f"<td>{r['calls']}</td>"
                f'<td style="color:{sr_color};font-weight:bold;">{sr}%</td>'
                f"<td>{r['avg_elapsed']}s</td>"
                f"{fm_cells}"
                f'<td style="font-size:0.85em;color:#dc2626;">{err_detail}</td></tr>'
            )

    table_rows = "\n".join(rows)

    # Per-call detail section
    detail_sections = []
    for plat, r in results.items():
        if r.get("mode") == "native_skill":
            continue
        call_rows = []
        for i, c in enumerate(r.get("call_details", [])):
            status = "OK" if c["success"] else f"FAIL({c['status_code']})"
            score = c.get("score", "-") if c["success"] else "-"
            violations = c.get("violations", "-") if c["success"] else "-"
            confirmed = c.get("confirmed", "-") if c["success"] else "-"
            call_rows.append(
                f"<tr><td>{i + 1}</td><td>{status}</td><td>{c['elapsed']}s</td>"
                f"<td>{score}</td><td>{violations}/{confirmed}</td></tr>"
            )
        if call_rows:
            detail_sections.append(
                f"<h3>{r['platform']} - 逐次调用明细</h3>\n"
                f'<table class="detail">\n'
                f"<tr><th>#</th><th>状态</th><th>耗时</th><th>评分</th><th>违规/确认</th></tr>\n"
                f"{chr(10).join(call_rows)}\n</table>"
            )

    detail_html = "\n".join(detail_sections)
    detail_col = (
        "" if use_fast else "<th>平均评分</th><th>违规/确认</th><th>复核耗时</th>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>VoiceGuard 四平台集成验证报告</title>
<style>
body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #333; }}
h1 {{ color: #1a56c4; }}
h2 {{ color: #1a56c4; margin-top: 1.5em; }}
h3 {{ color: #555; font-size: 1.1em; margin-top: 1em; }}
table {{ width: 100%; border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 0.9em; }}
th {{ background: #f5f5f5; }}
table.detail {{ max-width: 600px; }}
.detail td, .detail th {{ font-size: 0.85em; padding: 4px 8px; }}
.note {{ color: #666; font-size: 0.85em; margin: 1em 0; }}
.metric {{ background: #e8f4fd; padding: 2px 6px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>VoiceGuard 四平台集成验证报告</h1>
<p>生成时间：{timestamp} ｜ 模式：<span class="metric">{mode_label}</span> ｜ 样本音频：demo_finance_violation.wav</p>
<h2>总览</h2>
<table>
<tr><th>平台</th><th>接入模式</th><th>调用次数</th><th>成功率</th><th>平均耗时</th>{detail_col}<th>错误码分布</th></tr>
{table_rows}
</table>
<p class="note">达标标准：成功率 >= 95%。QwenWork 为原生 Skill 模式，不走 HTTP API，通过触发词在 QwenWork 桌面端验证。完整模式含 ASR(OpenVINO NPU/GPU) + 规则引擎 + 本地 LLM 语义复核(Qwen3-1.7B INT4 GPU)。</p>
{detail_html}
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="VoiceGuard 四平台集成验证")
    ap.add_argument("--calls", type=int, default=10, help="每平台调用次数")
    ap.add_argument("--port", type=int, default=8765, help="server.py 端口")
    ap.add_argument(
        "--fast", action="store_true", help="使用 /v1/qa/fast（仅规则引擎）"
    )
    args = ap.parse_args()

    # 健康检查
    import requests

    try:
        r = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=5)
        if r.status_code != 200:
            print(f"ERR: server not healthy on port {args.port}")
            sys.exit(1)
        print(f"Server healthy: {r.json()}")
    except Exception as e:
        print(f"ERR: 无法连接 server.py (port {args.port}): {e}")
        print(f"请先启动: python scripts/server.py --port {args.port}")
        sys.exit(1)

    print(
        f"\n开始集成验证：{args.calls} calls/platform, mode={'fast' if args.fast else 'full'}\n"
    )

    results = run_tests(args.calls, args.port, args.fast)

    # 生成报告
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_name = (
        "integration_report_fast.html" if args.fast else "integration_report_full.html"
    )
    report_path = os.path.join(OUTPUT_DIR, report_name)
    html = generate_report(results, args.fast)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n集成验证报告已写入: {report_path}")


if __name__ == "__main__":
    main()
