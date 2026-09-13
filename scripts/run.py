"""VoiceGuard 统一入口（SKILL.md 契约实现）。

用法：
  python run.py <音频文件> [--packs all|finance,telesales] [--format markdown|json|html]
                  [--no-model] [--cloud-enhance] [--device GPU|NPU|CPU]

状态：--no-model（Stage1 规则引擎模式）已可用；
模型模式：检测到 OpenVINO INT4 模型时自动启用语义复核。
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# Windows GBK 控制台保护：强制 stdout UTF-8（报告含 ✅/中文表格）
if hasattr(sys.stdout, "encoding") and (sys.stdout.encoding or "").lower() not in (
    "utf-8",
    "utf8",
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="VoiceGuard 端侧语音合规质检")
    ap.add_argument("audio", nargs="?", help="音频文件路径（--continue 时可省略）")
    ap.add_argument(
        "--packs", default="all", help="规则包：all / finance / telesales（逗号分隔）"
    )
    ap.add_argument(
        "--format", default="markdown", choices=["markdown", "json", "html"]
    )
    ap.add_argument(
        "--no-model", action="store_true", help="仅规则引擎（跳过语义复核）"
    )
    ap.add_argument(
        "--cloud-enhance", action="store_true", help="启用云端脱敏增强（默认关闭）"
    )
    ap.add_argument("--device", default="GPU", choices=["GPU", "NPU", "CPU"])
    ap.add_argument(
        "--reviewer-backend",
        default="auto",
        choices=["auto", "openvino", "ollama"],
        help="语义复核后端：auto(默认OpenVINO，降级Ollama) / openvino / ollama",
    )
    ap.add_argument("--ollama-model", default="qwen2.5:3b", help="Ollama 模型名")
    ap.add_argument("--continue", dest="cont", action="store_true", help="续传上次任务")
    ap.add_argument("--output", default=None, help="报告输出路径（默认打印到 stdout）")
    args = ap.parse_args()

    from pipeline import run_full_pipeline
    from report_gen import render_json, render_markdown, render_html

    if not args.audio:
        print("ERR: 请提供音频文件路径（或 --continue，暂未实现）", file=sys.stderr)
        sys.exit(2)

    packs = None if args.packs == "all" else [p.strip() for p in args.packs.split(",")]

    # D2 全链路编排：Stage1(ASR→规则) → Stage2(语义复核) → Stage3(报告)
    try:
        result = run_full_pipeline(
            args.audio,
            packs=packs,
            device=args.device,
            no_model=args.no_model,
            cloud_enhance=args.cloud_enhance,
            reviewer_backend=args.reviewer_backend,
            ollama_model=args.ollama_model,
        )
    except FileNotFoundError as e:
        print(f"ERR: 音频文件不存在: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERR: 管线执行失败: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    report = result.report
    if args.format == "json":
        text = render_json(report)
    elif args.format == "html":
        text = render_html(report)
    else:
        text = render_markdown(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"报告已写入: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
    # 硬退出：funasr/OpenVINO 的非守护后台线程在解释器清理阶段可能
    # 挂起或崩溃（rc!=0），导致上层技能（qa-ops-daily）等待/误判失败。
    # 产物已落盘并 flush，直接 os._exit 绕过清理。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
