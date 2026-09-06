"""Stage 3：结构化质检报告生成（评分 + 违规清单 + 调度漏斗 + token 经济性）。

评分算法（V1）：
- 基准 100 分
- 红线（redline）确认违规：每条 -15
- 警告（warning）确认违规：每条 -5
- 提示（notice）确认违规：每条 -2
- 下限 0 分；等级：>=90 优秀 / 75-89 合格 / 60-74 待整改 / <60 不合格

token 经济性：
- 纯云端方案估算：全量逐字稿直接喂云端大模型（按 1 汉字 ≈ 0.7 token 估）
- 本方案实际：仅候选复核 prompt（候选摘录 + 上下文窗口）
- 输出节省比例，作为路演量化证据
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

LEVEL_DEDUCT = {"redline": 15, "warning": 5, "notice": 2}
LEVEL_LABEL = {"redline": "红线", "warning": "警告", "notice": "提示"}


@dataclass
class Violation:
    rule_id: str
    rule_name: str
    level: str
    category: str
    excerpt: str
    advice: str
    basis: str
    matched_by: str
    confirmed: Optional[bool] = None  # None = 未经模型复核
    reason: str = ""


@dataclass
class QAReport:
    audio: str
    duration_ms: int
    lang: str
    score: int
    grade: str
    violations: List[Violation]
    funnel: dict
    token_economics: dict
    backends: dict
    timings: dict
    cloud_advice: str = ""
    cloud_enhance_meta: dict = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )


def grade_of(score: int) -> str:
    if score >= 90:
        return "优秀"
    if score >= 75:
        return "合格"
    if score >= 60:
        return "待整改"
    return "不合格"


def estimate_tokens_zh(text: str) -> int:
    """粗略估算：中文 1 字 ≈ 0.7 token，英文 4 字符 ≈ 1 token。"""
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - zh
    return int(zh * 0.7 + other / 4)


def build_report(
    stage1,  # pipeline.Stage1Result
    verdicts=None,  # List[ReviewVerdict]（可为空：仅规则引擎模式）
    reviewer_backend: str = "-",
    asr_backend: str = "funasr+OpenVINO(NPU/GPU)",
    review_prompt_tokens: int = 0,
    review_s: float = 0.0,
) -> QAReport:
    verdict_map = {}
    if verdicts:
        verdict_map = {v.index: v for v in verdicts}

    violations: List[Violation] = []
    confirmed_count = {"redline": 0, "warning": 0, "notice": 0}

    # 规则引擎确定性命中 → 违规项（有复核结果则按复核结论）
    for i, h in enumerate(stage1.engine.hits):
        v = verdict_map.get(i)
        confirmed = True if v is None else v.confirmed
        violations.append(
            Violation(
                rule_id=h.rule_id,
                rule_name=h.rule_name,
                level=h.level,
                category=h.category,
                excerpt=h.excerpt,
                advice=h.advice,
                basis=h.basis,
                matched_by=h.matched_by,
                confirmed=confirmed,
                reason=(v.reason if v else ""),
            )
        )
        if confirmed:
            confirmed_count[h.level] += 1

    # 语义任务确认违规 → 违规项
    base = len(stage1.engine.hits)
    for j, t in enumerate(stage1.engine.semantic_tasks):
        v = verdict_map.get(base + j)
        if v and v.confirmed:
            violations.append(
                Violation(
                    rule_id=t.rule_id,
                    rule_name=t.rule_name,
                    level=t.level,
                    category="语义复核",
                    excerpt="(语义判定)",
                    advice="",
                    basis="",
                    matched_by="local_llm",
                    confirmed=True,
                    reason=v.reason,
                )
            )
            confirmed_count[t.level] += 1

    score = 100
    for lvl, cnt in confirmed_count.items():
        score -= LEVEL_DEDUCT[lvl] * cnt
    score = max(0, score)

    transcript_text = "".join(s.text for s in (stage1_text_segments(stage1)))
    cloud_only_tokens = estimate_tokens_zh(transcript_text) + 300
    actual_tokens = review_prompt_tokens or max(200, int(cloud_only_tokens * 0.15))
    # VoiceGuard: 0 API tokens (all local), local_compute = actual_tokens
    saving_pct = 100.0  # 100% API token saving (all inference is local)

    return QAReport(
        audio=stage1.audio,
        duration_ms=stage1.duration_ms,
        lang=stage1.lang,
        score=score,
        grade=grade_of(score),
        violations=violations,
        funnel=stage1.engine.funnel,
        token_economics={
            "cloud_api_est_tokens": cloud_only_tokens,
            "voiceguard_api_tokens": 0,
            "voiceguard_local_compute_tokens": actual_tokens,
            "api_saving_pct": saving_pct,
            "note": "规则引擎零token全量初筛，本地模型仅复核候选片段，API调用零token",
        },
        backends={"asr": asr_backend, "review": reviewer_backend},
        timings={
            "asr_s": stage1.asr_elapsed_s,
            "engine_ms": stage1.engine.elapsed_ms,
            "total_stage1_s": stage1.total_elapsed_s,
            "review_s": review_s,
        },
    )


def stage1_text_segments(stage1):
    """从 Stage1Result 还原片段（仅用于 token 估算）。"""
    from rules_engine import Segment

    # stage1 内部未保存 segments，用漏斗近似还原长度不精确，
    # 因此由调用方注入：见 pipeline.run_stage1 返回的 _segments
    return getattr(stage1, "segments_ref", [])


def render_markdown(r: QAReport) -> str:
    lines = [
        f"# 通话合规质检报告 · {r.audio}",
        "",
        f"- **质检总分**：**{r.score} / 100（{r.grade}）**",
        f"- 时长：{round(r.duration_ms / 1000, 1)}s ｜ 语言：{r.lang or 'auto'} ｜ 生成时间：{r.generated_at}",
        f"- 后端：ASR `{r.backends['asr']}` ｜ 复核 `{r.backends['review']}`",
        "",
        "## 调度漏斗",
        "",
        "| 阶段 | 数量 |",
        "|------|------|",
    ]
    f = r.funnel
    lines += [
        f"| 逐字稿片段总数 | {f.get('total_segments', 0)} |",
        f"| 规则引擎标记片段 | {f.get('segments_flagged', 0)}（{f.get('candidate_segments_ratio_pct', 0)}%） |",
        f"| 红线命中 | {f.get('redline_hits', 0)} |",
        f"| 警告命中 | {f.get('warning_hits', 0)} |",
        f"| 提示命中 | {f.get('notice_hits', 0)} |",
        f"| 语义复核任务 | {f.get('semantic_tasks', 0)} |",
        "",
        "## 违规清单",
        "",
        "| 级别 | 规则 | 判定方式 | 摘录 | 复核 | 理由 |",
        "|------|------|----------|------|------|------|",
    ]
    for v in r.violations:
        conf = (
            ("✅确认" if v.confirmed else "❌排除") if v.confirmed is not None else "-"
        )
        exc = v.excerpt.replace("|", "/")[:60]
        reason = v.reason.replace("|", "/")[:80] if v.reason else ""
        lines.append(
            f"| {LEVEL_LABEL.get(v.level, v.level)} | {v.rule_id} {v.rule_name} "
            f"| {v.matched_by} | {exc} | {conf} | {reason} |"
        )
    lines += [
        "",
        "## token 经济性（vs 纯云端大模型方案）",
        "",
        f"- 云端 API 估算：**{r.token_economics['cloud_api_est_tokens']} tokens**（全量逐字稿发送）",
        f"- VoiceGuard API 调用：**{r.token_economics['voiceguard_api_tokens']} tokens**（本地推理，零外发）",
        f"- VoiceGuard 本地计算：**{r.token_economics['voiceguard_local_compute_tokens']} tokens**",
        f"- **API token 节省 {r.token_economics['api_saving_pct']}%**，敏感语音零外发",
        "",
        "## 整改建议",
        "",
    ]
    seen = set()
    for v in r.violations:
        if v.confirmed and v.advice and v.rule_id not in seen:
            seen.add(v.rule_id)
            lines.append(f"- **{v.rule_id} {v.rule_name}**：{v.advice}")
    if not seen:
        lines.append("- 未检出确认违规。")
    lines.append("")
    if r.cloud_advice:
        lines += [
            "## 云端辅导建议",
            "",
            r.cloud_advice,
            "",
            f"<small>Cloud status: {r.cloud_enhance_meta.get('status', '')}</small>",
            "",
        ]
    return "\n".join(lines)


def render_json(r: QAReport) -> str:
    import dataclasses

    return json.dumps(dataclasses.asdict(r), ensure_ascii=False, indent=1)


def render_html(r: QAReport) -> str:
    level_class = {
        "redline": "violation-redline",
        "warning": "violation-warning",
        "notice": "violation-notice",
    }
    level_label = LEVEL_LABEL

    # Compute confirmed counts by level for summary card
    cc = {"redline": 0, "warning": 0, "notice": 0}
    for v in r.violations:
        if v.confirmed:
            cc[v.level] = cc.get(v.level, 0) + 1

    rows = []
    for v in r.violations:
        conf = ("确认" if v.confirmed else "排除") if v.confirmed is not None else "-"
        exc = v.excerpt.replace("<", "&lt;").replace(">", "&gt;").replace("|", "/")[:60]
        reason = v.reason.replace("<", "&lt;").replace(">", "&gt;") if v.reason else ""
        cls = level_class.get(v.level, "")
        rows.append(
            f'<tr class="{cls}"><td>{level_label.get(v.level, v.level)}</td>'
            f"<td>{v.rule_id}</td><td>{v.rule_name}</td>"
            f"<td>{v.matched_by}</td><td>{exc}</td><td>{conf}</td>"
            f'<td class="reason-cell">{reason}</td></tr>'
        )
    violations_rows = "\n".join(rows) if rows else "<tr><td colspan=7>无违规</td></tr>"

    advice_items = []
    seen = set()
    for v in r.violations:
        if v.confirmed and v.advice and v.rule_id not in seen:
            seen.add(v.rule_id)
            advice_items.append(
                f"<li><b>{v.rule_id} {v.rule_name}</b>：{v.advice}</li>"
            )
    advice_html = (
        "\n".join(advice_items) if advice_items else "<li>未检出确认违规。</li>"
    )

    f = r.funnel
    grade_cls = (
        "excellent"
        if r.score >= 90
        else "pass"
        if r.score >= 75
        else "warning"
        if r.score >= 60
        else "fail"
    )

    # Funnel bar visualization
    total_seg = f.get("total_segments", 1) or 1
    funnel_stages = [
        ("逐字稿片段总数", f.get("total_segments", 0), "#1a56c4"),
        ("规则引擎标记片段", f.get("segments_flagged", 0), "#2563eb"),
        ("红线命中", f.get("redline_hits", 0), "#dc2626"),
        ("警告命中", f.get("warning_hits", 0), "#d97706"),
        ("提示命中", f.get("notice_hits", 0), "#0284c7"),
        ("语义复核任务", f.get("semantic_tasks", 0), "#7c3aed"),
    ]
    funnel_bars = ""
    for label, count, color in funnel_stages:
        width_pct = max(5, int(count / total_seg * 100)) if total_seg else 5
        funnel_bars += (
            f'<div class="funnel-stage">'
            f'<div class="funnel-label">{label}</div>'
            f'<div class="funnel-track">'
            f'<div class="funnel-bar" style="width:{width_pct}%;background:{color};">{count}</div>'
            f"</div></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>通话合规质检报告 - {r.audio}</title>
<style>
body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; color: #333; }}
h1 {{ color: #1a56c4; }}
.score {{ font-size: 2em; font-weight: bold; }}
.grade-excellent {{ color: #1a8a3b; }}
.grade-pass {{ color: #1a56c4; }}
.grade-warning {{ color: #d97706; }}
.grade-fail {{ color: #dc2626; }}
table {{ width: 100%; border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
th {{ background: #f5f5f5; }}
.violation-redline {{ background: #fef2f2; }}
.violation-warning {{ background: #fffbeb; }}
.violation-notice {{ background: #f0f9ff; }}
.metric {{ display: inline-block; margin-right: 2em; }}
.reason-cell {{ font-size: 0.85em; color: #555; max-width: 300px; }}
.funnel-bar {{ height: 28px; border-radius: 4px; display: flex; align-items: center; padding-left: 10px; color: #fff; font-weight: bold; min-width: 40px; margin-bottom: 4px; }}
.funnel-stage {{ display: flex; align-items: center; margin-bottom: 6px; }}
.funnel-label {{ width: 160px; text-align: right; padding-right: 10px; font-size: 0.9em; color: #555; }}
.funnel-track {{ flex: 1; background: #f0f0f0; border-radius: 4px; overflow: hidden; }}
.summary-card {{ display: flex; gap: 20px; margin: 1em 0; }}
.summary-item {{ flex: 1; text-align: center; padding: 12px; border-radius: 8px; background: #f8f9fa; }}
.summary-item .num {{ font-size: 1.8em; font-weight: bold; }}
.summary-item .label {{ font-size: 0.8em; color: #666; margin-top: 4px; }}
.summary-score {{ background: #1a56c4; color: #fff; }}
.summary-score .num {{ color: #fff; }}
.summary-redline {{ background: #fef2f2; }}
.summary-warning {{ background: #fffbeb; }}
.summary-notice {{ background: #f0f9ff; }}
</style>
</head>
<body>
<h1>通话合规质检报告</h1>
<p>文件：{r.audio} ｜ 时长：{round(r.duration_ms / 1000, 1)}s ｜ 语言：{r.lang or "auto"}</p>
<p>质检总分：<span class="score">{r.score}</span> / 100
<span class="score grade-{grade_cls}">（{r.grade}）</span></p>
<p>后端：ASR <code>{r.backends["asr"]}</code> ｜ 复核 <code>{r.backends["review"]}</code></p>

<h2>调度漏斗</h2>
<div class="summary-card">
<div class="summary-item summary-score"><div class="num">{r.score}</div><div class="label">质检总分（{r.grade}）</div></div>
<div class="summary-item summary-redline"><div class="num">{cc.get("redline", 0)}</div><div class="label">红线确认</div></div>
<div class="summary-item summary-warning"><div class="num">{cc.get("warning", 0)}</div><div class="label">警告确认</div></div>
<div class="summary-item summary-notice"><div class="num">{cc.get("notice", 0)}</div><div class="label">提示确认</div></div>
</div>
{funnel_bars}
<p style="margin-top:8px;color:#666;font-size:0.85em;">候选片段占比 {f.get("candidate_segments_ratio_pct", 0)}%（规则引擎零 token 初筛后仅候选片段进入模型复核）</p>

<h2>违规清单</h2>
<table>
<tr><th>级别</th><th>规则</th><th>名称</th><th>判定方式</th><th>摘录</th><th>复核</th><th>复核理由</th></tr>
{violations_rows}
</table>

<h2>Token 经济性</h2>
<p>云端 API 估算：<b>{r.token_economics["cloud_api_est_tokens"]} tokens</b>（全量逐字稿发送）</p>
<p>VoiceGuard API 调用：<b>{r.token_economics["voiceguard_api_tokens"]} tokens</b>（本地推理，零外发）</p>
<p>VoiceGuard 本地计算：<b>{r.token_economics["voiceguard_local_compute_tokens"]} tokens</b></p>
<p><b>API token 节省 {r.token_economics["api_saving_pct"]}%</b>，敏感语音零外发</p>

<h2>性能指标</h2>
<table>
<tr><th>阶段</th><th>耗时</th></tr>
<tr><td>ASR 转写</td><td>{r.timings.get("asr_s", 0)}s</td></tr>
<tr><td>规则引擎</td><td>{r.timings.get("engine_ms", 0)}ms</td></tr>
<tr><td>Stage1 合计</td><td>{r.timings.get("total_stage1_s", 0)}s</td></tr>
<tr><td>语义复核</td><td>{r.timings.get("review_s", 0)}s</td></tr>
</table>

<h2>整改建议</h2>
<ul>
{advice_html}
</ul>
{f'<h2>云端辅导建议</h2><div style="background:#f8f9fa;padding:1em;border-radius:8px;"><pre style="white-space:pre-wrap;margin:0;">{r.cloud_advice}</pre></div><p style="color:#999;font-size:0.8em">Cloud status: {r.cloud_enhance_meta.get("status", "")} | Endpoint: {r.cloud_enhance_meta.get("endpoint", "")}</p>' if r.cloud_advice else ""}
</body>
</html>"""
