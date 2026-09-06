"""质检规则引擎（分级调度第一级）- 零 token 确定性筛查。

设计目标：
1. 用关键词/正则/缺失检测对全量逐字稿做确定性初筛，零模型开销；
2. 命中项产出带 evidence_span 的候选违规，交由第二级（本地小模型）语义复核；
3. `semantic` 类规则不做确定性判断，直接生成复核任务（含语义提示与上下文）；
4. 输出漏斗统计（总片段数/被标记片段数/各级别命中数），作为路演可视化数据。

本模块不依赖任何模型与网络，保证纯本地、可离线、可审计。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LEVEL_ORDER = {"redline": 0, "warning": 1, "notice": 2}


@dataclass
class Segment:
    """逐字稿片段（ASR 输出的一行）。"""

    segment_id: int
    text: str
    start_ms: int = 0
    end_ms: int = 0
    speaker: str = ""


@dataclass
class RuleHit:
    """规则引擎命中记录（候选违规，待模型复核确认）。"""

    rule_id: str
    rule_name: str
    level: str
    category: str
    matched_by: str  # keyword | regex | absent
    segment_id: int  # absent 类为 -1（全篇）
    span_start: int  # 片段内字符偏移
    span_end: int
    excerpt: str  # 命中原文摘录
    basis: str = ""
    advice: str = ""


@dataclass
class SemanticTask:
    """语义复核任务（交由本地小模型的第二级输入）。"""

    rule_id: str
    rule_name: str
    level: str
    semantic_hint: str
    context_segments: List[int] = field(default_factory=list)
    context_text: str = ""


@dataclass
class EngineResult:
    hits: List[RuleHit] = field(default_factory=list)
    semantic_tasks: List[SemanticTask] = field(default_factory=list)
    funnel: Dict = field(default_factory=dict)
    elapsed_ms: int = 0
    packs_loaded: List[str] = field(default_factory=list)


def load_rule_packs(rules_dir: str, pack_ids: Optional[List[str]] = None) -> List[dict]:
    """加载规则包（可按 pack_id 过滤）。"""
    packs = []
    if not os.path.isdir(rules_dir):
        raise FileNotFoundError(f"规则目录不存在: {rules_dir}")
    for fn in sorted(os.listdir(rules_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(rules_dir, fn), "r", encoding="utf-8") as f:
            pack = json.load(f)
        if pack_ids and pack.get("pack_id") not in pack_ids:
            continue
        packs.append(pack)
    return packs


def _scope_segments(segments: List[Segment], scope: Optional[str]) -> List[Segment]:
    """按规则 scope 截取检测范围：first_60s / last_30s / 全篇。"""
    if not segments:
        return []
    if scope == "first_60s":
        t0 = segments[0].start_ms
        return [s for s in segments if s.start_ms - t0 <= 60_000]
    if scope == "last_30s":
        t_end = max(s.end_ms for s in segments)
        return [s for s in segments if s.end_ms >= t_end - 30_000]
    return segments


def _match_keyword(rule: dict, segments: List[Segment]) -> List[RuleHit]:
    hits = []
    for seg in _scope_segments(segments, rule.get("scope")):
        for kw in rule.get("patterns", []):
            idx = seg.text.find(kw)
            if idx >= 0:
                hits.append(
                    RuleHit(
                        rule_id=rule["rule_id"],
                        rule_name=rule["name"],
                        level=rule["level"],
                        category=rule.get("category", ""),
                        matched_by="keyword",
                        segment_id=seg.segment_id,
                        span_start=idx,
                        span_end=idx + len(kw),
                        excerpt=seg.text[max(0, idx - 10) : idx + len(kw) + 10],
                        basis=rule.get("basis", ""),
                        advice=rule.get("advice", ""),
                    )
                )
    return hits


def _match_regex(rule: dict, segments: List[Segment]) -> List[RuleHit]:
    hits = []
    for seg in _scope_segments(segments, rule.get("scope")):
        for pat in rule.get("regex", []):
            for m in re.finditer(pat, seg.text):
                hits.append(
                    RuleHit(
                        rule_id=rule["rule_id"],
                        rule_name=rule["name"],
                        level=rule["level"],
                        category=rule.get("category", ""),
                        matched_by="regex",
                        segment_id=seg.segment_id,
                        span_start=m.start(),
                        span_end=m.end(),
                        excerpt=seg.text[max(0, m.start() - 10) : m.end() + 10],
                        basis=rule.get("basis", ""),
                        advice=rule.get("advice", ""),
                    )
                )
    return hits


def _match_absent(rule: dict, segments: List[Segment]) -> List[RuleHit]:
    """缺失检测：触发词出现（说明在做相关行为）但必备表述全篇/范围内缺失。"""
    scope_segs = _scope_segments(segments, rule.get("scope"))
    scope_text = "".join(s.text for s in scope_segs)
    full_text = "".join(s.text for s in segments)

    triggers = rule.get("trigger_any")
    if triggers and not any(t in full_text for t in triggers):
        return []  # 未触发相关行为，不检查缺失

    absent_keys = rule.get("absent_any", [])
    for key in absent_keys:
        if re.search(key, full_text):
            return []  # 任一必备表述存在即通过
    return [
        RuleHit(
            rule_id=rule["rule_id"],
            rule_name=rule["name"],
            level=rule["level"],
            category=rule.get("category", ""),
            matched_by="absent",
            segment_id=-1,
            span_start=-1,
            span_end=-1,
            excerpt=f"全篇缺失必备表述：{'/'.join(absent_keys[:3])}",
            basis=rule.get("basis", ""),
            advice=rule.get("advice", ""),
        )
    ]


def run_engine(
    segments: List[Segment],
    packs: List[dict],
    semantic_context_radius: int = 2,
) -> EngineResult:
    """对逐字稿执行全部确定性规则 + 生成语义复核任务。"""
    t0 = time.perf_counter()
    result = EngineResult(packs_loaded=[p.get("pack_id", "?") for p in packs])
    flagged_segments = set()

    for pack in packs:
        for rule in pack.get("rules", []):
            mode = rule.get("match_mode", "keyword")
            if mode == "keyword":
                hits = _match_keyword(rule, segments)
            elif mode == "regex":
                hits = _match_regex(rule, segments)
            elif mode == "absent":
                hits = _match_absent(rule, segments)
            elif mode == "semantic":
                # 语义类规则：全文作为上下文生成复核任务
                ctx_ids = [s.segment_id for s in segments]
                result.semantic_tasks.append(
                    SemanticTask(
                        rule_id=rule["rule_id"],
                        rule_name=rule["name"],
                        level=rule["level"],
                        semantic_hint=rule.get("semantic_hint", ""),
                        context_segments=ctx_ids,
                        context_text="\n".join(
                            f"[{s.speaker or '说话人'}] {s.text}" for s in segments
                        )[:4000],
                    )
                )
                continue
            else:
                continue
            for h in hits:
                result.hits.append(h)
                if h.segment_id >= 0:
                    flagged_segments.add(h.segment_id)

    result.hits.sort(
        key=lambda h: (LEVEL_ORDER.get(h.level, 9), h.segment_id, h.span_start)
    )
    result.funnel = {
        "total_segments": len(segments),
        "segments_flagged": len(flagged_segments),
        "redline_hits": sum(1 for h in result.hits if h.level == "redline"),
        "warning_hits": sum(1 for h in result.hits if h.level == "warning"),
        "notice_hits": sum(1 for h in result.hits if h.level == "notice"),
        "semantic_tasks": len(result.semantic_tasks),
        "candidate_segments_ratio_pct": round(
            100.0 * len(flagged_segments) / len(segments), 1
        )
        if segments
        else 0,
    }
    result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return result


if __name__ == "__main__":
    # 自测：构造含违规话术的模拟逐字稿
    demo = [
        Segment(
            0,
            "喂，您好，这里是鑫源财富管理中心，给您推荐一款理财产品。",
            0,
            5000,
            "坐席",
        ),
        Segment(
            1, "这个产品保本保息，年化收益30%起步，稳赚不赔的。", 5000, 11000, "坐席"
        ),
        Segment(
            2, "跟存款一样安全，今天不买就没了，最后一小时。", 11000, 16000, "坐席"
        ),
        Segment(3, "我不需要，你们怎么又打来了，昨天刚打过。", 16000, 21000, "客户"),
        Segment(
            4,
            "您先别挂，我再说最后一点，您把验证码告诉我就能办。",
            21000,
            27000,
            "坐席",
        ),
    ]
    packs = load_rule_packs(os.path.join(os.path.dirname(__file__), "..", "rules"))
    res = run_engine(demo, packs)
    print(
        json.dumps(
            {
                "funnel": res.funnel,
                "hits": [
                    {
                        "rule": h.rule_id,
                        "name": h.rule_name,
                        "level": h.level,
                        "by": h.matched_by,
                        "excerpt": h.excerpt,
                    }
                    for h in res.hits
                ],
                "semantic_tasks": [t.rule_id for t in res.semantic_tasks],
                "elapsed_ms": res.elapsed_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
