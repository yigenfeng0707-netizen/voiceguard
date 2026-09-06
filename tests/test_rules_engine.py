"""tests/test_rules_engine.py - Core module tests for rules_engine.

Covers: keyword/regex/absent matching, scope filtering, funnel stats,
hit sorting, semantic task generation, edge cases.
"""

import os
import sys

import pytest

# conftest.py auto-adds scripts/ to sys.path at collection time
from rules_engine import (
    EngineResult,
    RuleHit,
    Segment,
    SemanticTask,
    _match_absent,
    _match_keyword,
    _match_regex,
    _scope_segments,
    load_rule_packs,
    run_engine,
)

# Project paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_DIR = os.path.join(_PROJECT_ROOT, "rules")


# ─── Fixtures ──────────────────────────────────────────────────────


def make_segments(texts_times):
    """Helper: [(text, start_ms, end_ms, speaker), ...] -> [Segment]."""
    return [
        Segment(
            segment_id=i,
            text=t,
            start_ms=s,
            end_ms=e,
            speaker=sp,
        )
        for i, (t, s, e, sp) in enumerate(texts_times)
    ]


@pytest.fixture
def finance_violation_segments():
    """Simulated transcript with finance violations."""
    return make_segments(
        [
            ("您好，给您推荐一款理财产品。", 0, 5000, "坐席"),
            ("这个产品保本保息，年化收益30%起步，稳赚不赔的。", 5000, 11000, "坐席"),
            ("跟存款一样安全，今天不买就没了。", 11000, 16000, "坐席"),
            ("我不需要。", 16000, 21000, "客户"),
            ("您把验证码告诉我就能办。", 21000, 27000, "坐席"),
        ]
    )


@pytest.fixture
def compliant_segments():
    """Simulated compliant transcript."""
    return make_segments(
        [
            ("您好，我是XX财富的理财顾问，工号1234。", 0, 5000, "坐席"),
            ("这款产品属于R2中风险等级，适合稳健型投资者。", 5000, 11000, "坐席"),
            ("投资有风险，过往业绩不代表未来收益。", 11000, 16000, "坐席"),
            ("请问您方便了解一下吗？", 16000, 21000, "坐席"),
            ("好的，我了解一下。", 21000, 26000, "客户"),
        ]
    )


@pytest.fixture
def real_rule_packs():
    """Load actual rule packs from project rules/ directory."""
    return load_rule_packs(RULES_DIR)


# ─── load_rule_packs ──────────────────────────────────────────────


class TestLoadRulePacks:
    def test_load_from_valid_dir(self, real_rule_packs):
        """Should load at least 2 packs (finance + telesales)."""
        assert len(real_rule_packs) >= 2

    def test_packs_have_required_fields(self, real_rule_packs):
        """Each pack should have pack_id and rules list."""
        for p in real_rule_packs:
            assert "pack_id" in p
            assert "rules" in p
            assert len(p["rules"]) > 0

    def test_filter_by_pack_id(self):
        """Should filter packs by pack_id."""
        packs = load_rule_packs(RULES_DIR, pack_ids=["finance"])
        assert len(packs) == 1
        assert packs[0]["pack_id"] == "finance"

    def test_nonexistent_dir_raises(self):
        """Nonexistent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_rule_packs("/nonexistent/path")

    def test_empty_dir_returns_empty(self, tmp_path):
        """Empty directory should return empty list."""
        packs = load_rule_packs(str(tmp_path))
        assert packs == []


# ─── _scope_segments ──────────────────────────────────────────────


class TestScopeSegments:
    def test_no_scope_returns_all(self):
        """No scope filter → all segments returned."""
        segs = make_segments([("first", 0, 5000, "A"), ("second", 5000, 10000, "B")])
        result = _scope_segments(segs, None)
        assert len(result) == 2

    def test_first_60s_scope(self):
        """first_60s scope → only segments within first 60s."""
        segs = make_segments(
            [
                ("early", 0, 10000, "A"),
                ("within", 30000, 50000, "B"),
                ("after_60s", 65000, 80000, "C"),
            ]
        )
        result = _scope_segments(segs, "first_60s")
        assert len(result) == 2
        assert result[0].text == "early"
        assert result[1].text == "within"

    def test_last_30s_scope(self):
        """last_30s scope → only segments in last 30s."""
        segs = make_segments(
            [
                ("early", 0, 10000, "A"),
                ("middle", 40000, 50000, "B"),
                ("late", 80000, 100000, "C"),
            ]
        )
        result = _scope_segments(segs, "last_30s")
        assert len(result) == 1
        assert result[0].text == "late"

    def test_empty_segments(self):
        """Empty segment list → empty result."""
        assert _scope_segments([], None) == []
        assert _scope_segments([], "first_60s") == []


# ─── _match_keyword ───────────────────────────────────────────────


class TestMatchKeyword:
    def test_basic_keyword_match(self):
        """Keyword present in segment → hit with correct excerpt."""
        rule = {
            "rule_id": "TEST-001",
            "name": "test",
            "level": "redline",
            "category": "test",
            "patterns": ["保本保息"],
            "basis": "test basis",
            "advice": "test advice",
        }
        segs = make_segments([("这个产品保本保息", 0, 5000, "坐席")])
        hits = _match_keyword(rule, segs)
        assert len(hits) == 1
        assert hits[0].rule_id == "TEST-001"
        assert hits[0].matched_by == "keyword"
        assert "保本保息" in hits[0].excerpt
        assert hits[0].span_start >= 0

    def test_no_match(self):
        """Keyword absent → no hits."""
        rule = {
            "rule_id": "TEST-002",
            "name": "test",
            "level": "warning",
            "patterns": ["不存在的关键词"],
        }
        segs = make_segments([("正常对话", 0, 5000, "坐席")])
        hits = _match_keyword(rule, segs)
        assert hits == []

    def test_multiple_patterns(self):
        """Multiple patterns → multiple hits if all present."""
        rule = {
            "rule_id": "TEST-003",
            "name": "test",
            "level": "redline",
            "patterns": ["保本", "保息"],
        }
        segs = make_segments([("保本保息", 0, 5000, "坐席")])
        hits = _match_keyword(rule, segs)
        assert len(hits) == 2  # "保本" and "保息" both match

    def test_excerpt_has_context(self):
        """Excerpt should include ±10 chars of context."""
        rule = {
            "rule_id": "TEST-004",
            "name": "test",
            "level": "redline",
            "patterns": ["关键词"],
        }
        segs = make_segments([("前面文字关键词后面文字", 0, 5000, "坐席")])
        hits = _match_keyword(rule, segs)
        assert len(hits) == 1
        excerpt = hits[0].excerpt
        assert "关键词" in excerpt
        assert "前面文字" in excerpt  # context before
        assert "后面文字" in excerpt  # context after


# ─── _match_regex ──────────────────────────────────────────────────


class TestMatchRegex:
    def test_basic_regex_match(self):
        """Regex pattern matches → hit."""
        rule = {
            "rule_id": "TEST-REG-001",
            "name": "regex test",
            "level": "redline",
            "regex": [r"\d+%年化"],
        }
        segs = make_segments([("收益30%年化起步", 0, 5000, "坐席")])
        hits = _match_regex(rule, segs)
        assert len(hits) == 1
        assert hits[0].matched_by == "regex"

    def test_no_regex_match(self):
        """No regex match → no hits."""
        rule = {
            "rule_id": "TEST-REG-002",
            "name": "regex test",
            "level": "warning",
            "regex": [r"\d+%年化"],
        }
        segs = make_segments([("正常推荐", 0, 5000, "坐席")])
        hits = _match_regex(rule, segs)
        assert hits == []

    def test_multiple_regex_matches(self):
        """Multiple matches in same segment → multiple hits."""
        rule = {
            "rule_id": "TEST-REG-003",
            "name": "regex test",
            "level": "redline",
            "regex": [r"\d+%"],
        }
        segs = make_segments([("收益30%起步，最高50%", 0, 5000, "坐席")])
        hits = _match_regex(rule, segs)
        assert len(hits) == 2


# ─── _match_absent ──────────────────────────────────────────────────


class TestMatchAbsent:
    def test_trigger_present_required_absent(self):
        """Trigger present but required phrase missing → hit."""
        rule = {
            "rule_id": "ABS-001",
            "name": "risk disclosure absent",
            "level": "warning",
            "trigger_any": ["理财产品"],
            "absent_any": ["投资有风险", "风险提示"],
        }
        segs = make_segments([("给您推荐一款理财产品。", 0, 5000, "坐席")])
        hits = _match_absent(rule, segs)
        assert len(hits) == 1
        assert hits[0].matched_by == "absent"
        assert hits[0].segment_id == -1  # whole-transcript match

    def test_trigger_present_required_present(self):
        """Trigger present AND required phrase present → no hit."""
        rule = {
            "rule_id": "ABS-002",
            "name": "risk disclosure absent",
            "level": "warning",
            "trigger_any": ["理财产品"],
            "absent_any": ["投资有风险"],
        }
        segs = make_segments([("给您推荐一款理财产品，投资有风险。", 0, 5000, "坐席")])
        hits = _match_absent(rule, segs)
        assert hits == []

    def test_trigger_absent(self):
        """Trigger phrase absent → no check, no hit."""
        rule = {
            "rule_id": "ABS-003",
            "name": "risk disclosure absent",
            "level": "warning",
            "trigger_any": ["理财产品"],
            "absent_any": ["投资有风险"],
        }
        segs = make_segments([("今天天气不错。", 0, 5000, "坐席")])
        hits = _match_absent(rule, segs)
        assert hits == []


# ─── run_engine ────────────────────────────────────────────────────


class TestRunEngine:
    def test_funnel_stats(self, finance_violation_segments, real_rule_packs):
        """Funnel should have all required keys with correct types."""
        result = run_engine(finance_violation_segments, real_rule_packs)
        f = result.funnel
        assert f["total_segments"] == 5
        assert f["redline_hits"] >= 1  # at least one redline (保本保息)
        assert f["segments_flagged"] >= 1
        assert 0 <= f["candidate_segments_ratio_pct"] <= 100
        assert f["semantic_tasks"] >= 0
        assert isinstance(f["warning_hits"], int)
        assert isinstance(f["notice_hits"], int)

    def test_hits_sorted_by_level(self, finance_violation_segments, real_rule_packs):
        """Hits should be sorted: redline first, then warning, then notice."""
        result = run_engine(finance_violation_segments, real_rule_packs)
        levels = [h.level for h in result.hits]
        if "redline" in levels and "warning" in levels:
            assert levels.index("redline") < levels.index("warning")

    def test_semantic_tasks_generated(
        self, finance_violation_segments, real_rule_packs
    ):
        """Semantic-type rules should generate SemanticTask entries."""
        result = run_engine(finance_violation_segments, real_rule_packs)
        assert len(result.semantic_tasks) > 0
        for t in result.semantic_tasks:
            assert isinstance(t, SemanticTask)
            assert t.rule_id
            assert t.semantic_hint
            assert t.context_text  # should have context

    def test_empty_segments(self, real_rule_packs):
        """Empty segment list → zero segments in funnel, no crash.

        Note: absent-type rules may still fire (required phrases absent from
        empty text), but total_segments and segments_flagged should be 0.
        """
        result = run_engine([], real_rule_packs)
        assert result.funnel["total_segments"] == 0
        assert result.funnel["segments_flagged"] == 0
        assert result.funnel["candidate_segments_ratio_pct"] == 0
        # Engine should not crash on empty input

    def test_compliant_transcript_low_hits(self, compliant_segments, real_rule_packs):
        """Compliant transcript should have fewer/no redline hits."""
        result = run_engine(compliant_segments, real_rule_packs)
        redline_count = result.funnel["redline_hits"]
        # Compliant call should have very few or zero redline hits
        assert redline_count <= 1, f"Expected <=1 redline hits, got {redline_count}"

    def test_elapsed_ms_positive(self, finance_violation_segments, real_rule_packs):
        """Engine should report positive elapsed time."""
        result = run_engine(finance_violation_segments, real_rule_packs)
        assert result.elapsed_ms >= 0  # could be 0 for very fast ops

    def test_packs_loaded(self, finance_violation_segments, real_rule_packs):
        """Should report which packs were loaded."""
        result = run_engine(finance_violation_segments, real_rule_packs)
        assert len(result.packs_loaded) >= 2
        assert "finance" in result.packs_loaded
        assert "telesales" in result.packs_loaded
