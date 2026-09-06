"""tests/test_report_gen.py - Tests for report generation.

Covers: grade_of scoring, estimate_tokens_zh, build_report structure,
render_json serialization (critical: dataclass → dict), render_html, render_markdown.
"""

import json
import os
import sys

import pytest

# conftest.py auto-adds scripts/ to sys.path at collection time
from report_gen import (
    QAReport,
    Violation,
    build_report,
    estimate_tokens_zh,
    grade_of,
    render_html,
    render_json,
    render_markdown,
)
from rules_engine import EngineResult, RuleHit, Segment


# ─── grade_of ─────────────────────────────────────────────────────


class TestGradeOf:
    def test_excellent(self):
        assert grade_of(95) == "优秀"
        assert grade_of(90) == "优秀"

    def test_pass(self):
        assert grade_of(89) == "合格"
        assert grade_of(75) == "合格"

    def test_warning(self):
        assert grade_of(74) == "待整改"
        assert grade_of(60) == "待整改"

    def test_fail(self):
        assert grade_of(59) == "不合格"
        assert grade_of(0) == "不合格"


# ─── estimate_tokens_zh ───────────────────────────────────────────


class TestEstimateTokens:
    def test_chinese_text(self):
        """Chinese text: ~0.7 tokens per char."""
        text = "你好世界"  # 4 chars
        tokens = estimate_tokens_zh(text)
        assert tokens == int(4 * 0.7)  # 2

    def test_english_text(self):
        """English text: ~1 token per 4 chars."""
        text = "Hello"  # 5 chars, all non-Chinese
        tokens = estimate_tokens_zh(text)
        assert tokens == int(5 / 4)  # 1

    def test_mixed_text(self):
        """Mixed CN+EN text."""
        text = "你好hello"  # 2 CN + 5 EN
        tokens = estimate_tokens_zh(text)
        expected = int(2 * 0.7 + 5 / 4)  # 1 + 1 = 2
        assert tokens == expected

    def test_empty_string(self):
        """Empty string → 0 tokens."""
        assert estimate_tokens_zh("") == 0


# ─── build_report ─────────────────────────────────────────────────


def make_mock_stage1(hits=None, semantic_tasks=None, segments=None):
    """Create a mock Stage1Result for build_report testing."""
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class MockStage1:
        audio: str = "test.wav"
        duration_ms: int = 10000
        lang: str = "zh"
        segments_ref: list = field(default_factory=list)
        engine: EngineResult = field(default_factory=EngineResult)
        asr_elapsed_s: float = 5.0
        total_elapsed_s: float = 5.0

    eng = EngineResult(
        hits=hits or [],
        semantic_tasks=semantic_tasks or [],
        funnel={
            "total_segments": len(segments) if segments else 0,
            "segments_flagged": 1,
            "redline_hits": sum(1 for h in (hits or []) if h.level == "redline"),
            "warning_hits": sum(1 for h in (hits or []) if h.level == "warning"),
            "notice_hits": sum(1 for h in (hits or []) if h.level == "notice"),
            "semantic_tasks": len(semantic_tasks or []),
            "candidate_segments_ratio_pct": 50.0,
        },
        elapsed_ms=8,
        packs_loaded=["finance", "telesales"],
    )
    return MockStage1(
        segments_ref=segments or [],
        engine=eng,
    )


class TestBuildReport:
    def test_no_violations(self):
        """No hits → score=100, grade=优秀."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        assert report.score == 100
        assert report.grade == "优秀"
        assert len(report.violations) == 0

    def test_redline_deduction(self):
        """One confirmed redline → -15 points."""
        hit = RuleHit(
            rule_id="FIN-001",
            rule_name="test",
            level="redline",
            category="test",
            matched_by="keyword",
            segment_id=0,
            span_start=0,
            span_end=4,
            excerpt="保本保息",
        )
        s1 = make_mock_stage1(hits=[hit])
        report = build_report(s1)
        assert report.score == 85  # 100 - 15

    def test_warning_deduction(self):
        """One confirmed warning → -5 points."""
        hit = RuleHit(
            rule_id="FIN-005",
            rule_name="test",
            level="warning",
            category="test",
            matched_by="keyword",
            segment_id=0,
            span_start=0,
            span_end=4,
            excerpt="限时",
        )
        s1 = make_mock_stage1(hits=[hit])
        report = build_report(s1)
        assert report.score == 95  # 100 - 5

    def test_score_floor_zero(self):
        """Many violations → score floors at 0."""
        hits = [
            RuleHit(
                rule_id=f"FIN-{i}",
                rule_name="test",
                level="redline",
                category="test",
                matched_by="keyword",
                segment_id=0,
                span_start=0,
                span_end=4,
                excerpt="x",
            )
            for i in range(20)
        ]
        s1 = make_mock_stage1(hits=hits)
        report = build_report(s1)
        assert report.score == 0  # 100 - 20*15 = -200 → max(0, -200) = 0
        assert report.grade == "不合格"

    def test_verdict_overrides_confirmed(self):
        """Verdict confirmed=False should override default True."""
        from reviewer import ReviewVerdict

        hit = RuleHit(
            rule_id="FIN-001",
            rule_name="test",
            level="redline",
            category="test",
            matched_by="keyword",
            segment_id=0,
            span_start=0,
            span_end=4,
            excerpt="保本保息",
        )
        s1 = make_mock_stage1(hits=[hit])
        verdicts = [
            ReviewVerdict(index=0, rule_id="FIN-001", confirmed=False, reason="合规")
        ]
        report = build_report(s1, verdicts=verdicts)
        assert report.score == 100  # overruled, no deduction
        assert report.violations[0].confirmed is False
        assert report.violations[0].reason == "合规"

    def test_token_economics_fields(self):
        """Token economics should have correct field names."""
        s1 = make_mock_stage1()
        report = build_report(s1, review_prompt_tokens=500)
        te = report.token_economics
        assert "cloud_api_est_tokens" in te
        assert "voiceguard_api_tokens" in te
        assert te["voiceguard_api_tokens"] == 0
        assert te["api_saving_pct"] == 100.0

    def test_timings_include_review_s(self):
        """Timings should include review_s when passed."""
        s1 = make_mock_stage1()
        report = build_report(s1, review_s=42.5)
        assert report.timings["review_s"] == 42.5


# ─── render_json (CRITICAL serialization fix) ────────────────────


class TestRenderJson:
    def test_violations_are_dicts_not_strings(self):
        """CRITICAL: Violations must be JSON objects, not repr strings.

        Regression test for the bug where json.dumps(default=str) serialized
        Violation dataclass objects as their __repr__ string.
        """
        hit = RuleHit(
            rule_id="FIN-001",
            rule_name="test",
            level="redline",
            category="test",
            matched_by="keyword",
            segment_id=0,
            span_start=0,
            span_end=4,
            excerpt="保本保息",
        )
        s1 = make_mock_stage1(hits=[hit])
        report = build_report(s1)
        text = render_json(report)
        data = json.loads(text)
        assert isinstance(data["violations"], list)
        assert len(data["violations"]) > 0
        assert isinstance(data["violations"][0], dict)  # NOT a string!
        assert "rule_id" in data["violations"][0]
        assert "confirmed" in data["violations"][0]
        assert "reason" in data["violations"][0]

    def test_all_fields_present(self):
        """All Violation fields should be in JSON output."""
        hit = RuleHit(
            rule_id="FIN-001",
            rule_name="保本保息",
            level="redline",
            category="收益承诺",
            matched_by="keyword",
            segment_id=0,
            span_start=0,
            span_end=4,
            excerpt="保本保息",
            basis="法规",
            advice="禁止",
        )
        s1 = make_mock_stage1(hits=[hit])
        report = build_report(s1)
        data = json.loads(render_json(report))
        v = data["violations"][0]
        for field in [
            "rule_id",
            "rule_name",
            "level",
            "category",
            "excerpt",
            "advice",
            "basis",
            "matched_by",
            "confirmed",
            "reason",
        ]:
            assert field in v, f"Missing field: {field}"

    def test_top_level_fields(self):
        """Top-level QAReport fields should be in JSON."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        data = json.loads(render_json(report))
        for field in [
            "audio",
            "duration_ms",
            "lang",
            "score",
            "grade",
            "violations",
            "funnel",
            "token_economics",
            "backends",
            "timings",
            "generated_at",
        ]:
            assert field in data, f"Missing top-level field: {field}"


# ─── render_html ──────────────────────────────────────────────────


class TestRenderHtml:
    def test_has_violation_table(self):
        """HTML should contain a violations table."""
        hit = RuleHit(
            rule_id="FIN-001",
            rule_name="test",
            level="redline",
            category="test",
            matched_by="keyword",
            segment_id=0,
            span_start=0,
            span_end=4,
            excerpt="保本保息",
        )
        s1 = make_mock_stage1(hits=[hit])
        report = build_report(s1)
        html = render_html(report)
        assert "<table>" in html
        assert "FIN-001" in html

    def test_has_reason_column(self):
        """HTML violations table should have a reason column."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        html = render_html(report)
        assert "复核理由" in html

    def test_has_score(self):
        """HTML should display the score."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        html = render_html(report)
        assert str(report.score) in html

    def test_has_token_economics(self):
        """HTML should show token economics."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        html = render_html(report)
        assert "API token" in html or "token" in html.lower()

    def test_has_timings(self):
        """HTML should show performance metrics."""
        s1 = make_mock_stage1()
        report = build_report(s1, review_s=10.5)
        html = render_html(report)
        assert "性能指标" in html
        assert "语义复核" in html


# ─── render_markdown ──────────────────────────────────────────────


class TestRenderMarkdown:
    def test_has_markdown_headers(self):
        """Markdown should have # headers."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        md = render_markdown(report)
        assert md.startswith("#")

    def test_has_score(self):
        """Markdown should display the score."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        md = render_markdown(report)
        assert str(report.score) in md

    def test_has_reason_column(self):
        """Markdown table should have a reason column."""
        s1 = make_mock_stage1()
        report = build_report(s1)
        md = render_markdown(report)
        assert "理由" in md
