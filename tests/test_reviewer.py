"""tests/test_reviewer.py - Core module tests for reviewer._parse_verdicts.

Covers: JSON array parsing, thinking tag stripping (critical bug regression),
individual JSON fallback, empty/malformed output, index matching,
default reason fallback for missing reason field.
"""

import os
import sys

import pytest

# conftest.py auto-adds scripts/ to sys.path at collection time
from reviewer import LocalReviewer, ReviewCandidate, ReviewVerdict


# ─── Helpers ─────────────────────────────────────────────────────


def make_candidate(index, rule_id="TEST-001", rule_name="test rule"):
    return ReviewCandidate(
        index=index,
        rule_id=rule_id,
        rule_name=rule_name,
        semantic_hint="test hint",
        excerpt="保本保息",
        context="坐席说保本保息",
    )


def make_candidates(n):
    return [make_candidate(i) for i in range(n)]


# ─── _parse_verdicts: Normal JSON ────────────────────────────────


class TestParseVerdictsNormal:
    def test_single_json_array(self):
        """Single candidate, valid JSON array → correct verdict."""
        cands = make_candidates(1)
        text = '[{"index": 0, "confirmed": true, "reason": "说话人说了违规话术"}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 1
        assert verdicts[0].confirmed is True
        assert verdicts[0].reason == "说话人说了违规话术"

    def test_multiple_json_array(self):
        """Multiple candidates, valid JSON array → all verdicts correct."""
        cands = make_candidates(3)
        text = (
            '[{"index": 0, "confirmed": true, "reason": "违规A"},'
            ' {"index": 1, "confirmed": false, "reason": "合规B"},'
            ' {"index": 2, "confirmed": true, "reason": "违规C"}]'
        )
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 3
        assert verdicts[0].confirmed is True
        assert verdicts[1].confirmed is False
        assert verdicts[2].confirmed is True
        assert verdicts[0].reason == "违规A"

    def test_confirmed_false(self):
        """confirmed: false → verdict.confirmed is False."""
        cands = make_candidates(1)
        text = '[{"index": 0, "confirmed": false, "reason": "未发现违规"}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert verdicts[0].confirmed is False
        assert verdicts[0].reason == "未发现违规"


# ─── _parse_verdicts: Thinking tag stripping (CRITICAL) ──────────


class TestParseVerdictsThinkingTags:
    """Regression tests for the critical bug where <think></think> tags
    were being swallowed into empty strings by the Python string parser,
    causing all model output to be deleted."""

    def _make_think_tag(self):
        """Construct think tags safely (same method as the fix)."""
        return chr(60) + "think" + chr(62), chr(60) + "/think" + chr(62)

    def test_closed_think_tag_stripped(self):
        """Think tags with content before JSON → JSON parsed correctly."""
        cands = make_candidates(1)
        open_t, close_t = self._make_think_tag()
        text = (
            f"{open_t}Let me analyze...{close_t}"
            + '[{"index": 0, "confirmed": true, "reason": "test"}]'
        )
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 1
        assert verdicts[0].confirmed is True
        assert verdicts[0].reason == "test"

    def test_unclosed_think_tag(self):
        """Unclosed think tag → should find JSON after think content."""
        cands = make_candidates(1)
        open_t, _ = self._make_think_tag()
        # Think tag opened but never closed, JSON appears later
        text = (
            f"{open_t}thinking thinking..."
            + '{"index": 0, "confirmed": true, "reason": "found"}'
        )
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 1
        assert verdicts[0].confirmed is True
        assert "found" in verdicts[0].reason

    def test_no_think_tag(self):
        """No think tags → JSON parsed directly."""
        cands = make_candidates(1)
        text = '[{"index": 0, "confirmed": true, "reason": "direct"}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert verdicts[0].reason == "direct"

    def test_empty_think_tag_not_swallowed(self):
        """Empty think tags (the original bug) should not corrupt output.

        The original bug: rfind("") returns len(text), causing all content
        to be deleted. This test ensures the fix handles this case.
        """
        cands = make_candidates(1)
        open_t, close_t = self._make_think_tag()
        # The exact pattern that caused the original bug
        text = (
            f"{open_t}{close_t}"
            + '[{"index": 0, "confirmed": true, "reason": "survived"}]'
        )
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 1
        assert verdicts[0].confirmed is True
        assert verdicts[0].reason == "survived"


# ─── _parse_verdicts: Individual JSON fallback ────────────────────


class TestParseVerdictsFallback:
    def test_individual_json_objects(self):
        """No array wrapper, individual JSON objects → parsed correctly."""
        cands = make_candidates(2)
        text = '{"index": 0, "confirmed": true, "reason": "A"} {"index": 1, "confirmed": false, "reason": "B"}'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 2
        assert verdicts[0].confirmed is True
        assert verdicts[1].confirmed is False

    def test_single_json_object(self):
        """Single JSON object (not in array) → parsed."""
        cands = make_candidates(1)
        text = '{"index": 0, "confirmed": true, "reason": "single"}'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 1
        assert verdicts[0].reason == "single"


# ─── _parse_verdicts: Edge cases ─────────────────────────────────


class TestParseVerdictsEdgeCases:
    def test_empty_text(self):
        """Empty string → all candidates get default (unconfirmed)."""
        cands = make_candidates(2)
        verdicts = LocalReviewer._parse_verdicts("", cands)
        assert len(verdicts) == 2
        assert all(v.confirmed is False for v in verdicts)

    def test_garbage_text(self):
        """Garbage text → all candidates unconfirmed with default reason."""
        cands = make_candidates(1)
        verdicts = LocalReviewer._parse_verdicts("this is not JSON at all", cands)
        assert len(verdicts) == 1
        assert verdicts[0].confirmed is False

    def test_missing_reason_field_confirmed(self):
        """Confirmed=true but no reason field → default reason."""
        cands = make_candidates(1)
        text = '[{"index": 0, "confirmed": true}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert verdicts[0].confirmed is True
        assert verdicts[0].reason != ""  # should have default reason
        assert "违规" in verdicts[0].reason

    def test_missing_reason_field_rejected(self):
        """Confirmed=false but no reason field → default reason."""
        cands = make_candidates(1)
        text = '[{"index": 0, "confirmed": false}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert verdicts[0].confirmed is False
        assert verdicts[0].reason != ""
        assert "未发现" in verdicts[0].reason

    def test_index_mismatch(self):
        """Model returns wrong index → candidate gets default (unconfirmed)."""
        cands = make_candidates(1)  # index=0
        text = '[{"index": 99, "confirmed": true, "reason": "wrong index"}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts) == 1
        assert verdicts[0].confirmed is False  # no match for index 0

    def test_reason_truncated(self):
        """Very long reason → truncated to 200 chars."""
        cands = make_candidates(1)
        long_reason = "A" * 500
        text = f'[{{"index": 0, "confirmed": true, "reason": "{long_reason}"}}]'
        verdicts = LocalReviewer._parse_verdicts(text, cands)
        assert len(verdicts[0].reason) <= 200

    def test_no_candidates(self):
        """Empty candidate list → empty verdicts."""
        verdicts = LocalReviewer._parse_verdicts(
            '[{"index": 0, "confirmed": true}]', []
        )
        assert verdicts == []


# ─── ReviewCandidate/ReviewVerdict dataclasses ───────────────────


class TestDataclasses:
    def test_review_candidate_fields(self):
        """ReviewCandidate should have all required fields."""
        c = ReviewCandidate(
            index=0,
            rule_id="FIN-001",
            rule_name="test",
            semantic_hint="hint",
            excerpt="excerpt",
            context="context",
        )
        assert c.index == 0
        assert c.rule_id == "FIN-001"
        assert c.context == "context"

    def test_review_verdict_fields(self):
        """ReviewVerdict should have all required fields."""
        v = ReviewVerdict(
            index=0,
            rule_id="FIN-001",
            confirmed=True,
            reason="test reason",
        )
        assert v.confirmed is True
        assert v.reason == "test reason"
