"""tests/test_transcribe.py - Pure function tests for transcribe utilities.

Covers: _strip_tags, _split_to_segments (boundary splitting, merge/split rules,
time distribution, edge cases). Does NOT test funasr model loading (heavy dep).
"""

import os
import sys

import pytest

# conftest.py auto-adds scripts/ to sys.path at collection time
from transcribe import (
    TranscriptSegment,
    _split_to_segments,
    _strip_tags,
    _TAG_RE,
    _BOUNDARY_RE,
    _MIN_SEG_CHARS,
    _MAX_SEG_CHARS,
)


# ─── _strip_tags ──────────────────────────────────────────────────


class TestStripTags:
    def test_strips_language_tag(self):
        """Language tags like <|zh|> should be removed."""
        assert _strip_tags("<|zh|>你好世界") == "你好世界"

    def test_strips_emotion_tag(self):
        """Emotion tags like <|HAPPY|> should be removed."""
        assert _strip_tags("<|HAPPY|>我很开心") == "我很开心"

    def test_strips_event_tag(self):
        """Event tags like <|Speech|> should be removed."""
        assert _strip_tags("<|Speech|>测试文本") == "测试文本"

    def test_strips_multiple_tags(self):
        """Multiple tags should all be removed."""
        result = _strip_tags("<|zh|><|HAPPY|>你好<|Speech|>世界")
        assert result == "你好世界"

    def test_no_tags(self):
        """Text without tags → unchanged."""
        assert _strip_tags("普通文本") == "普通文本"

    def test_empty_string(self):
        """Empty string → empty string."""
        assert _strip_tags("") == ""

    def test_tags_only(self):
        """Only tags → empty string."""
        assert _strip_tags("<|zh|><|HAPPY|>") == ""


# ─── _split_to_segments: Normal cases ─────────────────────────────


class TestSplitToSegments:
    def test_single_sentence(self):
        """Single sentence → one segment."""
        text = "这是一个完整的句子。"
        segs = _split_to_segments(text, 10000)
        assert len(segs) == 1
        assert segs[0].text == text
        assert segs[0].start_ms == 0
        assert segs[0].end_ms == 10000

    def test_multiple_sentences(self):
        """Multiple sentences (each >= _MIN_SEG_CHARS) → multiple segments.

        Short sentences (< 8 chars) are merged; use longer text to test split.
        """
        text = (
            "这是一个比较长的句子内容。又一个比较长的句子内容！最后还有一段较长的内容？"
        )
        segs = _split_to_segments(text, 30000)
        assert len(segs) == 3
        assert "第一" in segs[0].text or "比较长" in segs[0].text
        assert "又" in segs[1].text or "比较长" in segs[1].text
        assert "最后" in segs[2].text

    def test_time_distribution(self):
        """Time should be distributed proportionally to character count."""
        text = "短。这是一个比较长的句子内容。"
        segs = _split_to_segments(text, 30000)
        total_chars = sum(len(s.text) for s in segs)
        for seg in segs:
            expected_dur = int(30000 * len(seg.text) / total_chars)
            assert abs(seg.end_ms - seg.start_ms - expected_dur) <= 1

    def test_segment_ids_sequential(self):
        """Segment IDs should be 0, 1, 2, ..."""
        text = "第一句。第二句。第三句。"
        segs = _split_to_segments(text, 30000)
        for i, seg in enumerate(segs):
            assert seg.segment_id == i

    def test_contiguous_time(self):
        """Segments should have contiguous time (no gaps/overlaps)."""
        text = "第一句。第二句。第三句。"
        segs = _split_to_segments(text, 30000)
        for i in range(len(segs) - 1):
            assert segs[i].end_ms == segs[i + 1].start_ms

    def test_total_time_covers_duration(self):
        """Last segment's end_ms should equal total_ms."""
        text = "第一句。第二句。"
        segs = _split_to_segments(text, 20000)
        assert segs[-1].end_ms == 20000


# ─── _split_to_segments: Edge cases ───────────────────────────────


class TestSplitToSegmentsEdge:
    def test_empty_text(self):
        """Empty text → empty segments list."""
        segs = _split_to_segments("", 10000)
        assert segs == []

    def test_no_punctuation(self):
        """Text without punctuation → single segment."""
        text = "没有标点符号的长文本内容"
        segs = _split_to_segments(text, 10000)
        assert len(segs) == 1
        assert segs[0].text == text

    def test_english_boundaries(self):
        """English sentence boundaries (.!?) should also split."""
        text = "Hello world. How are you! I am fine."
        segs = _split_to_segments(text, 30000)
        assert len(segs) >= 2

    def test_short_segment_merged(self):
        """Segments shorter than _MIN_SEG_CHARS should be merged."""
        text = "长句子内容。短。又是一个长句子内容。"
        segs = _split_to_segments(text, 30000)
        # "短" (1 char) should be merged into adjacent segment
        for seg in segs:
            assert len(seg.text) >= _MIN_SEG_CHARS or len(segs) == 1

    def test_long_segment_split(self):
        """Segments longer than _MAX_SEG_CHARS should be split."""
        long_text = "测试" * 100 + "。"  # 200+ chars
        segs = _split_to_segments(long_text, 60000)
        for seg in segs:
            assert len(seg.text) <= _MAX_SEG_CHARS

    def test_zero_duration(self):
        """Zero duration → all segments have 0 time."""
        text = "测试。测试。"
        segs = _split_to_segments(text, 0)
        assert all(s.start_ms == 0 and s.end_ms == 0 for s in segs)


# ─── Regex pattern validation ─────────────────────────────────────


class TestRegexPatterns:
    def test_tag_regex_matches(self):
        """_TAG_RE should match SenseVoice tags."""
        assert _TAG_RE.search("<|zh|>")
        assert _TAG_RE.search("<|HAPPY|>")
        assert _TAG_RE.search("<|Speech|>")

    def test_tag_regex_no_false_match(self):
        """_TAG_RE should not match non-tag text."""
        assert not _TAG_RE.search("正常文本")
        assert not _TAG_RE.search("<not a tag>")

    def test_boundary_regex_chinese(self):
        """_BOUNDARY_RE should match Chinese sentence boundaries."""
        assert _BOUNDARY_RE.search("测试。")
        assert _BOUNDARY_RE.search("测试！")
        assert _BOUNDARY_RE.search("测试？")

    def test_boundary_regex_english(self):
        """_BOUNDARY_RE should match English sentence boundaries."""
        assert _BOUNDARY_RE.search("Hello world.")
        assert _BOUNDARY_RE.search("Hello!")

    def test_boundary_regex_no_false_match(self):
        """_BOUNDARY_RE should not match non-boundary text."""
        assert not _BOUNDARY_RE.search("测试文本")
        assert not _BOUNDARY_RE.search("3.14")
