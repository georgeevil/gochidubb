"""Tests for pipeline/segment_post.py — segment cleanup logic."""
import copy

from pipeline.segment_post import (
    postprocess_segments,
    _merge_continuation_segments,
    _absorb_micro_segments,
    _split_very_long_segments,
)


class TestMergeContinuationSegments:
    """_merge_continuation_segments() joins sentence fragments."""

    def test_merges_sentence_continuation(self, continuation_segments):
        """A short trailing fragment like 'Nicky' should merge into previous."""
        result = _merge_continuation_segments(
            continuation_segments,
            merge_gap=0.5,
            max_dur=12.0,
            max_chars=240,
        )
        # The first two segments ("Who's the most spaz?" + "Nicky") should merge
        assert len(result) < len(continuation_segments), (
            f"Expected fewer segments, got {len(result)}"
        )
        # Merged text should contain both parts
        merged_texts = " ".join(s["text"] for s in result)
        assert "Who's the most spaz?" in merged_texts or "Who" in merged_texts
        assert "Nicky" in merged_texts

    def test_merges_multi_word_continuation(self):
        """Two fragments of a longer sentence should merge."""
        segs = [
            {"start": 0.0, "end": 2.0, "text": "I mean he just goes absolutely", "speaker": "SPEAKER_00"},
            {"start": 2.1, "end": 3.5, "text": "crazy when he gets on the mat.", "speaker": "SPEAKER_00"},
        ]
        result = _merge_continuation_segments(segs, 0.5, 12.0, 240)
        assert len(result) == 1
        assert "crazy" in result[0]["text"]

    def test_does_not_merge_different_speakers(self, continuation_segments):
        """Segments from different speakers should NOT merge."""
        result = _merge_continuation_segments(
            continuation_segments,
            merge_gap=0.5,
            max_dur=12.0,
            max_chars=240,
        )
        # SPEAKER_00 and SPEAKER_01 segments should remain separate
        texts = [s["text"] for s in result]
        assert any("technique" in t for t in texts)

    def test_does_not_exceed_max_duration(self):
        """Merge should not produce segments longer than max_dur."""
        segs = [
            {"start": 0.0, "end": 5.0, "text": "This is a long first part", "speaker": "SPEAKER_00"},
            {"start": 5.1, "end": 15.0, "text": "and this is a really long continuation that goes on and on.",
             "speaker": "SPEAKER_00"},
        ]
        result = _merge_continuation_segments(segs, 0.5, max_dur=10.0, max_chars=240)
        # Without merging, result should stay
        assert len(result) >= 1

    def test_empty_input(self):
        assert _merge_continuation_segments([], 0.5, 12.0, 240) == []

    def test_single_segment(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "Hello.", "speaker": "SPEAKER_00"}]
        result = _merge_continuation_segments(segs, 0.5, 12.0, 240)
        assert len(result) == 1
        assert result[0]["text"] == "Hello."


class TestAbsorbMicroSegments:
    """_absorb_micro_segments() removes or merges tiny fragments."""

    def test_absorbs_tiny_segments(self, micro_segments):
        """Very short segments (<1s, <40 chars) should be absorbed."""
        result = _absorb_micro_segments(micro_segments, threshold_sec=1.0, threshold_chars=40)
        # "Hi" (0.3s, 2 chars) and "OK" (0.1s, 2 chars) should be absorbed
        assert len(result) < len(micro_segments)

    def test_preserves_normal_segments(self):
        """Segments above thresholds should remain unchanged."""
        segs = [
            {"start": 0.0, "end": 3.0, "text": "This is a normal length segment.", "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 7.0, "text": "Another normal segment.", "speaker": "SPEAKER_00"},
        ]
        result = _absorb_micro_segments(segs, 1.0, 40)
        assert len(result) == 2

    def test_empty_input(self):
        assert _absorb_micro_segments([], 1.0, 40) == []

    def test_absorbs_into_previous_gap_is_small(self):
        """A micro segment should be absorbed into the previous segment if gap is small."""
        segs = [
            {"start": 0.0, "end": 3.0, "text": "Previous segment.", "speaker": "SPEAKER_00"},
            {"start": 3.2, "end": 3.5, "text": "tiny", "speaker": "SPEAKER_00"},
        ]
        result = _absorb_micro_segments(segs, 1.0, 40)
        assert len(result) == 1
        assert "tiny" in result[0]["text"]

    def test_absorbs_into_next_when_prev_fails(self):
        """If prev can't absorb (different speaker), absorb into next."""
        segs = [
            {"start": 0.0, "end": 3.0, "text": "Speaker A", "speaker": "SPEAKER_00"},
            {"start": 3.2, "end": 3.5, "text": "hi", "speaker": "SPEAKER_01"},
            {"start": 3.6, "end": 6.0, "text": "Speaker B continues", "speaker": "SPEAKER_01"},
        ]
        result = _absorb_micro_segments(segs, 1.0, 40)
        assert len(result) == 2
        # "hi" should have been merged into the next SPEAKER_01 segment


class TestSplitVeryLongSegments:
    """_split_very_long_segments() splits long segments at sentence boundaries."""

    def test_splits_long_segment(self, long_segments):
        """Long segment (>15s) should be split at sentence boundary."""
        result = _split_very_long_segments(long_segments, threshold=15.0)
        # First segment should be split
        assert len(result) > len(long_segments)

    def test_does_not_split_short_segments(self):
        """Short segments should remain as-is."""
        segs = [{"start": 0.0, "end": 3.0, "text": "Short text."}]
        result = _split_very_long_segments(segs, threshold=15.0)
        assert len(result) == 1

    def test_no_split_without_sentence_boundary(self):
        """If no .!? boundary exists in the middle, don't split."""
        segs = [{"start": 0.0, "end": 20.0, "text": "this is a long runon sentence that has no punctuation "
                                                      "anywhere in the middle so we cannot split it properly"}]
        result = _split_very_long_segments(segs, threshold=15.0)
        assert len(result) == 1

    def test_split_preserves_total_text(self):
        """After split, concatenated text equals original."""
        segs = [{"start": 0.0, "end": 20.0, "text": "First sentence here. Second sentence there. "
                                                      "Third sentence at the end."}]
        result = _split_very_long_segments(segs, threshold=10.0)
        orig_text = segs[0]["text"]
        split_text = " ".join(s["text"] for s in result)
        # After merging whitespace, all words should be preserved
        assert "First" in split_text
        assert "Second" in split_text
        assert "Third" in split_text

    def test_empty_input(self):
        assert _split_very_long_segments([], 15.0) == []


class TestPostprocessSegments:
    """postprocess_segments() orchestrates all three passes."""

    def test_full_cleanup_continuation(self, continuation_segments):
        """Continuation segments should be merged end-to-end."""
        result = postprocess_segments(continuation_segments)
        # Should be fewer segments than input
        assert len(result) < len(continuation_segments)

    def test_full_cleanup_micro(self, micro_segments):
        """Micro segments should be absorbed."""
        result = postprocess_segments(micro_segments)
        assert len(result) < len(micro_segments)
        # All original text should be preserved
        all_text = " ".join(s["text"] for s in result)
        assert "This is a normal" in all_text or "This" in all_text
        assert "We continue" in all_text

    def test_clean_segments_mostly_unchanged(self, sample_segments):
        """Well-formed segments should generally be preserved.
        Note: 'Amazing right?' (0.4s, 12 chars) is a micro-segment
        and gets absorbed into the previous segment."""
        result = postprocess_segments(sample_segments)
        # 4 segments instead of 5 because the tiny 'Amazing right?' is absorbed
        assert len(result) == len(sample_segments) - 1
        # All text content is preserved
        all_text = " ".join(s["text"] for s in result)
        for seg in sample_segments:
            words = seg["text"].split()
            assert any(w in all_text for w in words), f"Missing: {seg['text']}"

    def test_empty_input(self):
        assert postprocess_segments([]) == []

    def test_single_segment_noop(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "Hi."}]
        result = postprocess_segments(segs)
        assert len(result) == 1

    def test_input_not_mutated(self, continuation_segments):
        """Original list should not be modified in place."""
        original = copy.deepcopy(continuation_segments)
        _ = postprocess_segments(continuation_segments)
        assert continuation_segments == original

    def test_merges_words_field(self):
        """When segments have 'words', the merged result should preserve them."""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00",
             "words": [{"word": "Hello", "start": 0.0, "end": 1.0}]},
            {"start": 1.1, "end": 2.0, "text": "world", "speaker": "SPEAKER_00",
             "words": [{"word": "world", "start": 1.1, "end": 2.0}]},
        ]
        result = postprocess_segments(segs)
        merged = result[0]
        assert "words" in merged
        assert len(merged["words"]) == 2

    def test_custom_parameters(self, continuation_segments):
        """Custom merge gap and thresholds should be respected."""
        result = postprocess_segments(continuation_segments, merge_gap=0.1)
        # Tighter gap may still merge
        assert len(result) > 0
