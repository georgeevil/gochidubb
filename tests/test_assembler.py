"""Tests for pipeline/assembler.py — SRT formatting, writing, and audio assembly."""
import os
import tempfile

import pytest

from pipeline.assembler import (
    MIN_SEGMENT_GAP,
    format_srt_time,
    plan_segment_fit,
    write_srt,
)


class TestFormatSrtTime:
    """format_srt_time() converts float seconds to SRT timestamp format."""

    def test_zero(self):
        assert format_srt_time(0.0) == "00:00:00,000"

    def test_simple_seconds(self):
        assert format_srt_time(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert format_srt_time(65.0) == "00:01:05,000"

    def test_hours(self):
        assert format_srt_time(3661.0) == "01:01:01,000"

    def test_milliseconds_rounding(self):
        # 0.12345 seconds → 123 ms
        result = format_srt_time(0.12345)
        assert result == "00:00:00,123" or result == "00:00:00,123"

    def test_large_value(self):
        assert format_srt_time(10000.0) == "02:46:40,000"

    def test_fractional_seconds(self):
        result = format_srt_time(12.345)
        assert result.endswith("345")
        assert result.startswith("00:00:12")


class TestWriteSrt:
    """write_srt() writes standard .srt format."""

    def test_writes_segments(self, sample_segments):
        """Writes valid SRT content with sequential numbering."""
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt.srt")
        try:
            write_srt(sample_segments, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()

            # Check structure: index, timestamp, text, blank line
            assert "1" in content
            assert "00:00:00,500" in content or "00:00:00,500" in content
            assert "Hello everyone" in content
            # Should have segment count entries
            assert content.count("-->") == len(sample_segments)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_uses_translated_text(self):
        """If translated_text exists, use it instead of text."""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "translated_text": "Hola"},
        ]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_translated.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "Hola" in content
            assert "Hello" not in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_empty_segments(self):
        """Empty segment list should produce empty SRT."""
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_empty.srt")
        try:
            write_srt([], tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert content == ""
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_output_is_utf8(self):
        """SRT should be UTF-8 encoded for non-ASCII support."""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Привет мир"},
        ]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_utf8.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "Привет мир" in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_prefers_dubbed_timings_when_present(self):
        """Subtitles must follow the audio, not the source it was made from.

        The SRT is first written before assembly, so a segment that had to be
        shifted or compressed leaves the file out of sync by exactly that
        amount — a 10s drift shipped this way.
        """
        segs = [{
            "start": 1.0, "end": 2.0, "text": "Hello",
            "placed_start": 3.5, "placed_end": 5.25,
        }]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_placed.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "00:00:03,500 --> 00:00:05,250" in content
            assert "00:00:01,000" not in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_falls_back_to_source_timings_when_unplaced(self):
        """A transcript that was never assembled has no placements."""
        segs = [{"start": 1.0, "end": 2.0, "text": "Hello"}]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_unplaced.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "00:00:01,000 --> 00:00:02,000" in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


class TestPlanSegmentFit:
    """plan_segment_fit() decides placement and time-compression together."""

    def test_leaves_a_fitting_segment_alone(self):
        start, speed = plan_segment_fit(
            seg_start=5.0, next_start=10.0, tts_dur=3.0, current_end=4.0)
        assert start == 5.0
        assert speed == 1.0

    def test_uses_the_pause_after_a_segment_before_compressing(self):
        """A 6s clip in a 4s slot followed by 4s of silence needs no stretch.

        Budgeting to the segment's own `end` would have compressed this;
        budgeting to the next segment's start is what makes the overrun free.
        """
        start, speed = plan_segment_fit(
            seg_start=1.0, next_start=9.0, tts_dur=6.0, current_end=0.0)
        assert start == 1.0
        assert speed == 1.0

    def test_compresses_to_exactly_fit_when_the_ceiling_allows(self):
        room = 5.0 - MIN_SEGMENT_GAP - 1.0
        start, speed = plan_segment_fit(
            seg_start=1.0, next_start=5.0, tts_dur=5.0, current_end=0.0)
        assert start == 1.0
        assert speed > 1.15
        assert 5.0 / speed == pytest.approx(room)

    def test_compression_stops_at_the_ceiling_and_spills_the_remainder(self):
        """Quality wins over sync once the stretch would be audible.

        The leftover isn't lost: the next segment starts late, so its own
        budget shrinks and it absorbs what's left rather than passing it on.
        """
        room = 5.0 - MIN_SEGMENT_GAP - 1.0
        _, speed = plan_segment_fit(
            seg_start=1.0, next_start=5.0, tts_dur=6.0, current_end=0.0,
            max_stretch=1.4)
        assert speed == pytest.approx(1.4)
        assert 6.0 / speed > room

    def test_never_stretches_short_audio_out(self):
        _, speed = plan_segment_fit(
            seg_start=1.0, next_start=20.0, tts_dur=0.5, current_end=0.0)
        assert speed == 1.0

    def test_pushes_a_segment_that_would_overlap_the_previous_one(self):
        start, _ = plan_segment_fit(
            seg_start=5.0, next_start=12.0, tts_dur=1.0, current_end=6.0)
        assert start == pytest.approx(6.0 + MIN_SEGMENT_GAP)

    def test_a_late_segment_compresses_harder_to_catch_up(self):
        """The drift is paid off by the segment that inherited it.

        This is the whole fix: previously a late segment kept its full
        duration and handed the delay to the next one, so the error grew
        monotonically instead of being absorbed.
        """
        on_time = plan_segment_fit(
            seg_start=5.0, next_start=10.0, tts_dur=6.0, current_end=0.0)[1]
        late = plan_segment_fit(
            seg_start=5.0, next_start=10.0, tts_dur=6.0, current_end=7.0)[1]
        assert late > on_time

    def test_respects_the_stretch_ceiling(self):
        _, speed = plan_segment_fit(
            seg_start=1.0, next_start=2.0, tts_dur=30.0, current_end=0.0,
            max_stretch=1.4)
        assert speed == pytest.approx(1.4)

    def test_does_not_compress_into_a_vanished_slot(self):
        """Past the next segment's start there is no room to budget against.

        Dividing by that would demand an absurd speed; the segment is simply
        allowed to run long and the ceiling handles the rest.
        """
        start, speed = plan_segment_fit(
            seg_start=5.0, next_start=6.0, tts_dur=4.0, current_end=8.0)
        assert start == pytest.approx(8.0 + MIN_SEGMENT_GAP)
        assert speed == 1.0

    def test_drift_does_not_accumulate_across_a_natural_pause(self):
        """A real gap in the source lets the dub resynchronise for free."""
        _, speed = plan_segment_fit(
            seg_start=30.0, next_start=36.0, tts_dur=4.0, current_end=25.0)
        assert speed == 1.0
