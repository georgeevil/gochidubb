"""Tests for pipeline/assembler.py — SRT formatting, writing, and audio assembly."""
import os
import tempfile

import pytest

from pipeline.assembler import format_srt_time, write_srt


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
