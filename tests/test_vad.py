"""Tests for pipeline/vad.py — VAD filtering logic and edge cases."""
from unittest.mock import patch

import pytest

from pipeline.vad import apply_vad_filter, get_speech_timestamps, SPEECH_RATIO_WARNING


class TestGetSpeechTimestamps:
    """get_speech_timestamps() wraps Silero VAD with graceful fallbacks."""

    @patch("torch.hub.load", side_effect=ImportError("no silero-vad"))
    def test_fallback_on_import_error(self, mock_load):
        """When silero-vad not installed, returns empty list (full audio pass-through)."""
        result = get_speech_timestamps("/fake/audio.wav")
        assert result == []

    @patch("torch.hub.load", side_effect=Exception("model load failed"))
    def test_fallback_on_any_exception(self, mock_load):
        result = get_speech_timestamps("/fake/audio.wav")
        assert result == []


class TestApplyVadFilter:
    """apply_vad_filter() decides when to filter vs pass-through."""

    @patch("pipeline.vad._get_duration_ffprobe", return_value=0.0)
    @patch("shutil.copy2")
    def test_passthrough_on_zero_duration(self, mock_copy, mock_dur):
        """If ffprobe returns 0 duration, copy original."""
        result_path, ratio = apply_vad_filter("/fake/input.wav", "/fake/output.wav")
        assert ratio == 1.0
        mock_copy.assert_called_once()

    @patch("pipeline.vad._get_duration_ffprobe", return_value=30.0)
    @patch("pipeline.vad.get_speech_timestamps", return_value=[])
    @patch("shutil.copy2")
    def test_passthrough_on_no_speech(self, mock_copy, mock_ts, mock_dur):
        """If VAD returns no timestamps, copy original and report 1.0."""
        result_path, ratio = apply_vad_filter("/fake/input.wav", "/fake/output.wav", threshold=0.5)
        assert ratio == 1.0
        mock_copy.assert_called_once()

    @patch("pipeline.vad._get_duration_ffprobe", return_value=30.0)
    @patch("pipeline.vad.get_speech_timestamps",
           return_value=[{"start": 0.0, "end": 29.0}])
    @patch("shutil.copy2")
    def test_skip_filter_when_high_speech_ratio(self, mock_copy, mock_ts, mock_dur):
        """If speech_ratio > 0.90, skip filtering and copy original (dense speech)."""
        result_path, ratio = apply_vad_filter("/fake/input.wav", "/fake/output.wav")
        assert ratio > 0.90
        mock_copy.assert_called_once()

    @patch("pipeline.vad._run_ffmpeg")  # prevent actual ffmpeg call
    @patch("shutil.copy2")  # prevent fallback copy
    @patch("pipeline.vad._get_duration_ffprobe", return_value=30.0)
    @patch("pipeline.vad.get_speech_timestamps",
           return_value=[{"start": 0.0, "end": 10.0}])
    def test_filter_applied_when_moderate_speech(self, mock_ts, mock_dur, mock_copy,
                                                  mock_ffmpeg):
        """~33% speech ratio should trigger actual filtering (not skipped, not fallback).
        The VAD applies SEGMENT_PAD (0.1s) padding around the 10s segment,
        so the actual ratio is slightly higher than 10/30 = 0.333."""
        result_path, ratio = apply_vad_filter("/fake/input.wav", "/fake/output.wav")
        assert 0.33 < ratio < 0.90  # moderate speech, below skip threshold
        # Should have called _run_ffmpeg (not fallback copy)
        mock_ffmpeg.assert_called_once()
        mock_copy.assert_not_called()

    @patch("pipeline.vad._run_ffmpeg")  # prevent actual ffmpeg call
    @patch("shutil.copy2")
    @patch("pipeline.vad._get_duration_ffprobe", return_value=10.0)
    @patch("pipeline.vad.get_speech_timestamps",
           return_value=[{"start": 0.0, "end": 1.0}])
    def test_low_speech_ratio_triggers_warning(self, mock_ts, mock_dur, mock_copy,
                                                mock_ffmpeg, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        apply_vad_filter("/fake/input.wav", "/fake/output.wav")
        # VAD padding adds 0.1s on each side → 1.2s speech / 10s total ≈ 12%
        # The threshold for warning is 15%, so it should fire
        assert any("only" in record.message and "%" in record.message
                   for record in caplog.records if "VAD" in record.message)
