"""Tests for pipeline/audio.py — audio extraction and separation utilities."""
from unittest.mock import patch

import pytest

from pipeline.audio import get_duration


class TestGetDuration:
    """get_duration() parses ffprobe output to get audio duration."""

    @patch("subprocess.run")
    def test_valid_duration(self, mock_run):
        """Normal ffprobe output returns a float."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "123.456\n"
        assert get_duration("/fake/audio.wav") == 123.456

    @patch("subprocess.run")
    def test_zero_duration(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "0.0\n"
        assert get_duration("/fake/audio.wav") == 0.0

    @patch("subprocess.run")
    def test_fallback_on_error(self, mock_run):
        """When subprocess returns non-zero, float conversion still works
        if stdout is parseable; the actual error path is ValueError/AttributeError in float()."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "N/A\n"
        assert get_duration("/fake/audio.wav") == 0.0

    @patch("subprocess.run")
    def test_fallback_on_non_numeric(self, mock_run):
        """When ffprobe returns non-numeric output, return 0.0."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "N/A\n"
        assert get_duration("/fake/audio.wav") == 0.0
