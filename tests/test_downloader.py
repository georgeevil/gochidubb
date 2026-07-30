"""Tests for pipeline/downloader.py — yt-dlp discovery and video download."""
from unittest.mock import patch

import pytest

from pipeline.downloader import _find_ytdlp


class TestFindYtdlp:
    """_find_ytdlp() locates the yt-dlp executable across platforms."""

    @patch("sys.platform", "win32")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_win32_not_found(self, mock_isfile, mock_which):
        """On Windows, if nothing is found, return None."""
        result = _find_ytdlp()
        assert result is None

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_linux_not_found(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result is None

    @patch("sys.platform", "darwin")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_macos_not_found(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result is None

    @patch("sys.platform", "darwin")
    @patch("shutil.which", side_effect=lambda c: "/usr/local/bin/yt-dlp" if c == "yt-dlp" else None)
    @patch("os.path.isfile", return_value=False)
    def test_macos_found_in_path(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result == "/usr/local/bin/yt-dlp"

    @patch("sys.platform", "win32")
    @patch("shutil.which", side_effect=lambda c: c if c == "yt-dlp.exe" else None)
    @patch("os.path.isfile", return_value=False)
    def test_win32_found_in_path(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result == "yt-dlp.exe"

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=True)
    def test_linux_found_as_file(self, mock_isfile, mock_which):
        """Even if which() fails, isfile check on candidate paths can find it."""
        # All isfile calls return True, so first candidate matches
        result = _find_ytdlp()
        assert result is not None
