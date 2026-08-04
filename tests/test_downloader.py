"""Tests for pipeline/downloader.py — yt-dlp discovery, metadata, cookies."""
from unittest.mock import patch

import pytest

from app.config import cfg
from pipeline.downloader import (
    _cookie_args,
    _find_ytdlp,
    _parse_probe_json,
    curate_metadata,
    probe_metadata,
)


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


# ── curate_metadata ──────────────────────────────────────────────────

FULL_INFO = {
    "id": "dQw4w9WgXcQ",
    "title": "Never Gonna Give You Up",
    "channel": "Rick Astley",
    "uploader": "RickAstleyVEVO",
    "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
    "duration": 213,
    "view_count": 1_400_000_000,
    "like_count": 16_000_000,
    "upload_date": "20091025",
    "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxres.jpg",
    "categories": ["Entertainment"],
    "tags": [f"tag{i}" for i in range(30)],
    "language": "en",
    "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "description": "x" * 3000,
}


class TestCurateMetadata:
    def test_full_info(self):
        m = curate_metadata(FULL_INFO)
        assert m["video_id"] == "dQw4w9WgXcQ"
        assert m["title"] == "Never Gonna Give You Up"
        assert m["channel"] == "Rick Astley"          # channel preferred
        assert m["channel_id"] == "UCuAXFkgsw1L7xaCfnd5JJOw"
        assert m["duration"] == 213
        assert m["view_count"] == 1_400_000_000
        assert m["like_count"] == 16_000_000
        assert m["upload_date"] == "20091025"
        assert m["thumbnail"].startswith("https://")
        assert m["categories"] == ["Entertainment"]
        assert len(m["tags"]) == 20                   # capped at 20
        assert m["language"] == "en"
        assert m["webpage_url"].endswith("dQw4w9WgXcQ")
        assert len(m["description"]) == 2000          # capped at 2000 chars
        assert m["is_music"] is False

    def test_uploader_fallback_for_channel(self):
        info = dict(FULL_INFO)
        del info["channel"]
        assert curate_metadata(info)["channel"] == "RickAstleyVEVO"

    def test_music_via_categories(self):
        info = {**FULL_INFO, "categories": ["Music"]}
        assert curate_metadata(info)["is_music"] is True

    @pytest.mark.parametrize("key", ["artist", "track"])
    def test_music_via_artist_or_track(self, key):
        info = {**FULL_INFO, key: "Rick Astley"}
        assert curate_metadata(info)["is_music"] is True

    def test_empty_artist_is_not_music(self):
        info = {**FULL_INFO, "artist": "", "track": None}
        assert curate_metadata(info)["is_music"] is False

    @pytest.mark.parametrize("title", [
        "Never Gonna Give You Up (Official Video)",
        "Never Gonna Give You Up (Official Music Video)",
        "SONG NAME - OFFICIAL VIDEO",
    ])
    def test_music_via_title_heuristic(self, title):
        info = {**FULL_INFO, "title": title}
        assert curate_metadata(info)["is_music"] is True

    def test_missing_keys(self):
        m = curate_metadata({})
        assert m["video_id"] is None
        assert m["title"] is None
        assert m["channel"] is None
        assert m["duration"] is None
        assert m["categories"] == []
        assert m["tags"] == []
        assert m["description"] == ""
        assert m["is_music"] is False

    def test_null_fields(self):
        """yt-dlp emits explicit nulls for unavailable fields."""
        m = curate_metadata({
            "id": "abc", "title": None, "categories": None,
            "tags": None, "description": None, "duration": None,
        })
        assert m["video_id"] == "abc"
        assert m["tags"] == []
        assert m["description"] == ""
        assert m["is_music"] is False


# ── _cookie_args ─────────────────────────────────────────────────────

class TestCookieArgs:
    def test_neither_set(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "")
        assert _cookie_args() == []

    def test_browser_only(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "firefox")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "")
        assert _cookie_args() == ["--cookies-from-browser", "firefox"]

    def test_file_only(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "/tmp/cookies.txt")
        assert _cookie_args() == ["--cookies", "/tmp/cookies.txt"]

    def test_both(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "chrome")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "/tmp/cookies.txt")
        assert _cookie_args() == [
            "--cookies-from-browser", "chrome",
            "--cookies", "/tmp/cookies.txt",
        ]

    def test_whitespace_is_empty(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "  ")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", " ")
        assert _cookie_args() == []


# ── probe JSON parsing (no subprocess) ───────────────────────────────

class TestParseProbeJson:
    def test_valid_dict(self):
        assert _parse_probe_json('{"id": "abc", "duration": 10}') == {
            "id": "abc", "duration": 10,
        }

    def test_invalid_json(self):
        assert _parse_probe_json("ERROR: not json") is None

    def test_empty(self):
        assert _parse_probe_json("") is None

    def test_none(self):
        assert _parse_probe_json(None) is None

    def test_non_dict_json(self):
        assert _parse_probe_json("[1, 2, 3]") is None


class TestProbeMetadataNonUrl:
    """Non-URL sources return None without ever invoking a subprocess."""

    @pytest.mark.parametrize("source", [
        "/path/to/local.mp4", "not a url", "", "ftp://weird",
    ])
    def test_non_url_returns_none(self, source):
        with patch("subprocess.run") as mock_run:
            assert probe_metadata(source) is None
        mock_run.assert_not_called()

    def test_none_source(self):
        assert probe_metadata(None) is None
