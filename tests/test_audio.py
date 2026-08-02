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


class TestSeparationNotice:
    """separate_background() falls back to silence rather than failing, which
    is invisible in the output — so it has to say what it could not do."""

    def test_reports_when_no_backend_is_installed(self, tmp_path, monkeypatch):
        import pipeline.audio as audio

        monkeypatch.setattr(audio, "_separate_demucs",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("demucs not installed — run: pip install demucs")))
        monkeypatch.setattr(audio, "_separate_audio_separator",
                            lambda *a, **k: (_ for _ in ()).throw(ImportError("no module")))
        monkeypatch.setattr(audio, "get_duration", lambda p: 10.0)
        monkeypatch.setattr(audio, "_make_silent_bg",
                            lambda dur, out: str(tmp_path / "background.wav"))

        src = tmp_path / "audio_hq.wav"
        src.write_bytes(b"RIFF")
        notices = []
        vocals, bg = audio.separate_background(str(src), str(tmp_path), notices=notices)

        assert vocals and bg, "the dub must still continue"
        assert [n["code"] for n in notices] == ["audio.separation_unavailable"]
        n = notices[0]
        assert n["severity"] == "warn"
        assert any("pip install demucs" in s for s in n["remediation"])
        assert "demucs: not installed" in n["detail"]

    def test_silent_when_separation_succeeds(self, tmp_path, monkeypatch):
        import pipeline.audio as audio
        monkeypatch.setattr(audio, "_separate_demucs",
                            lambda *a, **k: ("v.wav", "b.wav"))
        notices = []
        assert audio.separate_background("x.wav", str(tmp_path), notices=notices) == ("v.wav", "b.wav")
        assert notices == []

    def test_notices_argument_is_optional(self, tmp_path, monkeypatch):
        """Old callers must not crash."""
        import pipeline.audio as audio
        monkeypatch.setattr(audio, "_separate_demucs",
                            lambda *a, **k: ("v.wav", "b.wav"))
        assert audio.separate_background("x.wav", str(tmp_path)) == ("v.wav", "b.wav")
