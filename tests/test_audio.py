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


class TestTensorAudioIO:
    """load/save_audio_tensor must not route through torchcodec.

    torchaudio.load/save dispatch to torchcodec, which binds to a fixed set
    of FFmpeg majors. On a machine with a newer FFmpeg (Homebrew ffmpeg 8 →
    libavcodec.62) every call raises "Could not load libtorchcodec", which
    silently degraded demucs to a silent background and disabled VAD.
    """

    def test_round_trips_a_wav_without_torchaudio(self, tmp_path, monkeypatch):
        import numpy as np
        import soundfile as sf
        import torch

        from pipeline.audio import load_audio_tensor, save_audio_tensor

        # Any attempt to reach torchaudio's codec path is a failure
        def boom(*a, **k):
            raise RuntimeError("Could not load libtorchcodec")

        import torchaudio
        monkeypatch.setattr(torchaudio, "load", boom)
        monkeypatch.setattr(torchaudio, "save", boom)

        src = tmp_path / "in.wav"
        tone = np.sin(np.linspace(0, 40, 4000, dtype="float32")) * 0.5
        sf.write(src, np.stack([tone, tone], axis=1), 44100, subtype="PCM_16")

        wav, sr = load_audio_tensor(str(src))
        assert sr == 44100
        assert wav.shape == (2, 4000)          # (channels, time)
        assert wav.dtype == torch.float32

        out = tmp_path / "out.wav"
        save_audio_tensor(str(out), wav, sr)
        again, sr2 = load_audio_tensor(str(out))
        assert sr2 == 44100
        assert torch.allclose(wav, again, atol=1e-4)

    def test_overshooting_stems_are_scaled_not_clipped(self, tmp_path):
        import torch

        from pipeline.audio import load_audio_tensor, save_audio_tensor

        # Demucs stems can exceed ±1.0; 16-bit PCM would hard-clip them
        loud = torch.cat([torch.full((1, 100), 1.6), torch.full((1, 100), -0.8)], dim=1)
        out = tmp_path / "loud.wav"
        save_audio_tensor(str(out), loud, 44100)

        back, _ = load_audio_tensor(str(out))
        assert float(back.abs().max()) <= 1.0
        # Scaled, so the internal ratio survives — clipping would flatten it
        assert abs(float(back[0, 0] / back[0, 150]) - (1.6 / -0.8)) < 0.01

    def test_mono_1d_tensor_is_accepted(self, tmp_path):
        import torch

        from pipeline.audio import load_audio_tensor, save_audio_tensor

        save_audio_tensor(str(tmp_path / "m.wav"), torch.zeros(500), 16000)
        wav, sr = load_audio_tensor(str(tmp_path / "m.wav"))
        assert sr == 16000 and wav.shape == (1, 500)


class TestSeparationRemediationMatchesTheCause:
    """Telling someone to `pip install demucs` when demucs is installed and
    merely broken is how a real fix gets skipped for a day."""

    def _notice(self, tmp_path, monkeypatch, demucs_error):
        from pipeline import audio as A

        def boom(*a, **k):
            raise demucs_error

        monkeypatch.setattr(A, "_separate_demucs", boom)
        monkeypatch.setattr(A, "_separate_audio_separator",
                            lambda *a, **k: (_ for _ in ()).throw(ImportError("nope")))
        monkeypatch.setattr(A, "get_duration", lambda p: 1.0)
        monkeypatch.setattr(A, "_make_silent_bg",
                            lambda d, o: str(tmp_path / "background.wav"))
        src = tmp_path / "in.wav"
        src.write_bytes(b"RIFF0000WAVE")
        notices = []
        A.separate_background(str(src), str(tmp_path), notices=notices)
        return notices[0]

    def test_missing_demucs_says_install_it(self, tmp_path, monkeypatch):
        n = self._notice(tmp_path, monkeypatch,
                         RuntimeError("demucs not installed — run: pip install demucs"))
        assert any("pip install demucs" in r for r in n["remediation"])

    def test_broken_demucs_does_not_say_install_it(self, tmp_path, monkeypatch):
        n = self._notice(tmp_path, monkeypatch,
                         RuntimeError("demucs failed: RuntimeError: Could not load libtorchcodec"))
        joined = " ".join(n["remediation"])
        assert "IS installed" in joined
        assert "pip install demucs " not in joined, \
            "must not send the user to reinstall a package that is present"
        # The real reason still has to reach the user
        assert "libtorchcodec" in n["detail"]


class TestDemucsErrorClassification:
    """Only a failed *import* means "not installed". The old code wrapped the
    whole function body in `except ImportError`, so an ImportError raised deep
    inside demucs' own model loading surfaced as "demucs not installed" — and
    the user reinstalled a package that was never missing.

    Built on stub modules rather than the real demucs: another test in the
    suite replaces sys.modules["demucs"], and this assertion is about our
    error handling, not about demucs being importable.
    """

    def _stub_demucs(self, monkeypatch, get_model):
        import sys
        import types
        pkg = types.ModuleType("demucs")
        pkg.__path__ = []                      # make it a package
        apply_mod = types.ModuleType("demucs.apply")
        apply_mod.apply_model = lambda *a, **k: None
        pre_mod = types.ModuleType("demucs.pretrained")
        pre_mod.get_model = get_model
        monkeypatch.setitem(sys.modules, "demucs", pkg)
        monkeypatch.setitem(sys.modules, "demucs.apply", apply_mod)
        monkeypatch.setitem(sys.modules, "demucs.pretrained", pre_mod)

    def test_an_internal_import_error_is_not_reported_as_missing(self, monkeypatch):
        from pipeline import audio as A

        def get_model(_name):
            raise ImportError("cannot import name 'foo' from 'demucs.states'")

        self._stub_demucs(monkeypatch, get_model)
        with pytest.raises(RuntimeError) as ei:
            A._separate_demucs("nonexistent.wav", "/tmp")
        assert "not installed" not in str(ei.value)
        assert "demucs failed" in str(ei.value)

    def test_a_genuinely_absent_demucs_still_says_not_installed(self, monkeypatch):
        import sys
        from pipeline import audio as A

        import builtins
        real_import = builtins.__import__

        def fake(name, *a, **k):
            if name.startswith("demucs"):
                raise ImportError("No module named 'demucs'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake)
        monkeypatch.setitem(sys.modules, "demucs", None)
        with pytest.raises(RuntimeError, match="not installed"):
            A._separate_demucs("nonexistent.wav", "/tmp")
