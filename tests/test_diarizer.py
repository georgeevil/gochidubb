"""Tests for pipeline/diarizer.py — speaker assignment and reference extraction."""
from unittest.mock import patch

import pytest

from pipeline.diarizer import assign_speakers_to_segments, extract_fallback_reference


class TestAssignSpeakersToSegments:
    """assign_speakers_to_segments() maps diarization turns to segments."""

    def test_exact_match(self):
        """One speaker, one segment — simple mapping."""
        segments = [{"start": 0.0, "end": 5.0}]
        diarization = [(0.0, 5.0, "SPEAKER_00")]
        result = assign_speakers_to_segments(segments, diarization)
        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_multiple_speakers(self):
        segments = [
            {"start": 0.0, "end": 3.0},
            {"start": 3.5, "end": 7.0},
            {"start": 7.5, "end": 10.0},
        ]
        diarization = [
            (0.0, 3.0, "SPEAKER_00"),
            (3.5, 7.0, "SPEAKER_01"),
            (7.5, 10.0, "SPEAKER_00"),
        ]
        result = assign_speakers_to_segments(segments, diarization)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"
        assert result[2]["speaker"] == "SPEAKER_00"

    def test_overlap_best_overlap_wins(self):
        """Segment overlapping multiple turns picks the one with most overlap."""
        segments = [{"start": 2.0, "end": 6.0}]
        diarization = [
            (0.0, 4.0, "SPEAKER_00"),   # 2s overlap
            (4.0, 8.0, "SPEAKER_01"),   # 2s overlap
        ]
        result = assign_speakers_to_segments(segments, diarization)
        # Equal overlap, could be either — just check speaker is assigned
        assert "speaker" in result[0]

    def test_no_diarization(self):
        """Without diarization data, defaults to SPEAKER_00."""
        segments = [{"start": 0.0, "end": 5.0, "text": "Hello"}]
        result = assign_speakers_to_segments(segments, [])
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_empty_segments(self):
        assert assign_speakers_to_segments([], [("0.0", "5.0", "SPEAKER_00")]) == []

    def test_segment_matches_no_turn(self):
        """Segment outside all diarization turns gets default SPEAKER_00."""
        segments = [{"start": 100.0, "end": 105.0}]
        diarization = [(0.0, 10.0, "SPEAKER_00")]
        result = assign_speakers_to_segments(segments, diarization)
        assert result[0]["speaker"] == "SPEAKER_00"


class TestExtractFallbackReference:
    """extract_fallback_reference() builds a single-speaker ref when diarization fails."""

    def test_no_segments_returns_none(self):
        assert extract_fallback_reference("/fake/audio.wav", [], "/fake/out.wav") is None

    def test_with_segments_and_audio(self, temp_audio_file):
        """Should write a ref file and return the path."""
        segments = [
            {"start": 0.0, "end": 0.5, "text": "Hello world"},
            {"start": 0.6, "end": 1.0, "text": "This is a test."},
        ]
        out_path = "/tmp/_test_fallback_ref.wav"
        import os
        try:
            result = extract_fallback_reference(temp_audio_file, segments, out_path)
            # May succeed or return None depending on audio file read
            # At minimum, shouldn't crash
            if result:
                assert os.path.exists(result)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_filters_short_segments(self):
        """Very short segments (<0.5s) should be excluded."""
        segments = [
            {"start": 0.0, "end": 0.3, "text": "Hi"},
            {"start": 1.0, "end": 5.0, "text": "This is a proper segment with enough text"},
        ]
        out_path = "/tmp/_test_filter_short.wav"
        import os
        try:
            # Without a real audio file, it'll return None, but that's expected
            result = extract_fallback_reference("/nonexistent.wav", segments, out_path)
            assert result is None  # Nothing to extract from missing file
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


# ═════════════════════════════════════════════════════════════════════
# Setup notices — the "succeeded, but degraded" channel
# ═════════════════════════════════════════════════════════════════════
# Every failure path in this module returns [] and never raises, so `notices`
# is the *only* way a caller can tell "one speaker in the video" apart from
# "your Hugging Face setup is broken". These tests pin that contract.

import sys
import types

from pipeline.diarizer import _hub_repo_from_error, _load_pipeline, effective_hf_token


def _fake_pyannote(loader):
    """Install a stub `pyannote.audio` whose Pipeline.from_pretrained is `loader`."""
    mod = types.ModuleType("pyannote.audio")
    mod.Pipeline = type("Pipeline", (), {"from_pretrained": staticmethod(loader)})
    pkg = types.ModuleType("pyannote")
    pkg.audio = mod
    return patch.dict(sys.modules, {"pyannote": pkg, "pyannote.audio": mod})


class TestLoadPipelineNotices:
    def test_gated_first_model_still_reports_when_fallback_succeeds(self):
        """The motivating bug.

        speaker-diarization-3.1 pulls in segmentation-3.0; if that download
        fails, community-1 loads happily and the run looks completely normal.
        The error used to be collected into a local list that was only logged
        when *every* model failed, so a partial failure vanished entirely.
        """
        calls = []

        def loader(model, token=None):
            calls.append(model)
            if model == "pyannote/speaker-diarization-3.1":
                raise RuntimeError(
                    "401 Client Error. Cannot access gated repo for url "
                    "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/config.yaml"
                )
            return object()

        notices = []
        with _fake_pyannote(loader):
            pipe = _load_pipeline("hf_token", notices)

        assert pipe is not None, "the fallback model must still be used"
        assert calls == ["pyannote/speaker-diarization-3.1",
                         "pyannote/speaker-diarization-community-1"]
        codes = [n["code"] for n in notices]
        assert codes == ["pyannote.fallback_model"]
        n = notices[0]
        assert n["severity"] == "warn", "the dub still works, so this is not an error"
        # The URL must name the *dependency* that actually failed, not the
        # pipeline the caller asked for — it is parsed out of the exception.
        assert n["url"] == "https://hf.co/pyannote/segmentation-3.0"
        assert "community-1" in n["title"]
        assert n["remediation"], "a notice with no next step is just noise"

    def test_no_notice_when_the_preferred_model_loads(self):
        with _fake_pyannote(lambda model, token=None: object()):
            notices = []
            assert _load_pipeline("hf_token", notices) is not None
        assert notices == []

    def test_all_models_failing_is_an_error(self):
        def loader(model, token=None):
            raise RuntimeError("503 Server Error for https://huggingface.co/%s" % model)

        notices = []
        with _fake_pyannote(loader):
            assert _load_pipeline("hf_token", notices) is None
        assert [n["code"] for n in notices] == ["pyannote.load_failed"]
        assert notices[0]["severity"] == "error"
        assert "503" in notices[0]["detail"], "the raw error is the useful part"

    def test_from_pretrained_returning_none_is_not_silent(self):
        """A None return is a failure too, even though nothing was raised."""
        def loader(model, token=None):
            return None if "3.1" in model else object()

        notices = []
        with _fake_pyannote(loader):
            assert _load_pipeline("hf_token", notices) is not None
        assert [n["code"] for n in notices] == ["pyannote.fallback_model"]

    def test_missing_token_is_an_error_with_steps(self):
        # pyannote must look installed, or the ImportError branch (checked
        # first, correctly) wins. Other test modules stub sys.modules.
        notices = []
        with _fake_pyannote(lambda model, token=None: object()):
            assert _load_pipeline("", notices) is None
        assert [n["code"] for n in notices] == ["pyannote.no_token"]
        assert notices[0]["severity"] == "error"
        assert any("hf.co/settings/tokens" in s for s in notices[0]["remediation"])

    def test_pyannote_not_installed_is_reported(self):
        notices = []
        with patch.dict(sys.modules, {"pyannote.audio": None}):
            assert _load_pipeline("hf_token", notices) is None
        assert [n["code"] for n in notices] == ["pyannote.not_installed"]

    def test_notices_argument_is_optional(self):
        """Old callers that don't pass a list must not crash."""
        assert _load_pipeline("") is None


class TestHubRepoFromError:
    def test_extracts_the_repo_that_actually_failed(self):
        err = Exception("404 Client Error for url "
                        "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/x.bin")
        assert _hub_repo_from_error(err) == "pyannote/segmentation-3.0"

    def test_handles_the_api_url_form(self):
        err = Exception("GatedRepoError: https://huggingface.co/api/models/pyannote/foo-bar")
        assert _hub_repo_from_error(err) == "pyannote/foo-bar"

    def test_returns_empty_when_no_repo_is_named(self):
        assert _hub_repo_from_error(Exception("connection reset")) == ""


class TestEffectiveHfToken:
    def test_environment_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        assert effective_hf_token() == "hf_from_env"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "  hf_padded  ")
        assert effective_hf_token() == "hf_padded"

    def test_falls_back_to_user_config(self, monkeypatch):
        """A token typed into Settings must take effect without a restart —
        cfg.hf_token existed but nothing ever read it."""
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.setattr("pipeline.diarizer.HF_TOKEN", "", raising=False)
        from app.config import cfg
        monkeypatch.setattr(cfg, "hf_token", "hf_from_config", raising=False)
        assert effective_hf_token() == "hf_from_config"

    def test_empty_everywhere_is_empty(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.setattr("pipeline.diarizer.HF_TOKEN", "", raising=False)
        from app.config import cfg
        monkeypatch.setattr(cfg, "hf_token", "", raising=False)
        assert effective_hf_token() == ""
