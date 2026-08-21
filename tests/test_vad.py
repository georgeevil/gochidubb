"""Tests for pipeline/vad.py — VAD filtering logic and edge cases."""
from unittest.mock import patch


from pipeline.vad import (apply_vad_filter, get_speech_timestamps,
                          map_to_original, remap_segments, SPEECH_RATIO_WARNING)


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
        result_path, ratio, intervals = apply_vad_filter("/fake/input.wav", "/fake/output.wav")
        assert ratio == 1.0
        mock_copy.assert_called_once()

    @patch("pipeline.vad._get_duration_ffprobe", return_value=30.0)
    @patch("pipeline.vad.get_speech_timestamps", return_value=[])
    @patch("shutil.copy2")
    def test_passthrough_on_no_speech(self, mock_copy, mock_ts, mock_dur):
        """If VAD returns no timestamps, copy original and report 1.0."""
        result_path, ratio, intervals = apply_vad_filter("/fake/input.wav", "/fake/output.wav", threshold=0.5)
        assert ratio == 1.0
        mock_copy.assert_called_once()

    @patch("pipeline.vad._get_duration_ffprobe", return_value=30.0)
    @patch("pipeline.vad.get_speech_timestamps",
           return_value=[{"start": 0.0, "end": 29.0}])
    @patch("shutil.copy2")
    def test_skip_filter_when_high_speech_ratio(self, mock_copy, mock_ts, mock_dur):
        """If speech_ratio > 0.90, skip filtering and copy original (dense speech)."""
        result_path, ratio, intervals = apply_vad_filter("/fake/input.wav", "/fake/output.wav")
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
        result_path, ratio, intervals = apply_vad_filter("/fake/input.wav", "/fake/output.wav")
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
        # VAD padding adds 0.1s on each side → 1.2s speech / 10s total = 12%.
        # Pinned against the real constant so that lowering the threshold
        # fails here with the reason rather than on the assertion below.
        fixture_ratio = 1.2 / 10.0
        assert fixture_ratio < SPEECH_RATIO_WARNING, (
            f"fixture speech ratio {fixture_ratio:.0%} no longer sits below the "
            f"{SPEECH_RATIO_WARNING:.0%} warning threshold — adjust the mock"
        )
        apply_vad_filter("/fake/input.wav", "/fake/output.wav")
        assert any("only" in record.message and "%" in record.message
                   for record in caplog.records if "VAD" in record.message)


class TestLoadMono16k:
    """VAD must not use silero's read_audio() — it calls torchaudio.load(),
    which dies with "Could not load libtorchcodec" wherever FFmpeg is newer
    than the installed torchcodec supports."""

    def test_loads_without_torchaudio_load(self, tmp_path, monkeypatch):
        import numpy as np
        import soundfile as sf
        import torchaudio

        from pipeline.vad import _load_mono_16k

        monkeypatch.setattr(torchaudio, "load", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Could not load libtorchcodec")))

        p = tmp_path / "a.wav"
        sf.write(p, np.zeros((1600, 1), dtype="float32"), 16000)
        wav = _load_mono_16k(str(p))
        assert wav.ndim == 1, "silero expects a 1-D tensor"
        assert wav.shape[0] == 1600

    def test_downmixes_and_resamples(self, tmp_path, monkeypatch):
        import numpy as np
        import soundfile as sf
        import torchaudio

        from pipeline.vad import _load_mono_16k

        monkeypatch.setattr(torchaudio, "load", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Could not load libtorchcodec")))

        p = tmp_path / "st.wav"
        sf.write(p, np.zeros((44100, 2), dtype="float32"), 44100)
        wav = _load_mono_16k(str(p))
        assert wav.ndim == 1
        assert 15000 < wav.shape[0] < 17000, "1s of audio should become ~16000 samples"


class TestTimelineRemap:
    """The filter concatenates speech regions, so whisper's timestamps come
    back compressed. These cover the inversion that puts them back.

    Regression: a 60s clip whose speech started at 14.7s produced a dub that
    ran 0-28.4s and finished 30s before the picture did, because nothing
    mapped the compressed times back onto the video's timeline.
    """

    # Speech at 10-20s and 40-50s of a 60s source: 20s of audio reaches
    # whisper, and compressed t=15 is really t=45.
    INTERVALS = [(10.0, 20.0), (40.0, 50.0)]

    def test_identity_when_no_filtering(self):
        for t in (0.0, 7.5, 60.0):
            assert map_to_original(t, None) == t
            assert map_to_original(t, []) == t

    def test_first_region_is_offset_by_its_start(self):
        assert map_to_original(0.0, self.INTERVALS) == 10.0
        assert map_to_original(5.0, self.INTERVALS) == 15.0

    def test_second_region_skips_the_removed_gap(self):
        # 10s consumed by the first region, so compressed 10 == original 40
        assert map_to_original(10.0, self.INTERVALS) == 20.0
        assert map_to_original(15.0, self.INTERVALS) == 45.0
        assert map_to_original(20.0, self.INTERVALS) == 50.0

    def test_past_the_end_clamps_rather_than_extrapolating(self):
        assert map_to_original(99.0, self.INTERVALS) == 50.0

    def test_monotonic(self):
        prev = -1.0
        for i in range(0, 201):
            cur = map_to_original(i * 0.1, self.INTERVALS)
            assert cur >= prev
            prev = cur

    def test_remap_segments_moves_starts_ends_and_words(self):
        segs = [
            {"start": 0.0, "end": 5.0, "words": [{"start": 0.0, "end": 1.0}]},
            {"start": 12.0, "end": 15.0},
        ]
        remap_segments(segs, self.INTERVALS)
        assert segs[0]["start"] == 10.0
        assert segs[0]["end"] == 15.0
        assert segs[0]["words"][0]["end"] == 11.0
        assert segs[1]["start"] == 42.0
        assert segs[1]["end"] == 45.0

    def test_remap_is_a_noop_without_intervals(self):
        segs = [{"start": 1.0, "end": 2.0}]
        remap_segments(segs, None)
        assert segs == [{"start": 1.0, "end": 2.0}]

    def test_dub_spans_the_source_not_just_the_speech(self):
        """The user-visible symptom: last segment must land near the end of
        the source, not at the end of the compressed audio."""
        segs = [{"start": 19.0, "end": 20.0}]     # last second of filtered audio
        remap_segments(segs, self.INTERVALS)
        assert segs[0]["end"] == 50.0             # not 20.0
