"""Tests for pipeline/synthesizer.py — BaseTTSEngine abstraction and engine setup."""

import pytest

from pipeline.synthesizer import BaseTTSEngine, EdgeTTSFallback, resolve_guidance


class TestBaseTTSEngine:
    """BaseTTSEngine defines the TTS interface contract."""

    def test_can_instantiate_directly(self):
        """BaseTTSEngine is a concrete base class (not ABC) and can be instantiated."""
        engine = BaseTTSEngine()
        assert engine.name == "base"

    def test_default_sample_rate(self):
        """Default sample_rate should be 48000."""
        assert BaseTTSEngine.default_sample_rate == 48000

    def test_sample_rate_property(self):
        """When _sample_rate is None, returns default_sample_rate."""
        engine = BaseTTSEngine()
        assert engine.sample_rate == 48000

    def test_load_raises_not_implemented(self):
        """BaseTTSEngine.load() raises NotImplementedError (subclasses must implement)."""
        engine = BaseTTSEngine()
        with pytest.raises(NotImplementedError):
            engine.load()

    def test_unload_raises_not_implemented(self):
        """Both load() and unload() raise NotImplementedError in the base class."""
        engine = BaseTTSEngine()
        with pytest.raises(NotImplementedError):
            engine.unload()


class TestEdgeTTSFallback:
    """EdgeTTSFallback is the lightweight non-GPU TTS engine."""

    def test_instantiation(self):
        engine = EdgeTTSFallback()
        assert engine.name == "edge-tts"
        assert engine.default_sample_rate == 24000

    def test_is_loaded_by_default(self):
        engine = EdgeTTSFallback()
        assert engine.is_loaded is True

    def test_load_unload_noop(self):
        engine = EdgeTTSFallback()
        # These should not raise
        engine.load()
        engine.unload()

    def test_voice_map_has_common_languages(self):
        engine = EdgeTTSFallback()
        assert "en" in engine.VOICE_MAP
        assert "ru" in engine.VOICE_MAP
        assert "es" in engine.VOICE_MAP
        assert "fr" in engine.VOICE_MAP
        assert "de" in engine.VOICE_MAP
        assert "zh" in engine.VOICE_MAP
        assert "ja" in engine.VOICE_MAP
        assert "ko" in engine.VOICE_MAP
        assert "pt" in engine.VOICE_MAP
        assert "ar" in engine.VOICE_MAP
        assert "it" in engine.VOICE_MAP
        assert "bg" in engine.VOICE_MAP

    def test_voice_map_values_are_neural_voices(self):
        engine = EdgeTTSFallback()
        for voice in engine.VOICE_MAP.values():
            assert "Neural" in voice, f"Voice {voice} should be a Neural voice"

    def test_voice_map_covers_all_advertised_languages(self):
        """The README and GET /api/languages promise these 65 target codes,
        and /api/languages is derived from VOICE_MAP — so the map must keep
        every advertised language or the API silently shrinks."""
        advertised = {
            # Original 28
            "en", "ru", "es", "fr", "de", "it", "pt", "pl",
            "tr", "ja", "ko", "zh", "ar", "hi", "nl", "uk",
            "sv", "th", "vi", "cs", "ro", "hu", "bg", "el",
            "fi", "id", "no", "da",
            # Wave 2: big-audience targets
            "bn", "ur", "fa", "he", "sw", "tl", "ms", "ta",
            "te", "mr", "gu", "kn", "ml",
            # Wave 2: European gap-fillers
            "sk", "hr", "sr", "sl", "lt", "lv", "et", "ca",
            "is", "af", "mk", "sq", "bs", "cy",
            # Wave 2: Central Asia / Caucasus / mainland SE Asia
            "kk", "az", "uz", "ka", "mn", "ne", "si",
            "my", "km", "lo",
        }
        assert advertised <= set(EdgeTTSFallback.VOICE_MAP)


# ── Per-job VoxCPM overrides (CLD-189) ───────────────────────────────

class TestResolveGuidance:
    """resolve_guidance arbitrates three sources of cfg / inference steps:
    the per-job override, the global setting, and the cross-lingual floor.
    """

    def test_no_override_same_language_uses_the_globals(self):
        assert resolve_guidance(2.0, 8, is_cross_lingual=False) == (2.0, 8)

    def test_no_override_cross_lingual_raises_to_the_floor(self):
        """The behaviour every install had before overrides existed."""
        assert resolve_guidance(2.0, 8, is_cross_lingual=True) == (2.5, 14)

    def test_floor_never_lowers_a_higher_global(self):
        """It is a floor, not a setting: a global above it survives."""
        assert resolve_guidance(2.8, 20, is_cross_lingual=True) == (2.8, 20)

    def test_override_replaces_the_global(self):
        assert resolve_guidance(2.0, 8, is_cross_lingual=False,
                                cfg_override=2.6, steps_override=12) == (2.6, 12)

    def test_override_survives_the_cross_lingual_floor(self):
        """The point of the whole feature. Nearly every dub is cross-lingual,
        so if the floor could clamp an override up, asking for *less*
        guidance would silently do nothing."""
        cfg, steps = resolve_guidance(2.0, 8, is_cross_lingual=True,
                                      cfg_override=1.5, steps_override=6)
        assert (cfg, steps) == (1.5, 6)

    def test_one_override_leaves_the_other_on_the_floor(self):
        """Overriding cfg must not quietly disable the steps floor too."""
        assert resolve_guidance(2.0, 8, is_cross_lingual=True,
                                cfg_override=1.5) == (1.5, 14)
        assert resolve_guidance(2.0, 8, is_cross_lingual=True,
                                steps_override=6) == (2.5, 6)

    @pytest.mark.parametrize("unset", [None, 0, 0.0])
    def test_falsy_override_means_inherit(self, unset):
        """0 is the sentinel the config already uses for voxcpm_steps, and
        the form posts it when a knob is left on 'auto'."""
        assert resolve_guidance(2.0, 8, is_cross_lingual=True,
                                cfg_override=unset,
                                steps_override=unset) == (2.5, 14)

    def test_custom_cross_lingual_values_are_honoured(self):
        """Settings -> Voice & TTS can move the floor itself."""
        assert resolve_guidance(2.0, 8, is_cross_lingual=True,
                                xling_cfg=2.9, xling_steps=20) == (2.9, 20)

    def test_returns_the_types_voxcpm_needs(self):
        """inference_timesteps is a loop bound inside voxcpm; a float there
        is a TypeError deep in the model rather than here."""
        cfg, steps = resolve_guidance(2, 8, is_cross_lingual=False,
                                      cfg_override="2.4", steps_override="12")
        assert isinstance(cfg, float) and isinstance(steps, int)
        assert (cfg, steps) == (2.4, 12)


class TestOverrideReachesTheWorker:
    """resolve_guidance being right is only half of it — the number has to
    survive the trip into the subprocess. The worker reads cfg_value and
    inference_timesteps out of job.json and never consults config, so that
    file is the boundary worth asserting on.
    """

    @staticmethod
    def _job_json_for(monkeypatch, tmp_path, **synth_kwargs):
        import json
        import subprocess

        from pipeline.synthesizer import VoxCPMSynthesizer

        captured = {}

        class _Stop(RuntimeError):
            pass

        def fake_popen(argv, *a, **kw):
            # argv is [python, -u, tts_worker.py, --daemon, <job.json>]
            with open(argv[-1], encoding="utf-8") as f:
                captured["job"] = json.load(f)
            raise _Stop("worker launch intercepted")

        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        engine = VoxCPMSynthesizer(cfg_value=2.0, inference_timesteps=0)
        segments = [{"idx": 0, "start": 0.0, "end": 1.0, "text": "hi",
                     "translated_text": "salut", "speaker": "SPEAKER_00"}]
        with pytest.raises(_Stop):
            engine.synthesize_segments(
                segments, str(tmp_path / "out"), tts_speed="balanced",
                **synth_kwargs)
        return captured["job"]

    def test_per_job_values_land_in_the_worker_job_file(self, monkeypatch, tmp_path):
        job = self._job_json_for(monkeypatch, tmp_path, is_cross_lingual=True,
                                 cfg_override=1.6, steps_override=7)
        assert job["cfg_value"] == 1.6
        assert job["inference_timesteps"] == 7

    def test_without_an_override_the_worker_sees_the_old_values(self, monkeypatch, tmp_path):
        job = self._job_json_for(monkeypatch, tmp_path, is_cross_lingual=True)
        assert job["cfg_value"] == 2.5      # cross-lingual floor
        assert job["inference_timesteps"] == 14
