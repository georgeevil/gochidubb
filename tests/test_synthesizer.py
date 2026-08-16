"""Tests for pipeline/synthesizer.py — BaseTTSEngine abstraction and engine setup."""
from unittest.mock import patch

import pytest

from pipeline.synthesizer import BaseTTSEngine, EdgeTTSFallback


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
