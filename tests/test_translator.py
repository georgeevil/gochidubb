"""Tests for pipeline/translator.py — prompt building and glossary logic."""
import os
from unittest.mock import patch, mock_open

import pytest

from pipeline.translator import (
    _build_translation_prompt,
    _load_user_glossary,
    _is_large_model,
    _lm_studio_timeout,
    _lm_studio_max_concurrent,
)


class TestBuildTranslationPrompt:
    """_build_translation_prompt() constructs the LM Studio prompt."""

    def test_basic_prompt_structure(self):
        prompt = _build_translation_prompt(
            text="Hello world",
            source_lang="en",
            target_lang="ru",
        )
        assert "Hello world" in prompt
        assert "en to ru" in prompt or "english to russian" in prompt.lower()
        assert "Translation:" in prompt

    def test_includes_context_hint(self):
        prompt = _build_translation_prompt(
            text="Guard pass",
            source_lang="en",
            target_lang="ru",
            context_hint="Brazilian Jiu-Jitsu",
        )
        assert "Brazilian Jiu-Jitsu" in prompt

    def test_includes_glossary_terms(self):
        glossary = {
            "ru": {"guard": "гард", "pass": "проход"},
        }
        prompt = _build_translation_prompt(
            text="Guard pass",
            source_lang="en",
            target_lang="ru",
            glossary=glossary,
        )
        assert "guard" in prompt
        assert "гард" in prompt
        assert "pass" in prompt
        assert "проход" in prompt

    def test_builtin_glossary_structure(self):
        """The built-in glossary is keyed by domain 'BJJ', not source_lang.
        So glossary terms only appear when source_lang matches the domain key."""
        # _BUILTIN_GLOSSARY is {"BJJ": {"en": {...}}}, keyed by domain, not lang
        # So _BUILTIN_GLOSSARY.get("en", {}) returns {} — terms not injected by source_lang
        # The prompt won't contain glossary terms from built-in, but it should still work
        prompt = _build_translation_prompt(
            text="Side control",
            source_lang="en",
            target_lang="ru",
        )
        # The prompt is valid and contains the text to translate
        assert "Side control" in prompt
        assert "Translation:" in prompt

    def test_rules_section_present(self):
        prompt = _build_translation_prompt(
            text="Hello",
            source_lang="en",
            target_lang="es",
        )
        assert "Translate naturally" in prompt or "Rules" in prompt or "fluently" in prompt

    def test_does_not_include_explanations(self):
        """The prompt should tell the model not to add notes."""
        prompt = _build_translation_prompt(
            text="Hello",
            source_lang="en",
            target_lang="de",
        )
        assert "DO NOT add any explanations" in prompt

    def test_no_glossary_when_empty(self):
        """Empty glossary should not produce glossary section."""
        prompt = _build_translation_prompt(
            text="Hello",
            source_lang="en",
            target_lang="fr",
            glossary={},
        )
        # Should not contain glossary instructions since no terms provided
        assert "IMPORTANT:" not in prompt


class TestLoadUserGlossary:
    """_load_user_glossary() reads from presets/user_glossary.json."""

    @patch("os.path.exists", return_value=False)
    def test_no_glossary_file(self, mock_exists):
        assert _load_user_glossary() == {}

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{}')
    def test_empty_glossary_file(self, mock_file, mock_exists):
        assert _load_user_glossary() == {}

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open,
           read_data='{"domains": [{"target_lang": "ru", "terms": {"hello": "привет"}}]}')
    def test_loads_russian_terms(self, mock_file, mock_exists):
        result = _load_user_glossary()
        assert isinstance(result, dict)
        assert "hello" in result
        assert result["hello"] == "привет"

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open,
           read_data='{"domains": [{"target_lang": "es", "terms": {"hello": "hola"}}]}')
    def test_skips_non_russian(self, mock_file, mock_exists):
        """Only Russian terms are loaded based on current code."""
        result = _load_user_glossary()
        assert result == {}


class TestIsLargeModel:
    """_is_large_model() detects >=20B models from the model id, used to
    auto-serialize LM Studio translation for slow/CPU backends."""

    def test_detects_27b(self):
        assert _is_large_model("qwen/qwen3.6-27b") is True

    def test_detects_32b_and_72b(self):
        assert _is_large_model("Qwen/Qwen2.5-32B-Instruct") is True
        assert _is_large_model("Qwen/Qwen2.5-72B-Instruct") is True

    def test_small_models_not_large(self):
        assert _is_large_model("qwen/qwen3-8b") is False
        assert _is_large_model("google/gemma-3-4b-it") is False
        assert _is_large_model("Qwen/Qwen2.5-14B-Instruct") is False

    def test_decimal_param_count(self):
        """Models like '7.5b' should parse the decimal correctly."""
        assert _is_large_model("mistral/mistral-7.5b") is False
        assert _is_large_model("fake/model-24.5b") is True

    def test_no_param_count_is_not_large(self):
        assert _is_large_model("text-embedding-nomic") is False
        assert _is_large_model("") is False
        assert _is_large_model("gpt-oss") is False

    def test_custom_threshold(self):
        """A 14B model is large only if the threshold is lowered."""
        assert _is_large_model("Qwen/Qwen2.5-14B-Instruct", threshold=10) is True
        assert _is_large_model("qwen/qwen3-8b", threshold=10) is False


class TestLmStudioTunables:
    """_lm_studio_timeout / _lm_studio_max_concurrent read env vars lazily."""

    def test_timeout_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("LM_STUDIO_TIMEOUT", raising=False)
        assert _lm_studio_timeout() == 300

    def test_timeout_respects_env(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_TIMEOUT", "600")
        assert _lm_studio_timeout() == 600

    def test_timeout_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_TIMEOUT", "not-a-number")
        assert _lm_studio_timeout() == 300

    def test_timeout_falls_back_on_empty(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_TIMEOUT", "")
        assert _lm_studio_timeout() == 300

    def test_max_concurrent_default_auto(self, monkeypatch):
        monkeypatch.delenv("LM_STUDIO_MAX_CONCURRENT", raising=False)
        assert _lm_studio_max_concurrent() == 0  # 0 == auto

    def test_max_concurrent_respects_env(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_MAX_CONCURRENT", "1")
        assert _lm_studio_max_concurrent() == 1

    def test_max_concurrent_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_MAX_CONCURRENT", "xyz")
        assert _lm_studio_max_concurrent() == 0


class TestTranslationChunkSize:
    """_translation_chunk_size() reads TRANSLATION_CHUNK_SIZE env var."""

    def test_default_chunk_size(self, monkeypatch):
        monkeypatch.delenv("TRANSLATION_CHUNK_SIZE", raising=False)
        from pipeline.translator import _translation_chunk_size
        assert _translation_chunk_size() == 5

    def test_custom_chunk_size(self, monkeypatch):
        monkeypatch.setenv("TRANSLATION_CHUNK_SIZE", "10")
        from pipeline.translator import _translation_chunk_size
        assert _translation_chunk_size() == 10

    def test_chunk_size_minimum(self, monkeypatch):
        monkeypatch.setenv("TRANSLATION_CHUNK_SIZE", "0")
        from pipeline.translator import _translation_chunk_size
        assert _translation_chunk_size() == 1

    def test_chunk_size_maximum(self, monkeypatch):
        monkeypatch.setenv("TRANSLATION_CHUNK_SIZE", "100")
        from pipeline.translator import _translation_chunk_size
        assert _translation_chunk_size() == 15

    def test_chunk_size_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("TRANSLATION_CHUNK_SIZE", "invalid")
        from pipeline.translator import _translation_chunk_size
        assert _translation_chunk_size() == 5
