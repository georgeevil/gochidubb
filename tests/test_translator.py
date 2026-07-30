"""Tests for pipeline/translator.py — prompt building and glossary logic."""
from unittest.mock import patch, mock_open

import pytest

from pipeline.translator import _build_translation_prompt, _load_user_glossary


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
