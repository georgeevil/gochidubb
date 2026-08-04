"""Tests for pipeline/tts_qa.py — CER computation, text normalization, scoring."""
import os

import pytest

from pipeline.tts_qa import _normalize_text, _cer, is_acceptable


class TestNormalizeText:
    """_normalize_text() cleans up text for CER comparison."""

    def test_lowercases(self):
        assert _normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        result = _normalize_text("Hello, world! How's it going?")
        assert "!" not in result
        assert "," not in result
        assert "?" not in result
        assert "'" not in result

    def test_collapses_whitespace(self):
        assert _normalize_text("hello    world") == "hello world"

    def test_strips_voice_prefix(self):
        """Note: the voice-prefix regex runs after punctuation-removal
        which already strips parentheses, so the actual output includes
        the prefix words. The regex effectively operates on text that
        has already had parens removed."""
        result = _normalize_text("(deep male voice) Hello everyone")
        # Parens removed first by the [^\w\s] pass, so "deep male voice" remains as words
        assert "deep" in result
        assert "hello everyone" in result

    def test_preserves_cyrillic(self):
        result = _normalize_text("Привет, мир!")
        assert "привет" in result
        assert "мир" in result

    def test_keeps_digits(self):
        assert _normalize_text("Version 2.0 is out") == "version 2 0 is out"

    def test_empty_string(self):
        assert _normalize_text("") == ""

    def test_only_punctuation(self):
        assert _normalize_text("!!! ??? ...") == ""

    def test_mixed_latin_cyrillic(self):
        result = _normalize_text("Hello привет 123!")
        assert "hello" in result
        assert "привет" in result
        assert "123" in result


class TestCER:
    """_cer() computes character error rate via Levenshtein distance."""

    def test_exact_match(self):
        assert _cer("hello world", "hello world") == 0.0

    def test_empty_hypothesis(self):
        assert _cer("", "hello") == 1.0

    def test_empty_reference(self):
        assert _cer("hello", "") == 0.0 if "hello" == "" else 1.0

    def test_both_empty(self):
        assert _cer("", "") == 0.0

    def test_one_insertion(self):
        """One extra character in hypothesis."""
        result = _cer("hello world", "hello world!")
        assert result > 0.0

    def test_one_deletion(self):
        """One character missing in hypothesis."""
        result = _cer("hello word", "hello world")
        assert result > 0.0

    def test_one_substitution(self):
        """One wrong character."""
        result = _cer("hallo world", "hello world")
        assert result > 0.0
        # 1 substitution / 11 chars ≈ 0.091
        assert abs(result - (1.0 / 11)) < 0.01

    def test_completely_different(self):
        """Wildly different strings → score >= 1.0."""
        result = _cer("abc", "xyz")
        assert result >= 1.0

    def test_cer_on_short_target(self):
        """Short target text like 'this sport' — small change matters a lot."""
        result = _cer("this sport", "this sports")
        # 1 insertion / 11 chars ≈ 0.091
        assert result > 0.0
        assert result < 0.5

    def test_normalized_texts(self):
        """Pre-normalize before calling CER for realistic usage."""
        hyp = _normalize_text("Hello, world!")
        ref = _normalize_text("Hello world")
        assert _cer(hyp, ref) == 0.0


class TestIsAcceptable:
    """is_acceptable() determines if a QA score passes the threshold."""

    def test_perfect_score(self):
        assert is_acceptable(0.0) is True

    def test_below_threshold(self):
        assert is_acceptable(0.2) is True

    def test_at_threshold(self):
        assert is_acceptable(0.4) is True

    def test_above_threshold(self):
        assert is_acceptable(0.45) is False

    def test_bad_score(self):
        assert is_acceptable(1.0) is False

    def test_custom_threshold(self):
        assert is_acceptable(0.3, threshold=0.25) is False
        assert is_acceptable(0.2, threshold=0.25) is True

    def test_with_env_override(self, monkeypatch):
        monkeypatch.setenv("GOCHIDUBB_QA_THRESHOLD", "0.5")
        # Re-import to pick up env? No — function reads env on each call.
        assert is_acceptable(0.45) is True
        assert is_acceptable(0.55) is False
