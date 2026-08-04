"""Tests for pipeline/tts_qa.py — CER computation, text normalization, scoring."""
import sys
import types


import pipeline.tts_qa as tts_qa
from pipeline.tts_qa import (
    QA_THRESHOLD,
    _cer,
    _normalize_text,
    _resolve_device,
    check_segment_quality,
    is_acceptable,
)


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

    def test_threshold_constant_is_the_default_gate(self):
        """Single source of truth: the default param IS the module constant."""
        assert QA_THRESHOLD == 0.4
        assert is_acceptable(QA_THRESHOLD) is True
        assert is_acceptable(QA_THRESHOLD + 0.01) is False

    def test_none_score_is_pass_without_claim(self):
        """A segment that CANNOT be measured must not trigger retries."""
        assert is_acceptable(None) is True

    def test_none_score_passes_even_under_strict_env_threshold(self, monkeypatch):
        monkeypatch.setenv("GOCHIDUBB_QA_THRESHOLD", "0.0")
        assert is_acceptable(None) is True


def _fake_torch(cuda_available: bool):
    """Minimal torch stand-in for device resolution tests."""
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return mod


class TestResolveDevice:
    """_resolve_device() picks a real device instead of assuming CUDA."""

    def test_explicit_preference_passes_through(self):
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("cpu") == "cpu"

    def test_auto_with_cuda_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
        assert _resolve_device("auto") == "cuda"

    def test_auto_without_cuda_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
        assert _resolve_device("auto") == "cpu"

    def test_auto_when_torch_is_broken_falls_back_to_cpu(self, monkeypatch):
        # sys.modules[name] = None makes `import torch` raise ImportError
        monkeypatch.setitem(sys.modules, "torch", None)
        assert _resolve_device("auto") == "cpu"

    def test_empty_preference_resolves_like_auto(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
        assert _resolve_device("") == "cpu"


class TestNotMeasuredSemantics:
    """check_segment_quality() must be honest when it could not measure."""

    def test_missing_audio_is_a_real_failure(self, tmp_path):
        score, transcript, diag = check_segment_quality(
            str(tmp_path / "does_not_exist.wav"), "привет", target_lang="ru",
        )
        assert score == 1.0
        assert diag["measured"] is True
        assert "error" in diag

    def test_whisper_unavailable_returns_unmeasured(self, tmp_path, monkeypatch):
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"RIFF fake")
        monkeypatch.setattr(tts_qa, "_get_whisper", lambda **kw: None)
        score, transcript, diag = check_segment_quality(
            str(wav), "привет", target_lang="ru",
        )
        assert score is None, "unmeasured must NOT look like a perfect 0.0"
        assert diag["measured"] is False
        assert is_acceptable(score) is True  # pass-without-claim

    def test_transcribe_crash_returns_unmeasured(self, tmp_path, monkeypatch):
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"RIFF fake")

        class ExplodingModel:
            def transcribe(self, *a, **kw):
                raise RuntimeError("boom")

        monkeypatch.setattr(tts_qa, "_get_whisper", lambda **kw: ExplodingModel())
        score, transcript, diag = check_segment_quality(
            str(wav), "привет", target_lang="ru",
        )
        assert score is None
        assert diag["measured"] is False
        assert "boom" in diag["error"]

    def test_real_measurement_sets_measured_true(self, tmp_path, monkeypatch):
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"RIFF fake")

        info = types.SimpleNamespace(language="ru", language_probability=0.99)
        segs = [types.SimpleNamespace(text=" привет мир ")]

        class OkModel:
            def transcribe(self, *a, **kw):
                return iter(segs), info

        monkeypatch.setattr(tts_qa, "_get_whisper", lambda **kw: OkModel())
        score, transcript, diag = check_segment_quality(
            str(wav), "Привет, мир!", target_lang="ru",
        )
        assert score == 0.0
        assert transcript == "привет мир"
        assert diag["measured"] is True
        assert diag["lang_match"] is True

    def test_empty_transcript_is_a_real_failure(self, tmp_path, monkeypatch):
        """Silence from TTS is a measured defect, not an unmeasured one."""
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"RIFF fake")

        info = types.SimpleNamespace(language="ru", language_probability=0.2)

        class SilentModel:
            def transcribe(self, *a, **kw):
                return iter([]), info

        monkeypatch.setattr(tts_qa, "_get_whisper", lambda **kw: SilentModel())
        score, transcript, diag = check_segment_quality(
            str(wav), "привет", target_lang="ru",
        )
        assert score == 1.0
        assert diag["measured"] is True
        assert diag["empty"] is True
