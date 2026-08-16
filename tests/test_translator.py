"""Tests for pipeline/translator.py — prompt building, batching, glossary."""
from unittest.mock import patch, mock_open

import pytest

from pipeline import translator as T
from pipeline.translator import (
    _alignment_is_plausible,
    _batch_limits,
    _batch_output_budget,
    _build_batch_prompt,
    _build_translation_prompt,
    _chunk_lines,
    _load_user_glossary,
    _parse_numbered_lines,
    language_display_name,
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

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open,
           read_data='{"domains": [{"target_lang": "es", "terms": {"hello": "hola"}}]}')
    def test_other_target_languages_are_selectable(self, mock_file, mock_exists):
        assert _load_user_glossary("es") == {"hello": "hola"}


# ── Batching ─────────────────────────────────────────────────────────

class TestBatchLimits:
    """Batch size auto-scales with the model's context window."""

    def test_auto_scales_with_context_length(self, monkeypatch):
        monkeypatch.setattr(T, "TRANSLATE_BATCH_SIZE", 0)
        monkeypatch.setattr(T, "TRANSLATE_BATCH_CHARS", 0)

        monkeypatch.setattr(T, "LM_STUDIO_CONTEXT_LENGTH", 4096)
        small_lines, small_chars = _batch_limits()
        monkeypatch.setattr(T, "LM_STUDIO_CONTEXT_LENGTH", 32768)
        big_lines, big_chars = _batch_limits()

        assert big_lines > small_lines
        assert big_chars > small_chars
        # Never so large that one bad reply costs the whole transcript
        assert big_lines <= 64

    def test_explicit_setting_wins(self, monkeypatch):
        monkeypatch.setattr(T, "TRANSLATE_BATCH_SIZE", 7)
        monkeypatch.setattr(T, "LM_STUDIO_CONTEXT_LENGTH", 32768)
        assert _batch_limits()[0] == 7

    def test_caller_override_wins_over_env(self, monkeypatch):
        monkeypatch.setattr(T, "TRANSLATE_BATCH_SIZE", 7)
        assert _batch_limits(batch_size=3)[0] == 3


class TestChunkLines:
    def test_respects_the_line_cap(self, monkeypatch):
        monkeypatch.setattr(T, "TRANSLATE_BATCH_CHARS", 100_000)
        batches = _chunk_lines(["hi"] * 10, batch_size=4)
        assert [len(b) for b in batches] == [4, 4, 2]
        # Every line lands in exactly one batch, in order
        assert [i for b in batches for i in b] == list(range(10))

    def test_respects_the_char_cap(self, monkeypatch):
        monkeypatch.setattr(T, "TRANSLATE_BATCH_CHARS", 100)
        batches = _chunk_lines(["x" * 60] * 4, batch_size=50)
        assert [len(b) for b in batches] == [1, 1, 1, 1]

    def test_a_single_oversized_line_is_not_dropped(self, monkeypatch):
        monkeypatch.setattr(T, "TRANSLATE_BATCH_CHARS", 100)
        batches = _chunk_lines(["x" * 5000], batch_size=10)
        assert batches == [[0]]

    def test_empty_input(self):
        assert _chunk_lines([]) == []


class TestBatchOutputBudget:
    def test_scales_past_the_single_segment_default(self):
        """A batch needs room for every line's answer, not just one."""
        long_batch = ["A fairly long sentence of source text. " * 8] * 24
        assert _batch_output_budget(long_batch) > T.LM_STUDIO_MAX_OUTPUT_TOKENS

    def test_grows_with_the_batch_once_past_the_floor(self, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_MAX_OUTPUT_TOKENS", 512)
        small = _batch_output_budget(["Hello there friend"] * 4)
        big = _batch_output_budget(["Hello there friend"] * 40)
        assert big > small >= 512

    def test_never_below_the_single_segment_default(self):
        """Thinking models burn the first N tokens on reasoning regardless."""
        assert _batch_output_budget(["Hi"]) >= T.LM_STUDIO_MAX_OUTPUT_TOKENS


class TestParseNumberedLines:
    @pytest.mark.parametrize("reply", [
        "1. Привет\n2. Мир",
        "1) Привет\n2) Мир",
        "[1] Привет\n[2] Мир",
        "1: Привет\n2: Мир",
        "**1.** Привет\n**2.** Мир",
        "1.Привет\n2.Мир",
        "  1. Привет  \n\n  2. Мир  ",
    ])
    def test_accepts_the_numbering_styles_models_actually_use(self, reply):
        assert _parse_numbered_lines(reply, 2) == ["Привет", "Мир"]

    def test_drops_preamble_before_the_first_line(self):
        reply = "Sure, here is the translation:\n1. Привет\n2. Мир"
        assert _parse_numbered_lines(reply, 2) == ["Привет", "Мир"]

    def test_joins_a_wrapped_line(self):
        reply = "1. Привет\nмир\n2. Пока"
        assert _parse_numbered_lines(reply, 2) == ["Привет мир", "Пока"]

    def test_strips_think_blocks(self):
        reply = "<think>ru is Привет</think>\n1. Привет\n2. Мир"
        assert _parse_numbered_lines(reply, 2) == ["Привет", "Мир"]

    def test_rejects_a_missing_line(self):
        """The dangerous case: the model merged two subtitles. Returning a
        short list here would shift every later line onto the wrong audio."""
        assert _parse_numbered_lines("1. Привет\n3. Мир", 3) is None

    def test_rejects_a_short_reply(self):
        assert _parse_numbered_lines("1. Привет", 3) is None

    def test_rejects_an_unnumbered_reply(self):
        assert _parse_numbered_lines("Привет\nМир", 2) is None

    def test_rejects_an_empty_translation(self):
        assert _parse_numbered_lines("1. Привет\n2. \n3. Пока", 3) is None

    def test_rejects_more_lines_than_asked_for(self):
        """An extra numbered line means the model split one subtitle in two
        — same shifting hazard as dropping one, so don't trust the batch."""
        assert _parse_numbered_lines("1. Привет\n2. Мир\n3. Ещё", 2) is None

    def test_a_translation_starting_with_a_number_is_not_a_new_item(self):
        reply = "1. Первый шаг\n5. минут спустя\n2. Второй"
        assert _parse_numbered_lines(reply, 2) == ["Первый шаг 5. минут спустя",
                                                   "Второй"]


class TestAlignmentPlausibility:
    def test_accepts_normal_expansion(self):
        assert _alignment_is_plausible(["Hello there"], ["Привет, как дела"])

    def test_rejects_appended_commentary(self):
        src = ["Hi"]
        tgt = ["Привет. Note: this is an informal greeting used between "
               "friends; a formal alternative would be Здравствуйте."]
        assert not _alignment_is_plausible(src, tgt)

    def test_checks_every_line_against_the_target_language(self):
        src = ["Hola mamá", "Te amo mucho"]
        tgt = ["Здравей, мамо", "Te amo mucho"]
        assert not _alignment_is_plausible(src, tgt, "bg")


class TestRejectionReason:
    """_rejection_reason() is the single validator both translation paths use.

    The batch path had a length check; the single-line fallback had none, so
    a model that narrated the task instead of translating had its whole
    chain-of-thought accepted as the translation and synthesized into speech.
    """

    def test_accepts_a_real_translation(self):
        assert T._rejection_reason("Hola mamá", "Здравей, мамо", "bg") is None

    def test_accepts_weak_but_genuine_translation(self):
        # Poor quality is not this function's business — only "not a
        # translation at all" is.
        assert T._rejection_reason(
            " Hola, mamá. Te amo mucho.",
            "Здравей, мамо. Обичам те много.",
            "bg",
        ) is None

    def test_rejects_empty(self):
        assert T._rejection_reason("Hola", "   ", "bg") == "empty"

    def test_rejects_chain_of_thought_by_length(self):
        src = "Te amo mucho, gracias por apoyarme"
        tgt = ("The user wants me to translate a Spanish text into Russian. "
               "Source Text: ... Breakdown and Translation: " + "x" * 400)
        reason = T._rejection_reason(src, tgt, "ru")
        assert reason and "implausibly long" in reason

    def test_rejects_meta_prose_even_when_short(self):
        # A Latin-script target gets no script check, so the prose pattern is
        # the only thing standing between this and the TTS engine.
        reason = T._rejection_reason("Te amo", "Here is the translation: Je t'aime", "fr")
        assert reason and "about the task" in reason

    def test_rejects_untranslated_source_for_non_latin_target(self):
        reason = T._rejection_reason(
            "Hola mamá, te amo mucho", "Hola mamá, te amo mucho", "ru")
        assert reason and "wrong script" in reason

    def test_rejects_english_answer_for_non_latin_target(self):
        reason = T._rejection_reason(
            "Hola mamá, te amo mucho", "Hello mom, I love you very much", "ru")
        assert reason and "wrong script" in reason

    def test_allows_latin_target_to_look_latin(self):
        assert T._rejection_reason("Te amo mucho", "Je t'aime beaucoup", "fr") is None

    def test_no_script_rule_for_dual_script_languages(self):
        # Serbian, Kazakh, Azerbaijani, Uzbek and Mongolian are all written in
        # Latin as well as Cyrillic; a script check would reject valid work.
        for lang in ("sr", "kk", "az", "uz", "mn"):
            assert T._rejection_reason(
                "Hello there friend", "Zdravo prijatelju moj", lang) is None

    def test_short_answers_escape_the_script_check(self):
        assert T._rejection_reason("Sí", "OK", "ru") is None

    def test_accepts_cjk_targets(self):
        assert T._rejection_reason(
            "Hola mamá, te amo", "こんにちは、お母さん、大好きです", "ja") is None
        assert T._rejection_reason("Hola mamá", "你好，妈妈，我爱你", "zh") is None
        assert T._rejection_reason("Hola mamá", "안녕하세요 엄마 사랑해요", "ko") is None

    def test_unknown_target_language_skips_the_script_check(self):
        assert T._rejection_reason("Hello", "Anything at all here", "xx") is None


class TestBuildBatchPrompt:
    def test_numbers_every_line(self):
        prompt = _build_batch_prompt(["Hello", "World"], "ru")
        assert "1. Hello" in prompt
        assert "2. World" in prompt

    def test_states_the_exact_line_count(self):
        prompt = _build_batch_prompt(["a", "b", "c"], "ru")
        assert "EXACTLY 3 lines" in prompt

    def test_includes_context_hint_and_glossary(self):
        prompt = _build_batch_prompt(
            ["Guard pass"], "ru",
            context_hint="Brazilian Jiu-Jitsu",
            glossary={"guard": "гард"},
        )
        assert "Brazilian Jiu-Jitsu" in prompt
        assert "guard → гард" in prompt

    def test_previous_lines_are_marked_as_context_not_input(self):
        prompt = _build_batch_prompt(
            ["Then you sweep."], "ru",
            prev_pairs=[("Take the guard.", "Возьми гард.")],
        )
        assert "Take the guard." in prompt
        assert "Возьми гард." in prompt
        assert "do NOT translate" in prompt
        # The context lines must not be numbered — only the work is
        assert "1. Then you sweep." in prompt
        assert "1. Take the guard." not in prompt


class TestLanguageDisplayName:
    """language_display_name() spells language codes out in LLM prompts."""

    def test_known_codes_map_to_names(self):
        assert language_display_name("bg") == "Bulgarian"
        assert language_display_name("ru") == "Russian"
        assert language_display_name("en") == "English"

    def test_regional_codes_reduce_to_the_base_code(self):
        assert language_display_name("bg-BG") == "Bulgarian"
        assert language_display_name(" PT ") == "Portuguese"

    def test_unknown_codes_pass_through_unchanged(self):
        assert language_display_name("xx") == "xx"
        assert language_display_name("") == ""

    def test_batch_prompt_spells_the_language_out(self):
        prompt = _build_batch_prompt(["Hello"], "bg")
        assert "into Bulgarian" in prompt
        assert "into bg" not in prompt

    def test_legacy_prompt_spells_both_languages_out(self):
        prompt = _build_translation_prompt(
            text="Здравей", source_lang="bg", target_lang="en")
        assert "from Bulgarian to English" in prompt


class TestTranslateTexts:
    """translate_texts() — the metadata (title/description) helper, 3D."""

    def test_translates_each_string_in_order(self, monkeypatch):
        import asyncio
        calls = []

        async def fake_translate_text(text, target_lang, model="m", **kw):
            calls.append((text, target_lang, model))
            return f"[{target_lang}] {text}"

        monkeypatch.setattr(T, "translate_text", fake_translate_text)
        out = asyncio.run(T.translate_texts(
            ["My Title", "A description"], "ru", model="qwen3:8b"))
        assert out == ["[ru] My Title", "[ru] A description"]
        assert [c[0] for c in calls] == ["My Title", "A description"]
        assert all(c[2] == "qwen3:8b" for c in calls)

    def test_empty_list(self, monkeypatch):
        import asyncio

        async def fake_translate_text(*a, **kw):  # pragma: no cover
            raise AssertionError("must not be called")

        monkeypatch.setattr(T, "translate_text", fake_translate_text)
        assert asyncio.run(T.translate_texts([], "ru")) == []

    def test_failures_propagate(self, monkeypatch):
        import asyncio

        async def fake_translate_text(*a, **kw):
            raise T.LMStudioError("backend down")

        monkeypatch.setattr(T, "translate_text", fake_translate_text)
        with pytest.raises(T.LMStudioError):
            asyncio.run(T.translate_texts(["Title"], "ru"))

    def test_context_hint_forwarded(self, monkeypatch):
        import asyncio
        seen = {}

        async def fake_translate_text(text, target_lang, model="m", **kw):
            seen.update(kw)
            return text

        monkeypatch.setattr(T, "translate_text", fake_translate_text)
        asyncio.run(T.translate_texts(["x"], "ru", context_hint="a vlog"))
        assert seen.get("context_hint") == "a vlog"
