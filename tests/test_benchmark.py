"""Tests for tools/gochidubb_benchmark.py — grading, metrics, fixtures.

None of these touch LM Studio; they exercise the scoring that turns raw
translations into a verdict, plus the shape of the checked-in fixtures.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gochidubb_benchmark", ROOT / "tools" / "gochidubb_benchmark.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def _fixture(n=3, gap=0.5, seg=4.0, texts=None):
    segs, t = [], 1.0
    for i in range(n):
        segs.append({"start": t, "end": t + seg, "speaker": "SPEAKER_00",
                     "text": (texts[i] if texts else f"This is source line {i}.")})
        t += seg + gap
    return {"id": "t", "source_lang": "en", "duration": t + 2.0,
            "context_hint": "", "segments": segs}


class TestFixtureCorpus:
    """The checked-in fixtures are data other people will rely on."""

    def test_every_fixture_loads_and_is_well_formed(self):
        fixtures = B.load_fixtures()
        assert fixtures, "no benchmark fixtures found"
        for f in fixtures:
            for key in ("id", "source_lang", "origin", "duration",
                        "challenges", "segments"):
                assert key in f, f"{f.get('id')} missing {key}"
            assert f["origin"] in ("real", "constructed")
            assert f["segments"], f"{f['id']} has no segments"

    def test_segments_are_ordered_and_non_overlapping(self):
        for f in B.load_fixtures():
            prev_end = -1.0
            for s in f["segments"]:
                assert s["end"] > s["start"], f"{f['id']} has an empty segment"
                assert s["start"] >= prev_end, f"{f['id']} segments overlap"
                prev_end = s["end"]
            assert f["segments"][-1]["end"] <= f["duration"], \
                f"{f['id']} runs past its stated duration"

    def test_constructed_fixtures_say_so(self):
        """Anyone reading a result has to know what is real and what is not."""
        for f in B.load_fixtures():
            if f["origin"] == "constructed":
                assert "not transcribed" in f["note"].lower()


class TestTtsRate:
    def test_dense_scripts_are_slower_per_character(self):
        # A Chinese character carries far more than a Latin one, so the same
        # character count takes much longer to speak.
        assert B.tts_rate_for("zh") < B.tts_rate_for("hi") < B.tts_rate_for("es")

    def test_unknown_language_gets_the_base_rate(self):
        assert B.tts_rate_for("xx") == B.BASE_TTS_RATE

    def test_accepts_region_codes(self):
        assert B.tts_rate_for("zh-CN") == B.tts_rate_for("zh")


class TestMeasure:
    def test_counts_lines_that_fell_back_to_source(self):
        f = _fixture(3)
        texts = [s["text"] for s in f["segments"]]      # nothing translated
        m = B.measure(f, "ru", texts, 1.0)
        assert m["fallback"] == 3

    def test_a_real_translation_has_no_fallback(self):
        f = _fixture(2)
        m = B.measure(f, "ru", ["Это первая строка.", "Это вторая строка."], 1.0)
        assert m["fallback"] == 0
        assert m["script_ok"] == 1.0

    def test_flags_output_in_the_wrong_script(self):
        f = _fixture(2)
        m = B.measure(f, "ru", ["Still English here", "And here as well"], 1.0)
        assert m["script_ok"] == 0.0

    def test_latin_targets_get_no_script_score(self):
        f = _fixture(2)
        m = B.measure(f, "es", ["Hola a todos", "Segunda linea aqui"], 1.0)
        assert m["script_ok"] is None

    def test_notices_dropped_digits(self):
        f = _fixture(1, texts=["She was 18 months old and had 40 days left."])
        kept = B.measure(f, "ru", ["Ей было 18 месяцев и оставалось 40 дней."], 1.0)
        lost = B.measure(f, "ru", ["Ей было несколько месяцев."], 1.0)
        assert kept["digits_kept"] == 1.0
        assert lost["digits_kept"] == 0.0

    def test_no_digits_in_source_means_nothing_to_score(self):
        f = _fixture(1, texts=["Hello there my friend."])
        assert B.measure(f, "ru", ["Привет, друг мой."], 1.0)["digits_kept"] is None

    def test_longer_translations_drift_further(self):
        f = _fixture(3, gap=0.2, seg=3.0)
        short = B.measure(f, "es", ["Hola."] * 3, 1.0)
        long = B.measure(f, "es", ["Hola " * 40] * 3, 1.0)
        assert long["drift_sec"] > short["drift_sec"]


class TestGrade:
    def _run(self, **kw):
        base = {"fallback": 0, "segments": 7, "meta_prose": 0, "script_ok": 1.0,
                "digits_kept": 1.0, "len_ratio": 1.1, "drift_sec": 1.0,
                "elapsed": 10.0, "texts": []}
        base.update(kw)
        return base

    def test_clean_runs_are_good(self):
        assert B.grade([self._run(), self._run()])["grade"] == "good"

    def test_any_meta_prose_is_disqualifying(self):
        """Narrating the task once means it will do it again on a real job."""
        g = B.grade([self._run(), self._run(meta_prose=1)])
        assert g["grade"] == "unusable"

    def test_a_single_timeout_is_disqualifying(self):
        g = B.grade([self._run(), {"error": "timeout after 300s", "elapsed": 300}])
        assert g["grade"] == "unusable"

    def test_all_runs_failing_reports_the_reason(self):
        g = B.grade([{"error": "timeout after 300s", "elapsed": 300}])
        assert g["grade"] == "unusable"
        assert "timeout" in g["reason"]

    def test_occasional_fallback_is_risky_not_fatal(self):
        g = B.grade([self._run(fallback=1), self._run()])
        assert g["grade"] == "risky"

    def test_heavy_fallback_is_fatal(self):
        g = B.grade([self._run(fallback=5), self._run(fallback=5)])
        assert g["grade"] == "unusable"

    def test_wrong_script_is_risky(self):
        assert B.grade([self._run(script_ok=0.8)])["grade"] == "risky"

    def test_drift_never_decides_a_verdict(self):
        """Drift rests on estimated per-script speech rates.

        Grading on it would rank models by which script they were pointed at.
        It is reported as a diagnostic and nothing more.
        """
        assert B.grade([self._run(drift_sec=40.0)])["grade"] == "good"
        assert B.grade([self._run(drift_sec=40.0)])["drift_sec"] == 40.0

    def test_length_ratio_never_decides_a_verdict(self):
        # Character counts are not comparable across scripts.
        assert B.grade([self._run(len_ratio=0.6)])["grade"] == "good"
        assert B.grade([self._run(len_ratio=1.9)])["grade"] == "good"

    def test_dropping_most_numbers_is_risky(self):
        assert B.grade([self._run(digits_kept=0.2)])["grade"] == "risky"

    def test_spelling_one_number_out_is_allowed(self):
        """"three years" -> "три года" loses digits and is still correct."""
        assert B.grade([self._run(digits_kept=0.7)])["grade"] == "good"

    def test_worst_run_decides_not_the_average(self):
        """A model that is excellent two runs in three is still unusable."""
        g = B.grade([self._run(), self._run(), self._run(meta_prose=3)])
        assert g["grade"] == "unusable"

    def test_reports_spread_across_runs(self):
        g = B.grade([self._run(elapsed=5.0), self._run(elapsed=25.0)])
        assert g["spread_sec"] == pytest.approx(20.0)


class TestRenderReport:
    def test_renders_a_matrix_from_results(self):
        verdict = {"grade": "good", "median_sec": 12, "len_ratio": 1.1,
                   "drift_sec": 1.0}
        data = {
            "fixtures": ["es_mothers_day"], "targets": ["ru", "hi"],
            "results": {"some/model": {"es_mothers_day": {
                "ru": {"verdict": verdict},
                "hi": {"verdict": dict(verdict, grade="unusable")}}}},
        }
        out = B.render_report(data, B.load_fixtures())
        assert "some/model" in out
        assert B.GRADE_MARK["good"] in out
        assert B.GRADE_MARK["unusable"] in out
        assert "Spanish source" in out

    def test_matrix_cell_shows_the_worst_source_language(self):
        """A model that fails on one source must not look clean in the matrix."""
        good = {"grade": "good", "median_sec": 10, "len_ratio": 1.0, "drift_sec": 0.5}
        bad = dict(good, grade="unusable")
        data = {
            "fixtures": ["es_mothers_day", "en_clinic_thanks"],
            "targets": ["ru"],
            "results": {"m": {"es_mothers_day": {"ru": {"verdict": good}},
                              "en_clinic_thanks": {"ru": {"verdict": bad}}}},
        }
        report = B.render_report(data, B.load_fixtures())
        row = next(ln for ln in report.splitlines() if ln.startswith("| `m` |"))
        assert B.GRADE_MARK["unusable"] in row
        assert B.GRADE_MARK["good"] not in row


class TestIsTranslationPair:
    """Spanish→Spanish is not a translation, and scoring it as one reads as
    a total failure: the correct output is the source text, which every
    metric here counts as the model having fallen back."""

    def test_different_languages_are_a_pair(self):
        assert B.is_translation_pair("es", "ru")

    def test_same_language_is_not(self):
        assert not B.is_translation_pair("es", "es")

    def test_region_variants_are_the_same_language(self):
        assert not B.is_translation_pair("pt", "pt-BR")
        assert not B.is_translation_pair("es-419", "es")

    def test_case_insensitive(self):
        assert not B.is_translation_pair("ES", "es")

    def test_a_model_that_only_ever_failed_shows_no_time(self):
        """"0s" would read as instant when it was really a timeout."""
        data = {
            "fixtures": ["es_mothers_day"], "targets": ["ru"],
            "results": {"slow/model": {"es_mothers_day": {"ru": {
                "runs": [{"error": "timeout after 300s", "elapsed": 300}] * 2}}}},
        }
        report = B.render_report(data, B.load_fixtures())
        row = next(ln for ln in report.splitlines()
                   if ln.startswith("| `slow/model` |"))
        assert B.GRADE_MARK["unusable"] in row
        assert "0s" not in row
        assert "n/a" in row


class TestChecklists:
    """Hand-written semantic checks — the only signal here that judges whether
    a translation is right rather than merely well-formed."""

    def test_every_checked_in_checklist_is_valid(self):
        import json
        import re
        fixtures = {f["id"]: f for f in B.load_fixtures()}
        found = list(B.CHECKLIST_DIR.glob("*.json"))
        assert found, "no checklists found"
        for path in found:
            doc = json.loads(path.read_text(encoding="utf-8"))
            fid, lang = path.stem.rsplit(".", 1)
            assert doc["fixture"] == fid, f"{path.name} names a different fixture"
            assert doc["target_lang"] == lang, f"{path.name} names a different language"
            assert fid in fixtures, f"{path.name} targets an unknown fixture"
            n = len(fixtures[fid]["segments"])
            assert doc["checks"], f"{path.name} has no checks"
            for check in doc["checks"]:
                assert 0 <= check["segment"] < n, \
                    f"{path.name}: segment {check['segment']} is out of range"
                assert check.get("why"), f"{path.name}: {check['label']} has no why"
                re.compile(check["expect"])
                if check.get("reject"):
                    re.compile(check["reject"])
            for pen in doc.get("penalties", []):
                re.compile(pen["pattern"])
                assert pen.get("why")

    def test_loads_by_fixture_and_language(self):
        assert B.load_checklist("es_mothers_day", "bg") is not None
        assert B.load_checklist("es_mothers_day", "bg-BG") is not None
        assert B.load_checklist("es_mothers_day", "xx") is None
        assert B.load_checklist("no_such_fixture", "bg") is None

    def test_a_correct_translation_passes(self):
        checklist = {"checks": [
            {"segment": 0, "label": "thanks", "expect": "благодар"}]}
        r = B.score_checklist(checklist, ["Благодаря ти за всичко."])
        assert r["passed"] == 1 and r["score"] == 1.0 and not r["failed"]

    def test_reject_overrides_a_matching_expect(self):
        """"Моля" is 'please'; a line can contain the stem and still be wrong."""
        checklist = {"checks": [
            {"segment": 0, "label": "gracias", "expect": "благодар|Моля",
             "reject": r"\bМоля\b"}]}
        r = B.score_checklist(checklist, ["Моля за всичко."])
        assert r["passed"] == 0
        assert r["failed"] == ["gracias"]

    def test_penalties_count_across_the_whole_translation(self):
        checklist = {"checks": [], "penalties": [
            {"label": "russianism", "pattern": "скуча"}]}
        r = B.score_checklist(checklist, ["скучам по теб", "и скучам пак"])
        assert r["penalty_hits"] == 2
        assert r["penalties"][0]["label"] == "russianism"

    def test_a_missing_segment_fails_rather_than_crashing(self):
        checklist = {"checks": [
            {"segment": 5, "label": "late line", "expect": "x"}]}
        r = B.score_checklist(checklist, ["only one line"])
        assert r["failed"] == ["late line"]

    def test_bulgarian_checklist_reproduces_the_hand_scoring(self):
        """This checklist was ported from scoring done by hand; if it drifts
        from those numbers the port is wrong, not the models."""
        checklist = B.load_checklist("es_mothers_day", "bg")
        good = [
            "Обичам те много. Благодаря, че ме подкрепи във всичко, което направи. Благодаря.",
            "Здравей, мамо. Пожелавам ти щастлив Ден на майката. Обичам те много.",
            "Здравей, мамо. Как си? Искам да ти кажа, че те обичам много и си най-добрата майка в света.",
            "Здравей, мамо. Обичам те много. Благодаря за всичко и се надявам да имаш щастлив Ден на майката.",
            "Здравей, мамо. Благодаря за това, което направи и те обичам.",
            "Здравей, мамо. Благодаря за всичко. Искам да знаеш, че те обичам много и ми липсваш много. Довиждане.",
            "Здравей, мамо. Благодаря за всичко, което направи за мен и се наслаждавай на този ден.",
        ]
        assert B.score_checklist(checklist, good)["passed"] == 13

        # The real aya-expanse-8b output that prompted this whole benchmark.
        aya = [
            "Ти ме обичаваш много. Моля за поддръжка в всичко, което си делала. Моля.",
            "Здравей, мамо. Желаем ти щастлив День на матицата. Обичам те много.",
            "Здравей, мамо. Как ти е? Хвали я да обичам те много и си най-добрата мамо на свет.",
            "Здравей, мамо. Обичам те много. Моля за всичко, което си делала за мен.",
            "Здравей, мамо. Моля за всеките, които си спреш и обичам те.",
            "Здравей, мамо. Моля за всеките. Знай, че обичам те много и скучам по тебе много. До слагание.",
            "Здравей, мамо. Моля за всеките, които си делала за мен и да наслаждаш се от този ден.",
        ]
        scored = B.score_checklist(checklist, aya)
        assert scored["passed"] <= 3, "aya scored 2/13 by hand"
        assert scored["penalty_hits"] >= 8, "its Russianisms are the whole point"


class TestSemanticGrading:
    def _run(self, semantic=None, **kw):
        base = {"fallback": 0, "segments": 7, "meta_prose": 0, "script_ok": 1.0,
                "digits_kept": 1.0, "len_ratio": 1.1, "drift_sec": 1.0,
                "elapsed": 10.0, "texts": [], "semantic": semantic}
        base.update(kw)
        return base

    def _sem(self, passed, total=12, penalties=0):
        return {"passed": passed, "total": total,
                "score": passed / total, "failed": [],
                "penalties": [], "penalty_hits": penalties}

    def test_full_marks_is_good(self):
        assert B.grade([self._run(self._sem(12))])["grade"] == "good"

    def test_mostly_wrong_meaning_is_unusable_despite_clean_mechanics(self):
        """This is what disqualifies aya-expanse-8b: every mechanical check
        passes and the translation still says the wrong thing."""
        g = B.grade([self._run(self._sem(2, 13))])
        assert g["grade"] == "unusable"

    def test_a_few_misses_are_risky(self):
        assert B.grade([self._run(self._sem(10, 12))])["grade"] == "risky"

    def test_grammar_penalties_alone_make_it_risky(self):
        g = B.grade([self._run(self._sem(12, 12, penalties=4))])
        assert g["grade"] == "risky"

    def test_the_worst_run_sets_the_semantic_score(self):
        g = B.grade([self._run(self._sem(12)), self._run(self._sem(4))])
        assert g["semantic"]["passed"] == 4
        assert g["grade"] == "unusable"

    def test_absent_checklist_leaves_grading_mechanical(self):
        g = B.grade([self._run(None)])
        assert g["grade"] == "good"
        assert g["semantic"] is None


class TestCjkNumerals:
    """Han numerals are separate codepoints; without them a correctly
    localized 十八 reads as a dropped number."""

    def _fixture_with_numbers(self):
        return _fixture(1, texts=["She was 18 months old and 40 days in."])

    def test_han_numerals_count_as_kept(self):
        f = self._fixture_with_numbers()
        m = B.measure(f, "zh", ["她十八个月大，已经四十天了。"], 1.0)
        assert m["digits_kept"] == 1.0

    def test_ascii_digits_still_count_for_cjk(self):
        f = self._fixture_with_numbers()
        m = B.measure(f, "zh", ["她18个月大，已经40天了。"], 1.0)
        assert m["digits_kept"] == 1.0

    def test_japanese_uses_the_han_pattern_too(self):
        f = self._fixture_with_numbers()
        assert B.measure(f, "ja", ["十八ヶ月で四十日。"], 1.0)["digits_kept"] == 1.0

    def test_korean_is_still_checked(self):
        """Korean can spell numbers as Hangul (십팔), which no pattern counts —
        but models keep ASCII digits in practice, and a spelled-out number
        lands in the same tolerance every other language gets. Skipping the
        check outright let a model drop a number and still score clean."""
        f = self._fixture_with_numbers()
        kept = B.measure(f, "ko", ["그녀는 18개월이었고 40일이 지났습니다."], 1.0)
        assert kept["digits_kept"] == 1.0
        dropped = B.measure(f, "ko", ["그녀는 아기였습니다."], 1.0)
        assert dropped["digits_kept"] == 0.0

    def test_han_numerals_do_not_leak_into_non_cjk_targets(self):
        f = self._fixture_with_numbers()
        m = B.measure(f, "ru", ["Ей было十八месяцев"], 1.0)
        assert m["digits_kept"] == 0.0

    def test_cjk_output_is_much_shorter_than_its_source(self):
        """The length intuition inverts for CJK, which is why length ratio
        never decides a verdict."""
        f = self._fixture_with_numbers()
        zh = B.measure(f, "zh", ["她十八个月大，已经四十天了。"], 1.0)
        es = B.measure(f, "es", ["Tenia 18 meses y llevaba 40 dias."], 1.0)
        assert zh["len_ratio"] < 0.5 < es["len_ratio"]

    def test_cjk_script_check_accepts_real_translations(self):
        f = _fixture(1, texts=["Hello mother, I love you very much."])
        for lang, text in [("zh", "你好，妈妈，我非常爱你。"),
                           ("ja", "こんにちは、お母さん、大好きです。"),
                           ("ko", "안녕하세요 엄마, 정말 사랑해요.")]:
            assert B.measure(f, lang, [text], 1.0)["script_ok"] == 1.0

    def test_cjk_script_check_rejects_an_english_answer(self):
        f = _fixture(1, texts=["Hello mother."])
        for lang in ("zh", "ja", "ko"):
            m = B.measure(f, lang, ["Hello mother, I love you very much."], 1.0)
            assert m["script_ok"] == 0.0, lang


class TestMeasuredRates:
    def test_a_measured_rate_beats_the_fallback(self, monkeypatch):
        monkeypatch.setattr(B, "_RATES_CACHE",
                            {"hi": {"voxcpm_chars_per_sec": 7.25}})
        assert B.tts_rate_for("hi") == 7.25
        assert B.drift_is_measured("hi")

    def test_unmeasured_languages_fall_back_and_say_so(self, monkeypatch):
        monkeypatch.setattr(B, "_RATES_CACHE", {})
        assert not B.drift_is_measured("hi")
        # Falls back to the per-script guess rather than the base rate.
        assert B.tts_rate_for("hi") < B.BASE_TTS_RATE

    def test_region_codes_resolve_to_the_measured_rate(self, monkeypatch):
        monkeypatch.setattr(B, "_RATES_CACHE",
                            {"zh": {"voxcpm_chars_per_sec": 3.2}})
        assert B.tts_rate_for("zh-CN") == 3.2

    def test_a_recorded_rates_file_is_well_formed(self):
        """Skipped until someone runs the measurement tool."""
        if not B.RATES_FILE.exists():
            pytest.skip("no speech_rates.json recorded yet")
        import json
        data = json.loads(B.RATES_FILE.read_text(encoding="utf-8"))
        assert data["anchor_lang"] and data["anchor_rate"] > 0
        for lang, r in data["rates"].items():
            assert r["voxcpm_chars_per_sec"] > 0, lang
            assert r["scale"] > 0, lang


class TestFixtureDiscovery:
    def test_non_fixture_json_in_the_directory_is_ignored(self, tmp_path,
                                                          monkeypatch):
        """speech_rates.json lives beside the fixtures and is not one."""
        import json
        monkeypatch.setattr(B, "FIXTURE_DIR", tmp_path)
        (tmp_path / "speech_rates.json").write_text(
            json.dumps({"engine": "edge-tts", "rates": {}}), encoding="utf-8")
        (tmp_path / "real.json").write_text(json.dumps({
            "id": "real", "source_lang": "en", "origin": "constructed",
            "duration": 5.0, "challenges": [], "note": "not transcribed",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
        }), encoding="utf-8")
        got = B.load_fixtures()
        assert [f["id"] for f in got] == ["real"]


class TestDegenerateChecklists:
    """A checklist that can produce no score must not be treated as one."""

    def test_a_checklist_with_no_checks_is_treated_as_absent(self, tmp_path,
                                                             monkeypatch):
        import json
        monkeypatch.setattr(B, "CHECKLIST_DIR", tmp_path)
        monkeypatch.setattr(B, "_CHECKLIST_CACHE", {})
        (tmp_path / "f.bg.json").write_text(
            json.dumps({"fixture": "f", "target_lang": "bg", "checks": []}),
            encoding="utf-8")
        assert B.load_checklist("f", "bg") is None

    def test_grading_survives_a_scoreless_semantic_result(self):
        """score is None when a checklist has no checks; comparing those
        raised a TypeError before this guard."""
        run = {"fallback": 0, "segments": 1, "meta_prose": 0, "script_ok": 1.0,
               "digits_kept": 1.0, "len_ratio": 1.0, "drift_sec": 0.0,
               "elapsed": 1.0, "texts": [],
               "semantic": B.score_checklist({"checks": []}, ["x"])}
        g = B.grade([run, run])
        assert g["grade"] == "good"
        assert g["semantic"] is None

    def test_checklists_are_cached_not_reread(self, monkeypatch):
        monkeypatch.setattr(B, "_CHECKLIST_CACHE", {})
        first = B.load_checklist("es_mothers_day", "bg")
        assert first is B.load_checklist("es_mothers_day", "bg")

    def test_report_generation_does_not_mutate_its_input(self):
        """render_report is a read path; a results file written after it must
        not have picked up rescored fields."""
        import copy
        data = {
            "fixtures": ["es_mothers_day"], "targets": ["bg"],
            "results": {"m": {"es_mothers_day": {"bg": {"runs": [{
                "fallback": 0, "segments": 7, "meta_prose": 0,
                "script_ok": 1.0, "digits_kept": 1.0, "len_ratio": 1.0,
                "drift_sec": 0.0, "elapsed": 1.0,
                "texts": ["Обичам те много."] * 7,
            }]}}}},
        }
        before = copy.deepcopy(data)
        B.render_report(data, B.load_fixtures())
        assert data == before


class TestReviewRegressions:
    """Each of these is a defect a review caught in the working tree."""

    def test_report_recomputes_drift_from_current_rates(self, monkeypatch):
        """The report marks drift measured by consulting today's rates file,
        so a stale stored drift would be presented as measured."""
        fixture = next(f for f in B.load_fixtures() if f["id"] == "es_mothers_day")
        texts = ["Здравей, мамо. Обичам те много."] * len(fixture["segments"])
        data = {
            "fixtures": ["es_mothers_day"], "targets": ["bg"],
            "results": {"m": {"es_mothers_day": {"bg": {"runs": [{
                "fallback": 0, "segments": len(fixture["segments"]),
                "meta_prose": 0, "script_ok": 1.0, "digits_kept": None,
                "len_ratio": 1.0, "elapsed": 1.0,
                "drift_sec": 999.0,          # nonsense left over from an old sweep
                "texts": texts,
            }]}}}},
        }
        verdict = B._verdicts(data, B.load_fixtures())["m"]["es_mothers_day"]["bg"]
        assert verdict["drift_sec"] != 999.0

    def test_penalties_match_case_insensitively(self):
        """A wrong-language failure produces sentence-initial forms, so a
        lowercase penalty pattern has to match the capitalised word."""
        checklist = {"checks": [], "penalties": [
            {"label": "bulgarian bleed", "pattern": r"\bздравей\b"}]}
        r = B.score_checklist(checklist, ["Здравей, мамо."])
        assert r["penalty_hits"] == 1

    def test_semantic_cell_describes_one_run_not_a_splice(self):
        """Reporting the worst score together with the highest penalty count
        seen anywhere described a run that never happened."""
        def run(passed, penalties):
            return {"fallback": 0, "segments": 7, "meta_prose": 0,
                    "script_ok": 1.0, "digits_kept": 1.0, "len_ratio": 1.0,
                    "drift_sec": 0.0, "elapsed": 1.0, "texts": [],
                    "semantic": {"passed": passed, "total": 13,
                                 "score": passed / 13, "failed": [],
                                 "penalties": [], "penalty_hits": penalties}}
        sem = B.grade([run(3, 11), run(9, 12)])["semantic"]
        assert (sem["passed"], sem["penalty_hits"]) == (3, 11)
