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
