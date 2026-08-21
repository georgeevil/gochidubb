"""Tests for pipeline/quality.py scorers + the assembler loudnorm parser.

Everything here is pure-function testing over synthetic segment dicts.
conftest fixtures provide realistic pre-Wave-1 segments (no qa/avg_logprob
keys) — exactly the "old job" shape the available=False paths must handle;
the new-key shapes are built locally.
"""
import math

import pytest

from pipeline.assembler import _parse_loudnorm_stderr
from pipeline.quality import (
    LEN_RATIO_HIGH,
    STRETCH_THRESHOLD,
    asr_quality,
    full_report,
    loudness_quality,
    rollup,
    timing_quality,
    translation_quality,
    tts_quality,
)


# ── Local fixtures with the Wave-1 keys ──────────────────────────────────

def _seg(idx, **kw):
    base = {"idx": idx, "start": idx * 2.0, "end": idx * 2.0 + 1.8,
            "text": f"source line {idx}", "speaker": "SPEAKER_00"}
    base.update(kw)
    return base


@pytest.fixture
def scored_segments():
    """Segments carrying ASR confidence + translation + measured TTS QA."""
    return [
        _seg(0, avg_logprob=-0.15, no_speech_prob=0.01,
             word_conf_mean=0.95, word_conf_min=0.80,
             translated_text="перевод строки ноль",
             qa={"score": 0.05, "measured": True, "cer": 0.05,
                 "detected_lang": "ru", "lang_match": True,
                 "attempts": 1, "tier": 1}),
        _seg(1, avg_logprob=-0.40, no_speech_prob=0.02,
             word_conf_mean=0.90, word_conf_min=0.70,
             translated_text="перевод строки один",
             qa={"score": 0.12, "measured": True, "cer": 0.12,
                 "detected_lang": "ru", "lang_match": True,
                 "attempts": 1, "tier": 2}),
        _seg(2, avg_logprob=-1.90, no_speech_prob=0.75,
             translated_text="п",  # ratio 1/13 → low outlier
             qa={"score": 0.9, "measured": True, "cer": 0.7,
                 "detected_lang": "en", "lang_match": False,
                 "attempts": 3, "tier": 0}),
        _seg(3, avg_logprob=-0.30, no_speech_prob=0.03,
             translated_text="source line 3",  # identical → untranslated
             qa={"score": None, "measured": False, "cer": None,
                 "detected_lang": "", "lang_match": None,
                 "attempts": 1, "tier": 2}),
    ]


@pytest.fixture
def placements():
    """tts_placements.json rows as _save_placements writes them."""
    return [
        {"idx": 0, "src_start": 0.0, "src_end": 5.0,
         "dub_start": 0.0, "dub_end": 4.0},           # 0.8x — compressed
        {"idx": 1, "src_start": 6.0, "src_end": 10.0,
         "dub_start": 6.0, "dub_end": 11.0},          # 1.25x — stretched
        {"idx": 2, "src_start": 12.0, "src_end": 14.0,
         "dub_start": 13.0, "dub_end": 15.1},         # 1.05x + 1000ms drift
        {"idx": 3, "src_start": 15.0, "src_end": 15.0,
         "dub_start": 15.0, "dub_end": 16.0},         # zero src_dur → skipped
    ]


# ── asr_quality ──────────────────────────────────────────────────────────

class TestAsrQuality:
    def test_normal(self, scored_segments):
        q = asr_quality(scored_segments)
        assert q["available"] is True
        assert q["n_scored"] == 4
        assert q["mean_logprob"] == pytest.approx(-0.6875, abs=1e-3)
        assert q["high_no_speech_count"] == 1
        assert q["high_no_speech_indices"] == [2]
        # worst-first ordering by avg_logprob
        assert q["low_conf_segments"][0] == 2
        assert q["word_conf_mean"] == pytest.approx(0.925, abs=1e-3)
        assert q["word_conf_min"] == 0.70

    def test_old_segments_unavailable(self, sample_segments):
        assert asr_quality(sample_segments) == {"available": False}

    def test_empty(self):
        assert asr_quality([]) == {"available": False}

    def test_worst_five_cap(self):
        segs = [_seg(i, avg_logprob=-0.1 * i) for i in range(10)]
        q = asr_quality(segs)
        assert len(q["low_conf_segments"]) == 5
        assert q["low_conf_segments"] == [9, 8, 7, 6, 5]


# ── translation_quality ──────────────────────────────────────────────────

class TestTranslationQuality:
    def test_normal(self, scored_segments):
        q = translation_quality(scored_segments)
        assert q["available"] is True
        assert q["untranslated_count"] == 1
        assert q["untranslated_indices"] == [3]
        assert q["outlier_count"] == 1
        assert q["outlier_indices"] == [2]
        assert q["len_ratio_mean"] > 0

    def test_high_ratio_outlier(self):
        segs = [_seg(0, translated_text="x" * 100)]  # 100/13 ≫ LEN_RATIO_HIGH
        q = translation_quality(segs)
        assert q["outlier_indices"] == [0]
        assert q["len_ratio_mean"] > LEN_RATIO_HIGH

    def test_untranslated_when_no_key_present(self, sample_segments):
        # No segment even has translated_text → stage never ran
        assert translation_quality(sample_segments) == {"available": False}

    def test_empty(self):
        assert translation_quality([]) == {"available": False}


# ── tts_quality ──────────────────────────────────────────────────────────

class TestTtsQuality:
    def test_normal(self, scored_segments):
        q = tts_quality(scored_segments)
        assert q["available"] is True
        assert q["n_with_qa"] == 4
        assert q["qa_measured_pct"] == 75.0
        assert q["mean_cer"] == pytest.approx((0.05 + 0.12 + 0.7) / 3, abs=1e-3)
        assert q["tier_histogram"] == {"1": 1, "2": 2, "0": 1}
        assert q["degraded_count"] == 1
        assert q["degraded_indices"] == [2]
        assert q["worst_segments"][0] == 2  # highest CER first
        assert q["lang_match_rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_old_segments_unavailable(self, sample_segments):
        assert tts_quality(sample_segments) == {"available": False}

    def test_all_unmeasured(self):
        segs = [_seg(i, qa={"score": None, "measured": False, "cer": None,
                            "detected_lang": "", "lang_match": None,
                            "attempts": 1, "tier": 2}) for i in range(3)]
        q = tts_quality(segs)
        assert q["available"] is True
        assert q["qa_measured_pct"] == 0.0
        assert "mean_cer" not in q


# ── timing_quality ───────────────────────────────────────────────────────

class TestTimingQuality:
    def test_normal(self, placements):
        q = timing_quality(placements)
        assert q["available"] is True
        assert q["n_placed"] == 3          # zero-duration row skipped
        assert q["stretched_count"] == 1   # only 1.25x is > STRETCH_THRESHOLD
        assert q["max_stretch"] == pytest.approx(1.25, abs=1e-3)
        assert q["mean_abs_drift_ms"] == pytest.approx(1000 / 3, abs=1.0)
        assert q["worst_segments"][0] == 1
        assert STRETCH_THRESHOLD == 1.05   # 1.05x row must NOT count

    def test_empty(self):
        assert timing_quality([]) == {"available": False}
        assert timing_quality(None) == {"available": False}

    def test_all_invalid_rows(self):
        rows = [{"idx": 0, "src_start": 1.0, "src_end": 1.0,
                 "dub_start": 1.0, "dub_end": 2.0}]
        assert timing_quality(rows) == {"available": False}


# ── loudness_quality ─────────────────────────────────────────────────────

class TestLoudnessQuality:
    def test_passthrough(self):
        q = loudness_quality({"input_i": -27.6, "output_i": -16.5,
                              "output_lra": 11.2, "output_tp": -1.5})
        assert q == {"available": True, "input_i": -27.6, "output_i": -16.5,
                     "lra": 11.2, "true_peak": -1.5}

    def test_missing(self):
        assert loudness_quality(None) == {"available": False}
        assert loudness_quality({}) == {"available": False}
        assert loudness_quality({"normalization_type": "dynamic"}) == \
            {"available": False}


# ── rollup ───────────────────────────────────────────────────────────────

class TestRollup:
    def test_scores_in_bounds(self, scored_segments, placements):
        rep = full_report(scored_segments, placements,
                          {"input_i": -27.0, "output_i": -16.2,
                           "output_lra": 11.0, "output_tp": -1.5})
        r = rep["rollup"]
        assert 0 <= r["overall"] <= 100
        for stage in ("asr", "translation", "tts", "timing", "loudness"):
            entry = r["stages"][stage]
            assert entry["available"] is True
            assert 0 <= entry["score"] <= 100

    def test_bounds_with_pathological_inputs(self):
        # everything terrible → every score still clamped to [0, 100]
        segs = [_seg(i, avg_logprob=-5.0, no_speech_prob=0.99,
                     translated_text="",  # untranslated
                     qa={"score": 2.0, "measured": True, "cer": 2.0,
                         "detected_lang": "en", "lang_match": False,
                         "attempts": 3, "tier": 0}) for i in range(10)]
        pl = [{"idx": i, "src_start": i, "src_end": i + 1.0,
               "dub_start": i + 5.0, "dub_end": i + 9.0} for i in range(10)]
        r = full_report(segs, pl, {"output_i": -40.0})["rollup"]
        for entry in r["stages"].values():
            if entry["available"]:
                assert 0 <= entry["score"] <= 100
        assert 0 <= r["overall"] <= 100

    def test_nothing_available(self, sample_segments):
        r = rollup(asr_quality(sample_segments),
                   translation_quality(sample_segments),
                   tts_quality(sample_segments),
                   timing_quality([]), loudness_quality(None))
        assert r["overall"] is None
        assert all(not s["available"] for s in r["stages"].values())
        assert r["verdicts"] == []

    def test_verdict_degraded_tts_suggests_retry(self, scored_segments):
        r = full_report(scored_segments, [], None)["rollup"]
        tts_verdicts = [v for v in r["verdicts"] if v["stage"] == "tts"]
        actions = {v["suggested_action"] for v in tts_verdicts}
        assert "retry_tts" in actions
        degraded = next(v for v in tts_verdicts
                        if v["suggested_action"] == "retry_tts")
        assert 2 in degraded["segment_indices"]

    def test_verdict_untranslated_suggests_retranslate(self):
        segs = [_seg(i, translated_text="" if i < 3 else f"пер {i}")
                for i in range(10)]
        r = full_report(segs, [], None)["rollup"]
        v = next(v for v in r["verdicts"]
                 if v["stage"] == "translation"
                 and v["suggested_action"] == "retranslate")
        assert v["severity"] == "serious"  # 30% untranslated ≥ 10%
        assert v["segment_indices"] == [0, 1, 2]

    def test_verdict_stretched_names_counts(self):
        pl = [{"idx": i, "src_start": i * 10.0, "src_end": i * 10.0 + 4.0,
               "dub_start": i * 10.0, "dub_end": i * 10.0 + 5.6}
              for i in range(7)]  # all 1.4x stretched
        r = rollup({}, {}, {}, timing_quality(pl), {})
        v = next(v for v in r["verdicts"] if v["stage"] == "timing")
        assert "7/7" in v["message"]
        assert v["suggested_action"] == "edit_translations"
        assert v["severity"] == "warn"

    def test_verdict_shape(self, scored_segments, placements):
        for v in full_report(scored_segments, placements, None)["rollup"]["verdicts"]:
            assert v["severity"] in ("info", "warn", "serious")
            assert isinstance(v["segment_indices"], list)
            assert v["suggested_action"] in (
                "retry_tts", "edit_translations", "regenerate_segment",
                "retranslate", None)

    def test_empty_everything(self):
        rep = full_report([], [], None)
        assert rep["rollup"]["overall"] is None


# ── loudnorm stderr parsing (assembler helper) ───────────────────────────

REAL_STDERR = """\
size=N/A time=00:03:21.06 bitrate=N/A speed=41.3x
video:0kB audio:18856kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
[Parsed_loudnorm_3 @ 0x600002bcc0b0]
{
\t"input_i" : "-27.61",
\t"input_tp" : "-4.47",
\t"input_lra" : "18.06",
\t"input_thresh" : "-39.20",
\t"output_i" : "-16.58",
\t"output_tp" : "-1.50",
\t"output_lra" : "14.78",
\t"output_thresh" : "-27.71",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.58"
}
"""


class TestParseLoudnormStderr:
    def test_realistic_block(self):
        d = _parse_loudnorm_stderr(REAL_STDERR)
        assert d["input_i"] == pytest.approx(-27.61)
        assert d["output_i"] == pytest.approx(-16.58)
        assert d["output_tp"] == pytest.approx(-1.50)
        assert d["output_lra"] == pytest.approx(14.78)
        assert d["normalization_type"] == "dynamic"

    def test_inf_values_dropped(self):
        text = REAL_STDERR.replace('"-27.61"', '"-inf"')
        d = _parse_loudnorm_stderr(text)
        assert "input_i" not in d
        assert d["output_i"] == pytest.approx(-16.58)
        assert all(v is None or isinstance(v, str) or math.isfinite(v)
                   for v in d.values())

    def test_garbage_returns_empty(self):
        assert _parse_loudnorm_stderr("") == {}
        assert _parse_loudnorm_stderr("frame=1 fps=0 size=2kB") == {}
        assert _parse_loudnorm_stderr("{ not json ") == {}

    def test_ignores_unrelated_json_blocks(self):
        text = '{"other": "block"}\n' + REAL_STDERR
        assert _parse_loudnorm_stderr(text)["input_i"] == pytest.approx(-27.61)

    def test_feeds_loudness_quality(self):
        q = loudness_quality(_parse_loudnorm_stderr(REAL_STDERR))
        assert q["available"] is True
        assert q["output_i"] == pytest.approx(-16.58)


class TestSourceBleedDetection:
    """The uk→bg failure: Ukrainian words and letters inside otherwise-
    Bulgarian lines. Every line differed from its source, so the existing
    untranslated check saw nothing."""

    def _seg(self, i, src, tgt):
        return {"idx": i, "text": src, "translated_text": tgt}

    def test_partial_bleed_is_invisible_without_the_language_pair(self):
        from pipeline.quality import translation_quality
        segs = [self._seg(0, "Набагато легше якось на голову.",
                          "Набагато по-лесно, каквото и да е на главата.")]
        out = translation_quality(segs)          # no pair given
        assert out["untranslated_count"] == 0    # the old check sees nothing
        assert "bleed_count" not in out

    def test_verbatim_source_word_is_caught_with_the_pair(self):
        from pipeline.quality import translation_quality
        segs = [self._seg(0, "Набагато легше якось на голову.",
                          "Набагато по-лесно, каквото и да е на главата.")]
        out = translation_quality(segs, target_lang="bg", source_lang="uk")
        assert out["bleed_count"] == 1
        assert "набагато" in out["bleed_examples"][0]

    def test_foreign_letter_is_caught(self):
        from pipeline.quality import translation_quality
        segs = [self._seg(0, "Тибетське-китайське місто Кіронг.",
                          "Тибетско-китайски град Кіронг.")]
        out = translation_quality(segs, target_lang="bg", source_lang="uk")
        assert out["bleed_count"] == 1
        assert "і" in out["bleed_examples"][0]

    def test_clean_translation_is_not_flagged(self):
        from pipeline.quality import translation_quality
        segs = [self._seg(0, "Остання моя ніч в Китаї.",
                          "Последната ми нощ в Китай.")]
        out = translation_quality(segs, target_lang="bg", source_lang="uk")
        assert out["bleed_count"] == 0

    def test_cross_script_repeats_are_not_flagged(self):
        """A name carried across scripts is correct, not bleed."""
        from pipeline.quality import translation_quality
        segs = [self._seg(0, "Barcelona is lovely", "Барселона is lovely")]
        out = translation_quality(segs, target_lang="bg", source_lang="en")
        assert out["bleed_count"] == 0

    def test_bleed_lowers_the_score_so_it_matches_the_verdict(self):
        """A 100 score beside a 'serious' verdict teaches people to
        distrust the number."""
        from pipeline.quality import translation_quality, rollup
        segs = [self._seg(i, "Набагато легше", "Набагато по-лесно")
                for i in range(4)]
        tr = translation_quality(segs, target_lang="bg", source_lang="uk")
        r = rollup({"available": False}, tr, {"available": False},
                   {"available": False}, {"available": False})
        assert r["stages"]["translation"]["score"] < 60


class TestQualityGate:
    """Gating is policy: is this good enough to spend the next stage on."""

    def _report(self, bleed, n):
        segs = []
        for i in range(n):
            if i < bleed:
                segs.append({"idx": i, "text": "Набагато легше",
                             "translated_text": "Набагато по-лесно"})
            else:
                segs.append({"idx": i, "text": "Остання моя ніч",
                             "translated_text": "Последната ми нощ"})
        from pipeline.quality import full_report
        return full_report(segs, target_lang="bg", source_lang="uk")

    def test_a_clean_translation_passes(self):
        from pipeline.quality import gate
        g = gate(self._report(0, 10), "translation")
        assert g["pass"] and not g["reasons"]

    def test_widespread_bleed_trips_the_gate(self):
        from pipeline.quality import gate
        g = gate(self._report(3, 10), "translation")
        assert not g["pass"]
        assert any("source-language material" in r for r in g["reasons"])

    def test_a_single_bleed_in_a_long_job_does_not_trip_it(self):
        """One place name in fifty lines is not a reason to stop."""
        from pipeline.quality import gate
        g = gate(self._report(1, 50), "translation")
        assert g["pass"]

    def test_an_unscorable_stage_passes_rather_than_blocking(self):
        from pipeline.quality import gate, full_report
        g = gate(full_report([]), "translation")
        assert g["pass"]

    def test_failing_gate_carries_actionable_verdicts(self):
        from pipeline.quality import gate
        g = gate(self._report(5, 10), "translation")
        assert not g["pass"]
        assert g["verdicts"], "a paused job must say what to do about it"
        assert any(v.get("suggested_action") for v in g["verdicts"])
