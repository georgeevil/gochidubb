"""Per-stage quality scoring over already-persisted pipeline artifacts.

Pure functions only — every scorer takes plain lists/dicts (checkpoint
segments, tts_placements.json rows, the loudnorm measurement from
metrics.json) and returns a JSON-safe dict. No file I/O, no model loads:
this module must stay cheap enough to call from a request handler.

Each scorer returns ``{"available": False}``-style dicts when the input
predates the signals it needs (old jobs have no ``seg["qa"]`` /
``avg_logprob``), so callers never have to guard for missing keys.

``rollup()`` converts the scorer outputs into 0-100 per-stage scores, an
overall weighted score, and a list of actionable verdicts whose
``suggested_action`` maps to existing server routes (retry_tts,
edit_translations, regenerate_segment, retranslate) so agents can act on
the report mechanically.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

log = logging.getLogger("gochidubb.quality")

# ── Thresholds (module constants so tests and docs agree) ────────────────

# ASR: faster-whisper avg_logprob is ~0 for confident speech and drops
# toward -1.5 and below for mumbled / noisy / hallucinated segments.
ASR_LOGPROB_GOOD = -0.2      # mean at or above this scores 100
ASR_LOGPROB_BAD = -1.5       # mean at or below this scores 0
ASR_NO_SPEECH_HIGH = 0.5     # no_speech_prob above this = "probably not speech"

# Translation length ratio (translated chars / source chars). Russian and
# Spanish legitimately run 10-30% longer than English; 2x or 0.3x means the
# LLM merged, dropped or padded lines.
LEN_RATIO_HIGH = 2.0
LEN_RATIO_LOW = 0.3

# TTS roundtrip QA (see pipeline/tts_qa.py — CER 0.0 perfect, gate at 0.4).
TTS_CER_BAD = 0.4            # mirrors tts_qa.QA_THRESHOLD
TTS_CER_SCALE = 0.8          # CER at which the score component reaches 0

# Timing: assembler stretches audio with atempo when the dub overruns its
# slot; >1.05 is where a stretch was actually applied (see assembler.py).
STRETCH_THRESHOLD = 1.05
STRETCH_SEVERE = 1.15        # assembler's own cap for non-emotion segments

# Loudness targets mirror pipeline/assembler.py LN_I.
LOUDNESS_TARGET_I = -16.0
LOUDNESS_TOLERANCE = 2.0     # LUFS deviation before we start flagging

# Weights for the overall score. Only available stages participate; the
# weights of missing stages are redistributed proportionally.
STAGE_WEIGHTS = {
    "asr": 0.20,
    "translation": 0.25,
    "tts": 0.30,
    "timing": 0.15,
    "loudness": 0.10,
}

WORST_N = 5  # how many "worst segment" indices each scorer reports


# ── Small pure helpers ───────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _finite(v: Any) -> Optional[float]:
    """Return v as a finite float, else None (guards ffmpeg '-inf' etc.)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def _percentile(vals: List[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 100]) of a non-empty list."""
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _idx(seg: dict, fallback: int) -> int:
    v = seg.get("idx")
    return v if isinstance(v, int) else fallback


# ── Per-stage scorers ────────────────────────────────────────────────────

def asr_quality(segments: List[dict]) -> Dict[str, Any]:
    """Aggregate faster-whisper confidence signals persisted per segment.

    Uses ``avg_logprob`` / ``no_speech_prob`` (Wave-1 additions) plus the
    optional ``word_conf_mean`` / ``word_conf_min`` aggregates.
    """
    segments = segments or []
    with_lp = [(i, s, _finite(s.get("avg_logprob"))) for i, s in enumerate(segments)]
    with_lp = [(i, s, lp) for i, s, lp in with_lp if lp is not None]
    if not with_lp:
        return {"available": False}

    logprobs = [lp for _, _, lp in with_lp]
    high_no_speech = [
        _idx(s, i) for i, s in enumerate(segments)
        if (_finite(s.get("no_speech_prob")) or 0.0) > ASR_NO_SPEECH_HIGH
    ]
    worst = sorted(with_lp, key=lambda t: t[2])[:WORST_N]

    out: Dict[str, Any] = {
        "available": True,
        "n_segments": len(segments),
        "n_scored": len(with_lp),
        "mean_logprob": round(_mean(logprobs), 4),
        "p10_logprob": round(_percentile(logprobs, 10), 4),
        "high_no_speech_count": len(high_no_speech),
        "high_no_speech_indices": high_no_speech[:20],
        "low_conf_segments": [_idx(s, i) for i, s, _ in worst],
    }
    wc_means = [v for v in (_finite(s.get("word_conf_mean")) for s in segments)
                if v is not None]
    wc_mins = [v for v in (_finite(s.get("word_conf_min")) for s in segments)
               if v is not None]
    if wc_means:
        out["word_conf_mean"] = round(_mean(wc_means), 4)
    if wc_mins:
        out["word_conf_min"] = round(min(wc_mins), 4)
    return out


def translation_quality(segments: List[dict]) -> Dict[str, Any]:
    """Untranslated lines + length-ratio distribution of the translation."""
    segments = segments or []
    considered = [(i, s) for i, s in enumerate(segments)
                  if (s.get("text") or "").strip()]
    if not considered or not any("translated_text" in s for _, s in considered):
        return {"available": False}

    untranslated: List[int] = []
    ratios: List[float] = []
    outliers: List[int] = []
    for i, s in considered:
        src = (s.get("text") or "").strip()
        tgt = (s.get("translated_text") or "").strip()
        if not tgt or tgt == src:
            untranslated.append(_idx(s, i))
            continue
        ratio = len(tgt) / len(src)
        ratios.append(ratio)
        if ratio > LEN_RATIO_HIGH or ratio < LEN_RATIO_LOW:
            outliers.append(_idx(s, i))

    out: Dict[str, Any] = {
        "available": True,
        "n_segments": len(considered),
        "untranslated_count": len(untranslated),
        "untranslated_indices": untranslated[:20],
        "outlier_indices": outliers[:20],
        "outlier_count": len(outliers),
    }
    if ratios:
        out["len_ratio_mean"] = round(_mean(ratios), 3)
        out["len_ratio_p90"] = round(_percentile(ratios, 90), 3)
    return out


def tts_quality(segments: List[dict]) -> Dict[str, Any]:
    """Aggregate the per-segment roundtrip-QA dicts written by tts_worker.

    ``seg["qa"]`` = {score, measured, cer, detected_lang, lang_match,
    attempts, tier} — tier 0 marks "all retries failed QA, kept last output"
    (degraded).
    """
    segments = segments or []
    qa_segs = [(i, s, s.get("qa")) for i, s in enumerate(segments)
               if isinstance(s.get("qa"), dict)]
    if not qa_segs:
        return {"available": False}

    measured = [(i, s, qa) for i, s, qa in qa_segs if qa.get("measured")]
    cers = [(i, s, _finite(qa.get("cer"))) for i, s, qa in measured]
    cers = [(i, s, c) for i, s, c in cers if c is not None]

    tier_hist: Dict[str, int] = {}
    degraded: List[int] = []
    for i, s, qa in qa_segs:
        tier = qa.get("tier")
        tier_hist[str(tier)] = tier_hist.get(str(tier), 0) + 1
        if tier == 0:
            degraded.append(_idx(s, i))

    lang_checked = [qa for _, _, qa in measured if qa.get("lang_match") is not None]
    worst = sorted(cers, key=lambda t: -t[2])[:WORST_N]

    out: Dict[str, Any] = {
        "available": True,
        "n_segments": len(segments),
        "n_with_qa": len(qa_segs),
        "qa_measured_pct": round(100.0 * len(measured) / len(qa_segs), 1),
        "tier_histogram": tier_hist,
        "degraded_count": len(degraded),
        "degraded_indices": degraded[:20],
        "worst_segments": [_idx(s, i) for i, s, _ in worst],
    }
    if cers:
        vals = [c for _, _, c in cers]
        out["mean_cer"] = round(_mean(vals), 4)
        out["p90_cer"] = round(_percentile(vals, 90), 4)
    if lang_checked:
        out["lang_match_rate"] = round(
            sum(1 for qa in lang_checked if qa.get("lang_match")) / len(lang_checked), 3)
    return out


def timing_quality(placements: List[dict]) -> Dict[str, Any]:
    """Stretch/drift stats from tts_placements.json rows.

    Each row: {idx, src_start, src_end, dub_start, dub_end} — where the
    assembler actually placed each segment after atempo stretching and
    overlap-pushing (see server.py _save_placements).
    """
    placements = placements or []
    rows = []
    for i, p in enumerate(placements):
        src_d = _finite(p.get("src_end", 0)) or 0.0
        src_s = _finite(p.get("src_start", 0)) or 0.0
        dub_s = _finite(p.get("dub_start"))
        dub_e = _finite(p.get("dub_end"))
        src_dur = src_d - src_s
        if dub_s is None or dub_e is None or src_dur <= 0:
            continue
        stretch = (dub_e - dub_s) / src_dur
        drift_ms = abs(dub_s - src_s) * 1000.0
        rows.append((p.get("idx", i), stretch, drift_ms))
    if not rows:
        return {"available": False}

    stretches = [r[1] for r in rows]
    drifts = [r[2] for r in rows]
    stretched = [r for r in rows if r[1] > STRETCH_THRESHOLD]
    worst = sorted(rows, key=lambda r: -r[1])[:WORST_N]
    return {
        "available": True,
        "n_placed": len(rows),
        "stretched_count": len(stretched),
        "stretched_pct": round(100.0 * len(stretched) / len(rows), 1),
        "max_stretch": round(max(stretches), 3),
        "mean_stretch": round(_mean(stretches), 3),
        "mean_abs_drift_ms": round(_mean(drifts), 1),
        "worst_segments": [r[0] for r in worst],
    }


def loudness_quality(loudnorm: Optional[dict]) -> Dict[str, Any]:
    """Pass-through summary of the ffmpeg loudnorm measurement."""
    if not isinstance(loudnorm, dict) or not loudnorm:
        return {"available": False}
    out = {
        "available": True,
        "input_i": _finite(loudnorm.get("input_i")),
        "output_i": _finite(loudnorm.get("output_i")),
        "lra": _finite(loudnorm.get("output_lra", loudnorm.get("lra"))),
        "true_peak": _finite(loudnorm.get("output_tp", loudnorm.get("true_peak"))),
    }
    if out["output_i"] is None and out["input_i"] is None:
        return {"available": False}
    return out


# ── Rollup ───────────────────────────────────────────────────────────────
#
# Score mappings (documented here so a 63 is explainable):
#   asr          linear on mean avg_logprob between ASR_LOGPROB_BAD (0) and
#                ASR_LOGPROB_GOOD (100), minus 2 points per high-no_speech
#                segment (capped at 20).
#   translation  100 - 1.5 * untranslated% - 0.5 * outlier%.
#   tts          70% CER component (linear: CER 0 → 70, TTS_CER_SCALE → 0)
#                + 30% measured-coverage component, then scaled by
#                lang_match_rate when known.
#   timing       100 - 0.5 * stretched% - 200 * (max_stretch - STRETCH_SEVERE
#                when above it).
#   loudness     100 - 10 * LUFS deviation of output_i beyond
#                LOUDNESS_TOLERANCE from LOUDNESS_TARGET_I.
#   overall      weighted mean of available stage scores (STAGE_WEIGHTS,
#                renormalized over available stages).

def _score_asr(asr: Dict[str, Any]) -> Optional[float]:
    if not asr.get("available"):
        return None
    span = ASR_LOGPROB_GOOD - ASR_LOGPROB_BAD
    base = _clamp(100.0 * (asr["mean_logprob"] - ASR_LOGPROB_BAD) / span)
    penalty = min(20.0, 2.0 * asr.get("high_no_speech_count", 0))
    return _clamp(base - penalty)


def _score_translation(tr: Dict[str, Any]) -> Optional[float]:
    if not tr.get("available"):
        return None
    n = max(tr.get("n_segments", 0), 1)
    untranslated_pct = 100.0 * tr.get("untranslated_count", 0) / n
    outlier_pct = 100.0 * tr.get("outlier_count", 0) / n
    return _clamp(100.0 - 1.5 * untranslated_pct - 0.5 * outlier_pct)


def _score_tts(tts: Dict[str, Any]) -> Optional[float]:
    if not tts.get("available"):
        return None
    measured_pct = tts.get("qa_measured_pct", 0.0)
    if measured_pct <= 0:
        return None  # QA ran but never measured anything — nothing to score
    mean_cer = tts.get("mean_cer")
    cer_component = 70.0 if mean_cer is None else \
        _clamp(70.0 * (1.0 - mean_cer / TTS_CER_SCALE), 0.0, 70.0)
    coverage_component = 30.0 * measured_pct / 100.0
    score = cer_component + coverage_component
    lang_rate = tts.get("lang_match_rate")
    if lang_rate is not None:
        score *= lang_rate
    return _clamp(score)


def _score_timing(tm: Dict[str, Any]) -> Optional[float]:
    if not tm.get("available"):
        return None
    penalty = 0.5 * tm.get("stretched_pct", 0.0)
    overshoot = max(0.0, tm.get("max_stretch", 1.0) - STRETCH_SEVERE)
    penalty += 200.0 * overshoot
    return _clamp(100.0 - penalty)


def _score_loudness(ld: Dict[str, Any]) -> Optional[float]:
    if not ld.get("available") or ld.get("output_i") is None:
        return None
    deviation = abs(ld["output_i"] - LOUDNESS_TARGET_I)
    excess = max(0.0, deviation - LOUDNESS_TOLERANCE)
    return _clamp(100.0 - 10.0 * excess)


def _verdicts(asr, tr, tts, tm, ld) -> List[Dict[str, Any]]:
    v: List[Dict[str, Any]] = []

    def add(severity, stage, message, segment_indices=None, suggested_action=None):
        v.append({
            "severity": severity,
            "stage": stage,
            "message": message,
            "segment_indices": segment_indices or [],
            "suggested_action": suggested_action,
        })

    if asr.get("available"):
        n = asr.get("n_scored", 0)
        if asr["mean_logprob"] < -1.0:
            add("warn", "asr",
                f"Mean transcription confidence is low (avg_logprob "
                f"{asr['mean_logprob']:.2f}) — noisy source audio or heavy "
                f"accents; downstream translation inherits these errors.",
                asr.get("low_conf_segments"))
        hn = asr.get("high_no_speech_count", 0)
        if hn:
            add("warn" if hn > max(2, n // 20) else "info", "asr",
                f"{hn}/{n} segments look like non-speech "
                f"(no_speech_prob > {ASR_NO_SPEECH_HIGH}) — possible music or "
                f"noise transcribed as dialogue.",
                asr.get("high_no_speech_indices"))

    if tr.get("available"):
        n = max(tr.get("n_segments", 0), 1)
        uc = tr.get("untranslated_count", 0)
        if uc:
            pct = 100.0 * uc / n
            add("serious" if pct >= 10 else "warn", "translation",
                f"{uc}/{n} segments have no translation (empty or identical "
                f"to the source) — they play in the source language.",
                tr.get("untranslated_indices"), "retranslate")
        oc = tr.get("outlier_count", 0)
        if oc:
            add("warn", "translation",
                f"{oc}/{n} translations are length outliers (>"
                f"{LEN_RATIO_HIGH}x or <{LEN_RATIO_LOW}x the source length) — "
                f"likely merged, dropped or padded lines.",
                tr.get("outlier_indices"), "edit_translations")

    if tts.get("available"):
        dc = tts.get("degraded_count", 0)
        nq = max(tts.get("n_with_qa", 0), 1)
        if dc:
            pct = 100.0 * dc / nq
            add("serious" if pct >= 10 else "warn", "tts",
                f"{dc}/{nq} segments exhausted all QA retries and kept a "
                f"failing take (tier 0) — audibly garbled speech is likely.",
                tts.get("degraded_indices"), "retry_tts")
        lang_rate = tts.get("lang_match_rate")
        if lang_rate is not None and lang_rate < 0.9:
            add("warn", "tts",
                f"Only {lang_rate * 100:.0f}% of measured segments were "
                f"detected in the target language — the voice may drift into "
                f"source-language phonetics.",
                tts.get("worst_segments"), "regenerate_segment")
        p90 = tts.get("p90_cer")
        if p90 is not None and p90 > TTS_CER_BAD:
            add("warn", "tts",
                f"p90 CER is {p90:.2f} (gate {TTS_CER_BAD}) — the worst tenth "
                f"of segments is barely intelligible on ASR roundtrip.",
                tts.get("worst_segments"), "regenerate_segment")
        mp = tts.get("qa_measured_pct", 0.0)
        if 0 < mp < 50:
            add("info", "tts",
                f"QA only measured {mp:.0f}% of segments (whisper unavailable "
                f"or ASR failures) — CER stats cover a partial sample.")

    if tm.get("available"):
        sc = tm.get("stretched_count", 0)
        n = max(tm.get("n_placed", 0), 1)
        if sc:
            pct = tm.get("stretched_pct", 0.0)
            add("warn" if pct >= 10 else "info", "timing",
                f"{sc}/{n} segments were time-stretched >"
                f"{STRETCH_THRESHOLD}x (max {tm.get('max_stretch')}x) — "
                f"translations too long for their slots.",
                tm.get("worst_segments"), "edit_translations")
        if tm.get("mean_abs_drift_ms", 0.0) > 500:
            add("info", "timing",
                f"Mean start-time drift is {tm['mean_abs_drift_ms']:.0f}ms — "
                f"overlong segments are pushing later ones off their "
                f"original timestamps.")

    if ld.get("available") and ld.get("output_i") is not None:
        deviation = abs(ld["output_i"] - LOUDNESS_TARGET_I)
        if deviation > LOUDNESS_TOLERANCE:
            add("info", "loudness",
                f"Final track measured {ld['output_i']:.1f} LUFS vs the "
                f"{LOUDNESS_TARGET_I:.0f} LUFS target — playback platforms "
                f"will renormalize it.")

    return v


def rollup(asr: Dict[str, Any], translation: Dict[str, Any],
           tts: Dict[str, Any], timing: Dict[str, Any],
           loudness: Dict[str, Any]) -> Dict[str, Any]:
    """Combine scorer outputs into per-stage 0-100 scores + verdicts."""
    asr = asr or {}
    translation = translation or {}
    tts = tts or {}
    timing = timing or {}
    loudness = loudness or {}

    raw = {
        "asr": _score_asr(asr),
        "translation": _score_translation(translation),
        "tts": _score_tts(tts),
        "timing": _score_timing(timing),
        "loudness": _score_loudness(loudness),
    }
    stages = {
        name: {"available": score is not None,
               "score": None if score is None else round(score)}
        for name, score in raw.items()
    }

    avail = {n: s for n, s in raw.items() if s is not None}
    overall = None
    if avail:
        total_w = sum(STAGE_WEIGHTS[n] for n in avail)
        overall = round(sum(s * STAGE_WEIGHTS[n] for n, s in avail.items()) / total_w)

    return {
        "overall": overall,
        "stages": stages,
        "verdicts": _verdicts(asr, translation, tts, timing, loudness),
    }


def full_report(segments: List[dict],
                placements: Optional[List[dict]] = None,
                loudnorm: Optional[dict] = None) -> Dict[str, Any]:
    """Run every scorer + rollup. Convenience for the API route and CLI."""
    asr = asr_quality(segments)
    tr = translation_quality(segments)
    tts = tts_quality(segments)
    tm = timing_quality(placements or [])
    ld = loudness_quality(loudnorm)
    return {
        "asr": asr,
        "translation": tr,
        "tts": tts,
        "timing": tm,
        "loudness": ld,
        "rollup": rollup(asr, tr, tts, tm, ld),
    }
