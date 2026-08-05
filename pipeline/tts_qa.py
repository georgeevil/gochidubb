"""
TTS Quality Assurance — post-synthesis validation.
====================================================

VoxCPM (like most neural TTS) is stochastic — any given run has a small but
non-zero chance of producing garbled output ("eaten words"). The model has
a built-in `retry_badcase` mechanism, but it uses internal heuristics that
don't catch all failure modes, especially for cross-lingual synthesis where
the model drifts toward source-language phonetics.

This module adds a second layer: ROUNDTRIP VALIDATION.

For each generated segment:
  1. Run a lightweight ASR (faster-whisper "small" or "base") on the output WAV
  2. Compare ASR transcript to the target text we asked TTS to speak
  3. If character-error-rate (CER) is too high OR detected language doesn't
     match target language, flag the segment as bad
  4. Caller re-runs TTS with a different seed, up to N times

Why CER instead of WER:
  - Short segments ("этом спорте.") have few words — one wrong word = 50% WER.
    CER is smoother for short text.
  - Russian has agglutinative morphology — "спорт" vs "спорте" is a meaningful
    difference at word level but small CER difference; CER tracks intelligibility
    better than strict word matching.

Caching:
  - Whisper model is loaded lazily on first use; subsequent calls reuse it.
  - We use the smallest possible Whisper model (base) since we're checking
    intelligibility, not exact transcription. Base is ~7x faster than large-v3.
"""
import logging
import os
import re
import time
from typing import Optional, Tuple

log = logging.getLogger("gochidubb.tts_qa")

# Single source of truth for the pass/fail gate. Overridable at runtime
# via the GOCHIDUBB_QA_THRESHOLD env var (read on every is_acceptable call).
QA_THRESHOLD = 0.4

_whisper_model = None
_whisper_model_size = None


def _resolve_device(pref: str = "auto") -> str:
    """Resolve a device preference to a concrete faster-whisper device.

    "auto" → "cuda" when torch reports a usable CUDA device, else "cpu"
    (Apple Silicon / AMD / CPU-only boxes run whisper on CPU with int8 —
    slower but correct; the old hardcoded "cuda" default made the model
    fail to load off NVIDIA and QA silently scored everything as perfect).
    An explicit preference ("cuda"/"cpu") is passed through untouched.
    """
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        # torch missing/broken — CPU is the only safe assumption
        pass
    return "cpu"


def _get_whisper(device: str = "auto", model_size: str = "base"):
    """Load (or return cached) faster-whisper model for QA checks.
    We default to 'base' — small enough to be fast, large enough to
    transcribe intelligible speech reliably."""
    global _whisper_model, _whisper_model_size
    if _whisper_model is not None and _whisper_model_size == model_size:
        return _whisper_model
    device = _resolve_device(device)
    try:
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        log.info(f"[qa] Loading faster-whisper {model_size} on {device}...")
        t0 = time.time()
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute)
        _whisper_model_size = model_size
        log.info(f"[qa] Whisper QA model loaded in {time.time()-t0:.1f}s")
        return _whisper_model
    except Exception as e:
        log.warning(f"[qa] Could not load Whisper QA model: {e}")
        return None


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used before
    computing edit distance so minor formatting differences don't count."""
    # Keep letters (incl. Cyrillic) and digits, drop everything else
    s = re.sub(r"[^\w\s]", " ", s.lower(), flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    # Strip voice-design prefix if present, e.g. "(deep male voice)..."
    s = re.sub(r"^\([^)]*\)\s*", "", s)
    return s


def _cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate via Levenshtein distance / len(reference).
    Returns 0.0 for perfect match, 1.0+ for wildly different strings."""
    if not reference:
        return 0.0 if not hypothesis else 1.0

    h, r = hypothesis, reference
    # Standard DP edit distance
    m, n = len(h), len(r)
    if m == 0:
        return 1.0
    if n == 0:
        return 1.0

    # Use O(min(m,n)) memory
    if m > n:
        h, r = r, h
        m, n = n, m
    prev = list(range(m + 1))
    for j in range(1, n + 1):
        curr = [j] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if h[i-1] == r[j-1] else 1
            curr[i] = min(
                prev[i] + 1,       # deletion
                curr[i-1] + 1,     # insertion
                prev[i-1] + cost,  # substitution
            )
        prev = curr
    distance = prev[m]
    return distance / max(n, 1)


def check_segment_quality(
    audio_path: str,
    target_text: str,
    target_lang: str = "ru",
    whisper_size: str = "base",
) -> Tuple[Optional[float], str, dict]:
    """Run ASR on a TTS output and compare to what we asked TTS to speak.

    Returns:
        (score, transcript, diagnostics)
        - score: 0.0 (perfect) to 1.0+ (wildly wrong), or None when the
          check could not run at all (whisper unavailable / ASR crashed).
          The gate is is_acceptable(): score <= QA_THRESHOLD (0.4 by
          default) passes. For reporting, > ~0.6 is "bad — regenerate"
          territory and scores just above the gate are "suspect".
        - transcript: what Whisper heard
        - diagnostics: dict with 'cer', 'detected_lang', 'lang_match',
          'empty', 'measured' keys for logging/UI display.
          diag['measured'] is True only when ASR actually ran on the
          audio; False means "not measured" — callers must not treat the
          None score as evidence of quality (good or bad).
    """
    diag = {"cer": 1.0, "detected_lang": "", "lang_match": False, "empty": False,
            "measured": False}
    if not os.path.exists(audio_path):
        # A missing output file is a REAL failure, not an unmeasured one.
        diag["error"] = "audio file missing"
        diag["measured"] = True
        return 1.0, "", diag

    model = _get_whisper(model_size=whisper_size)
    if model is None:
        # Whisper unavailable — QA cannot run. Return an honest
        # "not measured" (score=None) instead of a fake perfect 0.0 so
        # downstream never mistakes a skipped check for a passing one.
        diag["error"] = "whisper unavailable"
        return None, "", diag

    try:
        # Force language for better small-model accuracy on the target tongue.
        # Whisper's "auto" on short noisy clips sometimes mis-IDs to English.
        segments, info = model.transcribe(
            audio_path,
            language=target_lang,
            beam_size=1,  # fast path; we don't need hypotheses here
            vad_filter=False,  # already trimmed upstream
            no_speech_threshold=0.6,
        )
        transcript = " ".join(s.text.strip() for s in segments).strip()
        diag["detected_lang"] = info.language
        diag["lang_match"] = info.language == target_lang
        diag["empty"] = not transcript
        diag["measured"] = True
    except Exception as e:
        log.warning(f"[qa] Whisper transcribe failed on {audio_path}: {e}")
        diag["error"] = str(e)
        return None, "", diag  # ASR crashed — not measured

    if not transcript:
        # TTS produced silence or unrecognizable output — very bad.
        # ASR DID run here, so this is a real measurement.
        return 1.0, "", diag

    # Language mismatch is a fatal defect: if we asked for Russian and
    # Whisper heard English, the TTS definitely went off the rails.
    # (Don't penalize for "lv"/"uk" etc. misdetections — those are often
    # close enough to target that TTS still sounds intelligible.)
    lang_penalty = 0.0
    if info.language_probability > 0.8 and info.language != target_lang:
        # Only major language mismatches (en/zh) are catastrophic;
        # related Slavic detections on Russian are usually fine.
        major_mismatches = {"en", "zh", "ja", "ko", "ar"}
        if info.language in major_mismatches and target_lang not in major_mismatches:
            lang_penalty = 0.5
            log.info(f"[qa] Language mismatch: asked {target_lang}, "
                     f"Whisper heard {info.language} "
                     f"(prob={info.language_probability:.2f})")

    # Compute CER against normalized target
    target_clean = _normalize_text(target_text)
    hyp_clean = _normalize_text(transcript)
    cer = _cer(hyp_clean, target_clean)
    diag["cer"] = round(cer, 3)

    score = cer + lang_penalty
    return score, transcript, diag


def is_acceptable(score: Optional[float], threshold: float = QA_THRESHOLD) -> bool:
    """Default quality gate. score <= QA_THRESHOLD (0.4) = OK.

    score=None means "not measured" (whisper unavailable / ASR crashed).
    That is accepted — pass-without-claim: a segment that CANNOT be
    measured must not trigger regeneration retries. Callers that need to
    distinguish "passed" from "unmeasured" must check diag['measured'].

    Tunable via GOCHIDUBB_QA_THRESHOLD env var if user wants stricter/looser."""
    if score is None:
        return True
    try:
        threshold = float(os.environ.get("GOCHIDUBB_QA_THRESHOLD", threshold))
    except ValueError:
        pass
    return score <= threshold
