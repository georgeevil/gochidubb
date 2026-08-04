"""Speaker diarization with pyannote.audio.

Returns speaker turns [(start, end, speaker_id)] and per-speaker reference audio.
Compatible with pyannote.audio 3.x and 4.x (API differences auto-handled).

Every failure mode here is *soft*: diarization is optional, and losing it must
never cost the caller a transcription they already paid for. The price of that
is that a broken setup looks like a clean run, so each degradation also appends
a structured notice (see pipeline/notices.py) to a caller-supplied list. The
caller decides whether to show it; this module's job is to stop throwing the
information away.
"""
import logging
import os
import re

import numpy as np
import soundfile as sf

from .notices import notice

log = logging.getLogger("gochidubb.diarizer")

# Snapshot at import for backwards compatibility. Prefer effective_hf_token(),
# which also sees a token typed into the Settings UI after startup.
HF_TOKEN = os.getenv("HF_TOKEN", "")

DIARIZATION_MODELS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
)


def effective_hf_token() -> str:
    """The token to actually use: environment first, then persisted config.

    `HF_TOKEN` above is captured at import. `cfg.hf_token` (app/config.py) is
    written by the Settings UI but was read by nothing, so a token entered in
    the browser silently did nothing. Import cfg lazily — pipeline/ does not
    otherwise depend on app/, and this must not become the reason the module
    fails to import.
    """
    # Default to the import-time snapshot rather than "", so an env var that
    # was never set still finds a token loaded from .env at startup.
    token = os.environ.get("HF_TOKEN", HF_TOKEN).strip()
    if token:
        return token
    try:
        from app.config import cfg
        return (cfg.hf_token or "").strip()
    except Exception:
        return ""


def _hub_repo_from_error(err: BaseException) -> str:
    """Which HF repo a download failure was really about.

    pyannote's own message names it (`Could not download Model from
    pyannote/segmentation-3.0`), but that goes to stdout via print(). The
    raised HfHubHTTPError carries the same repo inside its URL, which is the
    only machine-readable copy — and it is usually a *dependency* of the
    pipeline the user asked for, not the pipeline itself, so it cannot be
    hardcoded.
    """
    text = f"{type(err).__name__}: {err}"
    m = re.search(
        r"(?:huggingface\.co|hf\.co)/(?:api/models/)?"
        r"([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)",
        text,
    )
    return m.group(1) if m else ""

REFERENCE_TARGET = 30.0   # aim for ~30 seconds of clean speech per speaker
REFERENCE_MIN = 6.0
MIN_CHUNK = 1.5
MAX_CHUNK = 12.0


def _load_pipeline(token: str, notices: list | None = None):
    """Load pyannote diarization pipeline. Uses the new token= kwarg
    (pyannote.audio >= 3.0). `use_auth_token` was removed in 4.x.

    Candidates are tried in quality order and the first one that loads wins.
    Crucially, a *partial* failure — the preferred model is gated but a
    fallback loads — is reported rather than swallowed: that is the case where
    the run appears completely normal while quietly using a weaker model.
    """
    notices = notices if notices is not None else []
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("pyannote.audio not installed - diarization disabled")
        notices.append(notice(
            code="pyannote.not_installed",
            severity="error",
            subsystem="diarize",
            title="Speaker diarization is unavailable",
            detail="pyannote.audio is not installed, so every segment is "
                   "attributed to a single speaker.",
            remediation=["pip install pyannote.audio",
                         "Restart GoChiDUBB"],
            url="https://github.com/pyannote/pyannote-audio",
        ))
        return None

    if not token:
        log.warning(
            "HF_TOKEN is empty. Diarization needs a Hugging Face token. "
            "Set it in .env as HF_TOKEN=hf_xxx and install python-dotenv."
        )
        notices.append(notice(
            code="pyannote.no_token",
            severity="error",
            subsystem="diarize",
            title="No Hugging Face token — diarization is disabled",
            detail="pyannote's models are gated, so downloading them needs a "
                   "token. Without one every speaker is merged into one voice.",
            remediation=[
                "Create a read token at https://hf.co/settings/tokens",
                "Accept the conditions at https://hf.co/pyannote/speaker-diarization-3.1",
                "Put it in .env as HF_TOKEN=hf_… (or paste it in System → Setup)",
            ],
            url="https://hf.co/settings/tokens",
        ))
        return None

    errors = []
    for model in DIARIZATION_MODELS:
        try:
            pipe = Pipeline.from_pretrained(model, token=token)
            if pipe is None:
                # from_pretrained returns None instead of raising when the
                # pipeline config itself is unreadable.
                errors.append({"model": model, "repo": model,
                               "error": "from_pretrained returned None"})
                continue
            log.info(f"Diarization pipeline loaded: {model}")
            if errors:
                _note_fallback(notices, used=model, errors=errors)
            return pipe
        except Exception as e:
            errors.append({"model": model,
                           "repo": _hub_repo_from_error(e) or model,
                           "error": f"{type(e).__name__}: {e}"})
            continue

    joined = "\n".join(f"  {e['model']}: {e['error']}" for e in errors)
    log.warning(
        "Could not load any diarization pipeline. Check that:\n"
        "  1) Your HF_TOKEN is valid: https://huggingface.co/settings/tokens\n"
        "  2) You accepted the model terms at:\n"
        "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "     - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        "Attempt errors:\n" + joined
    )
    gated = next((e["repo"] for e in errors if e.get("repo")), "")
    notices.append(notice(
        code="pyannote.load_failed",
        severity="error",
        subsystem="diarize",
        title="No diarization model could be loaded",
        detail=f"All candidates failed, so all speech is treated as one "
               f"speaker.\n{joined}",
        remediation=[
            "Run System → Setup → Run checks — it distinguishes a bad token "
            "from unaccepted conditions from a network failure",
            f"If access is the problem, accept the conditions at https://hf.co/{gated}"
            if gated else "Accept the model conditions on huggingface.co",
            "Or retry the Diarize stage with 'Skip diarization' to continue "
            "the dub with a single voice",
        ],
        url=f"https://hf.co/{gated}" if gated else "https://hf.co/settings/tokens",
    ))
    return None


def _note_fallback(notices: list, used: str, errors: list) -> None:
    """Record that diarization ran, but not on the model we wanted.

    `pyannote/speaker-diarization-3.1` pulls in `pyannote/segmentation-3.0`,
    and if either download fails, `community-1` loads happily instead — same
    return value, quietly worse speaker separation.

    Deliberately neutral about *why*. pyannote prints a "the repository is
    private or gated" banner for **every** HfHubHTTPError
    (`pyannote/audio/utils/hf_hub.py:91-104`), including a timeout, a 503 or a
    rate limit — it is a guess printed as a diagnosis, and repeating it would
    send users off to accept conditions they already accepted. We report what
    is certain (the preferred model was unavailable, here is the raw error) and
    point at the deep check, which asks the Hub directly and can tell the two
    apart.
    """
    repo = next((e["repo"] for e in errors if e.get("repo")), "")
    failed = ", ".join(e["model"] for e in errors)
    log.warning(
        f"[diarize] Fell back to {used} — {failed} unavailable. "
        f"Run System → Setup → Run checks to see whether this is a permissions "
        f"problem or a transient download failure."
    )
    notices.append(notice(
        code="pyannote.fallback_model",
        severity="warn",
        subsystem="diarize",
        title=f"Diarization used the fallback model {used.split('/')[-1]}",
        detail=(f"{failed} could not be downloaded, so speakers were "
                f"identified with {used} instead. Speaker separation may be "
                f"less accurate.\n"
                + "\n".join(f"  {e['model']}: {e['error']}" for e in errors)),
        remediation=[
            "Run System → Setup → Run checks — it asks Hugging Face whether "
            "this is a permissions problem or just a failed download",
            f"If access is fine, re-run the Diarize stage; the download is "
            f"usually transient",
        ],
        url=f"https://hf.co/{repo}" if repo else f"https://hf.co/{used}",
    ))


def diarize_speakers(audio_path: str,
                     min_speakers: int | None = None,
                     max_speakers: int | None = None,
                     hf_token: str = "",
                     notices: list | None = None) -> list[tuple]:
    """Run diarization. Returns list of (start, end, speaker) tuples, or [] on failure.

    `notices` is an optional list to append setup notices to. Every failure
    path here returns [] rather than raising, so this list is the only way a
    caller can distinguish "one speaker" from "diarization is broken".
    """
    token = (hf_token or effective_hf_token()).strip()
    notices = notices if notices is not None else []
    if not token:
        log.warning("HF_TOKEN not set - diarization skipped (set it in .env)")
        _load_pipeline("", notices)   # emits pyannote.no_token
        return []

    pipe = _load_pipeline(token, notices)
    if pipe is None:
        return []

    # Move to GPU if available
    try:
        import torch as _t
        if _t.cuda.is_available():
            pipe.to(_t.device("cuda"))
    except Exception:
        pass

    try:
        import torch
        # Try preloading audio as a tensor to bypass broken torchcodec on Windows.
        # pyannote accepts {"waveform": tensor, "sample_rate": int} per
        # https://huggingface.co/pyannote/speaker-diarization-3.1
        try:
            import soundfile as sf
            import numpy as np
            wav, sr = sf.read(audio_path, always_2d=False)
            if wav.ndim == 1:
                wav = wav[np.newaxis, :]  # (1, time)
            elif wav.ndim == 2:
                wav = wav.T  # -> (channels, time)
            waveform = torch.from_numpy(wav.astype(np.float32))
            audio_input = {"waveform": waveform, "sample_rate": sr}
        except Exception as e:
            log.debug(f"Could not preload audio via soundfile: {e}; using path")
            audio_input = audio_path

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization = pipe(audio_input, **kwargs)

        # pyannote 4.x: returns DiarizeOutput (has .speaker_diarization attribute
        # which is the classic Annotation). 3.x returned Annotation directly.
        if hasattr(diarization, "speaker_diarization"):
            ann = diarization.speaker_diarization
        elif hasattr(diarization, "itertracks"):
            ann = diarization
        else:
            log.warning(f"Unknown diarization output type: {type(diarization)}")
            notices.append(notice(
                code="pyannote.unknown_output",
                severity="error",
                subsystem="diarize",
                title="Diarization returned an unrecognized result",
                detail=f"pyannote returned {type(diarization).__name__}, which "
                       f"this version of GoChiDUBB cannot read. All speech was "
                       f"attributed to one speaker.",
                remediation=["pip install -U pyannote.audio",
                             "Report the type above if it persists"],
            ))
            return []

        turns = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in ann.itertracks(yield_label=True)
        ]
        log.info(f"Diarization: {len(turns)} turns, "
                 f"{len(set(s for _,_,s in turns))} speakers")
        log.debug(
            f"[diarize] turns: {[(round(t[0],2), round(t[1],2), t[2]) for t in turns[:10]]}"
            f"{' ...' if len(turns) > 10 else ''}"
        )
        return turns
    except Exception as e:
        log.warning(f"Diarization inference failed: {e}")
        notices.append(notice(
            code="pyannote.inference_failed",
            severity="error",
            subsystem="diarize",
            title="Diarization crashed while analysing the audio",
            detail=f"The model loaded but failed on this file, so all speech "
                   f"was attributed to one speaker.\n{type(e).__name__}: {e}",
            remediation=[
                "Retry the Diarize stage",
                "If it repeats, retry with 'Skip diarization' to continue the dub",
            ],
        ))
        return []


def assign_speakers_to_segments(segments: list[dict], speaker_turns: list[tuple]) -> list[dict]:
    """Attach a 'speaker' field to each transcript segment by max temporal overlap."""
    if not speaker_turns:
        for s in segments:
            s.setdefault("speaker", "SPEAKER_00")
        return segments

    for seg in segments:
        s, e = seg["start"], seg["end"]
        best_overlap = 0.0
        best_spk = "SPEAKER_00"
        for ts, te, spk in speaker_turns:
            ov = max(0.0, min(e, te) - max(s, ts))
            if ov > best_overlap:
                best_overlap = ov
                best_spk = spk
        seg["speaker"] = best_spk
    return segments


def _total_duration(speaker_turns, spk) -> float:
    return sum(te - ts for ts, te, s in speaker_turns if s == spk)


def extract_speaker_audio(audio_path: str,
                          speaker_turns: list[tuple],
                          out_dir: str,
                          main_only: bool = False,
                          max_speakers: int | None = None) -> dict:
    """Build per-speaker reference WAVs by concatenating the cleanest chunks up to ~30s.

    - main_only=True  : keep only the speaker with the most total speech.
    - max_speakers=N  : keep only the top N speakers by total speech duration.

    Returns {speaker_id: path_to_wav}.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not speaker_turns:
        return {}

    # Group by speaker
    per_speaker: dict = {}
    for ts, te, spk in speaker_turns:
        per_speaker.setdefault(spk, []).append((ts, te))

    # Sort speakers by total duration desc
    speakers_sorted = sorted(per_speaker.keys(),
                             key=lambda k: -_total_duration(speaker_turns, k))
    if main_only and speakers_sorted:
        speakers_sorted = speakers_sorted[:1]
    elif max_speakers is not None and max_speakers > 0:
        speakers_sorted = speakers_sorted[:max_speakers]

    try:
        wav, sr = sf.read(audio_path)
    except Exception as e:
        log.warning(f"Could not read audio for references: {e}")
        return {}
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    refs = {}
    for spk in speakers_sorted:
        turns = sorted(per_speaker[spk], key=lambda x: -(x[1] - x[0]))
        selected = []
        total = 0.0
        seen = set()

        # Pass 1: durations in the sweet spot (1.5-12s, cleanest)
        for ts, te in turns:
            d = te - ts
            if MIN_CHUNK <= d <= MAX_CHUNK:
                selected.append((ts, te))
                seen.add((ts, te))
                total += d
                if total >= REFERENCE_TARGET:
                    break

        # Pass 2: short turns fill gaps
        if total < REFERENCE_MIN:
            for ts, te in turns:
                d = te - ts
                if 0.4 <= d < MIN_CHUNK and (ts, te) not in seen:
                    selected.append((ts, te))
                    seen.add((ts, te))
                    total += d
                    if total >= REFERENCE_TARGET:
                        break

        # Pass 3: long turns (capped) if still under-target
        if total < REFERENCE_MIN:
            for ts, te in turns:
                d = te - ts
                if d > MAX_CHUNK and (ts, te) not in seen:
                    capped_end = ts + min(d, 15.0)
                    selected.append((ts, capped_end))
                    seen.add((ts, te))
                    total += capped_end - ts
                    if total >= REFERENCE_TARGET:
                        break

        if not selected:
            log.warning(f"Speaker {spk}: no usable speech found")
            continue

        # Chronological order for natural-sounding reference
        selected.sort(key=lambda x: x[0])
        pieces = []
        for ts, te in selected:
            a = int(ts * sr)
            b = int(te * sr)
            if b > a and b <= len(wav):
                pieces.append(wav[a:b])

        if not pieces:
            continue

        merged = np.concatenate(pieces).astype(np.float32)
        # Peak-normalize so reference isn't clipped or too quiet
        peak = float(np.abs(merged).max())
        if peak > 0.01:
            merged = merged * (0.9 / peak)

        ref_path = os.path.join(out_dir, f"ref_{spk}.wav")
        sf.write(ref_path, merged, sr)
        refs[spk] = ref_path
        log.info(f"Speaker {spk}: {total:.1f}s reference -> {os.path.basename(ref_path)}")

    return refs


def extract_fallback_reference(audio_path: str,
                               segments: list[dict],
                               out_path: str,
                               duration: float = 30.0) -> str | None:
    """When diarization failed: build a single-speaker reference from the longest clean
    transcript segments. Use this for the main (single) speaker in the video.
    """
    if not segments:
        return None

    try:
        wav, sr = sf.read(audio_path)
    except Exception as e:
        log.warning(f"Could not read audio for fallback ref: {e}")
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Prefer medium-length segments (typical clear speech)
    sorted_segs = sorted(
        segments,
        key=lambda s: -(s.get("end", 0) - s.get("start", 0)),
    )

    selected = []
    total = 0.0
    # Pass 1: sweet spot
    for seg in sorted_segs:
        d = seg.get("end", 0) - seg.get("start", 0)
        text = (seg.get("text") or "").strip()
        if not text or d < MIN_CHUNK or d > MAX_CHUNK:
            continue
        selected.append((seg["start"], seg["end"]))
        total += d
        if total >= duration:
            break
    # Pass 2: any reasonable segment
    if total < REFERENCE_MIN:
        for seg in sorted_segs:
            d = seg.get("end", 0) - seg.get("start", 0)
            text = (seg.get("text") or "").strip()
            if not text or d < 0.5:
                continue
            pair = (seg["start"], seg["end"])
            if pair in selected:
                continue
            selected.append(pair)
            total += d
            if total >= duration:
                break

    if not selected:
        # Last resort: first `duration` seconds of audio
        nfr = min(len(wav), int(duration * sr))
        sf.write(out_path, wav[:nfr], sr)
        log.warning(f"Fallback reference: using first {nfr/sr:.1f}s of source audio")
        return out_path

    selected.sort(key=lambda x: x[0])
    pieces = []
    for ts, te in selected:
        a = int(ts * sr)
        b = int(te * sr)
        if b > a and b <= len(wav):
            pieces.append(wav[a:b])

    if not pieces:
        return None

    merged = np.concatenate(pieces).astype(np.float32)
    peak = float(np.abs(merged).max())
    if peak > 0.01:
        merged = merged * (0.9 / peak)
    sf.write(out_path, merged, sr)
    log.info(f"Fallback single-speaker reference: {total:.1f}s of clean speech")
    return out_path
