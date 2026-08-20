"""Voice Activity Detection — strip non-speech regions before Whisper.

WhisperX already uses Silero VAD internally, but running it explicitly
upfront lets us:
  1. Remove long silence / music intros that cause Whisper hallucinations
  2. Report how much of the audio is actually speech (diagnostic)
  3. Optionally gate transcription on minimum speech ratio

Uses silero-vad (ONNX-based, 1 MB model, no GPU required).
Falls back gracefully if silero-vad is not installed.
"""
import logging
import os
import subprocess
import tempfile

log = logging.getLogger("gochidubb.vad")

# Minimum ratio of speech to total audio — below this we warn the user
SPEECH_RATIO_WARNING = 0.15

# Padding added around each speech segment (seconds) to avoid clipping
SEGMENT_PAD = 0.1


def _run_ffmpeg(cmd, desc="", timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} failed: {r.stderr[:300]}")
    return r


def _load_mono_16k(audio_path: str):
    """Load audio as the 1-D 16 kHz float tensor Silero VAD expects."""
    from .audio import load_audio_tensor

    wav, sr = load_audio_tensor(audio_path)
    if wav.dim() > 1:
        wav = wav.mean(dim=0)          # (channels, time) → mono
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def get_speech_timestamps(audio_path: str, threshold: float = 0.5) -> list[dict]:
    """Return Silero VAD timestamps as list of {start, end} dicts (seconds).

    Falls back to [{start: 0, end: duration}] if silero-vad not installed,
    so callers don't need to special-case the missing-dependency path.
    """
    try:
        import torch
        import torchaudio  # noqa — needed by silero-vad load_silero_vad

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            verbose=False,
        )
        get_ts = utils[0]  # get_speech_timestamps is utils[0]

        # Deliberately not silero's own utils[2] read_audio(): it calls
        # torchaudio.load(), which since torchaudio 2.9 goes through
        # torchcodec and dies with "Could not load libtorchcodec" on any
        # machine whose FFmpeg is newer than 7 (e.g. Homebrew ffmpeg 8 on
        # Apple Silicon). See pipeline/audio.py::load_audio_tensor.
        wav = _load_mono_16k(audio_path)
        timestamps = get_ts(wav, model, sampling_rate=16000, threshold=threshold)
        result = [
            {"start": t["start"] / 16000, "end": t["end"] / 16000}
            for t in timestamps
        ]
        log.info(f"VAD: {len(result)} speech segments detected")
        return result
    except ImportError:
        log.debug("silero-vad not installed — VAD skipped, using full audio")
        return []
    except Exception as e:
        log.warning(f"VAD failed ({e}) — using full audio")
        return []


def _get_duration_ffprobe(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def map_to_original(t: float, intervals) -> float:
    """Map a timestamp from the VAD-filtered timeline back to the original.

    ``apply_vad_filter`` does not merely trim silence — it concatenates the
    speech regions, so the audio Whisper sees is *shorter* than the source and
    every timestamp after the first removed gap is early by the amount of
    silence cut before it. Without this inversion a 60s video whose speech
    starts at 14s gets a dub that begins at 0s and ends 30s before the picture
    does.

    ``intervals`` is the list of (start, end) speech regions in the ORIGINAL
    timeline, in order. ``None`` means no filtering happened, so the filtered
    timeline *is* the original one and t passes through unchanged.
    """
    if not intervals:
        return t
    consumed = 0.0
    for start, end in intervals:
        span = end - start
        if t <= consumed + span:
            return start + (t - consumed)
        consumed += span
    # Past the end of the last speech region — clamp to it rather than
    # extrapolating into silence we know nothing about.
    return intervals[-1][1]


def remap_segments(segments, intervals):
    """Return segments with start/end (and any word times) on the original
    timeline. Mutates nothing; safe to call when intervals is None."""
    if not intervals:
        return segments
    for seg in segments:
        if seg.get("start") is not None:
            seg["start"] = map_to_original(seg["start"], intervals)
        if seg.get("end") is not None:
            seg["end"] = map_to_original(seg["end"], intervals)
        for w in seg.get("words") or []:
            if w.get("start") is not None:
                w["start"] = map_to_original(w["start"], intervals)
            if w.get("end") is not None:
                w["end"] = map_to_original(w["end"], intervals)
    return segments


def apply_vad_filter(audio_path: str, output_path: str,
                     threshold: float = 0.5) -> tuple[str, float, list | None]:
    """Extract only speech regions from audio_path into output_path.

    Returns (output_path, speech_ratio, intervals):
      - speech_ratio is the fraction of the original audio that contains
        detected speech (0.0-1.0).
      - intervals is the list of (start, end) speech regions in the ORIGINAL
        timeline when filtering happened, or None when the audio was passed
        through untouched.

    **The caller must map timestamps back** with ``remap_segments`` when
    intervals is not None. The filter concatenates speech regions, so
    everything downstream of it is on a compressed timeline.

    If silero-vad isn't installed or VAD finds no segments, copies the
    original audio unchanged and returns speech_ratio=1.0 (conservative).

    This helps Whisper in two ways:
      1. Removes long music intros that cause hallucinations like
         "Translated by XYZ" or repeated filler phrases.
      2. Reduces total audio length → faster transcription.
    """
    total_dur = _get_duration_ffprobe(audio_path)
    if total_dur <= 0:
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, 1.0, None

    timestamps = get_speech_timestamps(audio_path, threshold=threshold)
    if not timestamps:
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, 1.0, None

    # Add padding and clamp to audio bounds
    padded = []
    for t in timestamps:
        s = max(0.0, t["start"] - SEGMENT_PAD)
        e = min(total_dur, t["end"] + SEGMENT_PAD)
        padded.append((s, e))

    # Merge overlapping/adjacent segments
    merged = []
    for s, e in sorted(padded):
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])

    speech_seconds = sum(e - s for s, e in merged)
    speech_ratio = speech_seconds / total_dur if total_dur > 0 else 1.0

    if speech_ratio < SPEECH_RATIO_WARNING:
        log.warning(
            f"VAD: only {speech_ratio*100:.0f}% speech detected in audio. "
            f"Background music or silence may affect transcription quality."
        )

    if speech_ratio > 0.90:
        # Almost all speech — skip filtering, not worth the overhead
        log.info(
            f"VAD: {speech_ratio*100:.0f}% speech — audio is dense, skipping filter"
        )
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, speech_ratio, None

    # Build ffmpeg filter: select speech intervals + concatenate
    # atrim=start=X:end=Y, then concat all pieces
    pieces = []
    for i, (s, e) in enumerate(merged):
        pieces.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )

    n = len(merged)
    concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_complex = ";".join(pieces) + f";{concat_inputs}concat=n={n}:v=0:a=1[out]"

    try:
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            output_path,
        ], "VAD filter", timeout=300)
        log.info(
            f"VAD: filtered {total_dur:.0f}s → {speech_seconds:.0f}s "
            f"({speech_ratio*100:.0f}% speech, {n} segments)"
        )
        return output_path, speech_ratio, [tuple(m) for m in merged]
    except Exception as e:
        log.warning(f"VAD ffmpeg filter failed ({e}) — using full audio")
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, 1.0, None
