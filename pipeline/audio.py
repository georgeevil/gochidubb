"""Audio extraction and vocal/background separation.

Separation priority:
  1. Demucs (htdemucs_ft) — best quality, ships with torch, no extra install
  2. audio-separator — alternative if installed
  3. Silent background fallback — if neither is available
"""
import logging
import os
import shutil
import subprocess

from .ffmpeg_run import run_ffmpeg
from .notices import notice

log = logging.getLogger("gochidubb.audio")


class SeparationAborted(Exception):
    """Raised out of demucs' per-segment callback to stop separation early.

    Demucs is the one pipeline stage that does minutes of uninterruptible work
    *inside* our own process — no subprocess to kill, and Python cannot cancel
    a thread. On a 40-minute source it can run for 20+ minutes, during which a
    server shutdown could not complete (asyncio joins the executor thread with
    no timeout) and a user cancel had no effect until the next stage boundary.

    `apply_model` calls its `callback` once per ~8s segment and deliberately
    re-raises whatever it throws (see demucs/apply.py, the `BaseException`
    handler added "so that a KeyboardInterrupt raised from a callback" works),
    so raising from there is the supported way to abort.
    """


def _run(cmd, desc="", timeout=None):
    """Run ffmpeg with a soft timeout: extended while the encode reports
    progress, killed only once progress stalls (see pipeline/ffmpeg_run.py)."""
    log.info(f"[{desc}] {' '.join(str(c) for c in cmd[:8])}")
    r = run_ffmpeg(cmd, desc=desc, logger=log, timeout=timeout)
    log.debug(f"[{desc}] ok stderr={r.stderr[:200]!r}")
    return r


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract 16kHz mono WAV for Whisper."""
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ], "extract audio")
    return audio_path


def extract_audio_hq(video_path: str, audio_path: str) -> str:
    """Extract 44.1kHz stereo WAV for background separation."""
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        audio_path,
    ], "extract HQ audio")
    return audio_path


# ─────────────────────────────────────────────────────────────
# Tensor audio I/O — deliberately NOT torchaudio.load / torchaudio.save
# ─────────────────────────────────────────────────────────────
# Since torchaudio 2.9 those two calls dispatch to torchcodec, which
# dlopen()s FFmpeg's shared libraries at runtime. torchcodec only ships
# loaders for FFmpeg majors 4–7 (libavcodec.58/59/60/61), so on a machine
# with a newer FFmpeg — Homebrew's ffmpeg 8 on Apple Silicon installs
# libavcodec.62 — every call raises "Could not load libtorchcodec" and the
# demucs and VAD stages silently degrade to "no background separation" and
# "no VAD". The FFmpeg *binary* we shell out to everywhere else is
# unaffected; it is only torchcodec's dylib-level binding that is version
# locked.
#
# Everything we open here is a PCM WAV this pipeline just wrote with the
# ffmpeg CLI, so libsndfile (via soundfile) reads and writes it directly
# with no codec layer involved. pipeline/diarizer.py already preloads audio
# the same way to keep pyannote off torchcodec.

def load_audio_tensor(path: str):
    """Load audio as a float32 (channels, time) tensor + its sample rate."""
    import torch
    try:
        import numpy as np  # noqa: F401 — soundfile returns ndarrays
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (time, ch)
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception as e:
        # Only reachable for a container libsndfile can't open; the FFmpeg
        # version lock applies here too, so this may well fail — but it
        # fails with a clear error instead of never being tried.
        log.debug(f"soundfile could not read {path} ({e}); trying torchaudio")
        import torchaudio
        return torchaudio.load(path)


def save_audio_tensor(path: str, wav, sr: int) -> None:
    """Write a (channels, time) float tensor as a 16-bit PCM WAV."""
    import torch  # noqa: F401 — wav is a torch tensor
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    data = wav.detach().cpu().float().numpy().T  # → (time, channels)
    peak = float(abs(data).max()) if data.size else 0.0
    if peak > 1.0:
        # Demucs stems can overshoot ±1.0. Scaling the whole file keeps its
        # internal balance; hard-clipping to 16-bit would not.
        log.debug(f"Scaling {os.path.basename(path)} down from peak {peak:.2f}")
        data = data / peak
    import soundfile as sf
    sf.write(path, data, int(sr), subtype="PCM_16")


def _demucs_device(torch):
    """Pick the best device demucs can run on: cuda → mps → cpu."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _separate_demucs(audio_path: str, output_dir: str,
                     should_abort=None) -> tuple[str, str]:
    """Separate vocals/background using Demucs (htdemucs_ft model).

    Demucs ships with torch — no extra install required if torch is present.
    Uses the fine-tuned htdemucs_ft model (best for speech/music separation).

    `should_abort` is an optional zero-arg predicate polled once per ~8s
    segment; when it returns True, SeparationAborted is raised out of
    `apply_model` instead of grinding on for another 20 minutes.

    Returns (vocals_path, background_path) as 44.1kHz stereo WAVs.
    """
    import torch

    log.info("Separating vocals with Demucs (htdemucs_ft)...")

    # Only the imports may report "not installed". This used to wrap the
    # whole body, so an ImportError raised anywhere inside — including from
    # deep in demucs' own model loading — surfaced as "demucs not installed"
    # and sent people off to `pip install demucs` for a package that was
    # already there.
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        import torchaudio
    except ImportError as e:
        raise RuntimeError(f"demucs not installed — run: pip install demucs ({e})")

    def _callback(_state):
        if should_abort is not None and should_abort():
            raise SeparationAborted("background separation aborted")

    try:
        model = get_model("htdemucs_ft")
        model.eval()

        wav, sr = load_audio_tensor(audio_path)
        # Demucs expects (batch, channels, time) at its native sample rate.
        # functional.resample is pure tensor math — no torchcodec involved.
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # mono → stereo

        # `device` must be passed explicitly. apply_model defaults it to
        # `mix.device` and moves the model there itself, so moving the model
        # to the accelerator beforehand accomplished nothing — separation ran
        # on CPU even on machines with a working GPU, at roughly one minute of
        # compute per two minutes of audio.
        device = _demucs_device(torch)
        log.info(f"Demucs device: {device}")
        try:
            with torch.no_grad():
                sources = apply_model(model, wav.unsqueeze(0), device=device,
                                      callback=_callback)
        except SeparationAborted:
            raise
        except Exception as e:
            if device == "cpu":
                raise
            # MPS in particular still misses ops that demucs uses, and the
            # failure is immediate rather than 20 minutes in. Losing the
            # background entirely is a worse outcome than a slow CPU run.
            log.warning(f"Demucs failed on {device} ({type(e).__name__}: {e}) "
                        f"— retrying on CPU")
            with torch.no_grad():
                sources = apply_model(model, wav.unsqueeze(0), device="cpu",
                                      callback=_callback)

        stems = model.sources  # e.g. ["drums", "bass", "other", "vocals"]
        vocals_idx = stems.index("vocals")

        vocals_wav = sources[0, vocals_idx].cpu()
        # Background = everything except vocals (sum non-vocal stems)
        bg_parts = [sources[0, i].cpu() for i in range(len(stems)) if i != vocals_idx]
        bg_wav = sum(bg_parts)

        vocals_path = os.path.join(output_dir, "vocals.wav")
        bg_path = os.path.join(output_dir, "background.wav")

        save_audio_tensor(vocals_path, vocals_wav, model.samplerate)
        save_audio_tensor(bg_path, bg_wav, model.samplerate)

        log.info("Demucs separation complete")
        return vocals_path, bg_path

    except SeparationAborted:
        # Not a demucs failure — the caller asked us to stop. Must not be
        # rewritten into a RuntimeError, or separate_background would treat
        # it as "demucs broken" and fall through to the silent background.
        raise
    except Exception as e:
        # Installed but broken — say which, so the notice can offer a
        # remedy that matches the actual cause instead of a reinstall.
        raise RuntimeError(f"demucs failed: {type(e).__name__}: {e}") from e


def _separate_audio_separator(audio_path: str, output_dir: str) -> tuple[str, str]:
    """Fallback separation using audio-separator library."""
    vocals = os.path.join(output_dir, "vocals.wav")
    bg = os.path.join(output_dir, "background.wav")

    from audio_separator.separator import Separator
    sep = Separator(output_dir=output_dir)
    sep.load_model()
    outputs = sep.separate(audio_path)
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        for out in outputs:
            if "vocal" in out.lower():
                shutil.move(out, vocals)
            elif any(k in out.lower() for k in ("instrumental", "bg", "no_vocal")):
                shutil.move(out, bg)
        if not os.path.exists(vocals) and outputs:
            shutil.move(outputs[0], vocals)
        if not os.path.exists(bg) and len(outputs) > 1:
            shutil.move(outputs[1], bg)
    log.info("audio-separator separation complete")
    return vocals, bg


def _make_silent_bg(duration: float, output_dir: str) -> str:
    """Create a silent WAV of the given duration as background placeholder."""
    bg = os.path.join(output_dir, "background.wav")
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"anullsrc=r=44100:cl=stereo:d={duration}",
        "-acodec", "pcm_s16le", bg,
    ], "silent bg")
    return bg


def separate_background(audio_path: str, output_dir: str,
                        notices: list | None = None,
                        should_abort=None) -> tuple[str, str]:
    """Separate vocals from background audio.

    Tries in order:
      1. Demucs (htdemucs_ft) — best quality
      2. audio-separator — alternative
      3. Silent background fallback

    Returns (vocals_path, background_path).

    The silent fallback is deliberate — losing the music must not fail a dub —
    but it is also indistinguishable from success in the output, so it fills
    `notices` (see pipeline/notices.py) with what to install. The user asked
    for background preservation and would otherwise just get a quiet video.
    """
    vocals = os.path.join(output_dir, "vocals.wav")
    bg = os.path.join(output_dir, "background.wav")
    notices = notices if notices is not None else []
    reasons = []
    # Whether the backends are *missing* or *present but broken* decides
    # what the notice should tell the user to do. Conflating the two sent
    # someone to `pip install demucs` for an already-installed package
    # whose real problem was a torchcodec/FFmpeg mismatch.
    demucs_missing = False

    # Try Demucs first
    try:
        return _separate_demucs(audio_path, output_dir,
                                should_abort=should_abort)
    except SeparationAborted:
        # Shutdown or cancel — propagate. Falling through to the other
        # backends would just start a second long computation nobody wants.
        raise
    except RuntimeError as e:
        if "not installed" in str(e):
            log.info("Demucs not installed, trying audio-separator...")
            reasons.append("demucs: not installed")
            demucs_missing = True
        else:
            log.warning(f"Demucs failed: {e}, trying audio-separator...")
            reasons.append(f"demucs: {e}")
    except Exception as e:
        log.warning(f"Demucs error: {e}, trying audio-separator...")
        reasons.append(f"demucs: {type(e).__name__}: {e}")

    # Try audio-separator
    try:
        return _separate_audio_separator(audio_path, output_dir)
    except ImportError:
        log.warning(
            "Neither demucs nor audio-separator installed — background music "
            "will be dropped. Fix: pip install demucs "
            "(see System → Setup in the UI)"
        )
        reasons.append("audio-separator: not installed")
    except Exception as e:
        log.warning(f"audio-separator failed: {e}")
        reasons.append(f"audio-separator: {type(e).__name__}: {e}")

    notices.append(notice(
        code="audio.separation_unavailable",
        severity="warn",
        subsystem="extract",
        title="Background music could not be separated",
        detail="You asked to keep the background, but no separation backend "
               "worked, so the dub gets a silent background instead of the "
               "original music and effects.\n  " + "\n  ".join(reasons),
        remediation=(
            [
                "pip install demucs      (best quality, ~2 GB of weights on first use)",
                "or: pip install audio-separator",
                "Then re-run the Extract stage on this job",
            ]
            if demucs_missing else
            [
                "demucs IS installed — it failed at runtime, so reinstalling "
                "will not help. The reason is quoted above.",
                "Check System \u2192 Setup for a failing dependency (a torchcodec "
                "or FFmpeg mismatch is the usual cause).",
                "Then re-run the Extract stage on this job",
            ]
        ),
        url="https://github.com/adefossez/demucs",
    ))

    # Fallback: copy audio as vocals, silent background
    shutil.copy2(audio_path, vocals)
    dur = get_duration(audio_path)
    bg = _make_silent_bg(dur, output_dir)
    return vocals, bg


def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        # ffprobe failed to read the container — e.g. an image (webp/jpg/png)
        # or a corrupted/truncated file copied into source_video.mp4. This is
        # a top cause of "pipeline breaks silently": duration becomes 0.0,
        # audio extraction yields an empty WAV, and transcription finds no
        # speech. Surface it loudly so the user sees the real reason.
        log.warning(
            f"[duration] ffprobe could not read {path}: "
            f"exit={r.returncode} stderr={r.stderr.strip()[:200]!r}"
        )
        return 0.0
    try:
        dur = float(r.stdout.strip())
    except (ValueError, AttributeError):
        log.warning(
            f"[duration] ffprobe returned non-numeric duration for {path}: "
            f"{r.stdout.strip()!r}"
        )
        return 0.0
    if dur <= 0:
        log.warning(
            f"[duration] {path} reported duration={dur}s — likely a still "
            f"image or empty/unsupported container; downstream stages will fail"
        )
    log.debug(f"[duration] {path} -> {dur:.3f}s")
    return dur
