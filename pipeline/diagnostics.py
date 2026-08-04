"""Environment diagnostics — turn "your setup is broken" into something the UI can show.

Two tiers, deliberately separated by cost:

  passive_checks()  no network, no model loads. Cheap enough to run on every
                    /api/system poll (5 s), so the banner is always current.
  deep_checks()     hits the network: validates the HF token, asks whether the
                    gated pyannote repos are actually accessible, pings the
                    translation backend. Only ever runs from the explicit
                    "Run checks" button — a local-first app should not phone
                    home on startup, and it must work offline.

The distinction matters for the motivating bug. `check_pyannote()` in
pipeline/models.py reports `{"ok": True, "has_token": True}` for a setup where
the token exists but the user never accepted the conditions for
`pyannote/segmentation-3.0` — a green light on a broken install. Only an
authenticated call to the Hub can tell those apart, and that call belongs
behind a button.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from .notices import merge_notices, notice

log = logging.getLogger("gochidubb.diagnostics")

# The repos pyannote actually needs. speaker-diarization-3.1 is a *pipeline*
# whose config pulls in segmentation-3.0; each is gated separately, and it is
# almost always the dependency people forget to accept.
PYANNOTE_REPOS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
)

# Last deep_checks() result, so /api/system can keep showing what the button
# found without re-probing the network every 5 seconds.
_last_deep: Optional[Dict[str, Any]] = None

# Notices discovered *while a job ran*, keyed by code. Some problems are only
# observable mid-pipeline — a model that downloads fine today and 503s during
# your dub leaves no trace any startup probe could find. Keeping them here is
# what lets the banner say "your last run was degraded" instead of the user
# finding out when the audio sounds wrong.
_runtime: Dict[str, Dict[str, Any]] = {}

# What a successful deep check is allowed to retire. Re-validating access is
# real evidence that a past download failure is no longer true; it says nothing
# about, say, a crash during inference, so those codes are not listed.
REVALIDATES = {
    "hf": ("pyannote.fallback_model", "pyannote.load_failed",
           "pyannote.no_token", "pyannote.bad_token", "pyannote.bad_token_format",
           "pyannote.gated_model", "hf.offline"),
    "translation": ("translation.unreachable", "translation.no_models"),
}


def record_runtime_notices(notices: List[Dict[str, Any]],
                           job_id: str = "") -> None:
    """Remember what a pipeline stage discovered about the environment."""
    for n in notices or ():
        if not isinstance(n, dict) or not n.get("code"):
            continue
        entry = dict(n)
        if job_id:
            entry["job_id"] = job_id
        entry["seen_at"] = time.time()
        _runtime[entry["code"]] = entry


def clear_runtime(codes: Optional[List[str]] = None) -> None:
    """Drop runtime notices — all of them, or a specific set."""
    if codes is None:
        _runtime.clear()
        return
    for c in codes:
        _runtime.pop(c, None)


def runtime_notices() -> List[Dict[str, Any]]:
    return list(_runtime.values())


def _check(cid: str, label: str, ok: bool, detail: str = "",
           severity: str = "error", hint: str = "") -> Dict[str, Any]:
    """One row in the subsystem checklist."""
    return {"id": cid, "label": label, "ok": bool(ok),
            "detail": detail, "severity": severity, "hint": hint}


# ═════════════════════════════════════════════════════════════════════
# Passive — no network
# ═════════════════════════════════════════════════════════════════════
def accelerator() -> Dict[str, Any]:
    """What torch will actually run on.

    pipeline/models.check_gpu() is CUDA-only, so on Apple Silicon it reports
    `ok: False` and the UI shows a permanent red "Torch broken!" badge on a
    machine where MPS works perfectly. A false alarm that never clears teaches
    users to ignore the warning area, which defeats the entire point of this
    module — so accelerator status comes from the MPS-aware probe in
    pipeline/metrics.py instead.
    """
    from .metrics import gpu_backend, gpu_snapshot

    backend = gpu_backend() or ""
    info: Dict[str, Any] = {"backend": backend, "ok": bool(backend)}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"] = bool(getattr(torch.backends, "mps", None)
                           and torch.backends.mps.is_available())
        if info["cuda"]:
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        elif info["mps"]:
            info["name"] = "Apple Silicon GPU (MPS)"
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"
        info["ok"] = False
    snap = gpu_snapshot()
    if snap:
        info["live"] = {k: round(v, 1) for k, v in snap.items()}
    return info


_TORCHCODEC_STATE: Optional[Dict[str, Any]] = None


def _installed_version(package: str) -> str:
    """Version of an installed distribution, without importing it.

    torchcodec raises on import when it can't bind FFmpeg, so asking the
    module for __version__ is exactly what we cannot do here.
    """
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return ""


def _torchcodec_ffmpeg_majors() -> List[int]:
    """FFmpeg majors this torchcodec build ships loaders for.

    It ships one `libtorchcodec_coreN` per supported FFmpeg major, and N is
    the major itself — 0.7.0 has core4–core7, 0.15.0 adds core8. Reading the
    filenames keeps the message right after an upgrade instead of baking in
    whatever was true when this was written. find_spec() locates the package
    without executing its __init__ (which is what raises).
    """
    try:
        import glob
        import importlib.util
        import os as _os
        import re as _re

        spec = importlib.util.find_spec("torchcodec")
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
        if not locations:
            return []
        majors = set()
        for pattern in ("*.dylib", "*.so*", "*.dll"):
            for path in glob.glob(_os.path.join(locations[0], pattern)):
                m = _re.search(r"libtorchcodec_core(\d+)", _os.path.basename(path))
                if m:
                    majors.add(int(m.group(1)))
        return sorted(majors)
    except Exception:
        return []


def _ffmpeg_major(version_text: str) -> str:
    """Leading major from an ffmpeg version string ("8.1.2_1" → "8")."""
    import re as _re
    m = _re.search(r"(\d+)", version_text or "")
    return m.group(1) if m else ""


def torchcodec_state(ffmpeg_version: str = "") -> Dict[str, Any]:
    """Can torchcodec load its FFmpeg bindings? Cached — it can't change.

    Two things can break it, and an M1 Mac with Homebrew hits both:
      1. FFmpeg major mismatch — a given torchcodec build ships loaders for
         a fixed set of FFmpeg majors (0.7.0 covers 4–7, i.e. libavcodec
         .58–.61). Homebrew's ffmpeg 8 installs libavcodec.62, which none
         of them bind to. torchcodec 0.15.0 added a core8 loader.
      2. torch ABI mismatch — a torchcodec built against an older libc10
         fails with "Symbol not found: __ZN3c1013MessageLogger…" even once
         FFmpeg resolves. Only a matching torchcodec release fixes that.

    `ok` is True even when it cannot load, because no code path in this
    pipeline decodes through torchcodec — pipeline/audio.load_audio_tensor
    and pipeline/vad._load_mono_16k go through libsndfile, and
    pipeline/diarizer preloads a waveform for pyannote. Reporting a red ✕
    for a component nothing uses is how a checklist becomes wallpaper (see
    accelerator() above). The claim is enforced, not assumed:
    tests/test_audio.py::TestTensorAudioIO and
    tests/test_vad.py::TestLoadMono16k make torchaudio.load/save raise, so
    reintroducing the dependency fails the suite rather than quietly making
    this row a lie.

    The detail lands in a single nowrap, ellipsis-truncated column
    (CheckList in static/index.html), so it has to stay short — roughly 70
    characters. `hint` is not rendered there at all (the template picks
    `detail || hint`) but is still served by /api/system, so the CLI and MCP
    clients carry the remedy.
    """
    global _TORCHCODEC_STATE
    if _TORCHCODEC_STATE is not None:
        return _TORCHCODEC_STATE

    state: Dict[str, Any] = {"ok": True, "detail": "", "hint": ""}
    try:
        import torchcodec
        state["detail"] = f"v{getattr(torchcodec, '__version__', '?')}"
    except ImportError:
        state["detail"] = "not installed · not required"
    except Exception:
        ver = _installed_version("torchcodec")
        majors = _torchcodec_ffmpeg_majors()
        system_major = _ffmpeg_major(ffmpeg_version)

        bits = ["unused"]
        if ver:
            bits.append(f"v{ver}")
        if majors and system_major and int(system_major) not in majors:
            span = (f"{majors[0]}–{majors[-1]}" if len(majors) > 1
                    else str(majors[0]))
            bits.append(f"binds FFmpeg {span}, system has {system_major}")
        else:
            # Either we couldn't read the loaders, or FFmpeg matches and the
            # breakage is the torch ABI. Don't guess at which.
            bits.append("cannot load its FFmpeg bindings")
        state["detail"] = " · ".join(bits)
        state["hint"] = ("Only needed for torchaudio.load(); a newer build "
                         "supports newer FFmpeg: pip install -U torchcodec")
    _TORCHCODEC_STATE = state
    return state


def passive_checks(status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Local-only environment assessment.

    `status` is an already-computed get_system_status() dict when the caller
    has one (the /api/system handler does), so we don't re-run the same
    subprocess probes twice per poll.
    """
    if status is None:
        from .models import get_system_status
        status = get_system_status()

    checks: List[Dict[str, Any]] = []
    notices: List[Dict[str, Any]] = []

    def sub(key: str) -> Dict[str, Any]:
        v = status.get(key)
        return v if isinstance(v, dict) else {}

    # ── Core, required for any dub ────────────────────────────────────
    py = sub("python")
    checks.append(_check("python", "Python", py.get("ok", False),
                         py.get("version", ""), hint=py.get("note", "")))

    ff = sub("ffmpeg")
    checks.append(_check("ffmpeg", "ffmpeg", ff.get("ok", False),
                         ff.get("version", "") or ff.get("path", "")))
    if not ff.get("ok"):
        notices.append(notice(
            code="ffmpeg.missing",
            severity="error",
            subsystem="system",
            title="ffmpeg is not installed",
            detail="Audio extraction, assembly and the final mux all shell out "
                   "to ffmpeg. No dub can complete without it.",
            remediation=[
                "macOS:  brew install ffmpeg",
                "Debian/Ubuntu:  sudo apt install ffmpeg",
                "Windows:  download from https://ffmpeg.org/download.html and add it to PATH",
            ],
            url="https://ffmpeg.org/download.html",
        ))

    yt = sub("yt_dlp")
    checks.append(_check("yt_dlp", "yt-dlp", yt.get("ok", False),
                         yt.get("path", "")))
    if not yt.get("ok"):
        notices.append(notice(
            code="yt_dlp.missing",
            severity="warn",
            subsystem="download",
            title="yt-dlp is not installed",
            detail="Uploaded files still work; dubbing from a URL does not.",
            remediation=["pip install -U yt-dlp"],
            url="https://github.com/yt-dlp/yt-dlp",
        ))

    wh = sub("whisper")
    checks.append(_check("whisper", "Whisper (transcription)", wh.get("ok", False),
                         wh.get("type", "") or wh.get("error", "")))
    if not wh.get("ok"):
        notices.append(notice(
            code="whisper.missing",
            severity="error",
            subsystem="transcribe",
            title="No Whisper backend is available",
            detail=wh.get("error", "") or "Transcription cannot run.",
            remediation=["pip install faster-whisper"],
        ))

    vox, edge = sub("voxcpm"), sub("edge_tts")
    tts_ok = vox.get("ok", False) or edge.get("ok", False)
    checks.append(_check("tts", "Text-to-speech", tts_ok,
                         "VoxCPM2" if vox.get("ok") else
                         ("edge-tts only" if edge.get("ok") else
                          vox.get("error", ""))))
    if not tts_ok:
        notices.append(notice(
            code="tts.missing",
            severity="error",
            subsystem="tts",
            title="No TTS engine is available",
            detail=vox.get("error", "") or "Neither VoxCPM2 nor edge-tts imported.",
            remediation=["pip install voxcpm", "or: pip install edge-tts"],
        ))
    elif not vox.get("ok"):
        notices.append(notice(
            code="voxcpm.missing",
            severity="warn",
            subsystem="tts",
            title="VoxCPM2 unavailable — voice cloning is off",
            detail=vox.get("error", "") or
                   "Dubs will fall back to edge-tts, which cannot clone the "
                   "source speaker's voice.",
            remediation=["pip install voxcpm"],
        ))

    # Background separation is opt-in per job ("keep background music"), so a
    # missing backend is a checklist row, not a notice — nagging everyone who
    # never uses the feature is how a warning area becomes wallpaper. The
    # notice is emitted at runtime by the extract stage, when it actually
    # matters. See pipeline/audio.separate_background.
    dem = sub("demucs")
    checks.append(_check("bg_separation", "Background separation",
                         dem.get("ok", False),
                         "demucs" if dem.get("ok") else
                         "not installed — background music is dropped when "
                         "'keep background' is on",
                         severity="warn",
                         hint=dem.get("hint", "pip install demucs")))

    # Always a pass: nothing here decodes through torchcodec (see
    # torchcodec_state). The row exists because torch, torchaudio and
    # pyannote print an alarming "Could not load libtorchcodec" on import,
    # and with no row at all the obvious conclusion is that the dub broke.
    tc = torchcodec_state(ff.get("version", ""))
    checks.append(_check("torchcodec", "torchcodec (optional decoder)", True,
                         tc["detail"], severity="warn", hint=tc["hint"]))

    # ── Diarization ───────────────────────────────────────────────────
    from .diarizer import effective_hf_token
    token = effective_hf_token()
    pya = sub("pyannote")
    checks.append(_check("pyannote", "Speaker diarization", pya.get("ok", False),
                         pya.get("error", ""), severity="warn"))
    checks.append(_check("hf_token", "Hugging Face token", bool(token),
                         "set" if token else "not set", severity="warn"))
    if pya.get("ok") and not token:
        notices.append(notice(
            code="pyannote.no_token",
            severity="error",
            subsystem="diarize",
            title="No Hugging Face token — diarization is disabled",
            detail="pyannote's models are gated. Without a token every speaker "
                   "is merged into a single voice.",
            remediation=[
                "Create a read token at https://hf.co/settings/tokens",
                "Accept the conditions at https://hf.co/pyannote/speaker-diarization-3.1"
                " and https://hf.co/pyannote/segmentation-3.0",
                "Put it in .env as HF_TOKEN=hf_… or paste it in System → Setup",
            ],
            url="https://hf.co/settings/tokens",
        ))
    elif not pya.get("ok"):
        notices.append(notice(
            code="pyannote.not_installed",
            severity="warn",
            subsystem="diarize",
            title="pyannote.audio is not installed",
            detail=pya.get("error", "") or
                   "Multi-speaker videos will be dubbed with one voice.",
            remediation=["pip install pyannote.audio"],
        ))
    elif token and not token.startswith("hf_"):
        # pyannote silently drops non-hf_ tokens (it assumes a pyannoteAI API
        # key), so the download then fails as if no token were set at all.
        notices.append(notice(
            code="pyannote.bad_token_format",
            severity="warn",
            subsystem="diarize",
            title="Hugging Face token does not look like an HF token",
            detail="pyannote ignores tokens that don't start with 'hf_', so "
                   "gated model downloads will fail as if unauthenticated.",
            remediation=["Create a read token at https://hf.co/settings/tokens"],
            url="https://hf.co/settings/tokens",
        ))

    # ── Accelerator ───────────────────────────────────────────────────
    acc = accelerator()
    checks.append(_check("gpu", "Accelerator", acc.get("ok", False),
                         acc.get("name", "") or acc.get("backend", "") or "CPU only",
                         severity="warn",
                         hint=acc.get("torch_error", "")))
    if acc.get("torch_error"):
        notices.append(notice(
            code="torch.broken",
            severity="error",
            subsystem="system",
            title="PyTorch failed to load",
            detail=acc["torch_error"],
            remediation=["Reinstall torch for your platform: "
                         "https://pytorch.org/get-started/locally/"],
            url="https://pytorch.org/get-started/locally/",
        ))
    elif not acc.get("ok"):
        notices.append(notice(
            code="gpu.cpu_only",
            severity="info",
            subsystem="system",
            title="No GPU detected — running on CPU",
            detail="Transcription and synthesis will be several times slower.",
            remediation=[],
        ))

    # ── QA roundtrip ──────────────────────────────────────────────────
    notices.extend(_tts_qa_notices(acc))

    return {"checks": checks, "notices": notices, "accelerator": acc}


def _tts_qa_notices(acc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Whisper-roundtrip QA is hardcoded to CUDA and fails open.

    `pipeline/tts_qa.py:44` defaults `device="cuda"` and its only caller (line
    130) never overrides it, so off NVIDIA the model never loads,
    `check_segment_quality()` scores everything as perfect, and the log still
    prints "QA: Whisper roundtrip enabled". Every segment reports qa=0.00
    whether it is good or not.

    Failing open is the right call — a broken QA check must not block a dub —
    but silently is not, because the scores then look like evidence.
    """
    if acc.get("cuda"):
        return []
    return [notice(
        code="tts_qa.device_unavailable",
        severity="warn",
        subsystem="tts",
        title="TTS quality checks are not running",
        detail="Whisper-roundtrip QA (used on cross-lingual dubs) asks for its "
               "model on CUDA, which this machine does not have, so it fails to "
               "load and every segment is scored as perfect. The qa=0.00 values "
               "in the logs mean 'not measured', not 'good'.",
        remediation=[
            "Read the qa= numbers as meaningless on this machine",
            "Judge output by listening, or run on an NVIDIA GPU",
        ],
    )]


# ═════════════════════════════════════════════════════════════════════
# Deep — network, on demand only
# ═════════════════════════════════════════════════════════════════════
def _probe_hf(token: str) -> Dict[str, Any]:
    """Validate the token and test access to each gated pyannote repo.

    Synchronous (huggingface_hub is a sync library); the caller runs it off
    the event loop.
    """
    out: Dict[str, Any] = {"token_valid": None, "user": "", "repos": {},
                           "offline": False}
    if not token:
        return out
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (GatedRepoError, HfHubHTTPError,
                                           RepositoryNotFoundError)
    except Exception as e:
        out["error"] = f"huggingface_hub unavailable: {e}"
        return out

    api = HfApi()
    try:
        who = api.whoami(token=token)
        out["token_valid"] = True
        out["user"] = who.get("name", "") if isinstance(who, dict) else ""
    except Exception as e:
        name = type(e).__name__
        # A network failure is not a bad token — saying so would send the user
        # off to regenerate a perfectly good credential.
        if "Connection" in name or "Timeout" in name or "DNS" in name:
            out["offline"] = True
            out["error"] = f"{name}: {e}"
            return out
        out["token_valid"] = False
        out["error"] = f"{name}: {e}"
        return out

    for repo in PYANNOTE_REPOS:
        try:
            api.model_info(repo, token=token)
            out["repos"][repo] = {"ok": True, "reason": ""}
        except GatedRepoError as e:
            out["repos"][repo] = {"ok": False, "reason": "gated",
                                  "error": str(e)[:300]}
        except RepositoryNotFoundError as e:
            # The Hub returns 404 for gated repos you have no access to, so
            # "not found" and "not accepted" are the same response.
            out["repos"][repo] = {"ok": False, "reason": "gated_or_missing",
                                  "error": str(e)[:300]}
        except HfHubHTTPError as e:
            out["repos"][repo] = {"ok": False, "reason": "http",
                                  "error": str(e)[:300]}
        except Exception as e:
            out["repos"][repo] = {"ok": False, "reason": "error",
                                  "error": f"{type(e).__name__}: {e}"[:300]}
    return out


async def deep_checks(token: str = "") -> Dict[str, Any]:
    """Run the network probes. Caches and returns the full report."""
    global _last_deep
    import asyncio

    from .diarizer import effective_hf_token
    from .models import get_system_status

    token = (token or effective_hf_token()).strip()
    status = get_system_status()
    base = passive_checks(status)
    checks: List[Dict[str, Any]] = list(base["checks"])
    # Only network findings are cached. Passive notices are recomputed on every
    # poll, so caching a copy here would resurrect problems the user has since
    # fixed — the banner would keep showing a missing ffmpeg they just installed.
    notices: List[Dict[str, Any]] = []

    # ── Hugging Face ──────────────────────────────────────────────────
    hf = await asyncio.to_thread(_probe_hf, token)
    if token:
        if hf.get("offline"):
            checks.append(_check("hf_access", "Hugging Face reachable", False,
                                 "no network", severity="warn"))
            notices.append(notice(
                code="hf.offline",
                severity="warn",
                subsystem="diarize",
                title="Could not reach huggingface.co",
                detail=hf.get("error", ""),
                remediation=["Check your internet connection and run the "
                             "checks again"],
            ))
        elif hf.get("token_valid") is False:
            checks.append(_check("hf_access", "Hugging Face token valid", False,
                                 hf.get("error", "")[:120]))
            notices.append(notice(
                code="pyannote.bad_token",
                severity="error",
                subsystem="diarize",
                title="Hugging Face rejected your token",
                detail=hf.get("error", ""),
                remediation=["Create a new read token at https://hf.co/settings/tokens",
                             "Update HF_TOKEN in .env, or paste it in System → Setup"],
                url="https://hf.co/settings/tokens",
            ))
        else:
            who = hf.get("user", "")
            checks.append(_check("hf_access", "Hugging Face token valid", True,
                                 f"authenticated as {who}" if who else "ok"))
            blocked = [r for r, v in hf.get("repos", {}).items() if not v.get("ok")]
            for repo in blocked:
                checks.append(_check(f"hf_repo:{repo}", f"Access to {repo}",
                                     False, hf["repos"][repo].get("reason", ""),
                                     severity="warn"))
            for repo, v in hf.get("repos", {}).items():
                if v.get("ok"):
                    checks.append(_check(f"hf_repo:{repo}", f"Access to {repo}",
                                         True, "accepted"))
            if not blocked:
                # The Hub says access is fine, so a past download failure was
                # transient — retire it rather than leaving a warning the user
                # cannot act on.
                clear_runtime(list(REVALIDATES["hf"]))
            if blocked:
                notices.append(notice(
                    code="pyannote.gated_model",
                    severity="warn",
                    subsystem="diarize",
                    title="Gated pyannote models are not accessible",
                    detail="Your token is valid but these repos have not been "
                           "unlocked, so diarization silently falls back to a "
                           "lower-quality model:\n  " + "\n  ".join(blocked),
                    remediation=[f"Visit https://hf.co/{r} and accept the user "
                                 f"conditions" for r in blocked]
                                + ["Then re-run these checks"],
                    url=f"https://hf.co/{blocked[0]}",
                ))

    # ── Translation backend ───────────────────────────────────────────
    from .translator import USE_LM_STUDIO, check_ollama
    backend = "LM Studio" if USE_LM_STUDIO else "Ollama"
    try:
        ok, models = await check_ollama()
    except Exception as e:
        ok, models = False, []
        log.debug(f"[diagnostics] backend probe failed: {e}")
    checks.append(_check("translation", f"{backend} reachable", ok,
                         f"{len(models)} model(s)" if ok else "not responding"))
    if ok and models:
        clear_runtime(list(REVALIDATES["translation"]))
    if not ok:
        notices.append(notice(
            code="translation.unreachable",
            severity="error",
            subsystem="translate",
            title=f"{backend} is not responding",
            detail="Translation will fail for every segment.",
            remediation=(
                [f"Start {backend} and load a model",
                 "Check LM_STUDIO_URL in .env points at the right host:port"]
                if USE_LM_STUDIO else
                ["Run: ollama serve", "Check OLLAMA_URL in .env"]
            ),
        ))
    elif not models:
        notices.append(notice(
            code="translation.no_models",
            severity="error",
            subsystem="translate",
            title=f"{backend} is running but has no models loaded",
            detail="Translation needs at least one loaded model.",
            remediation=(["Load a model in LM Studio's Developer tab"]
                         if USE_LM_STUDIO else ["ollama pull qwen2.5:14b"]),
        ))

    _last_deep = {
        "ran_at": time.time(),
        "checks": checks,
        # The report is what the Setup tab renders, so it must be the whole
        # picture — passive, whatever past runs observed (minus anything just
        # revalidated above), and the network findings.
        "notices": merge_notices(base["notices"], runtime_notices(), notices),
        "deep_notices": merge_notices(notices),              # cached layer
        "hf": {k: v for k, v in hf.items() if k != "error"},
        "backend": backend,
        "models": models,
    }
    return _last_deep


def last_deep() -> Optional[Dict[str, Any]]:
    """Most recent deep_checks() result, or None if the button was never used."""
    return _last_deep


def current_notices(status: Optional[Dict[str, Any]] = None,
                    extra: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Everything worth showing right now: passive + runtime + last deep + extras.

    Order is precedence: later layers win ties, so a confirmed finding (the Hub
    said this repo really is gated) supersedes the passive guess (a token
    exists, so probably fine) for the same code.
    """
    passive = passive_checks(status)["notices"]
    deep = (_last_deep or {}).get("deep_notices", [])
    return merge_notices(passive, runtime_notices(), deep, extra or [])
