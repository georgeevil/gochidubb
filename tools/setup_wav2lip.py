#!/usr/bin/env python3
"""Set up Wav2Lip for GoChiDUBB's optional lip-sync step.

Usage:
    python tools/setup_wav2lip.py                    # clone, build venv, patch
    python tools/setup_wav2lip.py --show-patches     # print the patches, change nothing
    python tools/setup_wav2lip.py --verify-only      # check an existing install
    python tools/setup_wav2lip.py --download-checkpoint   # also fetch the ~400 MB GAN weights

Why a script rather than four lines in the README:

Upstream Wav2Lip (Rudrabha, 2020) pins ``librosa==0.7.0``, ``numpy==1.17.1``,
``numba==0.48`` and ``torch==1.1.0``. Those cannot share a venv with what
VoxCPM and faster-whisper need, and numpy 1.17 does not build on Python 3.11
at all — so the old advice of "pip install its requirements into your venv"
breaks the dubbing pipeline in exchange for a lip-sync step that still will
not run. Wav2Lip therefore gets its own interpreter, and the server is pointed
at it through ``GOCHIDUBB_WAV2LIP_PYTHON``.

Two source patches are needed on top of that, both small and both applied
idempotently here:

1. ``inference.py`` picks between CUDA and CPU only. On Apple Silicon that
   silently means CPU. The patch adds an MPS branch.
2. ``audio.py`` calls ``librosa.filters.mel`` with positional arguments, which
   librosa made keyword-only in 0.10. This is the only librosa incompatibility
   on the inference path — ``load_wav`` and ``stft`` already pass keywords, and
   the removed ``librosa.output.write_wav`` sits in ``save_wav``, which
   inference never calls.

Not patched, deliberately: ``face_detection/utils.py`` uses ``np.int`` (removed
in numpy 1.24), but only inside a landmark helper that the s3fd detection path
never calls, so it stays dormant. Left alone rather than carrying an edit that
is never exercised.

The face-detection weights (s3fd) download themselves on first run via
``torch.utils.model_zoo``. Only the ~400 MB generator checkpoint is manual, and
only because a download that large should be something you opt into.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_DIR = PROJECT_ROOT / "Wav2Lip"
REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"

# Mirror of the GAN checkpoint. The upstream README links to Google Drive,
# which needs a browser and has rotted more than once; this file is
# byte-identical to two other public mirrors (xet hash 40dfad7c…).
CHECKPOINT_URL = (
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
)

# Modern replacements for the 2020 pins. Deliberately unpinned: the point is to
# ride the same wheels the rest of the machine already has, and the two patches
# below are what make that safe.
DEPS = ["torch", "numpy", "scipy", "opencv-python", "librosa", "tqdm"]

# (file, old, new, why). Applied with an exact-occurrence check so a change
# upstream surfaces as a clear failure instead of a silent no-op.
PATCHES = [
    (
        "inference.py",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "device = ('cuda' if torch.cuda.is_available()\n"
        "          else 'mps' if torch.backends.mps.is_available()\n"
        "          else 'cpu')",
        "use MPS on Apple Silicon instead of falling through to CPU",
    ),
    (
        "audio.py",
        "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
        "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,",
        "librosa >= 0.10 made these arguments keyword-only",
    ),
]


def _say(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def _venv_python(repo: Path) -> Path:
    """Interpreter inside the Wav2Lip venv, per-platform."""
    if os.name == "nt":
        return repo / "venv-w2l" / "Scripts" / "python.exe"
    return repo / "venv-w2l" / "bin" / "python"


def show_patches() -> int:
    for fname, old, new, why in PATCHES:
        _say(f"\n── {fname} — {why}")
        for line in old.splitlines():
            _say(f"  - {line}")
        for line in new.splitlines():
            _say(f"  + {line}")
    _say("")
    return 0


def clone(repo: Path) -> None:
    if (repo / "inference.py").exists():
        _say(f"✓ repo already at {repo}")
        return
    if repo.exists() and any(repo.iterdir()):
        raise SystemExit(f"✗ {repo} exists but has no inference.py — move it aside first")
    if not shutil.which("git"):
        raise SystemExit("✗ git not found on PATH")
    _say(f"→ cloning {REPO_URL}")
    _run(["git", "clone", "--depth", "1", REPO_URL, str(repo)])
    _say(f"✓ cloned to {repo}")


def build_venv(repo: Path) -> Path:
    py = _venv_python(repo)
    if not py.exists():
        _say("→ creating venv-w2l")
        _run([sys.executable, "-m", "venv", str(repo / "venv-w2l")])
    _say(f"→ installing {', '.join(DEPS)} (torch is a large download)")
    _run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    _run([str(py), "-m", "pip", "install", "--quiet", *DEPS])
    _say(f"✓ venv ready at {py}")
    return py


def patch(repo: Path) -> None:
    for fname, old, new, why in PATCHES:
        target = repo / fname
        if not target.exists():
            raise SystemExit(f"✗ {fname} missing — is {repo} really a Wav2Lip clone?")
        text = target.read_text(encoding="utf-8")
        if new in text:
            _say(f"✓ {fname} already patched ({why})")
            continue
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"✗ {fname}: expected exactly 1 occurrence to patch, found {n}. "
                "Upstream changed — re-check the patch before forcing it."
            )
        target.write_text(text.replace(old, new), encoding="utf-8")
        _say(f"✓ patched {fname} ({why})")


def checkpoint_path(repo: Path) -> Path:
    return repo / "checkpoints" / "wav2lip_gan.pth"


def download_checkpoint(repo: Path) -> None:
    dest = checkpoint_path(repo)
    if dest.exists():
        _say(f"✓ checkpoint present ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _say(f"→ downloading ~400 MB to {dest}")
    tmp = dest.with_suffix(".part")

    def _progress(block: int, size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * size * 100 // total)
            print(f"\r   {pct}%", end="", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, tmp, _progress)  # noqa: S310
    print()
    tmp.rename(dest)
    _say(f"✓ checkpoint saved ({dest.stat().st_size / 1e6:.0f} MB)")


def verify(repo: Path) -> bool:
    """Import the stack in the Wav2Lip venv and report the device it will use."""
    py = _venv_python(repo)
    if not py.exists():
        _say(f"· no venv at {py} — re-run without --skip-install to build it")
        return False
    probe = (
        "import torch, librosa, cv2, numpy, scipy\n"
        "dev = ('cuda' if torch.cuda.is_available() "
        "else 'mps' if torch.backends.mps.is_available() else 'cpu')\n"
        "print(f'torch {torch.__version__} | librosa {librosa.__version__} "
        "| numpy {numpy.__version__} | device {dev}')\n"
    )
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        _say("✗ the Wav2Lip venv cannot import its stack:")
        _say("   " + (r.stderr or "").strip().splitlines()[-1])
        return False
    _say(f"✓ {r.stdout.strip()}")

    ckpt = checkpoint_path(repo)
    if not ckpt.exists():
        _say(f"· checkpoint not downloaded yet — {ckpt}")
        return False
    # torch >= 2.6 defaults to weights_only=True. A plain state_dict is fine
    # under that, but the generator checkpoint also carries optimizer state, so
    # prove it loads here rather than at the end of a dub.
    load = (
        "import torch, sys\n"
        f"cp = torch.load(r'{ckpt}', map_location='cpu')\n"
        "sys.exit(0 if 'state_dict' in cp else 1)\n"
    )
    r = subprocess.run([str(py), "-c", load], capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-1:] or ["no state_dict in checkpoint"]
        _say(f"✗ checkpoint did not load: {tail[0]}")
        _say("   If this is an UnpicklingError, add weights_only=False to "
             "torch.load in inference.py:_load")
        return False
    _say(f"✓ checkpoint loads ({ckpt.stat().st_size / 1e6:.0f} MB)")
    return True


def print_env(repo: Path) -> None:
    _say("\nAdd these to .env, then restart the server:\n")
    _say(f"    GOCHIDUBB_WAV2LIP_DIR={repo}")
    _say(f"    GOCHIDUBB_WAV2LIP_PYTHON={_venv_python(repo)}")
    _say("\nThe lip-sync toggle lights up on its own once the server sees both.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=None,
                    help=f"where to clone Wav2Lip (default: {DEFAULT_DIR})")
    ap.add_argument("--show-patches", action="store_true",
                    help="print the source patches and exit, changing nothing")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing install without modifying it")
    ap.add_argument("--skip-install", action="store_true",
                    help="clone and patch, but do not build the venv")
    ap.add_argument("--download-checkpoint", action="store_true",
                    help="also fetch the ~400 MB generator weights")
    args = ap.parse_args()

    if args.show_patches:
        return show_patches()

    # Path("") is Path("."), so the env var has to be checked as a string —
    # otherwise an unset GOCHIDUBB_WAV2LIP_DIR resolves to the cwd.
    env_dir = os.getenv("GOCHIDUBB_WAV2LIP_DIR", "").strip()
    repo = (args.dir or (Path(env_dir) if env_dir else DEFAULT_DIR)).resolve()

    if args.verify_only:
        return 0 if verify(repo) else 1

    clone(repo)
    patch(repo)
    if not args.skip_install:
        build_venv(repo)
    if args.download_checkpoint:
        download_checkpoint(repo)

    ok = verify(repo)
    if not ok and not checkpoint_path(repo).exists():
        _say("\nThe generator checkpoint is the one manual step:\n")
        _say(f"    curl -L -o {checkpoint_path(repo)} \\\n        {CHECKPOINT_URL}")
        _say("\n(or re-run this script with --download-checkpoint)")
    print_env(repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
