"""Download videos from YouTube or validate local paths (Windows-safe)."""
import json
import logging
import os
import shutil
import subprocess
import sys

from app.config import cfg

log = logging.getLogger("gochidubb.downloader")


def _find_ytdlp() -> str:
    """Find yt-dlp executable across platforms."""
    exe_dir = os.path.dirname(sys.executable)
    # On Windows venv: python.exe lives in venv\Scripts\, yt-dlp.exe is a sibling.
    # On Linux venv:   python lives in venv/bin/, yt-dlp is a sibling.
    if sys.platform.startswith("win"):
        candidates = [
            os.path.join(exe_dir, "yt-dlp.exe"),      # inside venv\Scripts
            os.path.join(exe_dir, "..", "Scripts", "yt-dlp.exe"),  # from venv root
            "yt-dlp.exe",                              # PATH
            "yt-dlp",
        ]
    else:
        candidates = [
            os.path.join(exe_dir, "yt-dlp"),
            "yt-dlp",
        ]

    for c in candidates:
        # shutil.which accepts full paths
        found = shutil.which(c)
        if found:
            return found
        # also accept if it's just a direct file
        if os.path.isfile(c):
            return c

    # Last resort: use Python module invocation
    return None  # caller will use `python -m yt_dlp`


def _base_cmd() -> list:
    """Resolve the yt-dlp invocation: binary if found, else `python -m yt_dlp`."""
    ytdlp = _find_ytdlp()
    if ytdlp is None:
        log.info("Using: python -m yt_dlp (no binary found)")
        return [sys.executable, "-m", "yt_dlp"]
    log.info(f"Using yt-dlp: {ytdlp}")
    return [ytdlp]


def _cookie_args() -> list:
    """yt-dlp cookie flags from user config (both may be set)."""
    args = []
    browser = (cfg.ytdlp_cookies_from_browser or "").strip()
    cookiefile = (cfg.ytdlp_cookiefile or "").strip()
    if browser:
        args += ["--cookies-from-browser", browser]
    if cookiefile:
        args += ["--cookies", cookiefile]
    return args


def _parse_probe_json(stdout: str):
    """Parse `--dump-single-json` output. Returns dict or None — never raises."""
    try:
        info = json.loads(stdout)
    except (TypeError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def probe_metadata(source: str):
    """Fetch video metadata for a URL via yt-dlp without downloading.

    Returns the full info dict, or None for non-URL sources and on ANY
    failure (timeout, non-zero exit, unparseable JSON) — never raises.
    """
    if not isinstance(source, str) or not (
        source.startswith("http://") or source.startswith("https://")
    ):
        return None

    cmd = _base_cmd() + [
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--no-check-certificates",
    ] + _cookie_args() + [source]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log.warning(f"[probe] Metadata probe timed out (60s): {source}")
        return None
    except Exception as e:
        log.warning(f"[probe] Metadata probe failed to run: {e}")
        return None

    if result.returncode != 0:
        stderr = result.stderr or ""
        hint = ""
        if "cookie" in stderr.lower() or "keyring" in stderr.lower():
            hint = (" — check ytdlp_cookies_from_browser setting; "
                    "firefox is most reliable on macOS")
        log.warning(f"[probe] Metadata probe failed for {source}: "
                    f"{stderr[:300]}{hint}")
        return None

    info = _parse_probe_json(result.stdout)
    if info is None:
        log.warning(f"[probe] Could not parse yt-dlp metadata JSON for {source}")
    return info


def curate_metadata(info: dict) -> dict:
    """Distill a full yt-dlp info dict into a small, persistable summary."""
    categories = info.get("categories") or []
    tags = info.get("tags") or []
    title = info.get("title") or ""
    title_low = title.lower()
    is_music = bool(
        ("Music" in categories)
        or info.get("artist")
        or info.get("track")
        or "official video" in title_low
        or "official music video" in title_low
    )
    description = info.get("description") or ""
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "upload_date": info.get("upload_date"),
        "thumbnail": info.get("thumbnail"),
        "categories": categories,
        "tags": tags[:20],
        "language": info.get("language"),
        "webpage_url": info.get("webpage_url"),
        "description": description[:2000],
        "is_music": is_music,
    }


def download_video(source: str, output_dir: str, info: dict | None = None) -> str:
    """
    Download from YouTube URL or copy local file.
    Returns path to the local video file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source_video.mp4")

    # Local file
    if os.path.isfile(source):
        _src_size = os.path.getsize(source)
        _src_ext = os.path.splitext(source)[1].lower() or "(no ext)"
        # Log the real on-disk format — ffmpeg/ffprobe are picky about input
        # containers, and a common silent-failure cause is uploading an image
        # (e.g. .webp/.jpg) or audio-only file that gets copied to
        # source_video.mp4 and then fails downstream with no obvious reason.
        log.info(
            f"[download] Local file source: {source} "
            f"(ext={_src_ext}, size={_src_size/1048576:.2f} MB)"
        )
        if os.path.abspath(source) != os.path.abspath(output_path):
            shutil.copy2(source, output_path)
        log.info(f"Local file ready: {output_path}")
        return output_path

    # YouTube / HTTP URL
    if not (source.startswith("http://") or source.startswith("https://")):
        raise ValueError(f"Not a file or URL: {source}")

    base_cmd = _base_cmd()
    cookie_args = _cookie_args()

    cmd = base_cmd + [
        "--no-playlist",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-check-certificates",
        "--retries", "3",
        "--socket-timeout", "30",
        "--no-warnings",
    ] + cookie_args + [source]

    log.info(f"Downloading: {source}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        log.warning(f"First attempt failed: {result.stderr[:200]}")
        # Fallback with simpler format
        cmd_fallback = base_cmd + [
            "--no-playlist",
            "-f", "best[ext=mp4]/best",
            "-o", output_path,
            "--no-check-certificates",
        ] + cookie_args + [source]
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            raise RuntimeError(f"YouTube download failed: {result.stderr[:300]}")

    # yt-dlp sometimes adds extensions; find the actual file
    if not os.path.exists(output_path):
        for f in os.listdir(output_dir):
            if f.startswith("source_video") and f.endswith((".mp4", ".mkv", ".webm")):
                actual = os.path.join(output_dir, f)
                if actual != output_path:
                    os.rename(actual, output_path)
                break

    if not os.path.exists(output_path):
        raise RuntimeError("Download completed but video file not found")

    size_mb = os.path.getsize(output_path) / 1048576
    log.info(f"Downloaded: {output_path} ({size_mb:.1f} MB)")

    # Preserve the full yt-dlp metadata (probed by the caller) next to the
    # video so later stages / users can inspect it. Best-effort only.
    if info:
        info_path = os.path.join(output_dir, "source_info.json")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            log.info(f"Wrote metadata: {info_path}")
        except Exception as e:
            log.warning(f"Could not write source_info.json: {e}")

    return output_path
