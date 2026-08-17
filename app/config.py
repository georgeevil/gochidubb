"""Application configuration — paths, env vars, and persisted user settings.

UserConfig is the single source of truth for runtime-changeable settings.
It loads from config-user.json at startup and saves back when changed.
All fields have sensible defaults so it works out of the box without a config file.

Usage:
    from app.config import cfg

    cfg.whisper_model          # e.g. "large-v3"
    cfg.set("whisper_model", "medium")   # persists to disk immediately
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger("gochidubb.config")

# ─── Project paths ──────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.resolve()
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
JOBS_DB = BASE / "jobs_db"
STATIC_DIR = BASE / "static"
PRESETS_DIR = BASE / "presets"
VOICE_PRESETS_DIR = PRESETS_DIR / "voices"
USER_GLOSSARY_FILE = PRESETS_DIR / "user_glossary.json"
CONFIG_FILE = BASE / "config-user.json"

for _d in (UPLOAD_DIR, OUTPUT_DIR, JOBS_DB, STATIC_DIR, VOICE_PRESETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class UserConfig:
    """Runtime-editable settings, persisted to config-user.json.

    All fields are optional — missing keys in config-user.json fall back to
    the dataclass defaults, so adding new settings never breaks existing installs.
    """

    # ── Whisper ───────────────────────────────────────────────────────
    whisper_model: str = "large-v3"       # large-v3 | medium | small | tiny
    auto_denoise: bool = True             # FFT denoise before transcription

    # ── VAD ───────────────────────────────────────────────────────────
    vad_enabled: bool = True              # strip silence before Whisper
    vad_threshold: float = 0.5           # Silero VAD threshold (0.3–0.8)

    # ── Background separation ──────────────────────────────────────────
    separation_backend: str = "demucs"   # "demucs" | "audio-separator" | "none"
    demucs_model: str = "htdemucs_ft"    # Demucs model name

    # ── Translation ────────────────────────────────────────────────────
    translation_model: str = "qwen3:8b"  # default Ollama model
    ollama_url: str = "http://localhost:11434"

    # ── VoxCPM ────────────────────────────────────────────────────────
    voxcpm_model: str = "openbmb/VoxCPM2"
    voxcpm_cfg: float = 2.0              # 1.5–3.0
    voxcpm_steps: int = 10              # 5–20

    # ── TTS ───────────────────────────────────────────────────────────
    tts_engine: str = "voxcpm"           # "voxcpm" | "f5tts" | "edge-tts"
    tts_speed: str = "balanced"          # "fast" | "balanced" | "quality"
    # Hardest time-compression the assembler may apply to keep a segment in
    # sync (atempo, pitch preserved). Up to ~1.15 is inaudible; the headroom
    # above that is what stops one overlong segment from pushing every later
    # one late. Raise it for dense dialogue, lower it if the dub sounds rushed.
    tts_max_stretch: float = 1.4
    warmup_on_start: bool = False        # pre-load VoxCPM at server start

    # ── Stage reuse (beta) ────────────────────────────────────────────
    # Skip a pipeline stage when a previous job already computed the same
    # thing from the same inputs. Download/extract/transcribe/diarize are
    # ~60% of pipeline time and none of it depends on the target language,
    # so a redub into a new language currently redoes all of it.
    #
    # Off by default: this is new, and a stale reuse is a silent failure.
    # See docs/stage-reuse.md.
    reuse_enabled: bool = False
    # Stages allowed to be reused, comma-separated. The source-only stages
    # are the safe, high-value set; translate and tts are opt-in on top.
    reuse_stages: str = "download,extract,transcribe,diarize"
    # Quality gates — an artifact that scored worse than this when it was
    # produced is recomputed rather than reused. 0 disables a gate.
    reuse_transcribe_min_word_conf: float = 0.45
    reuse_transcribe_max_no_speech: float = 0.35
    reuse_diarize_min_ref_sec: float = 1.0
    reuse_translate_max_fallback: float = 0.0
    reuse_translate_min_semantic: float = 0.67
    reuse_tts_max_failed_qa: float = 0.25
    qa_same_language: bool = False       # whisper-roundtrip QA on same-language dubs too

    # ── FFmpeg (extract / render) ─────────────────────────────────────
    # Soft timeout: while ffmpeg keeps reporting encode progress the deadline
    # is extended, so long videos can render past this. It only becomes a
    # hard kill once no progress is seen for ffmpeg_stall_timeout seconds.
    ffmpeg_timeout: int = 600
    ffmpeg_stall_timeout: int = 120

    # ── Downloader (yt-dlp) ───────────────────────────────────────────
    ytdlp_cookies_from_browser: str = ""   # e.g. "firefox", "chrome", "safari"
    ytdlp_cookiefile: str = ""             # path to Netscape cookies.txt
    max_source_duration_sec: int = 0       # 0 = no pre-download duration gate

    # ── Publishing ────────────────────────────────────────────────────
    # Attribution appended to published-video descriptions. {source_url}
    # is replaced with the original video URL (see pipeline/publisher.py).
    publish_description_template: str = "Original: {source_url}"

    # ── UI behaviour ──────────────────────────────────────────────────
    open_browser: bool = True
    server_port: int = 8910
    hf_token: str = ""                   # HuggingFace token for pyannote

    def set(self, key: str, value) -> None:
        """Set a config value and immediately persist to disk."""
        if not hasattr(self, key):
            raise KeyError(f"Unknown config key: {key!r}")
        setattr(self, key, value)
        self._save()

    def update(self, **kwargs) -> None:
        """Batch-update keys and persist once."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                log.warning(f"[config] Unknown key ignored: {k!r}")
        self._save()

    def to_dict(self) -> dict:
        return asdict(self)

    def _save(self) -> None:
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"[config] Could not save {CONFIG_FILE}: {e}")


def _load_config() -> UserConfig:
    """Load UserConfig from disk, merging over dataclass defaults."""
    c = UserConfig()

    # Layer 1: environment variables (highest priority)
    env_map = {
        "HF_TOKEN": "hf_token",
        "VOXCPM_MODEL": "voxcpm_model",
        "VOXCPM_CFG": "voxcpm_cfg",
        "VOXCPM_STEPS": "voxcpm_steps",
        "OLLAMA_URL": "ollama_url",
        "WHISPER_MODEL": "whisper_model",
        "GOCHIDUBB_OPEN_BROWSER": "open_browser",
        "GOCHIDUBB_WARMUP": "warmup_on_start",
        "GOCHIDUBB_QA_SAME_LANGUAGE": "qa_same_language",
        "GOCHIDUBB_TTS_MAX_STRETCH": "tts_max_stretch",
        "GOCHIDUBB_REUSE": "reuse_enabled",
        "GOCHIDUBB_REUSE_STAGES": "reuse_stages",
        "GOCHIDUBB_FFMPEG_TIMEOUT": "ffmpeg_timeout",
        "GOCHIDUBB_FFMPEG_STALL_TIMEOUT": "ffmpeg_stall_timeout",
        "YT_DLP_COOKIES_FROM_BROWSER": "ytdlp_cookies_from_browser",
        "YT_DLP_COOKIEFILE": "ytdlp_cookiefile",
        "MAX_SOURCE_DURATION_SEC": "max_source_duration_sec",
        "PUBLISH_DESCRIPTION_TEMPLATE": "publish_description_template",
    }
    for env_k, field_k in env_map.items():
        v = os.getenv(env_k)
        if v is None:
            continue
        cur = getattr(c, field_k)
        try:
            if isinstance(cur, bool):
                setattr(c, field_k, v.strip() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(c, field_k, int(v.strip()))
            elif isinstance(cur, float):
                setattr(c, field_k, float(v.strip()))
            else:
                setattr(c, field_k, v.strip())
        except (ValueError, TypeError) as e:
            log.warning(f"[config] env {env_k}={v!r}: {e}")

    # Layer 2: config-user.json (lower priority than env)
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(c, k):
                    # Don't let the file override env-set values
                    env_key = next(
                        (ek for ek, fk in env_map.items() if fk == k), None
                    )
                    if env_key and os.getenv(env_key) is not None:
                        continue
                    setattr(c, k, v)
        except Exception as e:
            log.warning(f"[config] Could not read {CONFIG_FILE}: {e}")

    return c


# Singleton — import and use `cfg` everywhere
cfg: UserConfig = _load_config()
