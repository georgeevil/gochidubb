"""System status + model catalog for plug-and-play UX."""
import logging
import os
import shutil
import subprocess
import sys
from typing import List, Dict, Any

log = logging.getLogger("gochidubb.models")

# ============================================================
# LM Studio Configuration
# ============================================================
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "1") == "1"
# LM_STUDIO_URL may be written with or without the "/v1" suffix, so build the
# endpoint from the normalized host root instead of concatenating onto it.
# See pipeline/translator.py for why that mattered.
from .translator import LM_STUDIO_MODELS_ENDPOINT  # noqa: E402

# ============================================================
# Dynamic Model Catalog from LM Studio
# ============================================================

async def fetch_lm_studio_models() -> List[Dict[str, Any]]:
    """Fetch available models from LM Studio API."""
    if not USE_LM_STUDIO:
        return []
    
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                LM_STUDIO_MODELS_ENDPOINT,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    # Convert to our catalog format
                    catalog = []
                    for m in models:
                        model_id = m.get("id", "")
                        # Parse size from model ID if possible
                        size_gb = _estimate_model_size(model_id)
                        catalog.append({
                            "id": model_id,
                            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                            "size": f"{size_gb:.1f} GB" if size_gb > 0 else "Unknown",
                            "vram": f"~{size_gb + 2:.0f} GB" if size_gb > 0 else "Unknown",
                            "description": _get_model_description(model_id),
                            "recommended": "qwen" in model_id.lower() or "gemma" in model_id.lower(),
                            "source": "lm_studio",
                        })
                    return catalog
    except Exception as e:
        log.warning(f"Failed to fetch LM Studio models: {e}")
        return []
    
    return []


def _estimate_model_size(model_id: str) -> float:
    """Estimate model size in GB based on model ID."""
    model_lower = model_id.lower()
    
    # Qwen models
    if "qwen" in model_lower:
        if "72b" in model_lower or "72" in model_lower:
            return 40.0
        elif "32b" in model_lower or "32" in model_lower:
            return 18.0
        elif "14b" in model_lower or "14" in model_lower:
            return 8.0
        elif "7b" in model_lower or "7" in model_lower:
            return 4.0
        elif "3b" in model_lower or "3" in model_lower:
            return 2.0
        elif "1.5b" in model_lower:
            return 1.0
    
    # Gemma models
    if "gemma" in model_lower:
        if "27b" in model_lower:
            return 16.0
        elif "12b" in model_lower:
            return 7.5
        elif "4b" in model_lower:
            return 3.0
        elif "2b" in model_lower:
            return 1.5
    
    # Llama models
    if "llama" in model_lower:
        if "70b" in model_lower:
            return 40.0
        elif "8b" in model_lower:
            return 4.5
        elif "3b" in model_lower:
            return 2.0
    
    # Mistral models
    if "mistral" in model_lower:
        if "large" in model_lower:
            return 20.0
        elif "nemo" in model_lower:
            return 7.0
        elif "7b" in model_lower:
            return 4.0
    
    # Default fallback
    if "b" in model_lower:
        try:
            import re
            match = re.search(r'(\d+)b', model_lower)
            if match:
                return float(match.group(1)) * 0.6  # Approximate GB per B
        except Exception:
            pass
    
    return 0.0


def _get_model_description(model_id: str) -> str:
    """Get a description for a model based on its ID."""
    model_lower = model_id.lower()
    
    if "qwen" in model_lower:
        if "72b" in model_lower:
            return "Best quality translation. Needs 40GB+ RAM."
        elif "32b" in model_lower:
            return "Excellent quality. Needs 18GB+ RAM."
        elif "14b" in model_lower:
            return "Great balance of quality and speed. Needs 8GB+ RAM."
        elif "7b" in model_lower:
            return "Good quality, fast. Needs 4GB+ RAM."
        elif "3b" in model_lower:
            return "Fast, good for CPU. Needs 2GB+ RAM."
        else:
            return "Qwen series translation model."
    
    if "gemma" in model_lower:
        if "27b" in model_lower:
            return "Google's largest Gemma. Excellent quality."
        elif "12b" in model_lower:
            return "Google's latest. Strong on European languages."
        elif "4b" in model_lower:
            return "Compact, fast, good quality."
        else:
            return "Gemma series from Google."
    
    if "llama" in model_lower:
        if "70b" in model_lower:
            return "Meta's largest. Best quality, needs 40GB+ RAM."
        elif "8b" in model_lower:
            return "Meta's latest. Good all-around."
        else:
            return "Llama series from Meta."
    
    if "mistral" in model_lower:
        if "large" in model_lower:
            return "Mistral Large. High quality."
        elif "nemo" in model_lower:
            return "Mistral Nemo. Good for European languages."
        else:
            return "Mistral series."
    
    if "aya" in model_lower:
        return "Cohere's multilingual specialist."
    
    return "Translation model available in LM Studio."


# Keep a fallback catalog for when LM Studio is not available
FALLBACK_CATALOG = {
    "translation": [
        {"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen 2.5 14B", "size": "8.0 GB", "vram": "~10 GB", 
         "description": "Great balance. Needs 8GB+ RAM.", "recommended": True},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B", "size": "4.0 GB", "vram": "~6 GB",
         "description": "Good quality, fast. Needs 4GB+ RAM."},
        {"id": "google/gemma-3-12b-it", "name": "Gemma 3 12B", "size": "7.5 GB", "vram": "~9 GB",
         "description": "Google's latest. Strong on European languages."},
        {"id": "google/gemma-3-4b-it", "name": "Gemma 3 4B", "size": "3.0 GB", "vram": "~4 GB",
         "description": "Compact, fast, good quality."},
        {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B", "size": "4.5 GB", "vram": "~6 GB",
         "description": "Meta's latest. Good all-around."},
        {"id": "mistral-nemo", "name": "Mistral Nemo", "size": "7.0 GB", "vram": "~9 GB",
         "description": "Good for European languages."},
        {"id": "aya-expanse:8b", "name": "Aya Expanse 8B", "size": "4.5 GB", "vram": "~6 GB",
         "description": "Specialized for multilingual translation."},
    ]
}

# This will be populated dynamically
MODEL_CATALOG = {"translation": []}


async def get_model_catalog() -> Dict[str, List[Dict]]:
    """Get the model catalog, dynamically fetching from LM Studio."""
    global MODEL_CATALOG
    
    if USE_LM_STUDIO:
        models = await fetch_lm_studio_models()
        if models:
            MODEL_CATALOG["translation"] = models
            log.info(f"Loaded {len(models)} models from LM Studio")
            return MODEL_CATALOG
    
    # Fallback to static catalog
    return FALLBACK_CATALOG


# ============================================================
# System Status Checks (modified for LM Studio)
# ============================================================

def check_python():
    major, minor = sys.version_info.major, sys.version_info.minor
    version = f"{major}.{minor}.{sys.version_info.micro}"
    # The 3.14 ceiling is `spaces` (<3.15), not VoxCPM — which declares only
    # >=3.10. 3.13+ resolves the same dependency set as 3.12 but is newer here,
    # so it reports ok with a caveat rather than failing the check.
    ok = major == 3 and 10 <= minor <= 14
    note = ""
    if not ok:
        note = "Python 3.10-3.14 required"
    elif minor > 12:
        note = f"Python {major}.{minor} works but is less tested — 3.11/3.12 are the well-worn path"
    return {
        "ok": ok,
        "version": version,
        "note": note,
    }


def check_ffmpeg():
    p = shutil.which("ffmpeg")
    version = ""
    if p:
        try:
            r = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=5)
            version = r.stdout.split("\n")[0] if r.stdout else ""
        except Exception:
            pass
    return {"ok": p is not None, "path": p or "", "version": version}


def check_ytdlp():
    binary = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    has_module = False
    try:
        import yt_dlp  # noqa
        has_module = True
    except ImportError:
        pass
    return {"ok": binary is not None or has_module, "path": binary or "", "module": has_module}


def check_whisper():
    try:
        import faster_whisper  # noqa
        return {"ok": True, "type": "faster-whisper"}
    except ImportError:
        pass
    except Exception as e:
        return {"ok": False, "type": "broken", "error": str(e)[:150]}
    try:
        import whisper  # noqa
        return {"ok": True, "type": "openai-whisper"}
    except ImportError:
        pass
    except Exception as e:
        return {"ok": False, "type": "broken", "error": str(e)[:150]}
    return {"ok": False, "type": "none"}


def check_voxcpm():
    try:
        import voxcpm  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False}
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}


def check_edge_tts():
    try:
        import edge_tts  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False}


def check_demucs():
    try:
        import demucs  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False, "hint": "pip install demucs"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}


def check_f5tts():
    try:
        import f5_tts  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False, "hint": "pip install f5-tts"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}


def check_silero_vad():
    try:
        import torch  # noqa
        return {"ok": True}
    except (ImportError, OSError):
        return {"ok": False, "hint": "pip install torch"}


def check_pyannote():
    try:
        import pyannote.audio  # noqa
        has_token = bool(os.getenv("HF_TOKEN", ""))
        return {"ok": True, "has_token": has_token}
    except ImportError:
        return {"ok": False, "has_token": False}
    except Exception as e:
        return {"ok": False, "has_token": False, "error": str(e)[:150]}


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ok": False, "name": "", "vram_gb": 0}
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = 0
        try:
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free = free_bytes / 1024**3
        except Exception:
            free = total
        return {
            "ok": True,
            "name": name,
            "vram_gb": round(total, 1),
            "vram_free_gb": round(free, 1),
        }
    except Exception as e:
        return {
            "ok": False,
            "name": "",
            "vram_gb": 0,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def check_lm_studio_binary():
    """Check if LM Studio is available (just a placeholder - actual check is async)."""
    return {"ok": True, "note": "LM Studio connection checked via API"}


def check_ollama_binary():
    return {"ok": shutil.which("ollama") is not None}


async def check_translation_backend() -> Dict[str, Any]:
    """Check which translation backend is available."""
    if USE_LM_STUDIO:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    LM_STUDIO_MODELS_ENDPOINT,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m.get("id", "") for m in data.get("data", [])]
                        return {
                            "ok": True,
                            "backend": "lm_studio",
                            "url": LM_STUDIO_URL,
                            "models": models,
                        }
        except Exception:
            pass
        return {"ok": False, "backend": "lm_studio", "url": LM_STUDIO_URL, "models": []}
    else:
        # Ollama check
        try:
            import httpx
            with httpx.Client(timeout=3) as client:
                response = client.get(f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return {"ok": True, "backend": "ollama", "models": models}
        except Exception:
            pass
        return {"ok": False, "backend": "ollama", "models": []}


def get_system_status():
    """Aggregate all system checks for the UI."""
    status = {
        "python": check_python(),
        "ffmpeg": check_ffmpeg(),
        "yt_dlp": check_ytdlp(),
        "whisper": check_whisper(),
        "voxcpm": check_voxcpm(),
        "f5tts": check_f5tts(),
        "demucs": check_demucs(),
        "silero_vad": check_silero_vad(),
        "edge_tts": check_edge_tts(),
        "pyannote": check_pyannote(),
        "gpu": check_gpu(),
        "ollama_binary": check_ollama_binary(),
    }
    
    # Add translation backend info
    # Note: This is sync, but the actual check is async.
    # The server will populate this properly.
    status["translation_backend"] = "lm_studio" if USE_LM_STUDIO else "ollama"
    
    return status