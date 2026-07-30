import os
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional, Callable, Tuple
import time
import re

log = logging.getLogger("tachidubb.translator")
# Add this logger definition
logger = logging.getLogger(__name__)


# ============================================================
# LM Studio Configuration
# ============================================================
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "")  # Auto-detect if empty
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "1") == "1"

# ============================================================
# Translation System
# ============================================================

# Built-in glossary for common terms
_BUILTIN_GLOSSARY = {
    "BJJ": {
        "en": {
            "guard": "гард",
            "pass": "проход",
            "sweep": "свип",
            "submission": "болевой приём",
            "choke": "удушение",
            "armbar": "рычаг локтя",
            "triangle": "треугольник",
            "mount": "маунт",
            "back": "спина",
            "side control": "боковой контроль",
        }
    }
}

# Try to load user glossary
USER_GLOSSARY_FILE = os.path.join(os.path.dirname(__file__), "..", "presets", "user_glossary.json")

def _load_user_glossary() -> Dict:
    """Load user-defined glossary terms."""
    if not os.path.exists(USER_GLOSSARY_FILE):
        return {}
    try:
        with open(USER_GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Flatten the glossary into a simple dict
            terms = {}
            for domain in data.get("domains", []):
                if domain.get("target_lang") == "ru":  # Only for Russian for now
                    terms.update(domain.get("terms", {}))
            return terms
    except Exception as e:
        log.warning(f"Failed to load user glossary: {e}")
        return {}


def _build_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    context_hint: str = "",
    glossary: Dict = None
) -> str:
    """Build a translation prompt for LM Studio."""
    if glossary is None:
        glossary = {}
    
    # Combine built-in and user glossary
    all_glossary = {}
    all_glossary.update(_BUILTIN_GLOSSARY.get(source_lang, {}).get(target_lang, {}))
    all_glossary.update(glossary.get(target_lang, {}))
    
    # Build glossary section
    glossary_text = ""
    if all_glossary:
        terms = "\n".join([f"  - {k}: {v}" for k, v in all_glossary.items()])
        glossary_text = f"""
IMPORTANT: Use these specific translations for the following terms:
{terms}
"""
    
    context_text = f"\nContext: {context_hint}" if context_hint else ""
    
    # Modern prompt format for Qwen models
    prompt = f"""You are a professional translator. Translate the following text from {source_lang} to {target_lang}.
{glossary_text}{context_text}

Rules:
1. Translate naturally and fluently
2. Keep the same tone and style
3. DO NOT add any explanations or notes
4. ONLY output the translated text

Text to translate:
{text}

Translation:"""
    
    return prompt


async def _translate_with_lm_studio(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    context_hint: str = "",
    glossary: Dict = None,
    max_retries: int = 3
) -> str:
    """Translate using LM Studio's OpenAI-compatible API."""
    if not text.strip():
        return text
    
    # Check if LM Studio is available
    lm_available = await check_lm_studio()
    if not lm_available:
        log.warning("LM Studio not available, falling back to mock translation")
        return f"[{target_lang}] {text}"
    
    prompt = _build_translation_prompt(text, source_lang, target_lang, context_hint, glossary)
    
    # Get model from environment or auto-detect
    model_name = LM_STUDIO_MODEL or await _get_default_model()
    
    headers = {
        "Content-Type": "application/json",
    }
    
    # LM Studio uses OpenAI-compatible format
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": f"You are a professional translator from {source_lang} to {target_lang}."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 512,
        "stop": ["\n\n", "Translation:"]
    }
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{LM_STUDIO_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        # Clean up the result
                        result = result.replace("Translation:", "").strip()
                        # Remove any remaining prompt artifacts
                        result = re.sub(r'^Text to translate:.*$', '', result, flags=re.MULTILINE)
                        result = result.strip()
                        if result:
                            log.debug(f"Translated: {text[:50]}... -> {result[:50]}...")
                            return result
                        else:
                            log.warning(f"Empty translation for: {text[:50]}...")
                    else:
                        error_text = await response.text()
                        log.warning(f"LM Studio API error (attempt {attempt+1}): {response.status} - {error_text}")
        except asyncio.TimeoutError:
            log.warning(f"LM Studio timeout (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            log.warning(f"LM Studio translation error (attempt {attempt+1}): {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    # If all retries fail, fallback to mock translation
    log.warning(f"All translation attempts failed for: {text[:50]}...")
    return f"[{target_lang}] {text}"


async def _get_default_model() -> str:
    """Get the first available model from LM Studio, preferring Qwen/Gemma."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{LM_STUDIO_URL}/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    if models:
                        # Prefer Qwen models for translation
                        qwen_models = [m for m in models if "qwen" in m.lower()]
                        if qwen_models:
                            # Prefer larger Qwen models first
                            for size in ["72b", "32b", "14b", "7b", "3b"]:
                                for m in qwen_models:
                                    if size in m.lower():
                                        log.info(f"Using Qwen model: {m}")
                                        return m
                            return qwen_models[0]
                        
                        # Try Gemma models
                        gemma_models = [m for m in models if "gemma" in m.lower()]
                        if gemma_models:
                            log.info(f"Using Gemma model: {gemma_models[0]}")
                            return gemma_models[0]
                        
                        # Try any instruct/chat model
                        instruct_models = [m for m in models if "instruct" in m.lower() or "chat" in m.lower()]
                        if instruct_models:
                            log.info(f"Using instruct model: {instruct_models[0]}")
                            return instruct_models[0]
                        
                        # Fallback to first model
                        log.info(f"Using default model: {models[0]}")
                        return models[0]
    except Exception as e:
        log.warning(f"Failed to get default model: {e}")
    
    # Last resort
    return "Qwen/Qwen2.5-14B-Instruct"

async def check_lm_studio() -> Tuple[bool, List[str]]:
    """Check if LM Studio is running and return available models."""
    if not USE_LM_STUDIO:
        return False, []
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{LM_STUDIO_URL}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    log.info(f"LM Studio running with models: {models}")
                    return True, models
                return False, []
    except Exception as e:
        log.debug(f"LM Studio not available: {e}")
        return False, []


async def check_translation_available() -> bool:
    """Check if any translation backend is available."""
    if USE_LM_STUDIO:
        available, _ = await check_lm_studio()
        return available
    else:
        # Fallback to Ollama
        from .ollama_check import check_ollama
        available, _ = await check_ollama()
        return available


async def translate_text(
    text: str,
    target_lang: str,
    model: str = "qwen/qwen3-8b",
    context_hint: Optional[str] = None,
) -> str:
    """
    Translate a single text using LM Studio.
    
    Args:
        text: Text to translate
        target_lang: Target language code (e.g., 'ru', 'zh', 'fr')
        model: LM Studio model to use
        context_hint: Optional context information
        
    Returns:
        Translated text
    """
    # Build the translation prompt
    prompt = f"Translate the following text to {target_lang}:"
    
    if context_hint:
        prompt += f"\nContext: {context_hint}"
    
    prompt += f"\n\nText: {text}\n\nTranslation:"
    
    # Call LM Studio API
    try:
        lm_studio_url = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1/chat/completions')
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            }
            
            async with session.post(lm_studio_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    translated = data['choices'][0]['message']['content'].strip()
                    return translated
                else:
                    log.error(f"LM Studio API error: {response.status}")
                    return text  # Fallback to original
                    
    except Exception as e:
        log.error(f"Translation error: {e}")
        return text  # Fallback to original


async def translate_segments(
    segments: List[Dict],
    target_lang: str,
    model: str = "qwen/qwen3-8b",
    max_concurrent: int = 5,
    context_hint: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> List[Dict]:
    """Translate segments with progress callback."""
    if not segments:
        return segments
    
    # Ensure max_concurrent is a valid integer
    try:
        max_concurrent = int(max_concurrent)
    except (ValueError, TypeError):
        logger.warning(f"Invalid max_concurrent value: {max_concurrent}, using default 5")
        max_concurrent = 5
    
    max_concurrent = max(1, min(max_concurrent, 20))
    
    total_segments = len(segments)
    logger.info(f"Translating {total_segments} segments to {target_lang} (max_concurrent={max_concurrent})")
    if context_hint:
        logger.info(f"Using context hint: {context_hint}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    start_time = asyncio.get_event_loop().time()
    
    async def translate_one(segment: Dict) -> Dict:
        nonlocal completed_count
        async with semaphore:
            try:
                text = segment.get('text', '')
                if not text:
                    segment['translated_text'] = ''
                    return segment
                
                # Translate with retry
                for attempt in range(3):
                    try:
                        translated = await translate_text(
                            text,
                            target_lang,
                            model,
                            context_hint=context_hint
                        )
                        segment['translated_text'] = translated
                        break
                    except Exception as e:
                        if attempt == 2:
                            logger.error(f"Translation failed after 3 attempts: {e}")
                            segment['translated_text'] = text
                        else:
                            await asyncio.sleep(1)
                
                # Update progress
                completed_count += 1
                if progress_callback:
                    try:
                        # Calculate ETA
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if completed_count > 0:
                            avg_time_per_item = elapsed / completed_count
                            remaining = total_segments - completed_count
                            eta_sec = avg_time_per_item * remaining
                        else:
                            eta_sec = 0
                        
                        # Call with both progress and ETA
                        if asyncio.iscoroutinefunction(progress_callback):
                            await progress_callback(completed_count, total_segments, eta_sec)
                        else:
                            progress_callback(completed_count, total_segments, eta_sec)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
                
                return segment
                
            except Exception as e:
                logger.error(f"Error translating segment: {e}")
                segment['translated_text'] = segment.get('text', '')
                completed_count += 1
                return segment
    
    tasks = [translate_one(seg) for seg in segments]
    translated_segments = await asyncio.gather(*tasks)
    
    logger.info(f"Translation complete for {len(translated_segments)} segments")
    return translated_segments


# ============================================================
# Model Management (for LM Studio)
# ============================================================

async def get_lm_studio_models() -> List[str]:
    """Get list of available models from LM Studio."""
    available, models = await check_lm_studio()
    return models if available else []


async def set_lm_studio_model(model: str) -> bool:
    """Set the default model to use for translations."""
    available, models = await check_lm_studio()
    if not available:
        return False
    
    if model not in models:
        log.warning(f"Model '{model}' not found in LM Studio. Available: {models}")
        return False
    
    global LM_STUDIO_MODEL
    LM_STUDIO_MODEL = model
    return True


# ============================================================
# Legacy Ollama compatibility (keep for now)
# ============================================================

async def check_ollama() -> Tuple[bool, List[str]]:
    """Legacy function for backward compatibility."""
    if USE_LM_STUDIO:
        return await check_lm_studio()
    
    # Original Ollama check
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


async def unload_ollama_model(model: str) -> None:
    """Legacy function - no-op for LM Studio."""
    if USE_LM_STUDIO:
        log.debug(f"LM Studio doesn't need model unloading")
        return
    
    # Original Ollama unload
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(
                f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/delete",
                json={"name": model}
            )
    except Exception:
        pass


async def ollama_pull_stream(model: str):
    """Legacy function - no-op for LM Studio."""
    if USE_LM_STUDIO:
        yield {"status": "LM Studio handles model loading separately", "percent": 100}
        return
    
    # Original Ollama pull
    import httpx
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/pull",
            json={"name": model}
        ) as response:
            async for chunk in response.aiter_bytes():
                if chunk:
                    try:
                        data = json.loads(chunk)
                        yield data
                    except json.JSONDecodeError:
                        continue