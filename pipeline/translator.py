import os
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional, Callable, Tuple
import time
import re
import math

log = logging.getLogger("tachidubb.translator")
# Add this logger definition
logger = logging.getLogger(__name__)


# ── Batch translation configuration ──────────────────────────────────
# Instead of translating one segment per LLM call (expensive for many
# short segments), we group segments into chunks and translate them all
# in a single call. The LLM returns numbered translations that we parse
# back into individual segments.
#
# TRANSLATION_CHUNK_SIZE env var controls the batch size (default 5).
# Larger chunks = fewer API calls but harder to parse and more risk of
# the model losing track of segment boundaries.
def _translation_chunk_size() -> int:
    """Number of segments to translate per LLM call."""
    try:
        val = int(os.getenv("TRANSLATION_CHUNK_SIZE", "5") or "5")
        return max(1, min(val, 15))
    except (ValueError, TypeError):
        return 5


# ── LM Studio availability cache ──────────────────────────────────────
# The UI polls /api/system every 5s, which calls check_ollama() →
# check_lm_studio() → GET /v1/models.  Without a cache this hits LM
# Studio's model endpoint 720+ times per hour, which is wasteful and
# adds noise to both logs and LM Studio's request view.
# 60s TTL is plenty: models don't change mid-session, and the translation
# path has its own timeout/retry logic if LM Studio happens to go down
# between cache refreshes.
_lm_studio_cache: dict = {
    "result": (False, []),
    "timestamp": 0.0,
}
_LM_STUDIO_CACHE_TTL = 60.0


# ============================================================
# LM Studio Configuration
# ============================================================
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "")  # Auto-detect if empty
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "1") == "1"


# ── Tunables for LM Studio translation ────────────────────────────────
# These are read lazily (at call time) so they always reflect the loaded
# .env, regardless of import order vs. dotenv initialization.
def _lm_studio_timeout() -> int:
    """Per-request timeout (seconds) for LM Studio chat completions.

    Large models on CPU (e.g. qwen3.6-27b) can take 30-90s per chunk, so the
    old hardcoded 60s was too tight: with concurrency=5 the requests piled
    up in LM Studio's serial queue, the client timed out and disconnected,
    and segments silently fell back to the untranslated source text. 300s is
    a safe default for big CPU rigs; GPU users can lower it.
    """
    try:
        return int(os.getenv("LM_STUDIO_TIMEOUT", "300") or "300")
    except (ValueError, TypeError):
        return 300


def _lm_studio_max_concurrent() -> int:
    """Max concurrent translation requests. 0 = auto (see _resolve_concurrency)."""
    try:
        return int(os.getenv("LM_STUDIO_MAX_CONCURRENT", "0") or "0")
    except (ValueError, TypeError):
        return 0


# Match a "<number>b" parameter count in a model id, e.g. "27b", "12b", "4b".
_LARGE_MODEL_RE = re.compile(r'(\d+(?:\.\d+)?)b', re.IGNORECASE)


def _is_large_model(model_id: str, threshold: float = 20) -> bool:
    """Heuristic: does this model id look like a >=threshold-B params model?

    Large models are slow per-token on CPU, and LM Studio serves completions
    serially per model — so concurrent requests just queue up and blow the
    client-side timeout. We use this to auto-serialize translation for big
    models (see _resolve_concurrency).

    NOTE: Uses the LARGEST number-b suffix found in the model id, not the
    first one.  Many model ids contain embedded version numbers followed by
    'b' (e.g. ``qwen3.6-27b``, ``gemma-4-26b-a4b-qat``) — taking the first
    match would see 3.6b / 4b / 2b instead of the true parameter count.
    """
    if not model_id:
        return False
    matches = _LARGE_MODEL_RE.findall(model_id)
    if not matches:
        return False
    try:
        return max(float(m) for m in matches) >= threshold
    except ValueError:
        return False

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
        log.debug("[translate] empty text, skipping")
        return text

    # Check if LM Studio is available (force refresh — we're about to
    # actually translate, so we want a fresh health check, not a cached
    # "it was fine 30s ago" answer).
    lm_available = await check_lm_studio(force_refresh=True)
    if not lm_available:
        log.warning(
            f"[translate] LM Studio not available — returning mock translation "
            f"for {target_lang} ({source_lang}→{target_lang}). Check LM_STUDIO_URL={LM_STUDIO_URL}"
        )
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
    url = f"{LM_STUDIO_URL}/chat/completions"
    log.debug(
        f"[translate] POST {url} model={model_name} "
        f"src={source_lang} tgt={target_lang} text={text[:50]!r}"
    )

    for attempt in range(max_retries):
        t0 = time.time()
        timeout_s = _lm_studio_timeout()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_s)
                ) as response:
                    log.debug(
                        f"[translate] {url} -> HTTP {response.status} "
                        f"in {time.time()-t0:.1f}s (attempt {attempt+1}/{max_retries})"
                    )
                    if response.status == 200:
                        data = await response.json()
                        msg = data.get("choices", [{}])[0].get("message", {})
                        result = (msg.get("content", "") or "").strip()
                        # Thinking/reasoning models (Qwen 3.6, Gemma 4, etc.)
                        # sometimes put the answer in reasoning_content instead
                        # of content.  Fall back to it when content is empty.
                        if not result:
                            result = (msg.get("reasoning_content", "") or "").strip()
                        # Clean up the result
                        result = result.replace("Translation:", "").strip()
                        # Remove any remaining prompt artifacts
                        result = re.sub(r'^Text to translate:.*$', '', result, flags=re.MULTILINE)
                        result = result.strip()
                        if result:
                            log.debug(f"Translated: {text[:50]}... -> {result[:50]}...")
                            return result
                        else:
                            log.warning(
                                f"[translate] LM Studio returned 200 but empty content "
                                f"for {text[:50]!r} — full response: {str(data)[:300]}"
                            )
                    else:
                        error_text = await response.text()
                        log.warning(f"LM Studio API error (attempt {attempt+1}): {response.status} - {error_text}")
        except asyncio.TimeoutError:
            log.warning(
                f"LM Studio timeout (attempt {attempt+1}/{max_retries}) after "
                f"{time.time()-t0:.1f}s (limit={timeout_s}s). Large model on CPU? "
                f"Raise LM_STUDIO_TIMEOUT / lower LM_STUDIO_MAX_CONCURRENT in .env"
            )
        except Exception as e:
            log.warning(f"LM Studio translation error (attempt {attempt+1}): {type(e).__name__}: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    # If all retries fail, fallback to mock translation
    log.warning(f"All translation attempts failed for: {text[:50]}... — using mock [{target_lang}] prefix")
    return f"[{target_lang}] {text}"


async def _get_default_model() -> str:
    """Get the first available model from LM Studio, preferring Qwen/Gemma."""
    url = f"{LM_STUDIO_URL}/models"
    log.debug(f"[get_default_model] GET {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
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
                else:
                    log.warning(f"[get_default_model] {url} returned HTTP {response.status}")
    except Exception as e:
        log.warning(f"[get_default_model] Failed to get default model: {type(e).__name__}: {e}")

    # Last resort
    log.warning(f"[get_default_model] no models detected — falling back to hardcoded Qwen/Qwen2.5-14B-Instruct")
    return "Qwen/Qwen2.5-14B-Instruct"

async def check_lm_studio(force_refresh: bool = False) -> Tuple[bool, List[str]]:
    """Check if LM Studio is running and return available models.

    Results are cached for ``_LM_STUDIO_CACHE_TTL`` seconds (default 60)
    to avoid hammering LM Studio's ``/v1/models`` endpoint every 5-second
    UI poll.  Pass ``force_refresh=True`` to bypass the cache (used by
    ``/api/lm_studio/models`` and the translation path).
    """
    if not USE_LM_STUDIO:
        log.debug("[check_lm_studio] USE_LM_STUDIO=0 — backend disabled")
        return False, []

    # Return cached result if still fresh
    if not force_refresh:
        elapsed = time.time() - _lm_studio_cache["timestamp"]
        if elapsed < _LM_STUDIO_CACHE_TTL:
            return _lm_studio_cache["result"]

    url = f"{LM_STUDIO_URL}/models"
    t0 = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                log.debug(
                    f"[check_lm_studio] GET {url} -> HTTP {response.status} "
                    f"in {time.time()-t0:.1f}s"
                )
                if response.status == 200:
                    data = await response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    log.debug(f"LM Studio running with models: {models}")
                    _lm_studio_cache["result"] = (True, models)
                    _lm_studio_cache["timestamp"] = time.time()
                    return True, models
                log.warning(f"[check_lm_studio] {url} returned HTTP {response.status}")
                _lm_studio_cache["result"] = (False, [])
                _lm_studio_cache["timestamp"] = time.time()
                return False, []
    except asyncio.TimeoutError:
        log.warning(f"[check_lm_studio] GET {url} timed out after {time.time()-t0:.1f}s")
        _lm_studio_cache["result"] = (False, [])
        _lm_studio_cache["timestamp"] = time.time()
        return False, []
    except Exception as e:
        # Connection refused / LM Studio not running. Keep at debug since the
        # UI polls this constantly and it's expected to be down sometimes.
        log.debug(f"[check_lm_studio] {url} not reachable: {type(e).__name__}: {e}")
        _lm_studio_cache["result"] = (False, [])
        _lm_studio_cache["timestamp"] = time.time()
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
    if not text or not text.strip():
        log.debug("[translate_text] empty text, returning as-is")
        return text

    # Build the translation prompt
    prompt = f"Translate the following text to {target_lang}:"

    if context_hint:
        prompt += f"\nContext: {context_hint}"

    prompt += f"\n\nText: {text}\n\nTranslation:"

    # Call LM Studio API. NOTE: LM_STUDIO_URL is the OpenAI-style base
    # (e.g. http://localhost:1234/v1), so we must append /chat/completions.
    # Earlier code POSTed to the raw env value, which either 404'd or hit
    # the root and silently fell back to the untranslated source text — a
    # prime "breaks silently" cause. Now we normalize the URL and log the
    # request/response so failures are visible.
    try:
        lm_studio_url = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')
        # Normalize: strip trailing slash and ensure it ends at the base (/v1)
        if lm_studio_url.endswith('/'):
            lm_studio_url = lm_studio_url.rstrip('/')
        # If someone already appended chat/completions in .env, keep it
        if not lm_studio_url.endswith('/chat/completions'):
            chat_url = f"{lm_studio_url}/chat/completions"
        else:
            chat_url = lm_studio_url

        log.debug(
            f"[translate_text] POST {chat_url} model={model} "
            f"tgt={target_lang} text={text[:50]!r}"
        )

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

            t0 = time.time()
            timeout_s = _lm_studio_timeout()
            async with session.post(
                chat_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                log.debug(
                    f"[translate_text] {chat_url} -> HTTP {response.status} "
                    f"in {time.time()-t0:.1f}s"
                )
                if response.status == 200:
                    data = await response.json()
                    try:
                        msg = data['choices'][0]['message']
                        translated = (msg.get('content', '') or '').strip()
                        # Thinking/reasoning models (Qwen 3.6, Gemma 4, etc.)
                        # sometimes put the answer in reasoning_content instead
                        # of content.  Fall back to it when content is empty.
                        if not translated:
                            translated = (msg.get('reasoning_content', '') or '').strip()
                    except (KeyError, IndexError, TypeError) as pe:
                        log.warning(
                            f"[translate_text] 200 but unexpected JSON shape: "
                            f"{str(data)[:300]} (parse error: {pe})"
                        )
                        return text
                    if translated:
                        log.debug(f"[translate_text] ok: {text[:40]!r} -> {translated[:40]!r}")
                        return translated
                    log.warning(f"[translate_text] 200 but empty content for {text[:50]!r}")
                    return text
                else:
                    error_text = await response.text()
                    log.warning(
                        f"[translate_text] LM Studio API error: {response.status} "
                        f"- {error_text[:300]}"
                    )
                    return text  # Fallback to original

    except asyncio.TimeoutError:
        log.warning(
            f"[translate_text] timeout after {_lm_studio_timeout()}s calling LM Studio "
            f"for {text[:50]!r} — model={model}. If this is a large model on CPU, "
            f"increase LM_STUDIO_TIMEOUT (and/or lower LM_STUDIO_MAX_CONCURRENT) in .env"
        )
        return text
    except Exception as e:
        log.warning(f"[translate_text] Translation error ({type(e).__name__}): {e}")
        return text  # Fallback to original


async def _translate_with_retry(
    text: str,
    target_lang: str,
    model: str = "qwen/qwen3-8b",
    context_hint: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Translate a single text with retry logic.

    This is a helper for batch translation fallback. It calls translate_text()
    multiple times if the result is the same as the input (soft failure).

    Returns the translated text, or the original if all retries fail.
    """
    if not text.strip():
        return text

    for attempt in range(max_retries):
        try:
            translated = await translate_text(text, target_lang, model, context_hint)
            text_stripped = text.strip()
            translated_stripped = (translated or '').strip()

            # Check if translation actually happened
            if translated_stripped and translated_stripped != text_stripped:
                return translated

            # Soft failure — retry with backoff
            if attempt < max_retries - 1:
                log.debug(
                    f"[_translate_with_retry] attempt {attempt+1} returned "
                    f"original text — retrying ({text[:40]!r})"
                )
                await asyncio.sleep(1 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"[_translate_with_retry] failed after {max_retries} attempts: "
                    f"{type(e).__name__}: {e}"
                )
            else:
                await asyncio.sleep(1 * (attempt + 1))

    return text


async def _translate_batch(
    texts: List[str],
    target_lang: str,
    model: str = "qwen/qwen3-8b",
    context_hint: Optional[str] = None,
) -> List[str]:
    """Translate multiple text segments in a single LLM call.

    Builds a numbered prompt with all segments, sends one request, and
    parses the numbered response back into individual translations.

    This is much more efficient than calling translate_text() N times,
    especially for large models where each call has significant overhead.

    Returns a list of translated strings. If parsing fails or the count
    doesn't match, returns the original texts as fallback.
    """
    if not texts:
        return []

    # Filter out empty/whitespace-only texts
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    if not non_empty_indices:
        return texts[:]

    non_empty_texts = [texts[i] for i in non_empty_indices]
    n = len(non_empty_texts)

    # Build numbered prompt
    numbered_segments = "\n".join(f"{i+1}. {text}" for i, text in enumerate(non_empty_texts))
    context_text = f"\nContext: {context_hint}" if context_hint else ""

    prompt = f"""Translate the following {n} text segments from the source language to {target_lang}.
Return each translation on its own line, prefixed with its number (e.g., "1. Translation here").
Do NOT add explanations, notes, or any extra text.
Keep the same tone and style as the original.{context_text}

Segments to translate:
{numbered_segments}

Translations:"""

    # Call LM Studio
    try:
        lm_studio_url = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')
        if lm_studio_url.endswith('/'):
            lm_studio_url = lm_studio_url.rstrip('/')
        if not lm_studio_url.endswith('/chat/completions'):
            chat_url = f"{lm_studio_url}/chat/completions"
        else:
            chat_url = lm_studio_url

        log.debug(f"[translate_batch] POST {chat_url} model={model} segments={n}")

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": f"You are a professional translator. Translate each numbered segment to {target_lang}. Return ONLY the numbered translations, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            t0 = time.time()
            timeout_s = _lm_studio_timeout()
            async with session.post(
                chat_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                elapsed = time.time() - t0
                log.debug(f"[translate_batch] {chat_url} -> HTTP {response.status} in {elapsed:.1f}s")

                if response.status != 200:
                    error_text = await response.text()
                    log.warning(f"[translate_batch] API error: {response.status} - {error_text[:300]}")
                    return texts[:]

                data = await response.json()
                try:
                    msg = data['choices'][0]['message']
                    content = (msg.get('content', '') or '').strip()
                    if not content:
                        content = (msg.get('reasoning_content', '') or '').strip()
                except (KeyError, IndexError, TypeError) as pe:
                    log.warning(f"[translate_batch] Unexpected JSON shape: {pe}")
                    return texts[:]

                if not content:
                    log.warning(f"[translate_batch] Empty response for {n} segments")
                    return texts[:]

                # Parse numbered translations
                # Match patterns like "1. text", "1) text", "1: text"
                parsed = {}
                current_num = None
                current_lines = []

                for line in content.split('\n'):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    # Check if this line starts a new numbered item
                    match = re.match(r'^(\d+)\s*[\.\)\:]\s*(.*)$', line_stripped)
                    if match:
                        # Save previous item
                        if current_num is not None:
                            parsed[current_num] = ' '.join(current_lines).strip()
                        current_num = int(match.group(1))
                        current_lines = [match.group(2)]
                    elif current_num is not None:
                        # Continuation line
                        current_lines.append(line_stripped)

                # Save last item
                if current_num is not None:
                    parsed[current_num] = ' '.join(current_lines).strip()

                log.debug(f"[translate_batch] Parsed {len(parsed)}/{n} translations")

                # Build result list
                result = texts[:]
                for i, orig_idx in enumerate(non_empty_indices):
                    num = i + 1
                    if num in parsed and parsed[num]:
                        result[orig_idx] = parsed[num]
                    else:
                        log.warning(f"[translate_batch] Missing translation for segment {num}")
                        # Keep original text as fallback

                return result

    except asyncio.TimeoutError:
        log.warning(f"[translate_batch] Timeout after {_lm_studio_timeout()}s for {n} segments")
        return texts[:]
    except Exception as e:
        log.warning(f"[translate_batch] Error ({type(e).__name__}): {e}")
        return texts[:]


async def translate_segments(
    segments: List[Dict],
    target_lang: str,
    model: str = "qwen/qwen3-8b",
    max_concurrent: int = 5,
    context_hint: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> List[Dict]:
    """Translate segments with progress callback.

    Uses batch translation by default: groups segments into chunks and
    translates each chunk in a single LLM call. Falls back to individual
    translation for segments that fail in batch mode.
    """
    if not segments:
        return segments

    # Resolve effective concurrency. LM Studio serves completions SERIALLY per
    # model (one inference queue), so firing N concurrent requests just makes
    # them wait their turn in LM Studio while each burns the full client-side
    # timeout on the clock. For large/slow models (esp. on CPU) this reliably
    # causes client timeouts + disconnects + silent fallback-to-source. So:
    #   - explicit LM_STUDIO_MAX_CONCURRENT env wins (1..20)
    #   - otherwise auto: 1 for large models (>=20B), 5 for small ones
    env_concurrency = _lm_studio_max_concurrent()
    if env_concurrency and env_concurrency > 0:
        max_concurrent = env_concurrency
    else:
        max_concurrent = 1 if _is_large_model(model) else 5
        log.debug(
            f"[translate_segments] auto concurrency: {max_concurrent} "
            f"(model={model!r} large={_is_large_model(model)})"
        )

    # Ensure max_concurrent is a valid integer
    try:
        max_concurrent = int(max_concurrent)
    except (ValueError, TypeError):
        logger.warning(f"Invalid max_concurrent value: {max_concurrent}, using default 5")
        max_concurrent = 5

    max_concurrent = max(1, min(max_concurrent, 20))

    total_segments = len(segments)
    chunk_size = _translation_chunk_size()
    num_chunks = math.ceil(total_segments / chunk_size)

    logger.info(
        f"Translating {total_segments} segments to {target_lang} "
        f"(chunks={num_chunks}, chunk_size={chunk_size}, max_concurrent={max_concurrent})"
    )
    if context_hint:
        logger.info(f"Using context hint: {context_hint}")
    log.debug(
        f"[translate_segments] enter: segments={total_segments} tgt={target_lang} "
        f"model={model} max_concurrent={max_concurrent} chunk_size={chunk_size}"
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    completed_segments = 0
    start_time = asyncio.get_event_loop().time()

    async def translate_chunk(chunk_idx: int, chunk_segments: List[Dict]) -> List[Dict]:
        """Translate a chunk of segments using batch translation."""
        nonlocal completed_segments

        async with semaphore:
            chunk_num = chunk_idx + 1
            texts = [seg.get('text', '') for seg in chunk_segments]

            log.debug(
                f"[translate_segments] chunk {chunk_num}/{num_chunks}: "
                f"{len(chunk_segments)} segments"
            )

            # Try batch translation
            translated_texts = await _translate_batch(
                texts, target_lang, model, context_hint
            )

            # Assign translations back to segments
            for seg, translated in zip(chunk_segments, translated_texts):
                original = seg.get('text', '').strip()
                translated_stripped = (translated or '').strip()

                # Check if translation actually happened
                if translated_stripped and translated_stripped != original:
                    seg['translated_text'] = translated
                else:
                    # Batch failed for this segment, try individual translation
                    log.debug(
                        f"[translate_segments] batch failed for segment, "
                        f"trying individual: {original[:40]!r}"
                    )
                    seg['translated_text'] = await _translate_with_retry(
                        original, target_lang, model, context_hint
                    )

            # Update progress
            completed_segments += len(chunk_segments)

            if progress_callback:
                try:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if completed_segments > 0:
                        avg_time_per_seg = elapsed / completed_segments
                        remaining = total_segments - completed_segments
                        eta_sec = avg_time_per_seg * remaining
                    else:
                        eta_sec = 0

                    # Progress message with chunk info
                    preview = chunk_segments[0].get('text', '')[:50] if chunk_segments else ''
                    msg = f"Chunk {chunk_num}/{num_chunks} — {preview!r}..."

                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(
                            completed_segments, total_segments, eta_sec, msg
                        )
                    else:
                        progress_callback(
                            completed_segments, total_segments, eta_sec, msg
                        )
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

            return chunk_segments

    # Split segments into chunks and translate concurrently
    chunks = [
        segments[i:i + chunk_size]
        for i in range(0, total_segments, chunk_size)
    ]

    tasks = [
        translate_chunk(idx, chunk)
        for idx, chunk in enumerate(chunks)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results (they're already modified in place, but gather returns them)
    translated_segments = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Chunk translation failed: {type(result).__name__}: {result}")
        elif isinstance(result, list):
            translated_segments.extend(result)

    # If gather failed completely, return original segments
    if not translated_segments:
        translated_segments = segments

    # Summarize how many segments actually got a real translation vs fell back
    # to the source text. A high fallback count (visible at INFO without the
    # debug flag) is the clearest sign translation silently failed.
    fallback_count = 0
    for orig, tr in zip(segments, translated_segments):
        if (tr.get('translated_text', '') or '').strip() == (orig.get('text', '') or '').strip():
            fallback_count += 1
    logger.info(
        f"Translation complete for {len(translated_segments)} segments "
        f"(fallback-to-source: {fallback_count})"
    )
    if fallback_count > 0:
        log.warning(
            f"[translate_segments] {fallback_count}/{len(translated_segments)} "
            f"segments fell back to the untranslated source text — check LM Studio "
            f"availability/model ({model}) and the [translate_text] logs above"
        )
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