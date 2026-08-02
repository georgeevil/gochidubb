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
# LM_STUDIO_URL is written differently by different people (and by our own
# .env.example): "http://localhost:1234", ".../v1", ".../api/v1", or even a
# full ".../v1/chat/completions". Callers used to concatenate paths onto it
# directly, so a base ending in "/v1" produced a POST to "/v1" — LM Studio
# answers that with "Unexpected endpoint or method. (POST /v1). Returning
# 200 anyway", the JSON has no "choices" key, and every segment silently
# fell back to its untranslated source text. Normalize once, here, and build
# every endpoint from the resulting host root.
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "")  # Auto-detect if empty
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "1") == "1"

# Thinking models (qwen3.x, deepseek-r1, gpt-oss …) spend output tokens on
# reasoning before they emit a single word of the answer. With the old
# 500-token cap the budget was exhausted mid-thought and the response came
# back with no message content at all — the "partially fails" symptom.
LM_STUDIO_TIMEOUT = float(os.getenv("LM_STUDIO_TIMEOUT", "300"))
# Measured on qwen/qwen3.6-27b: a single 8-word sentence produced 1606
# reasoning tokens before 14 tokens of answer. The old 500-token cap
# truncated that mid-thought, so the response contained no message at all —
# which is exactly what "translation partially/completely fails" looked like.
LM_STUDIO_MAX_OUTPUT_TOKENS = int(os.getenv("LM_STUDIO_MAX_OUTPUT_TOKENS", "4096"))
# "off" | "low" | "medium" | "high" | "on" | "" (don't send the field).
# Translation doesn't benefit from chain-of-thought, and disabling it makes
# a 27B thinking model roughly an order of magnitude faster per segment.
# LM Studio errors when a model doesn't support the setting, so we drop the
# field and retry once if that happens (see _LM_REASONING_SUPPORTED).
LM_STUDIO_REASONING = os.getenv("LM_STUDIO_REASONING", "off").strip().lower()
LM_STUDIO_CONTEXT_LENGTH = int(os.getenv("LM_STUDIO_CONTEXT_LENGTH", "0"))  # 0 = server default

# Some models accept `reasoning: "off"` and then think anyway — qwen3.6-27b
# does exactly that. Qwen's documented in-prompt switch does work, and it
# takes the same request from 1620 output tokens to 17, so we append it for
# Qwen models when reasoning is meant to be off. Gated by model family
# because on a model that doesn't know the token it would just be prompt
# text the model might echo. Set empty to disable.
LM_STUDIO_NO_THINK_SUFFIX = os.getenv("LM_STUDIO_NO_THINK_SUFFIX", "/no_think")
_NO_THINK_MODELS = ("qwen3", "qwen/qwen3", "qwq")
# Warn once (not per segment) when a model ignores the reasoning setting.
_LM_WARNED_IGNORED_REASONING = False


def _supports_no_think(model: str) -> bool:
    m = (model or "").lower()
    return bool(LM_STUDIO_NO_THINK_SUFFIX) and any(k in m for k in _NO_THINK_MODELS)


def _lm_studio_host(url: str) -> str:
    """Reduce any LM Studio URL spelling to its host root (no trailing path).

    >>> _lm_studio_host("http://localhost:1234/v1")
    'http://localhost:1234'
    >>> _lm_studio_host("http://localhost:1234/v1/chat/completions")
    'http://localhost:1234'
    """
    base = (url or "").strip().rstrip("/")
    for suffix in ("/api/v1/chat", "/v1/chat/completions", "/api/v1", "/v1",
                   "/chat/completions", "/api/v0"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/") or "http://localhost:1234"


LM_STUDIO_HOST = _lm_studio_host(LM_STUDIO_URL)
# Native LM Studio REST API (LM Studio 0.3.x+): richer response shape that
# separates reasoning from the answer, which is exactly what we need for a
# thinking model — we can drop the reasoning and keep the translation.
LM_STUDIO_CHAT_ENDPOINT = f"{LM_STUDIO_HOST}/api/v1/chat"
# OpenAI-compatible endpoint, used as a fallback on older LM Studio builds.
LM_STUDIO_COMPAT_ENDPOINT = f"{LM_STUDIO_HOST}/v1/chat/completions"
LM_STUDIO_MODELS_ENDPOINT = f"{LM_STUDIO_HOST}/v1/models"

# Resolved lazily on first use, then cached for the process:
#   _LM_USE_NATIVE      — False once /api/v1/chat has 404'd (old LM Studio)
#   _LM_SEND_REASONING  — False once the server rejects the reasoning field
_LM_USE_NATIVE: Optional[bool] = None
_LM_SEND_REASONING = bool(LM_STUDIO_REASONING)

# LM Studio serves one model instance at a time; firing 5 concurrent
# requests at a 27B model just queues them behind each other while making
# every individual request look slow enough to trip the timeout.
LM_STUDIO_MAX_CONCURRENT = int(os.getenv("LM_STUDIO_MAX_CONCURRENT", "0"))  # 0 = caller's choice

_THINK_BLOCK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

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


class LMStudioError(RuntimeError):
    """LM Studio was reachable but did not return usable content."""


def _strip_reasoning(text: str) -> str:
    """Remove inline <think> blocks some models emit in their message text."""
    return _THINK_BLOCK_RE.sub("", text or "").strip()


def _parse_native_response(data: Dict) -> str:
    """Pull the answer out of a POST /api/v1/chat response.

    `output` is a list of typed items — messages, reasoning, tool calls. A
    thinking model returns its chain-of-thought as separate `reasoning`
    items, so taking only the `message` items is what makes this endpoint
    work where the OpenAI-compatible one hands back a wall of <think>.
    """
    global _LM_WARNED_IGNORED_REASONING
    stats_in = data.get("stats") or {}
    reasoning_tokens = stats_in.get("reasoning_output_tokens") or 0
    if (LM_STUDIO_REASONING == "off" and reasoning_tokens > 50
            and not _LM_WARNED_IGNORED_REASONING):
        _LM_WARNED_IGNORED_REASONING = True
        log.warning(
            f"[lm_studio] Model ignored reasoning='off' and spent "
            f"{reasoning_tokens} tokens thinking about one segment. "
            f"Translation will be slow; consider a non-thinking model "
            f"(e.g. gemma / qwen2.5) for bulk translation."
        )

    items = data.get("output") or []
    message_parts, reasoning_len = [], 0
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            content = item.get("content")
            if isinstance(content, str):
                message_parts.append(content)
            elif isinstance(content, list):
                # Defensive: some builds nest content as [{type, text}]
                message_parts.extend(
                    c.get("content") or c.get("text") or ""
                    for c in content if isinstance(c, dict)
                )
        elif kind == "reasoning":
            reasoning_len += len(item.get("content") or "")

    text = _strip_reasoning("".join(message_parts))
    if text:
        return text

    # No answer. The usual cause is a thinking model that used its entire
    # output budget on reasoning — say so, instead of an empty-string error.
    stats = data.get("stats") or {}
    if reasoning_len or stats.get("reasoning_output_tokens"):
        raise LMStudioError(
            f"model returned only reasoning "
            f"({stats.get('reasoning_output_tokens', '?')} reasoning tokens, "
            f"{stats.get('total_output_tokens', '?')} total) — raise "
            f"LM_STUDIO_MAX_OUTPUT_TOKENS (currently "
            f"{LM_STUDIO_MAX_OUTPUT_TOKENS}) or set LM_STUDIO_REASONING=off"
        )
    raise LMStudioError(f"empty response (output={items!r:.200})")


def _lm_studio_error(body: str) -> Dict:
    """Parse LM Studio's error envelope.

    Two shapes exist: the structured
    `{"error": {"message", "type", "param", "code"}}` for request problems,
    and a bare `{"error": "Unexpected endpoint or method. (POST /v1)"}` for
    an unrecognized route. Returns {} when the body isn't either.
    """
    try:
        err = (json.loads(body) or {}).get("error")
    except (ValueError, TypeError):
        return {}
    if isinstance(err, dict):
        return err
    if isinstance(err, str):
        return {"message": err}
    return {}


def _parse_compat_response(data: Dict) -> str:
    """Pull the answer out of an OpenAI-compatible /v1/chat/completions body."""
    choices = data.get("choices")
    if not choices:
        raise LMStudioError(f"no 'choices' in response: {str(data)[:200]}")
    message = (choices[0] or {}).get("message") or {}
    # Newer LM Studio builds expose reasoning as a sibling field; older ones
    # inline it in the content as <think>…</think>.
    text = _strip_reasoning(message.get("content") or "")
    if not text:
        if message.get("reasoning_content") or message.get("reasoning"):
            raise LMStudioError(
                "model returned only reasoning — raise "
                "LM_STUDIO_MAX_OUTPUT_TOKENS or set LM_STUDIO_REASONING=off"
            )
        raise LMStudioError("empty message content")
    return text


async def lm_studio_chat(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: float = 0.1,
    max_output_tokens: int = 0,
    timeout: float = 0,
) -> str:
    """Send one prompt to LM Studio and return the model's answer.

    Prefers the native `POST /api/v1/chat` API and falls back to the
    OpenAI-compatible `POST /v1/chat/completions` on older LM Studio builds.
    Raises LMStudioError (rather than returning the input) so callers can
    retry — silently returning the source text is what made a broken
    endpoint look like "translation partially failed".
    """
    global _LM_USE_NATIVE, _LM_SEND_REASONING

    max_output_tokens = max_output_tokens or LM_STUDIO_MAX_OUTPUT_TOKENS
    timeout = timeout or LM_STUDIO_TIMEOUT
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    if LM_STUDIO_REASONING == "off" and _supports_no_think(model):
        prompt = f"{prompt} {LM_STUDIO_NO_THINK_SUFFIX}"

    async def _post(session, url, payload):
        async with session.post(url, json=payload, timeout=client_timeout) as r:
            body = await r.text()
            return r.status, body

    async with aiohttp.ClientSession() as session:
        # ── Native API ────────────────────────────────────────────────
        if _LM_USE_NATIVE is not False:
            payload = {
                "model": model,
                "input": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "stream": False,
                "store": False,
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            if LM_STUDIO_CONTEXT_LENGTH:
                payload["context_length"] = LM_STUDIO_CONTEXT_LENGTH
            if _LM_SEND_REASONING:
                payload["reasoning"] = LM_STUDIO_REASONING

            status, body = await _post(session, LM_STUDIO_CHAT_ENDPOINT, payload)

            if status == 200:
                if _LM_USE_NATIVE is None:
                    log.info(f"[lm_studio] Using native API: {LM_STUDIO_CHAT_ENDPOINT}")
                    _LM_USE_NATIVE = True
                return _parse_native_response(json.loads(body))

            err = _lm_studio_error(body)

            # The model doesn't accept the reasoning setting — drop it and
            # retry once, then remember for the rest of the process.
            if _LM_SEND_REASONING and err.get("param") == "reasoning":
                log.warning(
                    f"[lm_studio] Model '{model}' rejected reasoning="
                    f"{LM_STUDIO_REASONING!r} ({err.get('message', '')[:120]}); "
                    f"retrying without it"
                )
                _LM_SEND_REASONING = False
                payload.pop("reasoning", None)
                status, body = await _post(session, LM_STUDIO_CHAT_ENDPOINT, payload)
                if status == 200:
                    _LM_USE_NATIVE = True
                    return _parse_native_response(json.loads(body))
                err = _lm_studio_error(body)

            # LM Studio answers 404 for BOTH "no such route" and "no such
            # model". Only the former means we're talking to an older build
            # that lacks the native API — a bad model name must surface as
            # itself, not silently downgrade the endpoint for the whole run.
            if err.get("code") == "model_not_found":
                raise LMStudioError(
                    f"model '{model}' is not available in LM Studio — "
                    f"{err.get('message', '')[:200]}"
                )

            if status in (404, 405) and "unexpected endpoint" in body.lower():
                log.info(
                    f"[lm_studio] {LM_STUDIO_CHAT_ENDPOINT} not available "
                    f"({status}); falling back to the OpenAI-compatible endpoint"
                )
                _LM_USE_NATIVE = False
            else:
                raise LMStudioError(
                    f"HTTP {status} from /api/v1/chat: "
                    f"{err.get('message') or body[:300]}"
                )

        # ── OpenAI-compatible fallback ────────────────────────────────
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        status, body = await _post(session, LM_STUDIO_COMPAT_ENDPOINT, payload)
        if status != 200:
            raise LMStudioError(
                f"HTTP {status} from {LM_STUDIO_COMPAT_ENDPOINT}: {body[:300]}"
            )
        return _parse_compat_response(json.loads(body))


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
    
    for attempt in range(max_retries):
        try:
            result = await lm_studio_chat(
                prompt,
                model=model_name,
                system_prompt=f"You are a professional translator from "
                              f"{source_lang} to {target_lang}.",
                temperature=0.3,
            )
            result = _clean_translation(result)
            if result:
                log.debug(f"Translated: {text[:50]}... -> {result[:50]}...")
                return result
            log.warning(f"Empty translation for: {text[:50]}...")
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
                LM_STUDIO_MODELS_ENDPOINT,
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
                LM_STUDIO_MODELS_ENDPOINT,
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
    if not text.strip():
        return text

    # Build the translation prompt
    prompt = f"Translate the following text to {target_lang}:"

    if context_hint:
        prompt += f"\nContext: {context_hint}"

    prompt += f"\n\nText: {text}\n\nTranslation:"

    # Errors propagate on purpose: translate_segments() retries three times
    # and only then falls back to the source text. Swallowing the error here
    # made every failure look like a successful no-op translation.
    translated = await lm_studio_chat(
        prompt,
        model=model,
        system_prompt="You are a professional translator. Reply with the "
                      "translation only — no explanations, no notes.",
        temperature=0.1,
    )
    return _clean_translation(translated) or text


def _clean_translation(raw: str) -> str:
    """Strip the scaffolding models like to echo back around a translation."""
    result = _strip_reasoning(raw)
    # Drop a leading "Translation:" label and any echoed source block.
    result = re.sub(r'^\s*(Translation|Перевод)\s*:\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'^\s*Text( to translate)?\s*:.*$', '', result, flags=re.MULTILINE)
    # A model that doesn't recognize the /no_think control token may echo it.
    result = re.sub(r'/no_?think\b', '', result, flags=re.IGNORECASE)
    # Some models wrap the whole answer in quotes or a code fence. Only strip
    # a matched pair that wraps the entire string — a translation that merely
    # *contains* a quoted phrase must keep its punctuation.
    result = result.strip().strip('`').strip()
    _QUOTE_PAIRS = {'"': '"', "'": "'", '«': '»', '“': '”', '„': '“', '”': '”'}
    if len(result) > 1 and _QUOTE_PAIRS.get(result[0]) == result[-1]:
        inner = result[1:-1]
        if result[-1] not in inner:
            result = inner
    return result.strip()


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

    # LM Studio serves a single model instance, so parallel requests queue up
    # server-side while each one's clock runs — with a big thinking model that
    # turns into timeouts on the later segments. LM_STUDIO_MAX_CONCURRENT
    # (already in .env.example) caps it; honour it here.
    if USE_LM_STUDIO and LM_STUDIO_MAX_CONCURRENT > 0:
        if LM_STUDIO_MAX_CONCURRENT < max_concurrent:
            logger.info(
                f"Limiting translation concurrency to "
                f"{LM_STUDIO_MAX_CONCURRENT} (LM_STUDIO_MAX_CONCURRENT)"
            )
        max_concurrent = LM_STUDIO_MAX_CONCURRENT

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