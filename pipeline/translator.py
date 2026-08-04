import os
import json
import logging
import asyncio
import aiohttp
from collections import Counter
from typing import List, Dict, Optional, Callable, Tuple
import time
import re

log = logging.getLogger("gochidubb.translator")
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

# ── Batched translation ─────────────────────────────────────────────
# One request per subtitle line is the wrong shape for a model with a 16k+
# context window: a 10-minute video is ~150 segments, so ~150 round trips
# that each re-send the whole system prompt, re-pay prompt-processing and
# (on a thinking model) re-pay the reasoning warm-up — for ~10 words of
# payload. It also translates every line blind, so pronouns, tense and
# terminology drift between neighbouring subtitles.
#
# Instead we send N consecutive lines per request, numbered, and require the
# answer back with the same numbering. Alignment is never assumed: a reply
# that doesn't map 1:1 onto the input is discarded and the batch is split in
# half and retried, down to single lines. That matters more than the speed
# — a silently dropped line would shift every later subtitle onto the wrong
# audio segment.
#
# 0 = auto-size from LM_STUDIO_CONTEXT_LENGTH (or a conservative 8k guess).
TRANSLATE_BATCH_SIZE = int(os.getenv("TRANSLATE_BATCH_SIZE", "0") or 0)
TRANSLATE_BATCH_CHARS = int(os.getenv("TRANSLATE_BATCH_CHARS", "0") or 0)
# Lines of already-translated preceding dialogue shown to the model as
# context (not re-translated). 0 disables.
TRANSLATE_CONTEXT_LINES = int(os.getenv("TRANSLATE_CONTEXT_LINES", "2") or 0)
_DEFAULT_CONTEXT_LENGTH = 8192

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

def _load_user_glossary(target_lang: str = "ru") -> Dict:
    """Load user-defined glossary terms for one target language."""
    if not os.path.exists(USER_GLOSSARY_FILE):
        return {}
    try:
        with open(USER_GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Flatten the glossary into a simple dict
            terms = {}
            for domain in data.get("domains", []):
                if domain.get("target_lang") == target_lang:
                    terms.update(domain.get("terms", {}))
            return terms
    except Exception as e:
        log.warning(f"Failed to load user glossary: {e}")
        return {}


_GLOSSARY_CACHE: Dict[str, Dict] = {}


def _glossary_for(target_lang: str) -> Dict:
    """Glossary terms for a target language, read from disk once per process.

    The live translation path used to ignore the glossary entirely — only
    the unused `_translate_with_lm_studio` helper ever passed one — so
    presets/user_glossary.json had no effect on a real dub.
    """
    lang = (target_lang or "").strip() or "ru"
    if lang not in _GLOSSARY_CACHE:
        _GLOSSARY_CACHE[lang] = _load_user_glossary(lang)
    return _GLOSSARY_CACHE[lang]


def _glossary_block(glossary: Optional[Dict], limit: int = 60) -> str:
    """Render a flat {term: translation} map as a prompt section."""
    if not glossary:
        return ""
    terms = "\n".join(f"  - {k} → {v}" for k, v in list(glossary.items())[:limit])
    return f"\nAlways use these translations for these terms:\n{terms}\n"


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


class LMStudioFatalError(LMStudioError):
    """Retrying will not help — e.g. the requested model isn't loaded.

    Raised so the batch ladder gives up immediately instead of splitting a
    24-line batch into 24 doomed single-line requests (and then retrying
    each three times) for every batch in the job.
    """


class LMStudioBudgetError(LMStudioError):
    """The model used its whole output budget reasoning and never answered.

    The remedy is a BIGGER BUDGET, not a smaller batch. Measured on
    qwen/qwen3.6-27b with real subtitles, against a 4096-token budget:

        16 lines → 4094 reasoning tokens, 1 of answer   FAIL (226s)
         8 lines → 4094 reasoning tokens, 1 of answer   FAIL (245s)
         4 lines → 4094 reasoning tokens, 1 of answer   FAIL (243s)
        16 lines, budget 16384                          OK   (291s)

    The reasoning cost is essentially fixed per request, so halving the
    batch buys nothing and doubles how many times you pay it. Splitting on
    this error is actively harmful: 16→8→4→2→1 fails at every rung, five
    times over, at ~4 minutes each. That is how a 182-segment job spent
    1205s and translated zero lines.
    """


class LMStudioTimeoutError(LMStudioError):
    """The request ran past its clock.

    Unlike a budget exhaustion this one *does* scale with the batch, so a
    smaller batch is a sane response. Kept distinct because str() on the
    underlying TimeoutError is the empty string — logging it bare produced
    "Batch of 16 failed (attempt 1/2):" and told nobody anything.
    """


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
        raise LMStudioBudgetError(
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
            raise LMStudioBudgetError(
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
                raise LMStudioFatalError(
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

# Last model list logged at INFO by check_lm_studio. The status probe runs on
# every /api/jobs poll (~every 2s), so logging the full catalog each time buries
# the Logs tab. Only announce it when the set actually changes.
_last_logged_lm_models: Optional[List[str]] = None


async def check_lm_studio() -> Tuple[bool, List[str]]:
    """Check if LM Studio is running and return available models."""
    global _last_logged_lm_models

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
                    if models != _last_logged_lm_models:
                        log.info(f"LM Studio running with models: {models}")
                        _last_logged_lm_models = models
                    else:
                        log.debug(f"LM Studio running with models: {models}")
                    return True, models
                _last_logged_lm_models = None
                return False, []
    except Exception as e:
        log.debug(f"LM Studio not available: {e}")
        _last_logged_lm_models = None
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
    glossary: Optional[Dict] = None,
    prev_pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> str:
    """
    Translate a single text using LM Studio.

    This is the fallback path — bulk work goes through translate_segments(),
    which batches many lines into one request. A single line still ends up
    here when a batch's reply couldn't be aligned and was split down to one.

    Args:
        text: Text to translate
        target_lang: Target language code (e.g., 'ru', 'zh', 'fr')
        model: LM Studio model to use
        context_hint: Optional context information
        glossary: Optional {term: translation} map to enforce
        prev_pairs: Preceding (source, translation) lines, for continuity

    Returns:
        Translated text
    """
    if not text.strip():
        return text

    # Build the translation prompt
    prompt = f"Translate the following text to {target_lang}:"

    if context_hint:
        prompt += f"\nContext: {context_hint}"

    prompt += _glossary_block(glossary)
    prompt += _context_block(prev_pairs)

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


# ============================================================
# Batching
# ============================================================

def _batch_limits(batch_size: int = 0) -> Tuple[int, int]:
    """(max lines, max source chars) allowed in one translation request.

    Auto-sized from the model's context window when it is known. The line
    cap normally binds first — subtitle lines are short — but the char cap
    protects against a transcript of long monologue segments.
    """
    ctx = LM_STUDIO_CONTEXT_LENGTH or _DEFAULT_CONTEXT_LENGTH
    lines = batch_size or TRANSLATE_BATCH_SIZE or max(4, min(24, ctx // 512))
    chars = TRANSLATE_BATCH_CHARS or max(800, min(6000, ctx // 4))
    return max(1, min(int(lines), 64)), max(80, int(chars))


def _chunk_lines(texts: List[str], batch_size: int = 0) -> List[List[int]]:
    """Group line indices into batches under both the line and char caps."""
    max_lines, max_chars = _batch_limits(batch_size)
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_chars = 0
    for i, t in enumerate(texts):
        n = len(t)
        if cur and (len(cur) >= max_lines or cur_chars + n > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


def _batch_output_budget(texts: List[str]) -> int:
    """Output-token budget for a batch.

    LM_STUDIO_MAX_OUTPUT_TOKENS is the *reasoning* headroom: it exists
    because a thinking model spends tokens working before it writes a word
    of the answer. The answer itself needs its own allowance on top — so
    these are added, not max()'d.

    Taking max() was a real bug: a 16-line batch of real subtitles computed
    an answer estimate of 1852 tokens, max()'d back down to the 4096 floor,
    and qwen3.6-27b then spent 4094 of those 4096 tokens reasoning and
    emitted a single token of answer. Every batch in the job failed the
    same way.

    Non-Latin targets tokenize badly, so budget roughly 1.2 tokens per
    source character plus per-line numbering overhead.
    """
    chars = sum(len(t) for t in texts)
    answer = int(chars * 1.2) + 16 * len(texts) + 256
    return LM_STUDIO_MAX_OUTPUT_TOKENS + answer


def _batch_timeout(budget: int, concurrency: int = 1, stretch: int = 1) -> float:
    """Seconds to allow one batch request.

    Three multipliers, and leaving any of them out has burned an hour:

    * **budget** — LM_STUDIO_TIMEOUT is calibrated for a single segment,
      i.e. for one LM_STUDIO_MAX_OUTPUT_TOKENS. Generation time is roughly
      linear in tokens emitted, so a batch that may emit 3× as many needs
      3× the clock.
    * **concurrency** — LM Studio serves one model instance, so N in-flight
      requests each take ~N× as long in wall-clock. Sizing the clock for a
      solo request while running two of them is what made every batch of a
      182-segment job time out at ~400s and then split; the circuit breaker
      gave up after 8 and 166 lines came back in English.
    * **stretch** — set when a batch already timed out once and is being
      retried with a longer clock instead of being split.
    """
    scale = max(1.0, budget / max(LM_STUDIO_MAX_OUTPUT_TOKENS, 1))
    return LM_STUDIO_TIMEOUT * scale * max(1, concurrency) * max(1, stretch)


def _context_block(prev_pairs: Optional[List[Tuple[str, Optional[str]]]]) -> str:
    """Render preceding dialogue as read-only context for the prompt."""
    if not prev_pairs:
        return ""
    lines = []
    for src, tgt in prev_pairs:
        src = (src or "").strip()
        if not src:
            continue
        lines.append(f"  {src}" + (f"  →  {tgt.strip()}" if (tgt or "").strip() else ""))
    if not lines:
        return ""
    return (
        "\nPreceding dialogue, already translated — for continuity only, "
        "do NOT translate or repeat it:\n" + "\n".join(lines) + "\n"
    )


def _build_batch_prompt(
    texts: List[str],
    target_lang: str,
    context_hint: str = "",
    glossary: Optional[Dict] = None,
    prev_pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> str:
    """Build a numbered multi-line translation prompt."""
    numbered = "\n".join(f"{i + 1}. {t.strip()}" for i, t in enumerate(texts))
    n = len(texts)
    hint = f"\nThe material is about: {context_hint}\n" if context_hint else ""
    return f"""Translate each numbered subtitle line below into {target_lang}.
{hint}{_glossary_block(glossary)}{_context_block(prev_pairs)}
Rules:
- Output EXACTLY {n} lines, numbered 1 to {n}, in the same order.
- Format every line as "<number>. <translation>" and nothing else.
- Never merge, split, reorder, drop or add lines — line {n} of your output
  must be the translation of line {n} of the input.
- These lines are consecutive speech from one video: keep terminology,
  tense, register and speaker voice consistent across them.
- Keep each translation close in length to its source line — it has to be
  spoken in the same amount of time.
- If a line cannot be translated, repeat its source text for that number.
- No explanations, no notes, no commentary, no blank lines.

Lines to translate:
{numbered}

Translation:"""


# "1. text" / "1) text" / "[1] text" / "1: text" / "**1.** text" — models
# pick their own numbering style, so accept the common ones.
_NUMBERED_LINE_RE = re.compile(
    r"^\s*[*#]*\s*[\[(]?(\d{1,3})[\])]?\s*(?:[.):\]|\-]\s*|\s+)\**\s*(.*)$"
)


def _parse_numbered_lines(raw: str, expected: int) -> Optional[List[str]]:
    """Parse a numbered batch reply into exactly `expected` translations.

    Returns None when the reply cannot be mapped 1:1 onto the input — the
    caller then splits the batch rather than guessing, because a shifted
    mapping would put every later line onto the wrong audio segment.

    Numbers are only accepted in strict ascending sequence. That is what
    catches the dangerous failure (the model merged two lines and numbered
    1,2,4…) and it also stops a translation that happens to start with
    "2." from being mistaken for a new item. A reply with MORE numbered
    lines than we asked for is rejected too — that is a line the model
    split in two, which shifts everything after it.
    """
    out: List[Optional[str]] = [None] * expected
    cur = -1
    for line in _strip_reasoning(raw).splitlines():
        m = _NUMBERED_LINE_RE.match(line)
        if m and int(m.group(1)) == cur + 2:
            if cur + 1 >= expected:
                return None
            cur += 1
            out[cur] = m.group(2).strip()
            continue
        # Continuation of the current line (a model wrapped it, or emitted a
        # stray blank). Anything before line 1 is preamble — drop it.
        text = line.strip()
        if cur >= 0 and text:
            out[cur] = f"{out[cur]} {text}".strip()
    if cur + 1 != expected or any(not (v or "").strip() for v in out):
        return None
    return [_clean_translation(v) or v.strip() for v in out]


def _alignment_is_plausible(sources: List[str], targets: List[str]) -> bool:
    """Reject a parsed batch whose lines are wildly the wrong size.

    A model that ignores the format and writes commentary usually still
    numbers it, so the shape alone isn't proof. Length is: a subtitle
    translation is never 5× its source plus 80 characters.
    """
    for src, tgt in zip(sources, targets):
        if len(tgt) > max(80, len(src) * 5):
            return False
    return True


async def _translate_batch(
    texts: List[str],
    target_lang: str,
    model: str,
    context_hint: str,
    glossary: Optional[Dict],
    prev_pairs: Optional[List[Tuple[str, Optional[str]]]],
    budget_multiplier: int = 1,
    concurrency: int = 1,
    timeout_stretch: int = 1,
) -> Optional[List[str]]:
    """One request for many lines. None means "reply didn't line up".

    Raises LMStudioBudgetError (retry bigger) or LMStudioTimeoutError
    (retry smaller) so the caller can tell those two apart.
    """
    prompt = _build_batch_prompt(texts, target_lang, context_hint, glossary, prev_pairs)
    budget = min(_batch_output_budget(texts) * max(1, budget_multiplier),
                 _MAX_ABSOLUTE_BUDGET)
    try:
        raw = await lm_studio_chat(
            prompt,
            model=model,
            system_prompt=(
                f"You are a professional subtitle translator working into "
                f"{target_lang}. You reply with the same numbered lines you were "
                f"given, one translation per line, and nothing else."
            ),
            temperature=0.1,
            max_output_tokens=budget,
            timeout=_batch_timeout(budget, concurrency, timeout_stretch),
        )
    except (asyncio.TimeoutError, TimeoutError) as e:
        # str(TimeoutError()) is "" — without the type name this logged as
        # "Batch of 16 failed (attempt 1/2):" and told nobody anything.
        raise LMStudioTimeoutError(
            f"no reply for {len(texts)} lines within "
            f"{_batch_timeout(budget, concurrency, timeout_stretch):.0f}s "
            f"({type(e).__name__})"
        ) from e
    parsed = _parse_numbered_lines(raw, len(texts))
    if parsed is None:
        logger.warning(
            f"[translate] Batch of {len(texts)} came back mis-numbered; "
            f"splitting. Reply started: {(raw or '')[:120]!r}"
        )
        return None
    if not _alignment_is_plausible(texts, parsed):
        logger.warning(
            f"[translate] Batch of {len(texts)} parsed but a line is "
            f"implausibly long (model likely added commentary); splitting"
        )
        return None
    return parsed


class _Circuit:
    """Stops a whole job from grinding through doomed requests.

    When LM Studio is unreachable (or the wrong model is selected) every
    request fails the same way. Without this, a 150-segment job would split
    each batch down to single lines and retry each of those three times —
    hundreds of timeouts, tens of minutes, same outcome. After enough
    consecutive failures we stop asking and let the segments come back
    untranslated, which the retry-failed-only UI can fix in one click.
    """

    LIMIT = 8

    def __init__(self):
        self.consecutive_failures = 0
        self.reason = ""

    @property
    def open(self) -> bool:
        return self.consecutive_failures >= self.LIMIT

    def record_failure(self, err: Exception) -> None:
        self.consecutive_failures += 1
        # Always include the type: str() is empty for TimeoutError, which
        # made the log read "giving up ... Last error: " and hid the cause.
        self.reason = f"{type(err).__name__}: {err}".rstrip(": ")
        if self.consecutive_failures == self.LIMIT:
            logger.error(
                f"[translate] {self.LIMIT} consecutive request failures — "
                f"giving up on the rest of this job. Last error: "
                f"{self.reason[:200]}"
            )

    def record_success(self) -> None:
        self.consecutive_failures = 0


# A budget is a ceiling, not a reservation — a model that doesn't need the
# tokens doesn't generate them — so escalating in big steps costs nothing
# except on the model that really is looping. Measured on qwen/qwen3.6-27b,
# 16 real subtitle lines: it hit exactly budget-2 reasoning tokens at both
# 4096 and 5948 (truncated mid-thought both times) and only completed at
# 16384. Doubling would have burned two ~5 minute requests to get there.
_BUDGET_ESCALATION = 4
# Never ask for more output than the model could hold. LM Studio rejects or
# truncates past the loaded context; 0 means we weren't told, so stay modest.
_MAX_ABSOLUTE_BUDGET = LM_STUDIO_CONTEXT_LENGTH or 32768


class _BudgetTuner:
    """How much output budget this model actually needs, learned per job.

    A model that thinks past its budget does it on every request, not just
    one — so once a batch comes back as pure reasoning we raise the budget
    and keep the larger figure for the rest of the job. Without the shared
    multiplier each of a job's twelve batches would rediscover the same
    thing at ~5 minutes a go; with it, a 182-segment job wastes two
    requests instead of twenty-four.

    Escalation stops at MAX_MULTIPLIER: past that the model isn't
    budget-starved, it's looping, and the batch is better split.
    """

    MAX_MULTIPLIER = 16

    def __init__(self):
        self.multiplier = 1

    def escalate(self) -> bool:
        """Raise the budget. False when there is no headroom left."""
        if self.multiplier >= self.MAX_MULTIPLIER:
            return False
        self.multiplier = min(self.multiplier * _BUDGET_ESCALATION,
                              self.MAX_MULTIPLIER)
        logger.warning(
            f"[translate] Model spent its whole output budget reasoning; "
            f"raising it to {self.multiplier}× for the rest of this job"
        )
        return True


async def _translate_single(
    text: str,
    target_lang: str,
    model: str,
    context_hint: str,
    glossary: Optional[Dict],
    prev_pairs: Optional[List[Tuple[str, Optional[str]]]],
    semaphore: asyncio.Semaphore,
    circuit: "_Circuit",
) -> str:
    """Last resort: the original one-line-per-request path, with retries."""
    for attempt in range(3):
        if circuit.open:
            break
        try:
            async with semaphore:
                # Re-check under the lock: while this call was queued, the
                # requests ahead of it may have tripped the breaker.
                if circuit.open:
                    break
                out = await translate_text(
                    text, target_lang, model,
                    context_hint=context_hint,
                    glossary=glossary,
                    prev_pairs=prev_pairs,
                )
            if (out or "").strip():
                circuit.record_success()
                return out
            logger.warning(f"Empty translation for: {text[:50]}...")
        except LMStudioFatalError:
            raise
        except LMStudioBudgetError as e:
            # The server answered — with nothing but reasoning. Retrying a
            # single line won't change that, and counting it as a backend
            # failure would abandon the rest of the job over one stubborn
            # line. Give up on this line only.
            logger.warning(
                f"[translate] Model would not answer within its budget for "
                f"a single line ({e}); leaving it untranslated"
            )
            break
        except Exception as e:
            circuit.record_failure(e)
            if attempt == 2:
                logger.error(f"Translation failed after 3 attempts: {e}")
            else:
                logger.warning(f"Translation error (attempt {attempt + 1}): {e}")
        if attempt < 2:
            await asyncio.sleep(1 + attempt)
    # Leaving the source text in place is deliberate: _is_untranslated() in
    # server.py spots it and the UI can offer "retry failed segments only".
    return text


async def _translate_group(
    texts: List[str],
    target_lang: str,
    model: str,
    context_hint: str,
    glossary: Optional[Dict],
    prev_fn: Callable[[], List[Tuple[str, Optional[str]]]],
    semaphore: asyncio.Semaphore,
    on_done: Callable[[int], "asyncio.Future"],
    circuit: "_Circuit",
    tuner: "_BudgetTuner",
    concurrency: int = 1,
) -> List[str]:
    """Translate a group of lines, halving it on any alignment failure.

    The semaphore is held only for the duration of a request, never across
    the recursive split — otherwise a single stuck batch would deadlock a
    concurrency-1 server against itself.
    """
    if circuit.open:
        # Backend is down. Hand back the source text so the segments are
        # marked untranslated, instead of splitting into more dead requests.
        await on_done(len(texts))
        return list(texts)

    if len(texts) == 1:
        out = await _translate_single(
            texts[0], target_lang, model, context_hint, glossary,
            prev_fn(), semaphore, circuit,
        )
        await on_done(1)
        return [out]

    # Enough attempts to walk the budget from 1x to MAX_MULTIPLIER, plus one
    # ordinary retry. Each escalation is a real request, so this is bounded
    # deliberately rather than looping until something works.
    timeout_stretch = 1
    for attempt in range(5):
        if circuit.open:
            break
        try:
            async with semaphore:
                # Re-check under the lock: while this batch was queued, the
                # requests ahead of it may have tripped the breaker.
                if circuit.open:
                    break
                result = await _translate_batch(
                    texts, target_lang, model, context_hint, glossary, prev_fn(),
                    budget_multiplier=tuner.multiplier,
                    concurrency=concurrency,
                    timeout_stretch=timeout_stretch,
                )
            if result is not None:
                circuit.record_success()
                await on_done(len(texts))
                return result
            # A mis-aligned reply is the model's format failing, not the
            # server's — the circuit stays closed and we split instead.
            circuit.record_success()
            break
        except LMStudioFatalError:
            raise
        except LMStudioBudgetError as e:
            # Budget starvation, not a size problem: the reasoning cost is
            # near-constant per request, so a smaller batch fails the same
            # way (measured 16/8/4 lines, all identical). Give it more room
            # and re-send the same batch. Not a circuit failure — the server
            # answered, it just answered with nothing but thinking.
            circuit.record_success()
            if tuner.escalate():
                continue
            logger.warning(
                f"[translate] Batch of {len(texts)} still out of budget at "
                f"{tuner.multiplier}× ({e}); splitting as a last resort"
            )
            break
        except LMStudioTimeoutError as e:
            # Give the same batch a longer clock before splitting. Splitting
            # is the wrong first move on a model whose reasoning cost is
            # per-request: two halves each pay it again, so the job gets
            # slower, not faster. Only a batch that misses even the stretched
            # deadline is worth breaking up.
            if timeout_stretch == 1:
                timeout_stretch = 3
                logger.warning(
                    f"[translate] Batch of {len(texts)} timed out ({e}); "
                    f"retrying it with a {timeout_stretch}x longer clock"
                )
                continue
            logger.warning(
                f"[translate] Batch of {len(texts)} timed out again ({e}); "
                f"splitting"
            )
            circuit.record_failure(e)
            break
        except Exception as e:
            circuit.record_failure(e)
            logger.warning(
                f"[translate] Batch of {len(texts)} failed "
                f"(attempt {attempt + 1}): {type(e).__name__}: {e}"
            )
            if attempt >= 1:
                break
            await asyncio.sleep(1)

    mid = len(texts) // 2
    left = await _translate_group(
        texts[:mid], target_lang, model, context_hint, glossary,
        prev_fn, semaphore, on_done, circuit, tuner, concurrency,
    )
    right = await _translate_group(
        texts[mid:], target_lang, model, context_hint, glossary,
        lambda: list(zip(texts[:mid], left))[-max(TRANSLATE_CONTEXT_LINES, 1):],
        semaphore, on_done, circuit, tuner, concurrency,
    )
    return left + right


async def translate_segments(
    segments: List[Dict],
    target_lang: str,
    model: str = "qwen/qwen3-8b",
    max_concurrent: int = 5,
    context_hint: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    batch_size: int = 0,
) -> List[Dict]:
    """Translate segments in batches, with progress callback.

    Each segment gets `translated_text`. A segment whose translation failed
    outright keeps its source text there, which server.py treats as
    "untranslated" so the stage can be retried for those lines only.
    """
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
    context_hint = context_hint or ""
    glossary = _glossary_for(target_lang)

    total_segments = len(segments)

    # ── Build the work list ───────────────────────────────────────────
    # Empty segments never need a request. Short lines that repeat verbatim
    # ("Yeah.", "Okay, so…") are translated once and reused — transcripts
    # are full of them. Longer lines are always translated in place so they
    # keep their surrounding context.
    keys: List[Optional[str]] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        keys.append(" ".join(text.lower().split()) if text else None)
    repeats = {
        k for k, c in Counter(k for k in keys if k).items()
        if c > 1 and len(k) <= 60
    }

    unique_texts: List[str] = []
    slot_of: Dict[int, int] = {}      # segment index → unique-line index
    first_seen: Dict[str, int] = {}
    for i, s in enumerate(segments):
        key = keys[i]
        if key is None:
            s["translated_text"] = ""
            continue
        if key in repeats and key in first_seen:
            slot_of[i] = first_seen[key]
            continue
        slot = len(unique_texts)
        unique_texts.append((s.get("text") or "").strip())
        slot_of[i] = slot
        if key in repeats:
            first_seen[key] = slot

    if not unique_texts:
        logger.info("Nothing to translate — all segments are empty")
        return segments

    batches = _chunk_lines(unique_texts, batch_size)
    max_lines, max_chars = _batch_limits(batch_size)
    logger.info(
        f"Translating {total_segments} segments to {target_lang}: "
        f"{len(unique_texts)} unique line(s) in {len(batches)} request(s) "
        f"(≤{max_lines} lines / ≤{max_chars} chars each, "
        f"max_concurrent={max_concurrent}, model={model})"
    )
    if context_hint:
        logger.info(f"Using context hint: {context_hint}")
    if glossary:
        logger.info(f"Applying {len(glossary)} glossary term(s) for {target_lang}")

    results: List[Optional[str]] = [None] * len(unique_texts)
    total_lines = len(unique_texts)
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    start_time = asyncio.get_event_loop().time()

    async def on_done(n_lines: int) -> None:
        """Progress is counted in lines translated, out of unique lines —
        the caller derives its percentage from the pair we hand it."""
        nonlocal completed_count
        completed_count = min(completed_count + n_lines, total_lines)
        if not progress_callback:
            return
        try:
            elapsed = asyncio.get_event_loop().time() - start_time
            eta_sec = 0
            if completed_count > 0:
                per_item = elapsed / completed_count
                eta_sec = per_item * (total_lines - completed_count)
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(completed_count, total_lines, eta_sec)
            else:
                progress_callback(completed_count, total_lines, eta_sec)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    def prev_context(batch_no: int) -> List[Tuple[str, Optional[str]]]:
        """Tail of the previous batch, evaluated when the prompt is built.

        At concurrency 1 the previous batch is already done, so the model
        sees real translations. When batches run in parallel it may only
        see the source lines — still enough to disambiguate a pronoun.
        """
        if batch_no == 0 or TRANSLATE_CONTEXT_LINES <= 0:
            return []
        tail = batches[batch_no - 1][-TRANSLATE_CONTEXT_LINES:]
        return [(unique_texts[i], results[i]) for i in tail]

    circuit = _Circuit()
    tuner = _BudgetTuner()

    async def run_batch(batch_no: int, idxs: List[int]) -> None:
        out = await _translate_group(
            [unique_texts[i] for i in idxs],
            target_lang, model, context_hint, glossary,
            lambda: prev_context(batch_no),
            semaphore,
            on_done,
            circuit,
            tuner,
            max_concurrent,
        )
        for i, t in zip(idxs, out):
            results[i] = t

    await asyncio.gather(*(
        run_batch(bn, idxs) for bn, idxs in enumerate(batches)
    ))

    for i, s in enumerate(segments):
        slot = slot_of.get(i)
        if slot is None:
            continue
        s["translated_text"] = results[slot] or (s.get("text") or "")

    logger.info(f"Translation complete for {len(segments)} segments")
    return segments


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