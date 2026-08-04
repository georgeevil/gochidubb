"""Trend scout — discover popular YouTube videos worth dubbing.

Primary source: mostviewed.today's JSON API (undocumented; verified live
2026-08-04). All endpoints return a bare JSON list of entries shaped like:

    {id (YouTube video id), title, description, channel_title, thumb_url,
     duration ("189s"), category_id (YouTube taxonomy int; 10 = Music),
     is_short (0/1), view_count, captured_at, rank, prev_rank, rank_delta,
     is_new, is_re}

Variants: /api/leaderboard/global, /api/leaderboard/category/{slug},
/api/leaderboard/shorts, /api/trending/{country} (country is a name slug:
"usa", "uk", "india", "germany", "canada", "brazil", "australia" — ISO codes
are rejected with {"error": "Invalid country code"}). A browser User-Agent is
required; the default python UA can be blocked with a 403.

Because the API is undocumented it may vanish or change shape — every failure
falls back to `yt-dlp --flat-playlist --dump-json` on YouTube's trending feed,
and the response reports which source actually served (`"source"` key).

Results are cached in memory for 15 minutes per (source, category, country,
include_shorts). The clock is injectable (module attr `time_fn`) for tests.
"""
import asyncio
import json
import logging
import subprocess
import sys
import time

log = logging.getLogger("gochidubb.scout")

API_BASE = "https://mostviewed.today/api"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
REQUEST_TIMEOUT_SEC = 10
FALLBACK_TIMEOUT_SEC = 60
CACHE_TTL_SEC = 15 * 60
MUSIC_CATEGORY_ID = 10  # YouTube category taxonomy: 10 = Music
YT_TRENDING_URL = "https://www.youtube.com/feed/trending"
YT_TOP_MUSIC_PLAYLIST = (
    "https://www.youtube.com/playlist?list=PL4fGSI1pDJn6O1LS0XSdF3RyO0Rq_LDeI"
)
# yt-dlp fallback feeds, tried in order. YouTube removed /feed/trending in
# mid-2025 (verified dead 2026-08-04: "does not exist ... redirected to
# home page") but it's kept first in case it resurfaces for some regions or
# with cookies. The YouTube-maintained "Top 100 Music Videos Global"
# playlist still works and is trending-shaped; everything in it is music,
# hence the is_music hint.
FALLBACK_FEEDS = [
    (YT_TRENDING_URL, False),
    (YT_TOP_MUSIC_PLAYLIST, True),
]

# Injectable clock so tests can drive cache expiry deterministically.
time_fn = time.time

# (source, category, country, include_shorts) -> (fetched_at, fetched_limit, result)
_cache: dict = {}

# mostviewed.today wants country *name* slugs, not ISO codes. Accept the
# common ISO spellings anyway and translate.
_COUNTRY_ALIASES = {
    "us": "usa", "usa": "usa", "united-states": "usa",
    "gb": "uk", "uk": "uk", "united-kingdom": "uk",
    "in": "india", "india": "india",
    "de": "germany", "germany": "germany",
    "ca": "canada", "canada": "canada",
    "br": "brazil", "brazil": "brazil",
    "au": "australia", "australia": "australia",
}


def parse_duration_sec(value):
    """Parse a duration into whole seconds.

    Tolerates: int/float (189, 189.0), "189s", "189", "3:09", "1:02:03".
    Returns None when unparseable.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    if not s:
        return None
    if s.endswith("s"):
        s = s[:-1]
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        if len(nums) not in (2, 3):
            return None
        sec = 0
        for n in nums:
            sec = sec * 60 + n
        return sec
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_int(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_candidate(raw: dict) -> dict:
    """Map one mostviewed.today entry to the normalized candidate shape."""
    video_id = raw.get("id") or ""
    category_id = _to_int(raw.get("category_id"))
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": raw.get("title") or "",
        "channel": raw.get("channel_title") or "",
        "duration_sec": parse_duration_sec(raw.get("duration")),
        "view_count": _to_int(raw.get("view_count")),
        "rank": _to_int(raw.get("rank")),
        "rank_delta": _to_int(raw.get("rank_delta")),
        "category_id": category_id,
        "is_short": bool(raw.get("is_short")),
        "is_music": category_id == MUSIC_CATEGORY_ID,
        "thumb_url": raw.get("thumb_url") or "",
    }


def _endpoint_for(category, country) -> str:
    if country:
        slug = _COUNTRY_ALIASES.get(str(country).strip().lower(),
                                    str(country).strip().lower())
        return f"{API_BASE}/trending/{slug}"
    if category:
        slug = str(category).strip().lower()
        if slug == "shorts":
            return f"{API_BASE}/leaderboard/shorts"
        return f"{API_BASE}/leaderboard/category/{slug}"
    return f"{API_BASE}/leaderboard/global"


async def _fetch_mostviewed(category, country, limit: int) -> list:
    """GET the mostviewed.today endpoint, return the raw entry list."""
    import aiohttp

    url = _endpoint_for(category, country)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as sess:
        async with sess.get(url, params={"limit": limit}) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            data = await resp.json(content_type=None)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{url}: {data['error']}")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response shape from {url}: "
                           f"{type(data).__name__}")
    return data


def _ytdlp_cmd_prefix() -> list:
    """Base yt-dlp invocation, reusing downloader helpers when present.

    `_base_cmd` / `_cookie_args` land with the downloader-metadata work;
    until then fall back to `_find_ytdlp` (always present) with no cookies.
    """
    from pipeline import downloader as _dl

    base = getattr(_dl, "_base_cmd", None)
    if callable(base):
        cmd = list(base())
    else:
        ytdlp = _dl._find_ytdlp()
        cmd = [ytdlp] if ytdlp else [sys.executable, "-m", "yt_dlp"]
    cookies = getattr(_dl, "_cookie_args", None)
    if callable(cookies):
        cmd += list(cookies())
    return cmd


def parse_flat_playlist_lines(text: str, include_shorts: bool = False,
                              is_music: bool = False) -> list:
    """Parse `yt-dlp --flat-playlist --dump-json` output (one JSON per line)
    into the normalized candidate shape. Position in the feed becomes rank.
    `is_music` is a feed-level hint (flat entries carry no category)."""
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        video_id = entry.get("id") or ""
        if not video_id:
            continue
        url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        is_short = "/shorts/" in url
        if is_short and not include_shorts:
            continue
        candidates.append({
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": entry.get("title") or "",
            "channel": entry.get("uploader") or entry.get("channel") or "",
            "duration_sec": parse_duration_sec(entry.get("duration")),
            "view_count": _to_int(entry.get("view_count")),
            "rank": len(candidates) + 1,
            "rank_delta": None,
            "category_id": None,
            "is_short": is_short,
            "is_music": bool(is_music),
            "thumb_url": "",
        })
    return candidates


def _fetch_ytdlp_fallback(limit: int, include_shorts: bool) -> list:
    """Blocking yt-dlp fallback — run via asyncio.to_thread.

    Tries FALLBACK_FEEDS in order; first feed that yields candidates wins."""
    prefix = _ytdlp_cmd_prefix()
    errors = []
    for feed_url, is_music in FALLBACK_FEEDS:
        cmd = prefix + [
            "--flat-playlist",
            "--dump-json",
            "--playlist-end", str(max(limit * 2, limit + 5)),  # headroom for shorts filtering
            "--no-warnings",
            feed_url,
        ]
        log.info(f"[scout] yt-dlp fallback: {feed_url}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=FALLBACK_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            errors.append(f"{feed_url}: timed out after {FALLBACK_TIMEOUT_SEC}s")
            continue
        if result.returncode != 0 and not result.stdout.strip():
            errors.append(f"{feed_url}: {(result.stderr or '').strip()[:200]}")
            continue
        candidates = parse_flat_playlist_lines(
            result.stdout, include_shorts=include_shorts, is_music=is_music)
        if candidates:
            return candidates
        errors.append(f"{feed_url}: no candidates in output")
    raise RuntimeError("yt-dlp trending failed: " + "; ".join(errors))


async def fetch_trending(source: str = "mostviewed", category=None,
                         country=None, limit: int = 20,
                         include_shorts: bool = False) -> dict:
    """Fetch trending video candidates.

    Returns {"source": "mostviewed" | "ytdlp_fallback", "candidates": [...]}
    where each candidate is normalize_candidate()'s shape. Raises RuntimeError
    only when the primary AND the yt-dlp fallback both fail.
    """
    limit = max(1, int(limit or 20))
    key = (source, category or None, country or None, bool(include_shorts))
    now = time_fn()

    hit = _cache.get(key)
    if hit is not None:
        fetched_at, fetched_limit, cached = hit
        if now - fetched_at < CACHE_TTL_SEC and fetched_limit >= limit:
            return {"source": cached["source"],
                    "candidates": cached["candidates"][:limit]}

    primary_err = None
    candidates = None
    served_by = None
    if source == "mostviewed":
        try:
            raw = await _fetch_mostviewed(category, country, limit)
            candidates = [normalize_candidate(e) for e in raw]
            if not include_shorts:
                candidates = [c for c in candidates if not c["is_short"]]
            served_by = "mostviewed"
        except Exception as e:
            primary_err = e
            log.warning(f"[scout] mostviewed.today fetch failed: {e}")

    if candidates is None:
        try:
            candidates = await asyncio.to_thread(
                _fetch_ytdlp_fallback, limit, include_shorts)
            served_by = "ytdlp_fallback"
        except Exception as e:
            raise RuntimeError(
                "Trend discovery failed — "
                f"mostviewed.today: {primary_err or 'skipped'}; "
                f"yt-dlp fallback: {e}") from e

    result = {"source": served_by, "candidates": candidates}
    _cache[key] = (now, limit, result)
    return {"source": served_by, "candidates": candidates[:limit]}


def clear_cache() -> None:
    _cache.clear()


def find_cached_candidate(video_id: str):
    """Look a video_id up across all cached trending results (any key)."""
    for _fetched_at, _limit, cached in _cache.values():
        for cand in cached.get("candidates", []):
            if cand.get("video_id") == video_id:
                return cand
    return None


def annotate_already_processed(candidates: list, jobs: dict) -> None:
    """Mark candidates whose video_id matches an existing job's meta.

    Sets `already_processed` on every candidate and `existing_job_id` on
    matches. Matches against job["meta"]["video_id"] (populated by the
    downloader-metadata probe on URL submissions).
    """
    by_video_id = {}
    for job in jobs.values():
        vid = (job.get("meta") or {}).get("video_id")
        if vid:
            by_video_id.setdefault(vid, job.get("id"))
    for cand in candidates:
        job_id = by_video_id.get(cand.get("video_id"))
        cand["already_processed"] = job_id is not None
        if job_id is not None:
            cand["existing_job_id"] = job_id
