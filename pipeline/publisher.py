"""Publisher — upload finished dubs to video platforms (VK first, RuTube stub).

Abstractions:
  PublishMeta       platform-agnostic metadata for one upload
  Uploader (ABC)    validate_credentials / search_similar / upload
  VKUploader        real implementation over the VK API (v5.199, aiohttp)
  RuTubeUploader    planned — every method raises PublishError
  get_uploader()    factory by platform name

Also home of the duplicate-check scorer shared with the scout (Phase 4C):
  normalize_title / score_match / classify_matches — pure, unit-tested.

VK upload flow (confirmed against VK API docs / vk_api reference client):
  1. video.save(name, description, is_private, group_id?) ->
     {upload_url, owner_id, video_id, ...}
  2. multipart POST of the file to upload_url under field name "video_file"
     (streamed from disk — the file is never read fully into memory)
  3. upload response JSON contains {size, video_id}
  4. final URL: https://vk.com/video{owner_id}_{video_id}

Credentials come from app.secrets (secrets.json / VK_ACCESS_TOKEN env) —
never from UserConfig, which is exposed by the unauthenticated /api/config.
"""
import asyncio
import difflib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

log = logging.getLogger("gochidubb.publisher")

VK_API_VERSION = "5.199"
VK_API_BASE = "https://api.vk.com/method/"

# Duration gate for the duplicate scorer: candidates whose duration differs
# from ours by more than this are treated as unlikely matches.
DURATION_TOLERANCE_SEC = 5.0

# Min seconds between VK API calls. User tokens are limited to ~3 req/s;
# 0.35s keeps a sustained loop just under that.
_MIN_CALL_INTERVAL = 0.35

# Titles longer than this are truncated for video.save's `name` param.
_VK_TITLE_MAX = 128


class PublishError(Exception):
    """Publishing failed. The message is user-actionable — show it verbatim.

    `code` carries the platform error code when one exists (VK error_code).
    """

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


@dataclass
class PublishMeta:
    """Platform-agnostic metadata for one upload.

    is_private defaults to True so a first upload never goes public by
    accident — the human approver flips it deliberately.
    """

    title: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    is_private: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  Pure helpers — duplicate scoring (shared with scout / Phase 4C)
# ═══════════════════════════════════════════════════════════════════════

def normalize_title(s: str) -> str:
    """Lowercase, strip punctuation (unicode-aware), collapse whitespace."""
    s = (s or "").lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def score_match(
    candidate_title: str,
    candidate_duration,
    title: str,
    duration_sec,
    alt_title: Optional[str] = None,
) -> float:
    """Similarity score 0..1 between a search hit and our video.

    Title similarity = difflib.SequenceMatcher ratio over normalized titles,
    taking the best of `title` and `alt_title` (e.g. original vs translated).
    Duration gates the score: when both durations are known and differ by
    more than DURATION_TOLERANCE_SEC, the score is capped at ratio * 0.5.
    """
    cand = normalize_title(candidate_title)
    best = 0.0
    for t in (title, alt_title):
        if not t:
            continue
        ratio = difflib.SequenceMatcher(None, cand, normalize_title(t)).ratio()
        if ratio > best:
            best = ratio
    if candidate_duration is not None and duration_sec is not None:
        try:
            if abs(float(candidate_duration) - float(duration_sec)) > DURATION_TOLERANCE_SEC:
                best *= 0.5
        except (TypeError, ValueError):
            pass
    return max(0.0, min(1.0, best))


def classify_matches(matches: list) -> str:
    """Verdict over a list of {..., score} match dicts.

    "likely_duplicate" if any score >= 0.85, "possible" if any >= 0.6,
    else "clear" (including an empty list).
    """
    best = max((m.get("score", 0.0) for m in matches), default=0.0)
    if best >= 0.85:
        return "likely_duplicate"
    if best >= 0.6:
        return "possible"
    return "clear"


# ── Publish state machine (Phase 3C) ───────────────────────────────────
# The server's publish routes and upload worker validate every status change
# through can_transition() so the full matrix lives in one testable place.
#
#   stage    -> "staged"     allowed from anything except an in-flight upload
#                            (re-staging overwrites the previous publish dict)
#   approve  -> "approved"   only from "staged" or "failed" (retry path)
#   cancel   -> "cancelled"  from staged/approved/failed; never mid-upload
#   upload   -> "uploading"  worker-only, from "approved"

PUBLISH_STATUSES = (
    "staged", "approved", "uploading", "uploaded", "failed", "cancelled",
)

# Statuses that belong in the review inbox (GET /api/publish/pending).
PUBLISH_PENDING_STATUSES = ("staged", "approved", "uploading", "failed")

_PUBLISH_ACTION_SOURCES = {
    # None means "no publish dict yet" — staging a fresh job is always legal.
    "stage": {None, "staged", "approved", "uploaded", "failed", "cancelled"},
    "approve": {"staged", "failed"},
    "cancel": {"staged", "approved", "failed"},
    "upload": {"approved"},
}


def can_transition(from_status, action: str) -> bool:
    """True when `action` is a legal publish transition from `from_status`.

    `from_status` is the current publish.status (or None/"" when the job has
    never been staged). Unknown actions are never legal.
    """
    allowed = _PUBLISH_ACTION_SOURCES.get(action)
    if allowed is None:
        return False
    return (from_status or None) in allowed


def build_publish_meta(job: dict) -> PublishMeta:
    """Derive PublishMeta from a finished job dict.

    Title preference: translated title (job["meta_translated"]["title"],
    Phase 3D) -> job["title"] -> source_label -> job id.
    Description: translated description head + attribution line built from
    cfg.publish_description_template ({source_url} placeholder).
    Tags: job meta tags, capped at 10.
    """
    from app.config import cfg  # lazy: pipeline convention (see diarizer.py)

    mt = job.get("meta_translated") or {}
    meta = job.get("meta") or {}

    title = (
        (mt.get("title") or "").strip()
        or (job.get("title") or "").strip()
        or (job.get("source_label") or "").strip()
        or str(job.get("id", ""))
    )

    parts = []
    desc_head = (mt.get("description") or "").strip()
    if desc_head:
        parts.append(desc_head)

    source = str(job.get("source") or "")
    source_url = source if source.startswith("http") else str(meta.get("webpage_url") or "")
    if source_url:
        template = getattr(cfg, "publish_description_template", "Original: {source_url}")
        # .replace, not .format — a user template with stray braces must not crash.
        parts.append(template.replace("{source_url}", source_url))

    tags = [str(t) for t in (mt.get("tags") or meta.get("tags") or [])][:10]

    return PublishMeta(
        title=title,
        description="\n\n".join(parts),
        tags=tags,
        is_private=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Pure helpers — VK response parsing (kept aiohttp-free for testability)
# ═══════════════════════════════════════════════════════════════════════

_VK_ERROR_HINTS = {
    5: ("Your VK access token is invalid or expired. Generate a new token "
        "with the `video` scope and save it via POST /api/secrets."),
    9: ("VK flood control: too many similar actions in a row. "
        "Wait a few minutes before retrying."),
    15: ("VK denied access. The token likely lacks the `video` scope, or "
         "you don't have upload rights for the target group."),
    29: "VK rate limit reached for this method. Wait a while and retry.",
}

# Error codes that indicate a permissions/scope problem (not a hard failure)
# when probing video access during credential validation.
_VK_PERMISSION_CODES = (7, 15)


def parse_vk_envelope(payload) -> object:
    """Unwrap VK's response envelope.

    Success is {"response": ...} -> returns the response value.
    Failure is {"error": {"error_code", "error_msg"}} -> raises PublishError
    with a friendly message for the common codes.
    """
    if not isinstance(payload, dict):
        raise PublishError("VK returned an unexpected non-object response.")
    if "error" in payload:
        err = payload.get("error") or {}
        code = err.get("error_code")
        msg = err.get("error_msg") or "unknown error"
        detail = f"VK API error {code}: {msg}"
        hint = _VK_ERROR_HINTS.get(code)
        if hint:
            detail = f"{detail} — {hint}"
        raise PublishError(detail, code=code)
    if "response" not in payload:
        raise PublishError("VK response missing both 'response' and 'error'.")
    return payload["response"]


def parse_vk_save_response(resp) -> dict:
    """Validate the video.save response; must contain an upload_url."""
    if not isinstance(resp, dict) or not resp.get("upload_url"):
        raise PublishError(
            "VK video.save did not return an upload URL. "
            "Check that the token has the `video` scope."
        )
    return {
        "upload_url": resp["upload_url"],
        "owner_id": resp.get("owner_id"),
        "video_id": resp.get("video_id"),
    }


def parse_vk_upload_response(payload) -> dict:
    """Validate the JSON body returned by the upload_url POST.

    Expected shape: {"size": <bytes>, "video_id": <id>} (older docs call the
    id field "vid"). Anything with an "error" key, or missing the id, means
    the upload did not register.
    """
    if not isinstance(payload, dict) or payload.get("error"):
        err = payload.get("error") if isinstance(payload, dict) else payload
        raise PublishError(
            f"VK upload server rejected the file: {err!r}. "
            "Retry the publish to get a fresh upload URL."
        )
    video_id = payload.get("video_id") or payload.get("vid")
    if not video_id:
        raise PublishError(
            "VK upload finished but returned no video id — the upload "
            "likely did not register. Retry the publish."
        )
    return {"video_id": video_id, "size": payload.get("size")}


def vk_video_url(owner_id, video_id) -> str:
    return f"https://vk.com/video{owner_id}_{video_id}"


# ═══════════════════════════════════════════════════════════════════════
#  Uploader ABC + implementations
# ═══════════════════════════════════════════════════════════════════════

class Uploader(ABC):
    """One video platform. Implementations must be stateless between calls
    apart from credentials/throttling — the publish state machine (Phase 3C)
    owns all job state."""

    name: str = "abstract"

    @abstractmethod
    async def validate_credentials(self) -> dict:
        """Return identity/scope info ({user, video_scope_ok, ...}) or raise
        PublishError with an actionable message."""

    @abstractmethod
    async def search_similar(self, title: str, duration_sec=None) -> list:
        """Search the platform for videos similar to (title, duration_sec).
        Returns [{title, url, duration, score}] sorted by score desc."""

    @abstractmethod
    async def upload(self, video_path: str, meta: PublishMeta) -> dict:
        """Upload the file. Returns {platform_id, url, ...} on success."""


class VKUploader(Uploader):
    """VK Video over the public API (v5.199) with a user or group token.

    Credentials default to app.secrets (vk_access_token / vk_group_id,
    overridable by VK_ACCESS_TOKEN / VK_GROUP_ID env vars).
    """

    name = "vk"

    def __init__(self, access_token: Optional[str] = None, group_id: Optional[str] = None):
        from app.secrets import get_secret  # lazy: keeps module import light

        self._token = access_token if access_token is not None else get_secret("vk_access_token")
        self._group_id = group_id if group_id is not None else get_secret("vk_group_id")
        self._throttle_lock = asyncio.Lock()
        self._last_call = 0.0

    # ── Plumbing ───────────────────────────────────────────────────────
    async def _throttle(self) -> None:
        """Enforce a minimum interval between VK API calls (user tokens are
        limited to ~3 requests/second)."""
        async with self._throttle_lock:
            now = time.monotonic()
            wait = self._last_call + _MIN_CALL_INTERVAL - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def _call(self, method: str, **params):
        """GET one VK API method, unwrap the envelope, return the response."""
        if not self._token:
            raise PublishError(
                "No VK access token configured. Obtain a token with the "
                "`video` scope and save it via POST /api/secrets "
                '(key "vk_access_token").'
            )
        await self._throttle()
        query = {k: str(v) for k, v in params.items() if v is not None}
        query["access_token"] = self._token
        query["v"] = VK_API_VERSION
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(VK_API_BASE + method, params=query) as r:
                    try:
                        payload = await r.json(content_type=None)
                    except (json.JSONDecodeError, ValueError, aiohttp.ClientError):
                        raise PublishError(
                            f"VK returned a non-JSON response for {method} "
                            f"(HTTP {r.status}). Try again later."
                        )
        except PublishError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            raise PublishError(
                f"Could not reach the VK API ({method}): {e}. "
                "Check your network connection and retry."
            ) from e
        log.debug(f"[vk] {method} ok")
        return parse_vk_envelope(payload)

    # ── Uploader interface ─────────────────────────────────────────────
    async def validate_credentials(self) -> dict:
        """Identify the token owner and probe video-scope access.

        users.get works with any valid user token; a video.get probe that
        fails with a permission error means the token lacks the `video`
        scope (VK has no direct scope-introspection endpoint).
        """
        users = await self._call("users.get")
        user = users[0] if isinstance(users, list) and users else {}
        display = " ".join(
            p for p in (user.get("first_name"), user.get("last_name")) if p
        ) or str(user.get("id", ""))

        video_scope_ok = True
        scope_error = ""
        try:
            await self._call("video.get", count=1)
        except PublishError as e:
            if e.code in _VK_PERMISSION_CODES:
                video_scope_ok = False
                scope_error = str(e)
            else:
                raise

        result = {
            "user": display,
            "user_id": user.get("id"),
            "video_scope_ok": video_scope_ok,
        }
        if scope_error:
            result["video_scope_error"] = scope_error
        return result

    async def search_similar(self, title: str, duration_sec=None) -> list:
        """video.search by title, scored client-side against (title, duration)."""
        resp = await self._call("video.search", q=title, count=20)
        items = resp.get("items", []) if isinstance(resp, dict) else []
        matches = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cand_title = it.get("title", "") or ""
            cand_dur = it.get("duration")
            matches.append({
                "title": cand_title,
                "url": vk_video_url(it.get("owner_id"), it.get("id")),
                "duration": cand_dur,
                "score": round(score_match(cand_title, cand_dur, title, duration_sec), 3),
            })
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    async def upload(self, video_path: str, meta: PublishMeta) -> dict:
        """video.save -> streamed multipart POST -> {platform_id, url, ...}.

        No auto-retry in v1: VK upload URLs are single-use, so a failed or
        interrupted upload CANNOT be resumed against the same upload_url.
        To retry, the caller simply calls upload() again — that re-runs
        video.save and gets a fresh URL (the orphaned video.save entry on
        VK stays empty and is garbage-collected by VK).
        """
        path = Path(video_path)
        if not path.is_file():
            raise PublishError(f"Video file not found: {video_path}")

        save_params = {
            "name": (meta.title or path.stem)[:_VK_TITLE_MAX],
            "description": meta.description or "",
            "is_private": int(bool(meta.is_private)),
        }
        if self._group_id:
            save_params["group_id"] = self._group_id
        saved = parse_vk_save_response(await self._call("video.save", **save_params))

        size_mb = path.stat().st_size / 1048576
        log.info(f"[vk] uploading {path.name} ({size_mb:.1f} MB, private={meta.is_private})")

        # Generous total budget for multi-GB files; sock_read only bounds the
        # gap between reads on the response, not overall transfer time, so a
        # slow-but-alive upload is never starved mid-stream.
        timeout = aiohttp.ClientTimeout(total=1800, sock_connect=60, sock_read=300)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                with open(path, "rb") as fh:
                    form = aiohttp.FormData()
                    # aiohttp streams file objects chunk-by-chunk — the video
                    # is never read fully into memory.
                    form.add_field(
                        "video_file", fh,
                        filename=path.name,
                        content_type="video/mp4",
                    )
                    async with session.post(saved["upload_url"], data=form) as r:
                        body = await r.text()
                        http_status = r.status
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            raise PublishError(
                f"Upload to VK failed mid-transfer: {e}. Retry the publish — "
                "each retry requests a fresh upload URL (VK upload URLs are "
                "single-use, the interrupted one cannot be resumed)."
            ) from e

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            raise PublishError(
                f"VK upload server returned a non-JSON response "
                f"(HTTP {http_status}). Retry the publish."
            )
        uploaded = parse_vk_upload_response(payload)

        owner_id = saved.get("owner_id")
        video_id = uploaded.get("video_id") or saved.get("video_id")
        url = vk_video_url(owner_id, video_id)
        log.info(f"[vk] upload complete: {url}")
        return {
            "platform_id": f"{owner_id}_{video_id}",
            "url": url,
            "owner_id": owner_id,
            "video_id": video_id,
            "size": uploaded.get("size"),
        }


class RuTubeUploader(Uploader):
    """Placeholder — RuTube publishing is planned but not implemented yet."""

    name = "rutube"

    _MSG = "RuTube support is planned but not implemented yet"

    async def validate_credentials(self) -> dict:
        raise PublishError(self._MSG)

    async def search_similar(self, title: str, duration_sec=None) -> list:
        raise PublishError(self._MSG)

    async def upload(self, video_path: str, meta: PublishMeta) -> dict:
        raise PublishError(self._MSG)


def get_uploader(platform: str) -> Uploader:
    """Factory: 'vk' -> VKUploader, 'rutube' -> RuTubeUploader (stub)."""
    p = (platform or "").strip().lower()
    if p == "vk":
        return VKUploader()
    if p == "rutube":
        return RuTubeUploader()
    raise PublishError(
        f"Unknown publish platform: {platform!r}. Supported: 'vk' "
        "('rutube' is planned)."
    )
