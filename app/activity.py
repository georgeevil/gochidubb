"""Activity stream — what the agent (or a human, or the server) just did.

Concept 1a of the SaaS redesign makes an activity feed the home screen: agent
runs appear as cards showing the tool calls they made, with job progress and
cost underneath, and one-line system events below that. Nothing in GoChiDUBB
recorded any of it, because the pieces live in different processes:

  * The MCP server (`tools/gochidubb_mcp.py`) and the CLI are separate
    processes. They reach this server over plain HTTP, so from the server's
    side an agent's `gochidubb.dub(...)` call is indistinguishable from the
    browser's `POST /api/dub` — same route, same shape.
  * Job progress *is* recorded, but only as the current state of each job.
    "Transcribe finished 40s ago" is not something the jobs dict can answer.

So two things are needed, and this module is the second half of both:

  1. Callers identify themselves. `GoChiDUBBClient` sends an
     `X-GoChiDUBB-Client` header, which the MCP server sets to
     `mcp/<agent>`; a middleware in server.py turns header-carrying requests
     into `tool_call` events here. Requests without the header — i.e. the UI —
     are not recorded as tool calls, because they are not.
  2. The job runner calls `record_job(...)` as jobs change state.

Kept deliberately in-memory and bounded, exactly like `app/logbuf.py`: this is
a feed of what is happening now, not an audit trail. The append-only audit log
(phase 6) is a separate, persisted concern — do not conflate them, because an
audit record that a ring buffer can silently drop is worse than none.

Everything is redacted on the way in via `pipeline.notices.redact`, for the
same reason logbuf does it: the buffer is served over HTTP and GOCHIDUBB_HOST
can put that on the network.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from pipeline.notices import redact

# Roughly a day of ordinary use. The feed only ever renders the newest page of
# this, so the cap is about bounding memory, not about the UI.
DEFAULT_CAPACITY = 600

# Event kinds, matching the filter tabs in the design (All · Runs · Tool calls
# · System). "run" groups the tool calls an agent made; "tool_call" is one
# call; "job" is a pipeline state change; "system" is everything else.
KINDS = ("run", "tool_call", "job", "system")

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=DEFAULT_CAPACITY)
_seq = 0


def _append(kind: str, **fields: Any) -> Dict[str, Any]:
    """Add one event and return it (already redacted)."""
    global _seq
    ev: Dict[str, Any] = {"kind": kind, "ts": time.time()}
    for k, v in fields.items():
        if v is not None:
            ev[k] = v
    # redact() walks strings; anything that is not a string passes through.
    for k, v in list(ev.items()):
        if isinstance(v, str):
            ev[k] = redact(v)
    with _lock:
        _seq += 1
        ev["id"] = _seq
        _events.append(ev)
    return ev


def record_tool_call(tool: str, actor: str, *, job_id: Optional[str] = None,
                     status: Optional[int] = None, detail: Optional[str] = None,
                     ms: Optional[float] = None) -> Dict[str, Any]:
    """One agent/CLI call, named as the tool it corresponds to.

    `actor` is the raw client identity from the header (e.g. "mcp/claude-code"
    or "cli"), so the feed can say who did it.
    """
    return _append("tool_call", tool=tool, actor=actor, job_id=job_id,
                   status=status, detail=detail,
                   ms=round(ms, 1) if ms is not None else None)


def record_job(job_id: str, status: str, *, title: Optional[str] = None,
               stage: Optional[str] = None, actor: Optional[str] = None,
               detail: Optional[str] = None) -> Dict[str, Any]:
    """A job changed state. Called from the job runner on transitions."""
    return _append("job", job_id=job_id, status=status, title=title,
                   stage=stage, actor=actor, detail=detail)


def record_system(title: str, *, severity: str = "info",
                  detail: Optional[str] = None) -> Dict[str, Any]:
    """Server-level event — startup, a webhook delivery, a budget warning."""
    return _append("system", title=title, severity=severity, detail=detail)


def recent(limit: int = 100, kinds: Optional[List[str]] = None,
           since_id: int = 0) -> List[Dict[str, Any]]:
    """Newest-first events, optionally filtered by kind.

    `since_id` lets the UI poll for only what it has not seen; it is compared
    against the monotonic per-event id, not a timestamp, so two events in the
    same millisecond cannot hide each other.
    """
    with _lock:
        items = list(_events)
    if kinds:
        want = set(kinds)
        items = [e for e in items if e.get("kind") in want]
    if since_id:
        items = [e for e in items if e["id"] > since_id]
    items.reverse()
    return items[:max(0, limit)]


def last_id() -> int:
    with _lock:
        return _seq


def clear() -> None:
    """Drop everything. For tests."""
    global _seq
    with _lock:
        _events.clear()
        _seq = 0
