"""Audit log — an append-only record of things worth being able to prove.

Deliberately *not* ``app/activity.py``. The activity feed is a bounded ring
buffer answering "what is happening now"; entries fall off the back and that is
fine, because nobody relies on them. An audit trail makes the opposite promise:
once written, an entry stays written. A record that a deque can silently drop
is worse than no record at all, because it looks like a trail and is not one.

So this is a JSONL file, opened in append mode, one line per entry, fsync'd on
write. Nothing here ever updates or deletes a line.

What gets logged is the security-relevant surface: API keys created and
revoked, webhooks added and removed, and destructive job actions. Reads are
not logged — in local mode there is one user and it would be noise.

Entries are redacted on the way in, like every other buffer that can be served
over HTTP.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.notices import redact

log = logging.getLogger("gochidubb.audit")

BASE = Path(__file__).parent.parent.resolve()
AUDIT_FILE = BASE / "audit.jsonl"

_lock = threading.Lock()


def record(action: str, *, actor: str = "local", target: Optional[str] = None,
           detail: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """Append one entry. Never raises — auditing must not break the action.

    A failure to write is itself logged, so a full disk shows up somewhere
    rather than silently producing a trail with holes in it.
    """
    entry: Dict[str, Any] = {
        "ts": time.time(),
        "action": action,
        "actor": actor,
    }
    if target:
        entry["target"] = target
    if detail:
        entry["detail"] = detail
    entry.update(extra)
    for k, v in list(entry.items()):
        if isinstance(v, str):
            entry[k] = redact(v)

    line = json.dumps(entry, ensure_ascii=False) + "\n"
    try:
        with _lock:
            # Append mode, flushed and fsync'd: an entry that is worth auditing
            # is worth surviving a crash a millisecond later.
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
    except Exception as e:
        log.warning(f"[audit] could not append entry ({action}): {e}")
    return entry


def recent(limit: int = 200) -> List[Dict[str, Any]]:
    """Newest-first entries. Malformed lines are skipped, never fatal."""
    if not AUDIT_FILE.exists():
        return []
    try:
        with _lock:
            lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        log.warning(f"[audit] could not read log: {e}")
        return []
    out: List[Dict[str, Any]] = []
    # Walk backwards so a large file costs only what the caller asked for.
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
        if len(out) >= max(0, limit):
            break
    return out


def count() -> int:
    if not AUDIT_FILE.exists():
        return 0
    try:
        with _lock:
            return sum(1 for ln in AUDIT_FILE.read_text(encoding="utf-8").splitlines()
                       if ln.strip())
    except Exception:
        return 0
