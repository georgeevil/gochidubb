"""In-memory log capture, so nothing important is console-only.

TachiDUBB is a local app: the user launches it, a browser opens, and the
terminal is usually behind the browser window or gone entirely (`start.sh` runs
`serverctl foreground`, which writes no log file at all). Anything that only
reaches stderr effectively did not happen.

Two sources have to be captured, and only one of them is logging:

  1. `logging` records — everything the pipeline emits via `log.warning(...)`.
  2. Raw `print()` from third-party libraries. This is not a corner case: the
     message that motivated this module is printed by pyannote
     (`pyannote/audio/utils/hf_hub.py:96`) straight to stdout, carrying the
     exact URL the user must visit. A logging handler alone would miss it.

So a `RingHandler` goes on the root logger and `TeeStream` wraps stdout/stderr.

The trap: `logging.basicConfig()` installs a StreamHandler that *writes to*
`sys.stderr`. Tee stderr naively and every record is captured twice — once as a
record, once as the handler's own output. `install()` therefore rebinds the
existing stream handlers to the original stream objects before teeing, so
handler output bypasses the tee entirely.

Everything is redacted on the way in (see pipeline/notices.redact). The buffer
is served over HTTP, and TACHIDUBB_HOST can put that on the network, so a
credential that reaches the deque is one bind address away from the LAN.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from pipeline.notices import redact

DEFAULT_CAPACITY = 2000

# Level names in the order the UI filters on them.
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}


class LogRing:
    """Bounded, thread-safe, monotonically-numbered log store.

    `seq` never resets and never repeats, so a client can poll with
    `since_seq` and know it missed nothing — except what eviction dropped,
    which `dropped` reports honestly rather than pretending the gap is not
    there.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._buf: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self._dropped = 0

    @property
    def capacity(self) -> int:
        return self._buf.maxlen or 0

    def add(self, level: str, logger: str, message: str,
            ts: Optional[float] = None) -> Dict[str, Any]:
        entry = {
            "seq": 0,
            "ts": ts if ts is not None else time.time(),
            "level": (level or "INFO").upper(),
            "logger": logger or "",
            "message": redact(message),
        }
        with self._lock:
            self._seq += 1
            entry["seq"] = self._seq
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(entry)
        return entry

    def entries(self, level: str = "", limit: int = 300,
                since_seq: int = 0) -> Dict[str, Any]:
        """Newest-last slice of the buffer.

        `level` is a *minimum* — "WARNING" returns warnings, errors and
        criticals. Unknown level names are ignored rather than returning
        nothing, because a filter typo must not look like a quiet system.
        """
        min_rank = _LEVEL_RANK.get((level or "").upper(), -1)
        with self._lock:
            items = list(self._buf)
            dropped = self._dropped
            last = self._seq
        out = [
            e for e in items
            if e["seq"] > since_seq
            and (min_rank < 0 or _LEVEL_RANK.get(e["level"], 99) >= min_rank)
        ]
        if limit and limit > 0:
            out = out[-limit:]
        return {
            "entries": out,
            "next_seq": last,
            "dropped": dropped,
            "capacity": self.capacity,
        }

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._dropped = 0


# The process-wide buffer. Created eagerly so imports can reference it; it
# stays empty and costless until install() attaches it to anything.
ring = LogRing()


class RingHandler(logging.Handler):
    """Feeds every log record into a LogRing.

    Formatting is deliberately minimal — the UI renders level/logger/time
    itself, so storing the raw message keeps entries small and lets the client
    filter without re-parsing a formatted line.
    """

    def __init__(self, target: LogRing, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._ring = target

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg = f"{msg}\n{self.format_exception(record)}"
            self._ring.add(record.levelname, record.name, msg,
                           ts=record.created)
        except Exception:
            # A logging handler that raises breaks the thing it observes.
            self.handleError(record)

    def format_exception(self, record: logging.LogRecord) -> str:
        try:
            import traceback
            return "".join(traceback.format_exception(*record.exc_info))
        except Exception:
            return ""


class TeeStream:
    """Write-through wrapper that also captures completed lines.

    Only whole lines are stored: progress bars (`tqdm` writes partial lines
    ending in \\r) would otherwise flood the buffer with thousands of
    fragments. A trailing partial line is held until its newline arrives.
    """

    def __init__(self, stream, target: LogRing, name: str,
                 level: str = "INFO") -> None:
        self._stream = stream
        self._ring = target
        self._name = name
        self._level = level
        self._pending = ""
        self._lock = threading.Lock()

    # ── captured ─────────────────────────────────────────────────────
    def write(self, data) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._capture(data)
        except Exception:
            pass
        return len(data) if data else 0

    def _capture(self, data) -> None:
        if not data:
            return
        text = data if isinstance(data, str) else str(data)
        with self._lock:
            self._pending += text
            if "\n" not in self._pending:
                # Guard against a runaway line with no newline at all.
                if len(self._pending) > 8192:
                    line, self._pending = self._pending, ""
                    self._ring.add(self._level, self._name, line)
                return
            *lines, self._pending = self._pending.split("\n")
        for line in lines:
            # tqdm redraws with \r; keep only the final state of the line.
            line = line.rsplit("\r", 1)[-1].rstrip()
            if line:
                self._ring.add(self._level, self._name, line)

    # ── pass-through ─────────────────────────────────────────────────
    def flush(self):
        try:
            return self._stream.flush()
        except Exception:
            return None

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, item):
        return getattr(self._stream, item)


_installed = False
_original_stdout = None
_original_stderr = None


def install(capacity: int = DEFAULT_CAPACITY,
            capture_stdio: bool = True) -> LogRing:
    """Attach the ring to the root logger and (optionally) tee stdio.

    Idempotent — calling twice (e.g. under uvicorn --reload) is a no-op rather
    than a second handler duplicating every line.
    """
    global _installed, _original_stdout, _original_stderr, ring
    if _installed:
        return ring

    if capacity != ring.capacity:
        ring = LogRing(capacity)

    root = logging.getLogger()

    # Pin existing stream handlers to the *real* streams before teeing, so
    # their output does not come back around as captured stdio.
    _original_stdout, _original_stderr = sys.stdout, sys.stderr
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            if h.stream is sys.stderr:
                h.stream = _original_stderr
            elif h.stream is sys.stdout:
                h.stream = _original_stdout

    handler = RingHandler(ring)
    root.addHandler(handler)

    # uvicorn configures its loggers with propagate=False, so root handlers
    # never see them — and "address already in use" or a request traceback is
    # exactly what someone opens the log viewer to find.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        if not lg.propagate:
            lg.addHandler(handler)

    if capture_stdio:
        sys.stdout = TeeStream(_original_stdout, ring, "stdout", "INFO")
        sys.stderr = TeeStream(_original_stderr, ring, "stderr", "WARNING")

    # Third-party UserWarnings (torchcodec, pyannote) become log records
    # rather than bypassing logging entirely.
    logging.captureWarnings(True)

    _installed = True
    return ring


def uninstall() -> None:
    """Restore stdio and detach the handler. Used by tests."""
    global _installed, _original_stdout, _original_stderr
    if not _installed:
        return
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if isinstance(h, RingHandler):
                lg.removeHandler(h)
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr
    logging.captureWarnings(False)
    _installed = False


def snapshot(level: str = "", limit: int = 300,
             since_seq: int = 0) -> Dict[str, Any]:
    return ring.entries(level=level, limit=limit, since_seq=since_seq)
