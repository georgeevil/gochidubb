"""Progress-aware ffmpeg subprocess runner.

Rendering a long video can legitimately take longer than any fixed timeout
(x264 on a 2h source easily exceeds 10 minutes), while a genuinely hung
ffmpeg — bad filter graph, stuck demuxer, dead disk — should still be killed
promptly. A single hard timeout can't serve both, so `run_ffmpeg` treats the
timeout as a *soft* deadline:

  - ffmpeg is launched with `-nostats -progress pipe:1`, which emits
    key=value progress blocks (out_time, frame, total_size, ...) on stdout.
  - While those values keep advancing, the soft deadline is extended — a
    working render is never killed mid-encode.
  - Once the soft deadline has passed, the process is killed as soon as no
    progress has been observed for `stall_timeout` seconds.
  - A process that never reports progress (hung from the start, or a
    non-ffmpeg command) falls back to the plain hard timeout, same as the
    old `subprocess.run(..., timeout=...)` behaviour.

Defaults come from `cfg.ffmpeg_timeout` / `cfg.ffmpeg_stall_timeout`
(env: GOCHIDUBB_FFMPEG_TIMEOUT / GOCHIDUBB_FFMPEG_STALL_TIMEOUT).
"""
import logging
import os
import subprocess
import threading
import time
from collections import deque

log = logging.getLogger("gochidubb.ffmpeg")

# Monotonically increasing keys from ffmpeg's -progress output. If any of
# them changes value, the encode is moving. (speed=/bitrate= jitter even
# when the encoder is stuck, so they are deliberately excluded.)
PROGRESS_KEYS = ("out_time_ms", "out_time_us", "out_time", "frame", "total_size")

DEFAULT_TIMEOUT = 600
DEFAULT_STALL_TIMEOUT = 120


def _config_defaults():
    try:
        from app.config import cfg
        return (
            float(getattr(cfg, "ffmpeg_timeout", DEFAULT_TIMEOUT)),
            float(getattr(cfg, "ffmpeg_stall_timeout", DEFAULT_STALL_TIMEOUT)),
        )
    except Exception:
        return float(DEFAULT_TIMEOUT), float(DEFAULT_STALL_TIMEOUT)


def inject_progress_flags(cmd):
    """Insert `-nostats -progress pipe:1` after the ffmpeg executable, if the
    command is ffmpeg and doesn't already configure progress reporting."""
    cmd = [str(c) for c in cmd]
    if not os.path.basename(cmd[0]).lower().startswith("ffmpeg"):
        return cmd
    if "-progress" in cmd:
        return cmd
    return cmd[:1] + ["-nostats", "-progress", "pipe:1"] + cmd[1:]


def run_ffmpeg(cmd, desc="", logger=None, timeout=None, stall_timeout=None,
               poll_interval=1.0):
    """Run an ffmpeg command with a soft, progress-extended timeout.

    Returns a subprocess.CompletedProcess (stdout stripped of progress
    lines). Raises RuntimeError on non-zero exit or on timeout.
    """
    logger = logger or log
    cfg_timeout, cfg_stall = _config_defaults()
    if timeout is None:
        timeout = cfg_timeout
    if stall_timeout is None:
        stall_timeout = cfg_stall

    run_cmd = inject_progress_flags(cmd)
    proc = subprocess.Popen(
        run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace",
    )

    state = {"last_progress": time.monotonic(), "position": "", "seen": False}
    last_values = {}
    out_lines = deque(maxlen=200)
    err_lines = deque(maxlen=200)

    def _pump_stdout():
        for line in proc.stdout:
            key, sep, val = line.strip().partition("=")
            if sep and key in PROGRESS_KEYS:
                if val not in ("", "N/A") and last_values.get(key) != val:
                    last_values[key] = val
                    state["seen"] = True
                    state["last_progress"] = time.monotonic()
                    if key == "out_time":
                        state["position"] = val
            elif sep and key in ("progress", "speed", "bitrate", "fps",
                                 "stream_0_0_q", "dup_frames", "drop_frames"):
                pass  # progress-block noise, not real stdout
            else:
                out_lines.append(line)

    def _pump_stderr():
        for line in proc.stderr:
            err_lines.append(line)

    threads = [
        threading.Thread(target=_pump_stdout, daemon=True),
        threading.Thread(target=_pump_stderr, daemon=True),
    ]
    for t in threads:
        t.start()

    start = time.monotonic()
    extending_logged = False
    while True:
        try:
            proc.wait(timeout=poll_interval)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if now - start < timeout:
            continue
        # Past the soft deadline: keep going only while progress is fresh.
        if state["seen"] and now - state["last_progress"] < stall_timeout:
            if not extending_logged:
                logger.info(
                    f"[{desc}] exceeded soft timeout ({timeout:.0f}s) but ffmpeg "
                    f"is still making progress (at {state['position'] or '?'}) "
                    f"— extending until progress stalls for {stall_timeout:.0f}s"
                )
                extending_logged = True
            continue
        proc.kill()
        proc.wait()
        for t in threads:
            t.join(timeout=5)
        elapsed = time.monotonic() - start
        if state["seen"]:
            detail = (f"no encode progress for {stall_timeout:.0f}s "
                      f"(last position {state['position'] or 'unknown'})")
        else:
            detail = "no encode progress ever reported"
        raise RuntimeError(
            f"{desc} timed out after {elapsed:.0f}s: {detail}"
        )

    for t in threads:
        t.join(timeout=5)
    stdout = "".join(out_lines)
    stderr = "".join(err_lines)
    if proc.returncode != 0:
        logger.debug(
            f"[{desc}] exit={proc.returncode} stdout={stdout[:400]!r} "
            f"stderr={stderr[-400:]!r}"
        )
        raise RuntimeError(f"{desc} failed: {stderr[-400:]}")
    return subprocess.CompletedProcess(run_cmd, proc.returncode, stdout, stderr)
