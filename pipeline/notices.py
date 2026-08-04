"""Structured, user-facing setup notices.

GoChiDUBB runs on the user's own machine, so there is nobody watching a
terminal. Anything that tells the user "your environment is wrong, here is how
to fix it" has to reach the UI, not just `log.warning`. This module is the
shared vocabulary for that.

A notice is a plain dict (matching the rest of the codebase, which passes
free-form JSON around) with a **stable `code`**. The code — not the title — is
the identity: it is the dedupe key when merging notices from several probes,
the localStorage key the UI dismisses on, and the anchor tests assert against.
Titles and wording may be rewritten freely; codes may not.

Severity ladder:

  error   the feature is unusable — diarization cannot run at all
  warn    it ran, but degraded — a fallback model, QA silently disabled
  info    worth knowing, nothing is wrong

`warn` is the interesting one. Before this module the pipeline had no way to
say "succeeded, but not the way you think": a stage was ok or it raised. That
gap is what let a gated-model fallback look exactly like a clean run.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

# Ordered worst-first: index in this tuple is the sort/merge precedence.
SEVERITIES = ("error", "warn", "info")

_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def notice(code: str,
           severity: str = "warn",
           title: str = "",
           detail: str = "",
           remediation: Iterable[str] = (),
           url: str = "",
           subsystem: str = "") -> Dict[str, Any]:
    """Build one notice.

    Args:
        code: stable slug, `subsystem.problem` (e.g. "pyannote.gated_model").
        severity: one of SEVERITIES. Anything unrecognized becomes "warn" —
            a malformed severity must not make a real problem invisible.
        title: one line, shown in the banner. No trailing period.
        detail: the specifics (which model, which path, the raw error).
        remediation: ordered human steps. Rendered as a list in the UI, so
            each entry should stand alone.
        url: the single most useful link — the banner's primary action.
        subsystem: which part of the app is affected ("diarize", "translate",
            "tts", "system"). Used to group notices and to attach them to a
            pipeline stage.
    """
    if severity not in _SEVERITY_RANK:
        severity = "warn"
    return {
        "code": code,
        "severity": severity,
        "title": title or code,
        "detail": detail,
        "remediation": [str(r) for r in remediation],
        "url": url,
        "subsystem": subsystem,
    }


def severity_rank(n: Dict[str, Any]) -> int:
    """Sort key — lower is worse. Unknown severities sort last."""
    return _SEVERITY_RANK.get(n.get("severity", ""), len(SEVERITIES))


def merge_notices(*lists: Iterable[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Combine notice lists, one entry per code, worst severity wins.

    The same problem is legitimately detected by more than one probe — a
    missing HF token shows up in the passive check *and* in whatever the
    diarize stage recorded on its last run. Emitting it twice would be noise,
    and dropping the more severe copy would understate the problem, so we keep
    the worst and let a later, richer detection overwrite an equally-severe
    earlier one.
    """
    by_code: Dict[str, Dict[str, Any]] = {}
    for group in lists:
        for n in group or ():
            if not isinstance(n, dict) or not n.get("code"):
                continue
            code = n["code"]
            prev = by_code.get(code)
            if prev is None or severity_rank(n) <= severity_rank(prev):
                by_code[code] = n
    return sorted(by_code.values(), key=lambda n: (severity_rank(n), n["code"]))


def worst_severity(notices: Iterable[Dict[str, Any]]) -> str:
    """Highest severity present, or "" for an empty list."""
    ranks = [severity_rank(n) for n in notices or ()]
    if not ranks:
        return ""
    best = min(ranks)
    return SEVERITIES[best] if best < len(SEVERITIES) else ""


# ═════════════════════════════════════════════════════════════════════
# Redaction
# ═════════════════════════════════════════════════════════════════════
# Notices and captured log lines are served over HTTP. The server binds
# loopback by default, but GOCHIDUBB_HOST can open it to the network, and
# anything that quotes a config value or an exception message would then leak a
# credential. Scrub before storing, not before rendering: the buffer itself
# should never hold the secret, whatever the bind address happens to be.

REDACTED = "[redacted]"

# Order matters, most specific first. The generic key/value rule below happily
# matches "Authorization: Bearer" and would swallow the word "Bearer" while
# leaving the token itself in the clear, so the Bearer rule has to run first.
_SECRET_PATTERNS = (
    # HuggingFace user access tokens — the one credential this app really holds.
    (re.compile(r"\bhf_[A-Za-z0-9]{8,}"), REDACTED),
    # Bearer headers.
    (re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{8,})", re.IGNORECASE), r"\1" + REDACTED),
    # Generic "KEY=value" / "key: value" for anything that names itself a secret.
    (re.compile(
        r"((?:token|secret|api[_-]?key|password|authorization)\s*[=:]\s*)"
        r"([\"']?)([^\s\"',}]{4,})\2",
        re.IGNORECASE,
    ), r"\1" + REDACTED),
)


def redact(text: Any) -> str:
    """Replace anything that looks like a credential with [redacted]."""
    if text is None:
        return ""
    s = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        s = pattern.sub(replacement, s)
    return s


def mask_secret(value: str, keep: int = 4) -> str:
    """Render a secret as a recognizable stub: "hf_ab…" / "" when unset.

    Used by GET /api/config so the Settings UI can show *that* a token is
    configured without shipping it to the browser.
    """
    if not value:
        return ""
    if len(value) <= keep:
        return "…"
    return f"{value[:keep]}…"
