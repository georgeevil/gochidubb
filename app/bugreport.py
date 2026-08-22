"""Bug reports — package a failed job into something a maintainer can act on.

A GoChiDUBB failure happens on someone else's machine, with no telemetry and
no shared logs. This module turns the structured error a job already carries
(``last_error`` from ``server._set_job_error``) plus the log window around it
into a single report dict, and can deliver that report to an issue tracker.

Deliberate choices:

* **Deduplication by signature, not by message.** The same bug produces a
  slightly different message every time (paths, ids, durations). The
  signature hashes a *normalized* message together with the stage id, and the
  resulting ``gcd-sig:<hash>`` line is embedded in the issue body so a later
  occurrence finds the existing issue via search and lands as a comment
  instead of a duplicate.
* **Everything user-visible goes through ``redact``** (the same scrubber
  ``app/logbuf.py`` applies on log ingest — see pipeline/notices.redact).
  Reports leave the machine, so no config values, no ``_pending_args``, no
  transcript, and no credential-shaped strings may survive into one.
* **Delivery never raises and never logs the API key.** A bug report is a
  courtesy on top of a failure; it must not become a second failure. Sinks
  return a result dict either way.
* **Sinks are a protocol.** Linear is the one sink today; a future Slack
  sink is another class here plus a branch in :func:`get_sink`.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from pipeline.notices import redact

log = logging.getLogger("gochidubb.bugreport")

LINEAR_API_URL = "https://api.linear.app/graphql"

# The dedupe marker embedded verbatim in issue bodies and searched for on the
# next occurrence. Changing this prefix (or the hash) orphans existing issues.
SIG_PREFIX = "gcd-sig:"


# ── Signature ────────────────────────────────────────────────────────
# Order matters: Windows paths first (a drive letter + hex-ish tail would be
# half-eaten by the generic rules), then POSIX paths (two-plus components so
# short URL-ish fragments like "/v1" survive), then ids, then bare numbers.
_NORMALIZE_RULES = (
    (re.compile(r"[a-z]:\\[^\s'\"]+"), "<path>"),
    (re.compile(r"(?:/[\w.@-]+){2,}"), "<path>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<id>"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<hex>"),
    (re.compile(r"0x[0-9a-f]+"), "<hex>"),
    # No trailing \b: durations glue their unit to the digits ("120s",
    # "12.5s") and must still collapse, or every timeout files a new issue.
    (re.compile(r"\b\d+(?:\.\d+)?"), "<n>"),
)


def normalize_message(msg: str) -> str:
    """Collapse the run-specific parts of an error message.

    Two occurrences of the same bug must normalize identically even though
    their paths, ids and numbers differ — the normalized form is what gets
    hashed into the dedupe signature.
    """
    s = str(msg or "").lower().strip()
    for pattern, replacement in _NORMALIZE_RULES:
        s = pattern.sub(replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:300]


def error_signature(stage: str, message: str) -> str:
    """12-hex-char dedupe key for one (stage, normalized message) pair.

    The stage id is part of the hash so "translate timed out" and "tts timed
    out" stay distinct issues.
    """
    basis = f"{stage or ''}|{normalize_message(message)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


# ── Report assembly (pure) ───────────────────────────────────────────
# The whitelist of job fields a report may carry. Explicitly NOT here:
# transcript / transcript_raw (huge, and someone else's content),
# _pending_args (raw request payloads), and anything from config.
_JOB_FIELDS = (
    "id", "status", "title", "source_label", "source", "source_type",
    "target_lang", "source_lang", "model", "speaker_mode", "voice_preset",
    "voice_mode", "tts_speed", "wizard_mode", "mode", "created", "batch_id",
)


def _redacted_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy with every string value scrubbed."""
    return {k: (redact(v) if isinstance(v, str) else v) for k, v in d.items()}


def build_bug_report(job: Dict[str, Any], *, system: Dict[str, Any],
                     logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the full report for one failed job (pure — no I/O).

    Takes the in-memory job dict (the DB copy has large fields stripped and
    may lag the live run). ``logs`` should come from
    :func:`select_log_window`; entries are already redacted on ingest.
    """
    le = job.get("last_error") or {}
    stage = le.get("stage") or job.get("failed_stage") or ""
    message = le.get("message") or job.get("error") or ""
    return {
        "report_version": 1,
        "generated_at": time.time(),
        "job": _redacted_copy({k: job[k] for k in _JOB_FIELDS if k in job}),
        "last_error": _redacted_copy(le),
        "error_history": [_redacted_copy(h) for h in job.get("error_history") or []
                          if isinstance(h, dict)],
        "signature": error_signature(stage, message) if message else "",
        "logs": logs,
        "system": _redacted_copy(system),
    }


def select_log_window(snapshot_fn: Callable[..., Dict[str, Any]],
                      last_error: Optional[Dict[str, Any]],
                      limit: int = 80) -> List[Dict[str, Any]]:
    """The log lines worth attaching to a report.

    When the error carries a [log_from, log_to] ring-seq window (see
    ``server._set_job_error``), take that window's tail, padded forward with
    lines logged after the failure if the window is shorter than ``limit``.
    Without a window, just the newest ``limit`` lines. Entries are redacted
    at ingest time (see app/logbuf.LogRing.add), so they pass through as-is.
    """
    le = last_error or {}
    log_from = int(le.get("log_from") or 0)
    log_to = int(le.get("log_to") or 0)
    if not log_to:
        return list(snapshot_fn(limit=limit)["entries"])
    entries = snapshot_fn(limit=0, since_seq=max(0, log_from - 1))["entries"]
    window = [e for e in entries if e["seq"] <= log_to]
    window = window[-limit:]
    if len(window) < limit:
        after = [e for e in entries if e["seq"] > log_to]
        window = window + after[:limit - len(window)]
    return window


# ── Sinks ────────────────────────────────────────────────────────────
class Sink(Protocol):
    """Somewhere a report can be delivered. ``deliver`` never raises."""

    name: str

    async def deliver(self, report: Dict[str, Any], note: str = "") -> Dict[str, Any]:
        ...


def _fmt_log_lines(logs: List[Dict[str, Any]]) -> str:
    lines = []
    for e in logs or []:
        lines.append(f"[{e.get('level', '')}] {e.get('logger', '')} — "
                     f"{e.get('message', '')}")
    return "\n".join(lines)


class LinearSink:
    """Deliver reports to Linear: one issue per signature, comments after.

    Personal API keys are sent raw in the ``Authorization`` header (no
    ``Bearer`` prefix) — that is what Linear's GraphQL API expects for them.
    ``transport`` is a test seam forwarded to ``httpx.AsyncClient`` so tests
    can use ``httpx.MockTransport``.
    """

    name = "linear"

    _FIND_QUERY = (
        "query FindBySig($q: String!) { issueSearch(query: $q, first: 5) "
        "{ nodes { id identifier url title } } }"
    )
    _COMMENT_MUTATION = (
        "mutation AddComment($issueId: String!, $body: String!) "
        "{ commentCreate(input: {issueId: $issueId, body: $body}) "
        "{ success comment { url } } }"
    )
    _CREATE_MUTATION = (
        "mutation CreateIssue($input: IssueCreateInput!) "
        "{ issueCreate(input: $input) { success issue { id identifier url } } }"
    )

    def __init__(self, api_key: str, team_id: str, project_id: str = "",
                 transport=None):
        self._api_key = api_key
        self._team_id = team_id
        self._project_id = project_id
        self._transport = transport

    async def _gql(self, client, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        r = await client.post(
            LINEAR_API_URL,
            json={"query": query, "variables": variables},
            headers={"Authorization": self._api_key,
                     "Content-Type": "application/json"},
        )
        if r.status_code != 200:
            raise RuntimeError(f"Linear API returned HTTP {r.status_code}")
        payload = r.json()
        errors = payload.get("errors")
        if errors:
            msg = (errors[0] or {}).get("message") or "unknown GraphQL error"
            raise RuntimeError(f"Linear GraphQL error: {msg}")
        return payload.get("data") or {}

    # ── Body rendering ───────────────────────────────────────────────
    @staticmethod
    def _issue_content(report: Dict[str, Any], note: str) -> tuple:
        le = report.get("last_error") or {}
        job = report.get("job") or {}
        system = report.get("system") or {}
        sig = report.get("signature") or ""
        stage_label = le.get("stage_label") or le.get("stage") or "unknown stage"
        message = le.get("message") or "unknown error"
        title = f"[gochidubb] {stage_label}: {message[:90]}"
        rows = (
            ("Job id", job.get("id", "")),
            ("Stage", le.get("stage", "")),
            ("Error type", le.get("type", "")),
            ("Target language", job.get("target_lang", "")),
            ("Signature", sig),
            ("Platform", system.get("platform", "")),
            ("Python", system.get("python", "")),
            ("GPU", f"{system.get('gpu_backend', '')} {system.get('gpu') or ''}".strip()),
        )
        parts = ["| field | value |", "| --- | --- |"]
        parts += [f"| {k} | {v} |" for k, v in rows if v != ""]
        parts += ["",
                  f"`{SIG_PREFIX}{sig}` — dedupe key, do not remove: later "
                  "occurrences of this error are matched to this issue by "
                  "searching for this line."]
        if note:
            parts += ["", f"**User note:** {note}"]
        if le.get("traceback_tail"):
            parts += ["", "```text", le["traceback_tail"], "```"]
        logs = report.get("logs") or []
        if logs:
            parts += ["", f"<details><summary>Last {len(logs)} log lines</summary>",
                      "", "```text", _fmt_log_lines(logs), "```", "", "</details>"]
        return title, "\n".join(parts)

    @staticmethod
    def _comment_body(report: Dict[str, Any], note: str) -> str:
        le = report.get("last_error") or {}
        job = report.get("job") or {}
        system = report.get("system") or {}
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(le.get("ts") or time.time()))
        parts = [f"New occurrence — job `{job.get('id', '?')}`, "
                 f"target `{job.get('target_lang', '?')}`, {ts} UTC",
                 f"{system.get('platform', '')} · python {system.get('python', '')} · "
                 f"{system.get('gpu_backend', '')}"]
        if note:
            parts += ["", f"**User note:** {note}"]
        if le.get("traceback_tail"):
            parts += ["", "```text", le["traceback_tail"], "```"]
        logs = report.get("logs") or []
        if logs:
            parts += ["", f"<details><summary>Last {len(logs)} log lines</summary>",
                      "", "```text", _fmt_log_lines(logs), "```", "", "</details>"]
        return "\n".join(parts)

    # ── Delivery ─────────────────────────────────────────────────────
    async def deliver(self, report: Dict[str, Any], note: str = "") -> Dict[str, Any]:
        """Create or comment. Never raises; the key never reaches a log."""
        import httpx
        sig = report.get("signature") or ""
        result = {"ok": False, "sink": self.name, "action": "failed",
                  "url": "", "issue": "", "signature": sig, "error": ""}
        try:
            async with httpx.AsyncClient(timeout=10.0,
                                         transport=self._transport) as client:
                found = None
                if sig:
                    data = await self._gql(client, self._FIND_QUERY,
                                           {"q": SIG_PREFIX + sig})
                    nodes = (data.get("issueSearch") or {}).get("nodes") or []
                    found = nodes[0] if nodes else None
                if found:
                    data = await self._gql(
                        client, self._COMMENT_MUTATION,
                        {"issueId": found.get("id"),
                         "body": self._comment_body(report, note)})
                    cc = data.get("commentCreate") or {}
                    if not cc.get("success"):
                        raise RuntimeError("commentCreate reported success=false")
                    url = ((cc.get("comment") or {}).get("url")
                           or found.get("url") or "")
                    result.update(ok=True, action="commented", url=url,
                                  issue=found.get("identifier") or "")
                else:
                    title, body = self._issue_content(report, note)
                    inp = {"teamId": self._team_id, "title": title,
                           "description": body}
                    if self._project_id:
                        inp["projectId"] = self._project_id
                    data = await self._gql(client, self._CREATE_MUTATION,
                                           {"input": inp})
                    ic = data.get("issueCreate") or {}
                    if not ic.get("success"):
                        raise RuntimeError("issueCreate reported success=false")
                    issue = ic.get("issue") or {}
                    result.update(ok=True, action="created",
                                  url=issue.get("url") or "",
                                  issue=issue.get("identifier") or "")
        except Exception as e:
            # redact() as belt-and-braces; the exception text never contains
            # the key by construction, but a proxy error might echo headers.
            err = redact(f"{type(e).__name__}: {e}")
            result["error"] = err
            log.warning(f"[bugreport] Linear delivery failed: {err}")
        return result


# ── Sink selection ───────────────────────────────────────────────────
def get_sink() -> Optional[Sink]:
    """The configured delivery sink, or None.

    Linear is the only sink today. A future Slack sink is another class in
    this module plus a branch here on its own secrets.
    """
    from app.secrets import get_secret
    api_key = get_secret("linear_api_key")
    team_id = get_secret("linear_team_id")
    if not (api_key and team_id):
        return None
    return LinearSink(api_key, team_id,
                      project_id=get_secret("linear_project_id"))


def sink_configured() -> bool:
    """Presence check only — never touches the network."""
    return get_sink() is not None
