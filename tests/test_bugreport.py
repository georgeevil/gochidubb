"""Tests for app/bugreport.py — signatures, report assembly, Linear sink.

The error signature is a *contract with the outside world*: it is embedded
in Linear issue bodies as ``gcd-sig:<hash>`` and searched for on the next
occurrence. Changing the normalization rules or the hash orphans every
existing issue, so a golden-value test locks the algorithm.
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import logbuf  # noqa: E402
from app.bugreport import (  # noqa: E402
    SIG_PREFIX, LinearSink, build_bug_report, error_signature,
    normalize_message, select_log_window,
)


# ═════════════════════════════════════════════════════════════════════
# Normalization + signature
# ═════════════════════════════════════════════════════════════════════
class TestNormalizeMessage:
    def test_posix_paths_collapse(self):
        a = normalize_message("cannot open /Users/alice/videos/clip.mp4")
        b = normalize_message("cannot open /home/bob/other/file.mp4")
        assert a == b
        assert "<path>" in a

    def test_windows_paths_collapse(self):
        a = normalize_message(r"cannot open C:\Users\alice\clip.mp4")
        b = normalize_message(r"cannot open D:\videos\other.mp4")
        assert a == b
        assert "<path>" in a

    def test_short_url_ish_fragment_survives(self):
        # One-component absolute fragments like "/v1" are meaningful
        # (endpoint paths) and must not be swallowed by the POSIX rule.
        assert "/v1" in normalize_message("POST /v1 failed")

    def test_uuids_collapse(self):
        a = normalize_message("job 0f3aab12-1234-4abc-9def-001122334455 died")
        b = normalize_message("job ffffffff-0000-4a4a-8b8b-aabbccddeeff died")
        assert a == b
        assert "<id>" in a

    def test_hex_runs_collapse(self):
        a = normalize_message("chunk deadbeefcafe0123 failed at 0x7fff")
        b = normalize_message("chunk 0123456789abcdef failed at 0xdead")
        assert a == b
        assert "<hex>" in a

    def test_numbers_collapse(self):
        a = normalize_message("failed on segment 17 after 120 seconds")
        b = normalize_message("failed on segment 3 after 95 seconds")
        assert a == b
        assert "<n>" in a

    def test_unit_suffixed_numbers_collapse(self):
        # Durations glue the unit onto the digits ("120s", "12.5s"); the
        # number rule must still swallow them or every timeout gets its own
        # Linear issue.
        a = normalize_message("request failed after 120s (retry 3, 12.5s)")
        b = normalize_message("request failed after 95s (retry 1, 3.2s)")
        assert a == b
        assert "<n>s" in a

    def test_case_and_whitespace_insensitive(self):
        assert normalize_message("  CUDA   Error ") == normalize_message("cuda error")

    def test_truncated_to_300(self):
        assert len(normalize_message("x" * 1000)) <= 300


class TestErrorSignature:
    def test_same_bug_different_run_hashes_identically(self):
        a = error_signature("tts", "no audio at /tmp/job-1/seg 17.wav after 42 ms")
        b = error_signature("tts", "no audio at /tmp/job-9/seg 3.wav after 95 ms")
        assert a == b

    def test_stage_distinguishes(self):
        assert (error_signature("translate", "backend timed out")
                != error_signature("tts", "backend timed out"))

    def test_shape(self):
        sig = error_signature("tts", "boom")
        assert len(sig) == 12
        assert sig == sig.lower()
        int(sig, 16)  # valid hex

    def test_golden_value(self):
        """Locks the algorithm. If this fails you changed normalization or
        hashing — which orphans every Linear issue already carrying a
        gcd-sig line. Do that only on purpose."""
        assert error_signature(
            "tts", "CUDA out of memory. Tried to allocate 512.00 MiB on device 0"
        ) == "3bce138aacbb"


# ═════════════════════════════════════════════════════════════════════
# Report assembly
# ═════════════════════════════════════════════════════════════════════
def _fake_job():
    return {
        "id": "job-1",
        "status": "error",
        "title": "Some lecture",
        "target_lang": "ru",
        "model": "qwen3",
        "created": 1700000000.0,
        "error": "synth exploded with token=abcd1234secret attached",
        "last_error": {
            "stage": "tts",
            "stage_label": "Voice synthesis",
            "type": "RuntimeError",
            "message": "synth exploded with token=abcd1234secret attached",
            "traceback_tail": "Traceback...\ntoken=abcd1234secret\nRuntimeError",
            "ts": 1700000100.0,
            "log_from": 5,
            "log_to": 9,
        },
        "error_history": [
            {"stage": "tts", "message": "earlier fail 1", "ts": 1.0},
        ],
        # Must never leak:
        "transcript": [{"text": "private words"}],
        "transcript_raw": [{"text": "private words"}],
        "_pending_args": {"hf_token": "hf_abcdefghijkl"},
    }


class TestBuildBugReport:
    def test_shape(self):
        rep = build_bug_report(_fake_job(), system={"platform": "TestOS"},
                               logs=[{"seq": 1, "message": "m"}])
        assert rep["report_version"] == 1
        assert rep["generated_at"] > 0
        assert rep["job"]["id"] == "job-1"
        assert rep["job"]["target_lang"] == "ru"
        assert rep["last_error"]["stage"] == "tts"
        assert rep["error_history"][0]["message"] == "earlier fail 1"
        assert rep["system"]["platform"] == "TestOS"
        assert rep["logs"] == [{"seq": 1, "message": "m"}]
        assert rep["signature"] == error_signature(
            "tts", _fake_job()["last_error"]["message"])

    def test_absent_job_fields_omitted(self):
        rep = build_bug_report({"id": "j", "last_error": {"stage": "tts",
                                                          "message": "x"}},
                               system={}, logs=[])
        assert "target_lang" not in rep["job"]

    def test_secrets_redacted_everywhere(self):
        rep = build_bug_report(_fake_job(), system={}, logs=[])
        blob = json.dumps(rep)
        assert "abcd1234secret" not in blob
        assert "[redacted]" in rep["last_error"]["traceback_tail"]
        assert "[redacted]" in rep["last_error"]["message"]

    def test_large_and_private_fields_never_leak(self):
        rep = build_bug_report(_fake_job(), system={}, logs=[])
        blob = json.dumps(rep)
        assert "private words" not in blob
        assert "_pending_args" not in blob
        assert "hf_abcdefghijkl" not in blob

    def test_no_error_means_empty_signature(self):
        rep = build_bug_report({"id": "j", "status": "complete"},
                               system={}, logs=[])
        assert rep["signature"] == ""


# ═════════════════════════════════════════════════════════════════════
# Log window selection
# ═════════════════════════════════════════════════════════════════════
class TestSelectLogWindow:
    def _ring(self, n=120):
        ring = logbuf.LogRing(capacity=500)
        for i in range(1, n + 1):
            ring.add("INFO", "test", f"line {i}")
        return ring

    def test_error_window_honored(self):
        ring = self._ring()
        le = {"log_from": 30, "log_to": 40}
        out = select_log_window(ring.entries, le, limit=80)
        seqs = [e["seq"] for e in out]
        assert seqs[0] == 30
        # window is 11 lines; padded forward past log_to up to the limit
        assert 41 in seqs
        assert len(out) <= 80

    def test_long_window_keeps_tail(self):
        ring = self._ring()
        le = {"log_from": 1, "log_to": 110}
        out = select_log_window(ring.entries, le, limit=80)
        assert len(out) == 80
        assert out[-1]["seq"] == 110          # tail of the window, not the ring
        assert out[0]["seq"] == 31

    def test_no_error_falls_back_to_newest(self):
        ring = self._ring()
        out = select_log_window(ring.entries, None, limit=80)
        assert len(out) == 80
        assert out[-1]["seq"] == 120

    def test_limit_respected(self):
        ring = self._ring()
        out = select_log_window(ring.entries, {"log_from": 1, "log_to": 120},
                                limit=10)
        assert len(out) == 10


# ═════════════════════════════════════════════════════════════════════
# Linear sink (httpx.MockTransport — no network)
# ═════════════════════════════════════════════════════════════════════
httpx = pytest.importorskip("httpx")


def _report():
    job = _fake_job()
    return build_bug_report(job, system={"platform": "TestOS", "python": "3.11"},
                            logs=[{"seq": i, "level": "INFO", "logger": "t",
                                   "message": f"line {i}"} for i in range(5)])


def _transport(search_nodes, seen):
    def handler(request):
        payload = json.loads(request.content.decode("utf-8"))
        seen.append((request, payload))
        q = payload["query"]
        if "issueSearch" in q:
            return httpx.Response(200, json={
                "data": {"issueSearch": {"nodes": search_nodes}}})
        if "commentCreate" in q:
            return httpx.Response(200, json={
                "data": {"commentCreate": {"success": True,
                                           "comment": {"url": "https://linear.app/c/1"}}}})
        if "issueCreate" in q:
            return httpx.Response(200, json={
                "data": {"issueCreate": {"success": True,
                                         "issue": {"id": "iid", "identifier": "GCD-7",
                                                   "url": "https://linear.app/i/GCD-7"}}}})
        return httpx.Response(400, json={"errors": [{"message": "bad request"}]})
    return httpx.MockTransport(handler)


class TestLinearSink:
    def test_creates_when_no_match(self):
        seen = []
        sink = LinearSink("lin_api_KEY", "team-1",
                          transport=_transport([], seen))
        rep = _report()
        result = asyncio.run(sink.deliver(rep, note="I clicked dub"))
        assert result["ok"] is True
        assert result["action"] == "created"
        assert result["issue"] == "GCD-7"
        assert result["url"] == "https://linear.app/i/GCD-7"
        assert result["signature"] == rep["signature"]
        # Two calls: search then create.
        assert len(seen) == 2
        # Personal keys go raw — no Bearer prefix.
        for request, _ in seen:
            assert request.headers["Authorization"] == "lin_api_KEY"
        create_payload = seen[1][1]
        body = create_payload["variables"]["input"]["description"]
        assert f"{SIG_PREFIX}{rep['signature']}" in body
        assert "I clicked dub" in body
        assert create_payload["variables"]["input"]["teamId"] == "team-1"
        assert "projectId" not in create_payload["variables"]["input"]

    def test_project_id_forwarded_when_set(self):
        seen = []
        sink = LinearSink("k", "team-1", project_id="proj-9",
                          transport=_transport([], seen))
        asyncio.run(sink.deliver(_report()))
        assert seen[1][1]["variables"]["input"]["projectId"] == "proj-9"

    def test_comments_on_existing_issue(self):
        seen = []
        nodes = [{"id": "iid-0", "identifier": "GCD-3",
                  "url": "https://linear.app/i/GCD-3", "title": "t"}]
        sink = LinearSink("k", "team-1", transport=_transport(nodes, seen))
        result = asyncio.run(sink.deliver(_report()))
        assert result["ok"] is True
        assert result["action"] == "commented"
        assert result["issue"] == "GCD-3"
        assert result["url"] == "https://linear.app/c/1"
        assert seen[1][1]["variables"]["issueId"] == "iid-0"

    def test_search_query_uses_sig_prefix(self):
        seen = []
        rep = _report()
        sink = LinearSink("k", "team-1", transport=_transport([], seen))
        asyncio.run(sink.deliver(rep))
        assert seen[0][1]["variables"]["q"] == SIG_PREFIX + rep["signature"]

    def test_connect_error_never_raises(self):
        def boom(request):
            raise httpx.ConnectError("no route to host")
        sink = LinearSink("k", "team-1", transport=httpx.MockTransport(boom))
        result = asyncio.run(sink.deliver(_report()))
        assert result["ok"] is False
        assert result["action"] == "failed"
        assert result["error"]

    def test_graphql_error_becomes_failed_result(self):
        def denied(request):
            return httpx.Response(200, json={
                "errors": [{"message": "authentication failed"}]})
        sink = LinearSink("k", "team-1", transport=httpx.MockTransport(denied))
        result = asyncio.run(sink.deliver(_report()))
        assert result["ok"] is False
        assert result["action"] == "failed"
        assert "authentication failed" in result["error"]


# ═════════════════════════════════════════════════════════════════════
# Routes (handlers awaited directly — no HTTP client, no server start)
# ═════════════════════════════════════════════════════════════════════
# server.py transitively imports the heavy optional ML stack. Stub the
# modules that may not be installed in a test environment so the import
# works on a bare checkout.
for _name in ("voxcpm", "whisperx", "faster_whisper", "pyannote", "edge_tts",
              "demucs"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

server = pytest.importorskip("server")
from app import bugreport as bugreport_mod  # noqa: E402


class _Req:
    """Just enough of a Request for send_bug_report: .json()."""

    def __init__(self, body=None):
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _StubSink:
    name = "stub"

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def deliver(self, report, note=""):
        self.calls.append((report, note))
        return dict(self.result, signature=report.get("signature", ""))


class TestBugReportRoutes:
    def test_get_unknown_job_404(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {})
        resp = asyncio.run(server.get_bug_report("nope"))
        assert resp.status_code == 404

    def test_post_unknown_job_404(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {})
        resp = asyncio.run(server.send_bug_report("nope", _Req()))
        assert resp.status_code == 404

    def test_post_job_without_error_400(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": {"id": "j1",
                                                    "status": "complete"}})
        resp = asyncio.run(server.send_bug_report("j1", _Req()))
        assert resp.status_code == 400
        assert b"no recorded error" in resp.body

    def test_post_unconfigured_400(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: None)
        resp = asyncio.run(server.send_bug_report("j1", _Req()))
        assert resp.status_code == 400
        assert b"linear_api_key" in resp.body

    def test_get_returns_report_and_config_flag(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: None)
        d = asyncio.run(server.get_bug_report("j1"))
        assert d["report"]["job"]["id"] == "job-1"
        assert d["signature"] == d["report"]["signature"]
        assert d["linear_configured"] is False

    def test_post_success_delivers_and_echoes_result(self, monkeypatch):
        job = _fake_job()
        monkeypatch.setattr(server, "jobs", {"j1": job})
        sink = _StubSink({"ok": True, "sink": "stub", "action": "created",
                          "url": "https://linear.app/i/GCD-9",
                          "issue": "GCD-9", "error": ""})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: sink)
        d = asyncio.run(server.send_bug_report(
            "j1", _Req({"note": "was dubbing token=abcd1234secret"})))
        assert d == {"ok": True, "action": "created",
                     "url": "https://linear.app/i/GCD-9", "issue": "GCD-9",
                     "signature": sink.calls[0][0]["signature"]}
        report, note = sink.calls[0]
        assert report["job"]["id"] == "job-1"
        # The note is redacted before it reaches the sink.
        assert "abcd1234secret" not in note
        assert "[redacted]" in note

    def test_post_delivery_failure_502(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        sink = _StubSink({"ok": False, "sink": "stub", "action": "failed",
                          "url": "", "issue": "", "error": "HTTP 500"})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: sink)
        resp = asyncio.run(server.send_bug_report("j1", _Req({})))
        assert resp.status_code == 502
        assert b"HTTP 500" in resp.body

    def test_post_tolerates_missing_body(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        sink = _StubSink({"ok": True, "sink": "stub", "action": "commented",
                          "url": "u", "issue": "GCD-1", "error": ""})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: sink)
        d = asyncio.run(server.send_bug_report("j1", _Req()))   # body raises
        assert d["ok"] is True
        assert sink.calls[0][1] == ""
