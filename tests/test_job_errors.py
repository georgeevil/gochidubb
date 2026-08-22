"""Tests for structured job errors: _set_job_error / _clear_job_error.

The structure is what the UI's error panel and the "logs at failure" deep
link render, so the shape here is a contract: stage, message, traceback
tail, timestamp, and the [log_from, log_to] ring-seq window.
"""
import sys
import types
from pathlib import Path

import pytest

# server.py transitively imports the heavy optional ML stack. Stub the
# modules that may not be installed in a test environment so the import
# works on a bare checkout.
for _name in ("voxcpm", "whisperx", "faster_whisper", "pyannote", "edge_tts", "demucs"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
server = pytest.importorskip("server")

from app import logbuf  # noqa: E402


def _boom():
    raise ValueError("synthesis exploded")


class TestSetJobError:
    def test_records_structure_and_legacy_string(self):
        job = {"stage_id": "tts", "_stage_log_from": 3}
        try:
            _boom()
        except ValueError as e:
            err = server._set_job_error(job, e)

        assert job["error"] == "synthesis exploded"          # legacy consumers
        le = job["last_error"]
        assert le is err
        assert le["stage"] == "tts"
        assert le["type"] == "ValueError"
        assert le["message"] == "synthesis exploded"
        assert "_boom" in le["traceback_tail"]
        assert le["ts"] > 0
        assert le["log_from"] == 3
        assert le["log_to"] >= 0

    def test_explicit_stage_wins_over_job_stage_id(self):
        job = {"stage_id": "translate"}
        server._set_job_error(job, RuntimeError("x"), stage_id="download")
        assert job["last_error"]["stage"] == "download"

    def test_plain_string_message_has_no_traceback(self):
        job = {}
        server._set_job_error(job, "No saved state to retry", stage_id="tts")
        le = job["last_error"]
        assert le["message"] == "No saved state to retry"
        assert le["type"] == ""
        assert le["traceback_tail"] == ""

    def test_message_falls_back_to_type_name_for_empty_str(self):
        job = {}
        server._set_job_error(job, KeyError())
        assert job["last_error"]["type"] == "KeyError"
        assert job["error"]  # never an empty string in the UI

    def test_history_accumulates_and_is_capped(self):
        job = {}
        for i in range(12):
            server._set_job_error(job, RuntimeError(f"fail {i}"))
        hist = job["error_history"]
        assert len(hist) == 8                                # capped
        assert hist[-1]["message"] == "fail 11"              # newest kept
        # History entries stay small: no traceback duplication per attempt.
        assert all("traceback_tail" not in h for h in hist)

    def test_stage_label_resolved_from_pipeline_spec(self):
        job = {}
        sid = server.PIPELINE_STAGES[0]["id"]
        server._set_job_error(job, RuntimeError("x"), stage_id=sid)
        assert job["last_error"]["stage_label"] == server.STAGE_BY_ID[sid]["label"]


class TestClearJobError:
    def test_clears_error_but_keeps_history(self):
        job = {"stage_id": "tts"}
        server._set_job_error(job, RuntimeError("first failure"))
        job["failed_stage"] = "tts"
        job["stale_from_restart"] = True

        server._clear_job_error(job)

        assert "error" not in job
        assert "last_error" not in job
        assert "failed_stage" not in job
        assert "stale_from_restart" not in job
        assert len(job["error_history"]) == 1                # survives retries

    def test_noop_on_clean_job(self):
        job = {"status": "queued"}
        server._clear_job_error(job)
        assert job == {"status": "queued"}


class TestLogWindow:
    def test_log_to_tracks_ring_growth(self):
        job = {"_stage_log_from": logbuf.current_seq()}
        logbuf.ring.add("ERROR", "test", "stage output line")
        server._set_job_error(job, RuntimeError("x"))
        le = job["last_error"]
        assert le["log_to"] >= le["log_from"] + 1

    def test_current_seq_matches_newest_entry(self):
        entry = logbuf.ring.add("INFO", "test", "hello")
        assert logbuf.current_seq() == entry["seq"]
