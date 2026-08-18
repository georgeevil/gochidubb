"""Tests for the activity feed buffer (app/activity.py).

The feed's ordering and filtering are what the UI relies on, and they are pure
functions over an in-memory deque, so they are cheap to pin down here rather
than by watching a browser.
"""
import pytest

from app import activity


@pytest.fixture(autouse=True)
def _clean_buffer():
    activity.clear()
    yield
    activity.clear()


def test_recent_is_newest_first():
    activity.record_tool_call("dub", "mcp/claude-code")
    activity.record_job("job_a", "downloading")
    activity.record_tool_call("get_status", "mcp/claude-code")

    events = activity.recent()
    assert [e["kind"] for e in events] == ["tool_call", "job", "tool_call"]
    assert events[0]["tool"] == "get_status"
    # ids are monotonic, so newest-first means descending ids.
    assert [e["id"] for e in events] == sorted(
        (e["id"] for e in events), reverse=True)


def test_kinds_filter():
    activity.record_tool_call("dub", "cli")
    activity.record_job("job_a", "queued")
    activity.record_system("server started")

    assert {e["kind"] for e in activity.recent(kinds=["job"])} == {"job"}
    assert {e["kind"] for e in activity.recent(kinds=["job", "system"])} == {
        "job", "system"}
    assert activity.recent(kinds=["run"]) == []


def test_since_id_returns_only_newer_events():
    activity.record_tool_call("dub", "cli")
    second = activity.record_job("job_a", "queued")
    activity.record_job("job_a", "downloading")

    newer = activity.recent(since_id=second["id"])
    assert [e["id"] for e in newer] == [second["id"] + 1]
    # Asking from the newest id yields nothing rather than repeating it.
    assert activity.recent(since_id=activity.last_id()) == []


def test_since_id_distinguishes_events_in_the_same_instant():
    """Two events can share a timestamp; ids must still separate them.

    This is the reason the API pages on id and not on time.
    """
    a = activity.record_tool_call("get_status", "cli")
    b = activity.record_tool_call("get_status", "cli")
    assert a["id"] != b["id"]
    assert [e["id"] for e in activity.recent(since_id=a["id"])] == [b["id"]]


def test_limit_takes_the_newest():
    for i in range(10):
        activity.record_job("job_a", f"stage_{i}")
    events = activity.recent(limit=3)
    assert [e["status"] for e in events] == ["stage_9", "stage_8", "stage_7"]


def test_buffer_is_bounded():
    for i in range(activity.DEFAULT_CAPACITY + 25):
        activity.record_job("job_a", f"s{i}")
    # Oldest entries fall off; ids keep counting.
    assert len(activity.recent(limit=10_000)) == activity.DEFAULT_CAPACITY
    assert activity.last_id() == activity.DEFAULT_CAPACITY + 25


def test_secrets_are_redacted_on_the_way_in():
    activity.record_tool_call(
        "dub", "cli", detail="using token hf_abcdefghijklmnopqrstuvwxyz012345")
    detail = activity.recent()[0]["detail"]
    assert "hf_abcdefghijklmnopqrstuvwxyz012345" not in detail


def test_none_fields_are_dropped_not_stored_as_null():
    activity.record_tool_call("dub", "cli")
    ev = activity.recent()[0]
    assert "job_id" not in ev and "status" not in ev and "ms" not in ev
    assert ev["tool"] == "dub" and ev["actor"] == "cli"
