"""Tests for app/scout.py — trend discovery, normalization, cache, fallback.

No live network: the mostviewed.today shape comes from a saved fixture
(tests/fixtures/mostviewed_global.json, captured 2026-08-04) and the yt-dlp
fallback is fed canned --flat-playlist --dump-json lines.
"""
import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from app import scout

FIXTURE = Path(__file__).parent / "fixtures" / "mostviewed_global.json"


@pytest.fixture
def fixture_entries():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _clean_cache():
    scout.clear_cache()
    yield
    scout.clear_cache()
    scout.time_fn = __import__("time").time


# ── parse_duration_sec ───────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("189s", 189),
    (189, 189),
    (189.7, 189),
    ("189", 189),
    ("3:09", 189),
    ("1:02:03", 3723),
    ("0:59", 59),
    (None, None),
    ("", None),
    ("abc", None),
    ("1:2:3:4", None),
    (True, None),
])
def test_parse_duration_variants(value, expected):
    assert scout.parse_duration_sec(value) == expected


# ── normalize_candidate ──────────────────────────────────────────────

def test_normalize_candidate_music_entry(fixture_entries):
    cand = scout.normalize_candidate(fixture_entries[0])
    assert cand["video_id"] == "CwKb85C0ZuA"
    assert cand["url"] == "https://www.youtube.com/watch?v=CwKb85C0ZuA"
    assert cand["title"].startswith("Meek Mill")
    assert cand["channel"] == "Meek Mill"
    assert cand["duration_sec"] == 189          # "189s"
    assert cand["view_count"] == 491731
    assert cand["rank"] == 1
    assert cand["category_id"] == 10
    assert cand["is_music"] is True
    assert cand["is_short"] is False
    assert cand["thumb_url"].startswith("https://i.ytimg.com/")


def test_normalize_candidate_non_music(fixture_entries):
    cand = scout.normalize_candidate(fixture_entries[1])  # category_id 20
    assert cand["is_music"] is False
    assert cand["duration_sec"] == 3198


def test_normalize_candidate_missing_fields():
    cand = scout.normalize_candidate({"id": "abc123"})
    assert cand["video_id"] == "abc123"
    assert cand["title"] == ""
    assert cand["duration_sec"] is None
    assert cand["view_count"] is None
    assert cand["is_music"] is False
    assert cand["is_short"] is False


def test_normalize_candidate_short_flag_int():
    # API sends is_short as 0/1 ints
    assert scout.normalize_candidate({"id": "x", "is_short": 1})["is_short"] is True
    assert scout.normalize_candidate({"id": "x", "is_short": 0})["is_short"] is False


# ── fetch_trending: shorts filtering ─────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def _patch_primary(monkeypatch, entries, calls=None):
    async def fake_fetch(category, country, limit):
        if calls is not None:
            calls.append((category, country, limit))
        return entries

    monkeypatch.setattr(scout, "_fetch_mostviewed", fake_fetch)


def test_fetch_trending_filters_shorts(monkeypatch, fixture_entries):
    entries = fixture_entries + [dict(fixture_entries[0], id="short1",
                                      is_short=1, rank=6)]
    _patch_primary(monkeypatch, entries)

    result = _run(scout.fetch_trending(limit=10))
    assert result["source"] == "mostviewed"
    ids = [c["video_id"] for c in result["candidates"]]
    assert "short1" not in ids
    assert len(ids) == 5


def test_fetch_trending_keeps_shorts_when_asked(monkeypatch, fixture_entries):
    entries = fixture_entries + [dict(fixture_entries[0], id="short1",
                                      is_short=1, rank=6)]
    _patch_primary(monkeypatch, entries)

    result = _run(scout.fetch_trending(limit=10, include_shorts=True))
    ids = [c["video_id"] for c in result["candidates"]]
    assert "short1" in ids


# ── annotate_already_processed ───────────────────────────────────────

def test_annotate_already_processed_match(fixture_entries):
    candidates = [scout.normalize_candidate(e) for e in fixture_entries[:2]]
    jobs = {
        "j1": {"id": "j1", "meta": {"video_id": "CwKb85C0ZuA"}},
        "j2": {"id": "j2", "meta": {}},
        "j3": {"id": "j3"},  # no meta at all
    }
    scout.annotate_already_processed(candidates, jobs)
    assert candidates[0]["already_processed"] is True
    assert candidates[0]["existing_job_id"] == "j1"
    assert candidates[1]["already_processed"] is False
    assert "existing_job_id" not in candidates[1]


def test_annotate_already_processed_no_jobs(fixture_entries):
    candidates = [scout.normalize_candidate(fixture_entries[0])]
    scout.annotate_already_processed(candidates, {})
    assert candidates[0]["already_processed"] is False


# ── cache behavior (injected clock) ──────────────────────────────────

def test_cache_fresh_hit_then_expiry(monkeypatch, fixture_entries):
    calls = []
    _patch_primary(monkeypatch, fixture_entries, calls)
    clock = {"now": 1000.0}
    monkeypatch.setattr(scout, "time_fn", lambda: clock["now"])

    r1 = _run(scout.fetch_trending(limit=5))
    assert len(calls) == 1

    # Within TTL: served from cache, no second fetch.
    clock["now"] += scout.CACHE_TTL_SEC - 1
    r2 = _run(scout.fetch_trending(limit=5))
    assert len(calls) == 1
    assert r2["candidates"] == r1["candidates"]

    # Past TTL: refetches.
    clock["now"] += 2
    _run(scout.fetch_trending(limit=5))
    assert len(calls) == 2


def test_cache_key_varies_by_params(monkeypatch, fixture_entries):
    calls = []
    _patch_primary(monkeypatch, fixture_entries, calls)
    monkeypatch.setattr(scout, "time_fn", lambda: 1000.0)

    _run(scout.fetch_trending(limit=5))
    _run(scout.fetch_trending(category="music", limit=5))
    _run(scout.fetch_trending(country="usa", limit=5))
    assert len(calls) == 3


def test_cache_refetches_for_larger_limit(monkeypatch, fixture_entries):
    calls = []
    _patch_primary(monkeypatch, fixture_entries, calls)
    monkeypatch.setattr(scout, "time_fn", lambda: 1000.0)

    _run(scout.fetch_trending(limit=3))
    _run(scout.fetch_trending(limit=2))   # smaller: cache hit, sliced
    assert len(calls) == 1
    _run(scout.fetch_trending(limit=10))  # larger than fetched: refetch
    assert len(calls) == 2


# ── yt-dlp fallback ──────────────────────────────────────────────────

FLAT_LINES = "\n".join([
    json.dumps({"id": "vid1", "title": "First Trending",
                "uploader": "Chan One", "duration": 212.0,
                "view_count": 1000000,
                "url": "https://www.youtube.com/watch?v=vid1"}),
    json.dumps({"id": "vid2", "title": "Second Trending",
                "channel": "Chan Two", "duration": None,
                "view_count": None,
                "url": "https://www.youtube.com/watch?v=vid2"}),
    json.dumps({"id": "vid3", "title": "A Short",
                "uploader": "Chan Three", "duration": 42,
                "url": "https://www.youtube.com/shorts/vid3"}),
])


def test_parse_flat_playlist_lines():
    cands = scout.parse_flat_playlist_lines(FLAT_LINES)
    assert [c["video_id"] for c in cands] == ["vid1", "vid2"]  # short dropped
    assert cands[0]["channel"] == "Chan One"
    assert cands[0]["duration_sec"] == 212
    assert cands[0]["view_count"] == 1000000
    assert cands[0]["rank"] == 1
    assert cands[0]["is_music"] is False
    assert cands[1]["channel"] == "Chan Two"   # falls back to "channel" key
    assert cands[1]["duration_sec"] is None
    assert cands[1]["rank"] == 2
    assert cands[1]["url"] == "https://www.youtube.com/watch?v=vid2"


def test_parse_flat_playlist_lines_include_shorts():
    cands = scout.parse_flat_playlist_lines(FLAT_LINES, include_shorts=True)
    assert [c["video_id"] for c in cands] == ["vid1", "vid2", "vid3"]
    assert cands[2]["is_short"] is True
    # canonical watch URL even for shorts
    assert cands[2]["url"] == "https://www.youtube.com/watch?v=vid3"


def test_parse_flat_playlist_tolerates_garbage_lines():
    text = "not json\n\n" + FLAT_LINES + "\n{\"no_id\": true}\n"
    cands = scout.parse_flat_playlist_lines(text)
    assert [c["video_id"] for c in cands] == ["vid1", "vid2"]


def test_fallback_engages_when_primary_fails(monkeypatch):
    async def boom(category, country, limit):
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr(scout, "_fetch_mostviewed", boom)

    def fake_run(cmd, **kwargs):
        assert "--flat-playlist" in cmd and "--dump-json" in cmd
        assert scout.YT_TRENDING_URL in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=FLAT_LINES, stderr="")

    monkeypatch.setattr(scout.subprocess, "run", fake_run)

    result = _run(scout.fetch_trending(limit=5))
    assert result["source"] == "ytdlp_fallback"
    assert [c["video_id"] for c in result["candidates"]] == ["vid1", "vid2"]


def test_fallback_second_feed_carries_music_hint(monkeypatch):
    """If /feed/trending is dead (YouTube removed it mid-2025), the Top-100
    music playlist serves — and its candidates are flagged is_music."""
    async def boom(category, country, limit):
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr(scout, "_fetch_mostviewed", boom)

    def fake_run(cmd, **kwargs):
        if scout.YT_TRENDING_URL in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="",
                stderr="ERROR: [youtube:tab] trending: does not exist")
        assert scout.YT_TOP_MUSIC_PLAYLIST in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=FLAT_LINES, stderr="")

    monkeypatch.setattr(scout.subprocess, "run", fake_run)

    result = _run(scout.fetch_trending(limit=5))
    assert result["source"] == "ytdlp_fallback"
    assert all(c["is_music"] for c in result["candidates"])


def test_both_sources_failing_raises(monkeypatch):
    async def boom(category, country, limit):
        raise RuntimeError("HTTP 403")

    def run_boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(scout, "_fetch_mostviewed", boom)
    monkeypatch.setattr(scout.subprocess, "run", run_boom)

    with pytest.raises(RuntimeError) as exc:
        _run(scout.fetch_trending(limit=5))
    msg = str(exc.value)
    assert "HTTP 403" in msg and "yt-dlp" in msg
