"""Tests for reupload-only mode (Phase 3E) — pure server helpers.

Importing server is deliberate: normalize_job_mode / stages_for_mode are
module-level pure functions on the coordinator (they read PIPELINE_STAGES),
and server.py imports in ~2s with no model loading.
"""
import server


# ── normalize_job_mode ───────────────────────────────────────────────

def test_mode_default_is_dub():
    assert server.normalize_job_mode(None) == "dub"
    assert server.normalize_job_mode("") == "dub"


def test_mode_known_values():
    assert server.normalize_job_mode("dub") == "dub"
    assert server.normalize_job_mode("reupload") == "reupload"


def test_mode_case_and_whitespace():
    assert server.normalize_job_mode("  Reupload ") == "reupload"
    assert server.normalize_job_mode("DUB") == "dub"


def test_mode_unknown_rejected():
    assert server.normalize_job_mode("remix") is None
    assert server.normalize_job_mode("upload") is None
    assert server.normalize_job_mode(42) is None


# ── stages_for_mode ──────────────────────────────────────────────────

def test_dub_mode_walks_every_stage():
    assert server.stages_for_mode("dub") == server.STAGE_ORDER
    # Defensive: must be a copy, not the module list itself.
    assert server.stages_for_mode("dub") is not server.STAGE_ORDER


def test_default_mode_walks_every_stage():
    assert server.stages_for_mode("") == server.STAGE_ORDER
    assert server.stages_for_mode(None) == server.STAGE_ORDER


def test_reupload_mode_downloads_only():
    assert server.stages_for_mode("reupload") == ["download"]


def test_reupload_stage_is_a_real_stage():
    # The runner indexes stages_for_mode output into PIPELINE_STAGES —
    # every id it returns must exist there.
    for mode in ("dub", "reupload"):
        for sid in server.stages_for_mode(mode):
            assert sid in server.STAGE_ORDER
