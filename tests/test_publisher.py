"""Tests for pipeline/publisher.py — pure helpers only, no network."""
import asyncio

import pytest

from pipeline.publisher import (
    PublishError,
    PublishMeta,
    RuTubeUploader,
    VKUploader,
    build_publish_meta,
    classify_matches,
    get_uploader,
    normalize_title,
    parse_vk_envelope,
    parse_vk_save_response,
    parse_vk_upload_response,
    score_match,
    vk_video_url,
)


# ── normalize_title ──────────────────────────────────────────────────

def test_normalize_lowercase_and_punctuation():
    assert normalize_title("Hello, World! (Official Video)") == "hello world official video"


def test_normalize_collapses_whitespace():
    assert normalize_title("  a\t b \n c  ") == "a b c"


def test_normalize_unicode_kept():
    assert normalize_title("Привет — МИР!") == "привет мир"


def test_normalize_empty_and_none():
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


# ── score_match ──────────────────────────────────────────────────────

def test_score_exact_match():
    s = score_match("My Cool Video", 100, "My Cool Video!", 100)
    assert s == pytest.approx(1.0)


def test_score_translated_ish_title_via_alt():
    # Candidate matches the alt (translated) title, not the original.
    s = score_match(
        "Как приготовить пасту", 300,
        "How to cook pasta", 300,
        alt_title="Как приготовить пасту дома",
    )
    assert s > 0.8


def test_score_off_duration_capped():
    full = score_match("Same Title", 100, "Same Title", 100)
    off = score_match("Same Title", 200, "Same Title", 100)
    assert full == pytest.approx(1.0)
    assert off == pytest.approx(0.5)
    assert off <= full * 0.5 + 1e-9


def test_score_duration_within_tolerance_not_capped():
    assert score_match("Same Title", 104, "Same Title", 100) == pytest.approx(1.0)


def test_score_unknown_duration_not_capped():
    assert score_match("Same Title", None, "Same Title", 100) == pytest.approx(1.0)
    assert score_match("Same Title", 100, "Same Title", None) == pytest.approx(1.0)


def test_score_unrelated_titles_low():
    assert score_match("Cooking pasta at home", 100, "GPU benchmark review", 100) < 0.5


def test_score_bounds():
    s = score_match("", None, "", None)
    assert 0.0 <= s <= 1.0


# ── classify_matches ─────────────────────────────────────────────────

def test_classify_likely_duplicate():
    assert classify_matches([{"score": 0.3}, {"score": 0.85}]) == "likely_duplicate"


def test_classify_possible():
    assert classify_matches([{"score": 0.6}, {"score": 0.2}]) == "possible"


def test_classify_clear():
    assert classify_matches([{"score": 0.59}]) == "clear"
    assert classify_matches([]) == "clear"


# ── PublishMeta defaults ─────────────────────────────────────────────

def test_publish_meta_defaults_private():
    m = PublishMeta()
    assert m.is_private is True
    assert m.tags == []
    # default tags list must not be shared between instances
    m.tags.append("x")
    assert PublishMeta().tags == []


# ── build_publish_meta ───────────────────────────────────────────────

def _job(**over):
    job = {
        "id": "job42",
        "source": "https://youtube.com/watch?v=abc",
        "source_label": "https://youtube.com/watch?v=abc",
        "target_lang": "ru",
    }
    job.update(over)
    return job


def test_build_meta_prefers_translated_title(monkeypatch):
    job = _job(
        title="Original Title",
        meta_translated={"title": "Переведённый заголовок", "description": "Описание."},
    )
    m = build_publish_meta(job)
    assert m.title == "Переведённый заголовок"
    assert m.description.startswith("Описание.")
    assert "https://youtube.com/watch?v=abc" in m.description
    assert m.is_private is True


def test_build_meta_fallback_chain_without_meta_translated():
    assert build_publish_meta(_job(title="Plain Title")).title == "Plain Title"
    assert build_publish_meta(_job(source_label="A Label")).title == "A Label"
    assert build_publish_meta(_job(source_label="", source="")).title == "job42"


def test_build_meta_template_substitution(monkeypatch):
    from app.config import cfg
    monkeypatch.setattr(cfg, "publish_description_template", "Src -> {source_url} <-")
    m = build_publish_meta(_job())
    assert m.description == "Src -> https://youtube.com/watch?v=abc <-"


def test_build_meta_template_with_stray_braces(monkeypatch):
    from app.config import cfg
    monkeypatch.setattr(cfg, "publish_description_template", "{weird} {source_url}")
    m = build_publish_meta(_job())
    assert "{weird}" in m.description
    assert "https://youtube.com/watch?v=abc" in m.description


def test_build_meta_no_source_url_no_attribution():
    m = build_publish_meta(_job(source="/local/file.mp4", meta={}))
    assert "Original:" not in m.description


def test_build_meta_source_url_from_meta_webpage_url():
    m = build_publish_meta(_job(
        source="/uploads/file.mp4",
        meta={"webpage_url": "https://youtu.be/xyz"},
    ))
    assert "https://youtu.be/xyz" in m.description


def test_build_meta_tags_capped_at_10():
    m = build_publish_meta(_job(meta={"tags": [f"t{i}" for i in range(25)]}))
    assert len(m.tags) == 10
    assert m.tags[0] == "t0"


def test_build_meta_translated_tags_win():
    m = build_publish_meta(_job(
        meta={"tags": ["orig"]},
        meta_translated={"tags": ["переведено"]},
    ))
    assert m.tags == ["переведено"]


# ── VK envelope parsing ──────────────────────────────────────────────

def test_envelope_success():
    assert parse_vk_envelope({"response": {"items": []}}) == {"items": []}
    assert parse_vk_envelope({"response": 1}) == 1


@pytest.mark.parametrize("code,fragment", [
    (5, "invalid or expired"),
    (9, "flood control"),
    (15, "denied access"),
    (29, "rate limit"),
])
def test_envelope_known_error_codes(code, fragment):
    payload = {"error": {"error_code": code, "error_msg": "boom"}}
    with pytest.raises(PublishError) as ei:
        parse_vk_envelope(payload)
    assert fragment in str(ei.value)
    assert "boom" in str(ei.value)
    assert ei.value.code == code


def test_envelope_unknown_error_code():
    with pytest.raises(PublishError) as ei:
        parse_vk_envelope({"error": {"error_code": 100, "error_msg": "bad param"}})
    assert "100" in str(ei.value)
    assert ei.value.code == 100


def test_envelope_garbage():
    with pytest.raises(PublishError):
        parse_vk_envelope("<html>")
    with pytest.raises(PublishError):
        parse_vk_envelope({"neither": True})


# ── video.save / upload response parsing ─────────────────────────────

def test_save_response_ok():
    saved = parse_vk_save_response({
        "upload_url": "https://pu.vk.com/x", "owner_id": -123, "video_id": 456,
    })
    assert saved["upload_url"] == "https://pu.vk.com/x"
    assert saved["owner_id"] == -123
    assert saved["video_id"] == 456


def test_save_response_missing_upload_url():
    with pytest.raises(PublishError):
        parse_vk_save_response({"video_id": 1})


def test_upload_response_ok():
    up = parse_vk_upload_response({"size": 1234, "video_id": 789})
    assert up == {"video_id": 789, "size": 1234}


def test_upload_response_legacy_vid_key():
    assert parse_vk_upload_response({"vid": 5})["video_id"] == 5


def test_upload_response_error_or_empty():
    with pytest.raises(PublishError):
        parse_vk_upload_response({"error": "file corrupted"})
    with pytest.raises(PublishError):
        parse_vk_upload_response({"size": 0})


def test_vk_video_url():
    assert vk_video_url(-123, 456) == "https://vk.com/video-123_456"


# ── factory + stubs ──────────────────────────────────────────────────

def test_get_uploader_vk():
    up = get_uploader("vk")
    assert isinstance(up, VKUploader)
    assert up.name == "vk"


def test_get_uploader_rutube_stub_raises():
    up = get_uploader("RuTube")
    assert isinstance(up, RuTubeUploader)
    with pytest.raises(PublishError, match="planned"):
        asyncio.run(up.validate_credentials())
    with pytest.raises(PublishError, match="planned"):
        asyncio.run(up.upload("/tmp/x.mp4", PublishMeta(title="t")))


def test_get_uploader_unknown():
    with pytest.raises(PublishError):
        get_uploader("dailymotion")


# ── VKUploader guards (no network) ───────────────────────────────────

def test_vk_call_without_token_raises():
    up = VKUploader(access_token="", group_id="")
    with pytest.raises(PublishError, match="access token"):
        asyncio.run(up._call("users.get"))


def test_vk_upload_missing_file():
    up = VKUploader(access_token="tok", group_id="")
    with pytest.raises(PublishError, match="not found"):
        asyncio.run(up.upload("/nonexistent/video.mp4", PublishMeta(title="t")))


# ── can_transition (Phase 3C state machine) ──────────────────────────

def test_can_transition_full_matrix():
    """Exhaustive matrix: every (from_status, action) pair."""
    from pipeline.publisher import PUBLISH_STATUSES, can_transition

    expected = {
        # action: set of from-statuses where it is legal (None = never staged)
        "stage": {None, "staged", "approved", "uploaded", "failed", "cancelled"},
        "approve": {"staged", "failed"},
        "cancel": {"staged", "approved", "failed"},
        "upload": {"approved"},
    }
    all_from = [None] + list(PUBLISH_STATUSES)
    for action, legal in expected.items():
        for frm in all_from:
            assert can_transition(frm, action) is (frm in legal), (
                f"can_transition({frm!r}, {action!r})"
            )


def test_can_transition_never_from_uploading():
    from pipeline.publisher import can_transition
    for action in ("stage", "approve", "cancel", "upload"):
        assert can_transition("uploading", action) is False


def test_can_transition_unknown_action():
    from pipeline.publisher import can_transition
    assert can_transition("staged", "publish") is False
    assert can_transition(None, "") is False


def test_can_transition_empty_status_means_unstaged():
    from pipeline.publisher import can_transition
    # "" and None both mean "no publish dict yet" — staging is legal.
    assert can_transition("", "stage") is True
    assert can_transition(None, "stage") is True
    assert can_transition("", "approve") is False


def test_pending_statuses_cover_inbox_states():
    from pipeline.publisher import PUBLISH_PENDING_STATUSES
    assert set(PUBLISH_PENDING_STATUSES) == {
        "staged", "approved", "uploading", "failed"}
