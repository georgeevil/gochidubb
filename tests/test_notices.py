"""Tests for pipeline/notices.py — the setup-notice vocabulary."""
import pytest

from pipeline.notices import (
    SEVERITIES,
    mask_secret,
    merge_notices,
    notice,
    redact,
    severity_rank,
    worst_severity,
)


class TestNoticeFactory:
    def test_has_every_field_the_ui_renders(self):
        n = notice("pyannote.no_token", "error", "No token",
                   detail="d", remediation=["step one"], url="https://hf.co",
                   subsystem="diarize")
        for key in ("code", "severity", "title", "detail", "remediation",
                    "url", "subsystem"):
            assert key in n, f"UI reads {key}; a missing key renders as blank"

    def test_title_defaults_to_code(self):
        """A notice with no title still says something identifiable."""
        assert notice("ffmpeg.missing")["title"] == "ffmpeg.missing"

    def test_unknown_severity_becomes_warn_not_dropped(self):
        """A typo in severity must not make a real problem invisible."""
        assert notice("x.y", severity="critical")["severity"] == "warn"

    def test_remediation_is_always_a_list_of_strings(self):
        n = notice("x.y", remediation=("a", "b"))
        assert n["remediation"] == ["a", "b"]


class TestSeverityOrdering:
    def test_error_outranks_warn_outranks_info(self):
        ranks = [severity_rank(notice("x", s)) for s in SEVERITIES]
        assert ranks == sorted(ranks), "SEVERITIES must be ordered worst-first"

    def test_worst_severity_picks_the_worst(self):
        ns = [notice("a", "info"), notice("b", "error"), notice("c", "warn")]
        assert worst_severity(ns) == "error"

    def test_worst_severity_of_nothing_is_empty(self):
        assert worst_severity([]) == ""


class TestMergeNotices:
    def test_dedupes_on_code(self):
        merged = merge_notices([notice("a.b", "warn", "first")],
                               [notice("a.b", "warn", "second")])
        assert len(merged) == 1

    def test_later_layer_wins_a_tie(self):
        """Deep-probe results are layered last and must supersede guesses."""
        merged = merge_notices([notice("a.b", "warn", "passive guess")],
                               [notice("a.b", "warn", "confirmed by the Hub")])
        assert merged[0]["title"] == "confirmed by the Hub"

    def test_worse_severity_survives_regardless_of_order(self):
        merged = merge_notices([notice("a.b", "error", "hard")],
                               [notice("a.b", "info", "soft")])
        assert merged[0]["severity"] == "error"

    def test_sorted_worst_first(self):
        merged = merge_notices([notice("z.z", "info"), notice("a.a", "error")])
        assert [n["code"] for n in merged] == ["a.a", "z.z"]

    def test_ignores_junk_entries(self):
        merged = merge_notices([None, {}, {"no_code": 1}, notice("a.b")])
        assert [n["code"] for n in merged] == ["a.b"]

    def test_handles_none_lists(self):
        assert merge_notices(None, [notice("a.b")]) != []


class TestRedact:
    def test_scrubs_a_hugging_face_token(self):
        out = redact("using hf_QRSTuvWXyz0123456789abcdef for auth")
        assert "hf_QRST" not in out
        assert "[redacted]" in out

    def test_scrubs_key_value_secrets(self):
        assert "supersecretvalue" not in redact("HF_TOKEN=supersecretvalue")
        assert "hunter2000" not in redact('password: "hunter2000"')

    def test_scrubs_bearer_headers(self):
        assert "abcdef1234567890" not in redact("Authorization: Bearer abcdef1234567890")

    def test_leaves_ordinary_text_alone(self):
        msg = "Diarization pipeline loaded: pyannote/speaker-diarization-community-1"
        assert redact(msg) == msg

    def test_none_becomes_empty_string(self):
        assert redact(None) == ""


class TestMaskSecret:
    def test_keeps_a_recognizable_prefix(self):
        assert mask_secret("hf_abcdefghijklmnop") == "hf_a…"

    def test_empty_stays_empty_so_the_ui_can_say_not_set(self):
        assert mask_secret("") == ""

    def test_short_values_reveal_nothing(self):
        assert mask_secret("abc") == "…"
