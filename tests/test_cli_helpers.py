"""Pure-function tests for tools/gochidubb_cli.py helpers.

Only the logic factored out of the CLI handlers is covered here — no HTTP
mocking, per the repo's convention that tools/ is exercised end-to-end.
"""
from datetime import datetime, timedelta, timezone

import pytest

from tools.gochidubb_cli import parse_when, _fmt_count, _fmt_dur


# ── parse_when ───────────────────────────────────────────────────────

def test_parse_when_epoch_string():
    assert parse_when("1754300000") == 1754300000.0


def test_parse_when_epoch_float_string():
    assert parse_when("1754300000.5") == 1754300000.5


def test_parse_when_iso_naive_is_local_time():
    # Naive ISO must be interpreted as local time — exactly what
    # datetime.timestamp() does for naive datetimes.
    s = "2026-08-05T02:00:00"
    assert parse_when(s) == datetime(2026, 8, 5, 2, 0, 0).timestamp()


def test_parse_when_iso_space_separator():
    s = "2026-08-05 02:00"
    assert parse_when(s) == datetime(2026, 8, 5, 2, 0).timestamp()


def test_parse_when_iso_with_offset_is_exact():
    tz = timezone(timedelta(hours=3))
    expected = datetime(2026, 8, 5, 2, 0, tzinfo=tz).timestamp()
    assert parse_when("2026-08-05T02:00:00+03:00") == expected


def test_parse_when_strips_whitespace():
    assert parse_when("  1754300000  ") == 1754300000.0


@pytest.mark.parametrize("bad", ["", "   ", "tomorrow", "2026-13-45", "5pm"])
def test_parse_when_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_when(bad)


# ── table formatting helpers ─────────────────────────────────────────

def test_fmt_count():
    assert _fmt_count(None) == "?"
    assert _fmt_count(999) == "999"
    assert _fmt_count(1_234) == "1.2K"
    assert _fmt_count(5_600_000) == "5.6M"
    assert _fmt_count(2_100_000_000) == "2.1B"


def test_fmt_dur():
    assert _fmt_dur(None) == "?"
    assert _fmt_dur(5) == "0:05"
    assert _fmt_dur(125) == "2:05"
    assert _fmt_dur(3725) == "1:02:05"
