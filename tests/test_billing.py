"""Tests for the usage meter (app/billing.py).

Tier arithmetic is exactly the kind of thing that looks right and is off by a
band, so the boundaries get pinned down explicitly. The other thing worth
guarding is that failed work is never billed.
"""
import pytest

from app import billing


def test_first_band_is_linear():
    r = billing.price_minutes(100)
    assert r["cost"] == pytest.approx(100 * 0.080)
    assert r["rate"] == 0.080


def test_bands_are_cumulative_not_replacing():
    # 600 min = 500 @ 0.080 + 100 @ 0.065, NOT 600 @ 0.065.
    r = billing.price_minutes(600)
    assert r["cost"] == pytest.approx(500 * 0.080 + 100 * 0.065)
    assert r["rate"] == 0.065


def test_third_band():
    # 2,500 min = 500@.08 + 1500@.065 + 500@.05
    r = billing.price_minutes(2500)
    assert r["cost"] == pytest.approx(500 * 0.080 + 1500 * 0.065 + 500 * 0.050)
    assert r["rate"] == 0.050


@pytest.mark.parametrize("minutes,expected", [
    (0, 0.080), (499.9, 0.080),
    (500, 0.065), (1999.9, 0.065),
    (2000, 0.050), (10_000, 0.050),
])
def test_tier_boundaries(minutes, expected):
    """The rate must change exactly at the boundary, not one minute either side."""
    assert billing.tier_for(minutes) == expected


def test_zero_and_negative_minutes_cost_nothing():
    assert billing.price_minutes(0)["cost"] == 0
    assert billing.price_minutes(-5)["cost"] == 0


def test_next_tier_countdown():
    nxt = billing.minutes_to_next_tier(400)
    assert nxt["minutes"] == 100 and nxt["next_rate"] == 0.065
    assert billing.minutes_to_next_tier(2500) is None


def test_job_minutes_multiplies_by_language_count():
    assert billing.job_minutes({"duration": 120, "status": "complete"}) == 2.0
    multi = {"duration": 120, "status": "complete",
             "target_langs": ["fr", "es", "ja"]}
    assert billing.job_minutes(multi) == 6.0


@pytest.mark.parametrize("status", ["error", "cancelled", "failed"])
def test_failed_jobs_are_never_billed(status):
    assert billing.job_minutes({"duration": 600, "status": status}) == 0.0


def test_job_without_duration_is_not_guessed():
    assert billing.job_minutes({"status": "complete"}) == 0.0
    assert billing.job_minutes({"duration": 0, "status": "complete"}) == 0.0


def test_summarize_counts_only_real_work():
    jobs = [
        {"id": "a", "duration": 600, "status": "complete", "target_lang": "fr"},
        {"id": "b", "duration": 600, "status": "error", "target_lang": "es"},
        {"id": "c", "duration": 300, "status": "complete", "target_lang": "fr"},
    ]
    s = billing.summarize(jobs)
    assert s["minutes"] == pytest.approx(15.0)   # 10 + 5, the error ignored
    assert s["jobs_counted"] == 2
    assert s["jobs_unbilled"] == 1
    assert s["by_lang"]["fr"] == pytest.approx(15.0)
    assert "es" not in s["by_lang"]
    assert s["estimate"] is True


def test_summarize_honours_the_since_window():
    jobs = [
        {"id": "old", "duration": 600, "status": "complete", "created": 100},
        {"id": "new", "duration": 600, "status": "complete", "created": 900},
    ]
    assert billing.summarize(jobs, since=500)["jobs_counted"] == 1


def test_showcase_surcharge_applies_on_top():
    jobs = [{"id": "s", "duration": 600, "status": "complete",
             "batch_kind": "showcase", "target_lang": "fr"}]
    s = billing.summarize(jobs)
    assert s["showcase_minutes"] == pytest.approx(10.0)
    assert s["showcase_surcharge"] == pytest.approx(10.0 * 0.020)
    assert s["cost"] == pytest.approx(10.0 * 0.080 + 10.0 * 0.020)


def test_storage_is_free_up_to_the_included_allowance():
    jobs = [{"id": "a", "duration": 60, "status": "complete"}]
    assert billing.summarize(jobs, storage_gb=50)["storage_cost"] == 0
    assert billing.summarize(jobs, storage_gb=100)["storage_cost"] == 0
    over = billing.summarize(jobs, storage_gb=150)
    assert over["storage_cost"] == pytest.approx(50 * 0.02)


def test_bands_sum_to_the_total_cost():
    r = billing.price_minutes(3000)
    assert sum(b["cost"] for b in r["bands"]) == pytest.approx(r["cost"])


def test_every_band_is_reported_even_when_unreached():
    """The screen shows the whole schedule with your position in it.

    Reporting only the band you have reached hides the cheaper rates ahead,
    which is exactly what the first cut of this did.
    """
    r = billing.price_minutes(100)
    assert len(r["bands"]) == len(billing.TIERS)
    assert r["bands"][0]["minutes"] == pytest.approx(100)
    assert [b["minutes"] for b in r["bands"][1:]] == [0, 0]
    # And the unreached bands still advertise their rate.
    assert [b["rate"] for b in r["bands"]] == [0.080, 0.065, 0.050]
