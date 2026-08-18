"""Usage metering — what a hosted workspace *would* bill for this activity.

The design's Billing & Usage screen prices dubbing per minute of source, per
language, on a descending volume tier. This module implements exactly that
arithmetic over the jobs this server has actually run.

Read the honesty boundary carefully, because it is the whole point of the
module:

* The **minutes are real.** They come from ``job["duration"]`` — the measured
  length of the source media — multiplied by the number of target languages,
  which is how the design meters ("per minute of source dubbed, per
  language"). Nothing here is invented.
* The **money is an estimate.** GoChiDUBB in local mode bills nobody; there is
  no account, no invoice and no payment. The rates below are the ones the
  design publishes, applied to real usage so the screen shows what a hosted
  workspace would charge. Every figure it produces must be labelled as an
  estimate in the UI, never as an amount owed.

Rules taken from the design:
  · first 500 min          $0.080 / min
  · 500 – 2,000 min        $0.065 / min
  · 2,000+ min             $0.050 / min
  · showcase stitch        +$0.020 / min
  · failed jobs            never billed
  · storage                100 GB included, then $0.02 / GB / month
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

# (upper bound in minutes or None for "no limit", rate per minute)
TIERS: List[Tuple[Optional[float], float]] = [
    (500.0, 0.080),
    (2000.0, 0.065),
    (None, 0.050),
]

SHOWCASE_SURCHARGE = 0.020      # per minute, on stitched showcase jobs
STORAGE_INCLUDED_GB = 100.0
STORAGE_PER_GB = 0.02

# Statuses that are never billed. A job that failed cost the user nothing, and
# a cancelled one is the same story.
UNBILLED_STATUSES = {"error", "cancelled", "failed"}


def tier_for(total_minutes: float) -> float:
    """The marginal rate that the next minute would be charged at."""
    used = 0.0
    for upper, rate in TIERS:
        if upper is None or total_minutes < upper:
            return rate
        used = upper
    return TIERS[-1][1] if used else TIERS[0][1]


def price_minutes(total_minutes: float) -> Dict[str, Any]:
    """Cost of `total_minutes` under the tier schedule, with the breakdown.

    Tiers are cumulative: the first 500 minutes are always charged at the
    first rate even once the total is far past it.
    """
    total_minutes = max(0.0, float(total_minutes or 0.0))
    remaining = total_minutes
    lower = 0.0
    cost = 0.0
    bands: List[Dict[str, Any]] = []
    for upper, rate in TIERS:
        span = float("inf") if upper is None else max(0.0, upper - lower)
        used = min(max(remaining, 0.0), span)
        # Every band is emitted, including ones this usage has not reached —
        # the screen shows the whole schedule with your position in it, so
        # stopping at the current band would hide the cheaper rates ahead.
        bands.append({
            "from": lower,
            "to": upper,
            "rate": rate,
            "minutes": round(used, 2),
            "cost": round(used * rate, 4),
        })
        cost += used * rate
        remaining -= used
        if upper is not None:
            lower = upper
    return {
        "minutes": round(total_minutes, 2),
        "cost": round(cost, 2),
        "bands": bands,
        "rate": tier_for(total_minutes),
    }


def minutes_to_next_tier(total_minutes: float) -> Optional[Dict[str, Any]]:
    """How far to the next cheaper rate, or None when already on the last."""
    for upper, _rate in TIERS:
        if upper is not None and total_minutes < upper:
            idx = TIERS.index((upper, _rate))
            nxt = TIERS[idx + 1][1] if idx + 1 < len(TIERS) else None
            if nxt is None:
                return None
            return {"minutes": round(upper - total_minutes, 2), "next_rate": nxt}
    return None


def job_minutes(job: Dict[str, Any]) -> float:
    """Billable minutes for one job: source length × target languages.

    Unbilled statuses return 0. A job with no measured duration also returns
    0 rather than a guess — it never got far enough to have cost anything.
    """
    if (job.get("status") or "") in UNBILLED_STATUSES:
        return 0.0
    duration = job.get("duration") or 0.0
    if duration <= 0:
        return 0.0
    langs = job.get("target_langs")
    n = len(langs) if isinstance(langs, (list, tuple)) and langs else 1
    return (float(duration) / 60.0) * n


def is_showcase(job: Dict[str, Any]) -> bool:
    return (job.get("batch_kind") or "") == "showcase"


def summarize(jobs: Iterable[Dict[str, Any]], *,
              since: Optional[float] = None,
              storage_gb: float = 0.0) -> Dict[str, Any]:
    """Estimate for a set of jobs, optionally only those created after `since`.

    Returns real minutes alongside an estimated cost. Callers must present the
    money as an estimate — see the module docstring.
    """
    billable = 0.0
    showcase_minutes = 0.0
    counted = 0
    unbilled = 0
    by_lang: Dict[str, float] = {}

    for job in jobs:
        if since is not None and (job.get("created") or 0) < since:
            continue
        mins = job_minutes(job)
        if mins <= 0:
            if (job.get("status") or "") in UNBILLED_STATUSES:
                unbilled += 1
            continue
        counted += 1
        billable += mins
        if is_showcase(job):
            showcase_minutes += mins
        lang = job.get("target_lang") or "?"
        by_lang[lang] = round(by_lang.get(lang, 0.0) + mins, 2)

    priced = price_minutes(billable)
    surcharge = showcase_minutes * SHOWCASE_SURCHARGE
    storage_billable = max(0.0, float(storage_gb or 0.0) - STORAGE_INCLUDED_GB)
    storage_cost = storage_billable * STORAGE_PER_GB

    return {
        "minutes": round(billable, 2),
        "jobs_counted": counted,
        "jobs_unbilled": unbilled,
        "by_lang": dict(sorted(by_lang.items(), key=lambda kv: -kv[1])),
        "bands": priced["bands"],
        "rate": priced["rate"],
        "next_tier": minutes_to_next_tier(billable),
        "showcase_minutes": round(showcase_minutes, 2),
        "showcase_surcharge": round(surcharge, 2),
        "storage_gb": round(float(storage_gb or 0.0), 2),
        "storage_cost": round(storage_cost, 2),
        "cost": round(priced["cost"] + surcharge + storage_cost, 2),
        # Never let a caller forget what this number is.
        "estimate": True,
        "generated_at": time.time(),
    }
