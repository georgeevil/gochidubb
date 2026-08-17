#!/usr/bin/env python3
"""Benchmark local translation models on the fixtures in tests/fixtures/benchmark.

Answers one question: which of the models you actually have installed can be
trusted to translate a video into a given language, unattended?

Most of what it measures is mechanical reliability, and every one of those
metrics corresponds to a way a real dub has failed:

  fallback     the model did not return a usable translation and the segment
               kept its source text (see pipeline/translator.py). A model can
               be eloquent and still be unusable if it does this one line in
               seven.
  meta-prose   the model narrated the task instead of doing it. Observed on
               google/gemma-4-e4b, which does it on most runs.
  script       output written in the target language's script. Catches "the
               model answered in English" and "the model echoed the source".
  digits       numbers surviving into the translation, in whatever numeral
               system the target uses.
  drift        how far the dub would run behind the video, predicted from
               character count and a measured per-language speech rate.
  spread       slowest minus fastest run. A model whose output changes shape
               run to run cannot be tuned against.

None of those measure whether the translation is CORRECT, and the gap is not
small: aya-expanse-8b passes every one of them into Bulgarian while rendering
"gracias" as "please". That is what semantic checklists are for — hand-written,
per language pair, in tests/fixtures/benchmark/checklists/. Where one exists it
decides the verdict; where none exists the grade can only rule a model out.
Writing one needs no Python, only that you read the language.

Runs models in the outer loop so each one loads into LM Studio exactly once —
swapping per language would spend most of the wall clock on model loads.

    python tools/gochidubb_benchmark.py --targets es,pt,ru,hi,ar
    python tools/gochidubb_benchmark.py --models openai/gpt-oss-20b --runs 3
    python tools/gochidubb_benchmark.py --list
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import translator as T                      # noqa: E402
from pipeline.assembler import plan_segment_fit           # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "benchmark"
DEFAULT_OUT = ROOT / "docs" / "benchmarks"

# Characters of synthesized speech per second, used to predict how long a
# translation takes to say. Character counts are not comparable across scripts
# — a Chinese line is a third the length of its Spanish source and takes about
# as long — so this needs a per-language figure.
#
# 10.6 is MEASURED, from the real VoxCPM2 Bulgarian dub of job c5b12e16.
# Per-language rates come from tools/gochidubb_speech_rate.py, which
# synthesizes real translated lines and divides characters by audio seconds.
# The fallback table below is only used for languages nobody has measured yet;
# those are guesses and are marked as such wherever drift is reported.
BASE_TTS_RATE = 10.6
RATES_FILE = FIXTURE_DIR / "speech_rates.json"
FALLBACK_SCRIPT_SCALE = {
    "han": 0.30, "japanese": 0.42, "hangul": 0.55,
    "devanagari": 0.85, "arabic": 0.90, "thai": 0.80,
}
_RATES_CACHE = None


def _measured_rates() -> dict:
    global _RATES_CACHE
    if _RATES_CACHE is None:
        try:
            _RATES_CACHE = json.loads(
                RATES_FILE.read_text(encoding="utf-8")).get("rates", {})
        except Exception:
            _RATES_CACHE = {}
    return _RATES_CACHE


def tts_rate_for(lang: str) -> float:
    code = (lang or "")[:2].lower()
    measured = _measured_rates().get(code)
    if measured and measured.get("voxcpm_chars_per_sec"):
        return measured["voxcpm_chars_per_sec"]
    script = T._LANG_SCRIPT.get(code)
    return BASE_TTS_RATE * FALLBACK_SCRIPT_SCALE.get(script, 1.0)


def drift_is_measured(lang: str) -> bool:
    """Whether the drift estimate for this target rests on a measured rate."""
    return (lang or "")[:2].lower() in _measured_rates()


CHECKLIST_DIR = FIXTURE_DIR / "checklists"


_CHECKLIST_CACHE: dict = {}


def load_checklist(fixture_id: str, target: str):
    """Hand-written semantic checks for one (fixture, language), or None.

    These are the only thing in the benchmark that judges whether a
    translation is *correct* rather than merely well-formed. They exist per
    language pair because somebody who reads the language had to write them —
    see checklists/README.md.

    A checklist with no checks is treated as absent: it can produce no score,
    and returning it would make callers reason about a scoreless score.
    """
    key = (fixture_id, (target or "")[:2].lower())
    if key not in _CHECKLIST_CACHE:
        path = CHECKLIST_DIR / f"{key[0]}.{key[1]}.json"
        doc = None
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not doc.get("checks"):
                doc = None
        _CHECKLIST_CACHE[key] = doc
    return _CHECKLIST_CACHE[key]


def score_checklist(checklist, texts):
    """Run a checklist over one set of translations.

    Returns passed/total plus the labels that failed, so a score can always be
    traced back to the line that produced it.
    """
    passed, failed = 0, []
    for check in checklist.get("checks", []):
        idx = check["segment"]
        if idx >= len(texts):
            failed.append(check["label"])
            continue
        text = texts[idx] or ""
        ok = bool(re.search(check["expect"], text, re.IGNORECASE))
        if ok and check.get("reject") and re.search(
                check["reject"], text, re.IGNORECASE):
            ok = False
        if ok:
            passed += 1
        else:
            failed.append(check["label"])

    penalties = []
    for pen in checklist.get("penalties", []):
        # Case-insensitive, like the checks above. Matching case-sensitively
        # meant a penalty for "здравей" could not fire on "Здравей" — the
        # sentence-initial form the wrong-language failure actually produces.
        hits = sum(len(re.findall(pen["pattern"], t or "", re.IGNORECASE))
                   for t in texts)
        if hits:
            penalties.append({"label": pen["label"], "hits": hits})

    total = len(checklist.get("checks", []))
    return {
        "passed": passed, "total": total,
        "score": round(passed / total, 3) if total else None,
        "failed": failed,
        "penalties": penalties,
        "penalty_hits": sum(p["hits"] for p in penalties),
    }


def load_fixtures(only=None):
    """Every benchmark fixture in FIXTURE_DIR.

    Files that are not fixtures live here too (speech_rates.json), so identify
    a fixture by its shape rather than by being a .json in the directory.
    """
    out = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or "id" not in doc or "segments" not in doc:
            continue
        if only and doc["id"] not in only:
            continue
        out.append(doc)
    return out


def predict_drift(segments, texts, rate, total_duration, max_stretch=1.4):
    """Seconds the dub would run behind the source, worst segment."""
    next_starts = [segments[i + 1]["start"] for i in range(len(segments) - 1)]
    next_starts.append(total_duration)
    current_end = worst = 0.0
    for i, (seg, text) in enumerate(zip(segments, texts)):
        tts_dur = len(text) / rate
        start, speed = plan_segment_fit(
            seg["start"], next_starts[i], tts_dur, current_end, max_stretch)
        worst = max(worst, start - seg["start"])
        current_end = start + tts_dur / speed
    return worst


# Python's \d already covers Arabic-Indic (٤٠) and Devanagari (४०) numerals as
# well as ASCII. Han numerals are separate codepoints, so zh/ja need them
# added or a correctly-localized "十八" reads as a dropped number.
#
# 一 also means "one" as an ordinary word, so this over-counts on Chinese
# prose. That is the safe direction: the metric only flags *substantial* loss,
# so over-counting risks passing a model, never failing a good one.
_HAN_NUMERALS = "〇零一壹二貳贰三參叁四肆五伍六陸陆七柒八捌九玖十拾百佰千仟万萬億兆"
_DIGITS = re.compile(r"\d")
_DIGITS_CJK = re.compile(r"[\d" + _HAN_NUMERALS + r"]")

# Korean can write Sino-Korean numbers as ordinary Hangul (십팔 = eighteen),
# which no pattern can count. It is still worth checking: across every Korean
# run recorded so far the models kept ASCII digits, and a spelled-out number
# lands the same "spelled a number as a word" tolerance (digits_kept >= 0.5)
# that every other language already gets. Skipping the check outright meant a
# model that simply dropped "18 months" still earned a clean verdict.


def _digit_re(target: str):
    """The numeral pattern that is meaningful for this target language."""
    script = T._LANG_SCRIPT.get((target or "")[:2].lower())
    return _DIGITS_CJK if script in ("han", "japanese") else _DIGITS


def is_translation_pair(source: str, target: str) -> bool:
    """True when translating source→target is actually a translation.

    Compares the base subtag, so es/es-419 and pt/pt-BR count as the same
    language. The correct "translation" of a line into its own language is the
    line itself, which every metric here would read as a total failure.
    """
    return (source or "")[:2].lower() != (target or "")[:2].lower()


def measure(fixture, target, texts, elapsed):
    segments = fixture["segments"]
    sources = [s["text"] for s in segments]
    n = len(sources)

    fallback = sum(1 for s, t in zip(sources, texts) if t.strip() == s.strip())
    meta = sum(1 for t in texts if T._META_PROSE_RE.search(t or ""))

    script = T._LANG_SCRIPT.get(target[:2].lower())
    if script:
        good = 0
        for t in texts:
            ratio, letters = T._script_ratio(t, script)
            good += 1 if (letters < 8 or ratio >= T._MIN_SCRIPT_RATIO) else 0
        script_ok = good / n
    else:
        script_ok = None                      # Latin target: nothing to check

    # Digits carry facts, and dropping them is a translation error a machine
    # can see. The pattern is chosen per target so a correctly localized
    # number still counts: "18" -> "١٨" or "十八" both score as kept.
    digit_re = _digit_re(target)
    src_digits = sum(len(_DIGITS.findall(s)) for s in sources)
    out_digits = sum(len(digit_re.findall(t)) for t in texts)
    digits_kept = (min(out_digits / src_digits, 1.0) if src_digits else None)

    checklist = load_checklist(fixture["id"], target)
    semantic = score_checklist(checklist, texts) if checklist else None

    ratios = [len(t) / max(1, len(s.strip())) for s, t in zip(sources, texts)]
    rate = tts_rate_for(target)
    return {
        "fallback": fallback,
        "segments": n,
        "meta_prose": meta,
        "script_ok": script_ok,
        "digits_kept": digits_kept,
        "semantic": semantic,
        "len_ratio": round(sum(ratios) / n, 3),
        "drift_sec": round(predict_drift(
            segments, texts, rate, fixture["duration"]), 2),
        "elapsed": round(elapsed, 1),
        "texts": texts,
    }


async def run_one(model, fixture, target, timeout):
    segs = [dict(s) for s in fixture["segments"]]
    t0 = time.time()
    try:
        await asyncio.wait_for(
            T.translate_segments(
                segs, target, model=model, max_concurrent=1,
                context_hint=fixture.get("context_hint") or "",
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"error": f"timeout after {timeout}s", "elapsed": timeout}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "elapsed": round(time.time() - t0, 1)}
    texts = [s.get("translated_text") or "" for s in segs]
    return measure(fixture, target, texts, time.time() - t0)


def grade(runs):
    """Collapse repeat runs of one (model, fixture, target) into a verdict.

    Only fully-measured, script-independent signals decide the grade: did it
    answer at all, did it answer in the target language, did it narrate the
    task, did it drop facts. Drift and length ratio are reported but never
    graded on — drift depends on estimated per-script speech rates, and a
    character-count ratio is not comparable across scripts (Arabic output is
    shorter than its Latin source in characters while taking about as long to
    say). Grading on either would rank models by the script they were pointed
    at rather than by how well they did.
    """
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {"grade": "unusable", "reason": runs[0].get("error", "failed")}
    segments = ok[0]["segments"]
    fallback = sum(r["fallback"] for r in ok) / len(ok)
    meta = sum(r["meta_prose"] for r in ok) / len(ok)
    script = min((r["script_ok"] for r in ok if r["script_ok"] is not None),
                 default=None)
    elapsed = sorted(r["elapsed"] for r in ok)
    digits = min((r["digits_kept"] for r in ok if r["digits_kept"] is not None),
                 default=None)

    # Hand-written semantic checks, when someone who reads the language has
    # contributed them. This is the only signal here that judges whether the
    # translation is *right*; without it the grade can only rule a model out.
    sem_runs = [r["semantic"] for r in ok
                if r.get("semantic") and r["semantic"].get("score") is not None]
    sem = None
    if sem_runs:
        # The worst run, whole. Splicing its score together with the highest
        # penalty count seen in any run described no single run: a cell read
        # "3/13 +12" when the 3/13 run had 11 penalties and the 12-penalty
        # run scored differently.
        worst = min(sem_runs, key=lambda s: (s["score"], -s["penalty_hits"]))
        sem = dict(worst)

    if len(ok) < len(runs) or fallback > segments * 0.2 or meta > 0:
        g = "unusable"
    # A model that gets less than two thirds of the checked meaning right is
    # not producing a usable dub, however clean its mechanics are. This is
    # what disqualifies aya-expanse-8b, which passes every mechanical check.
    elif sem and sem["score"] < 0.67:
        g = "unusable"
    # Spelling one number out as a word ("three years" → "три года") is a
    # legitimate choice, so only substantial loss counts against a model.
    elif (fallback > 0 or (script is not None and script < 0.99)
          or (digits is not None and digits < 0.5)):
        g = "risky"
    elif sem and (sem["score"] < 1.0 or sem["penalty_hits"] > 0):
        g = "risky"
    else:
        g = "good"
    return {
        "grade": g, "fallback": round(fallback, 2), "meta_prose": round(meta, 2),
        "script_ok": None if script is None else round(script, 3),
        "digits_kept": None if digits is None else round(digits, 3),
        "semantic": sem,
        "drift_sec": max(r["drift_sec"] for r in ok),
        "median_sec": elapsed[len(elapsed) // 2],
        "spread_sec": round(elapsed[-1] - elapsed[0], 1),
        "runs": len(ok), "segments": segments,
        "len_ratio": round(sum(r["len_ratio"] for r in ok) / len(ok), 3),
    }


GRADE_MARK = {"good": "✅", "risky": "⚠️", "unusable": "❌"}
LANG_NAMES = dict(T.LANGUAGE_NAMES, en="English")


def _verdicts(data, fixtures):
    """{model: {fixture: {target: verdict}}}, re-graded from the raw runs.

    Grades are recomputed rather than read back, so a change to what counts as
    usable re-scores an existing results.json instead of costing another few
    hours of model time. Same-language pairs are dropped: older result files
    may contain them, and they always look like total failures.
    """
    by_id = {f["id"]: f for f in fixtures}
    src_of = {fid: f["source_lang"] for fid, f in by_id.items()}
    out = {}
    for model, per_fixture in data["results"].items():
        for fid, per_target in per_fixture.items():
            for target, cell in per_target.items():
                if not is_translation_pair(src_of.get(fid, ""), target):
                    continue
                # Everything derived from the stored translations is
                # recomputed rather than read back, so a checklist or a speech
                # rate added after a sweep re-scores the results already on
                # disk instead of costing another few hours of model time.
                #
                # Drift especially: the report marks a figure as measured by
                # looking up today's speech_rates.json, so a drift value left
                # over from the old guessed rates would be presented as
                # measured when it is nothing of the kind.
                #
                # Rescored onto copies: this is a read path, and silently
                # rewriting the caller's data would corrupt a results file if
                # anything wrote it back afterwards.
                checklist = load_checklist(fid, target)
                fixture = by_id.get(fid)
                runs = []
                for run in cell.get("runs") or []:
                    if "error" in run or not run.get("texts"):
                        runs.append(run)
                        continue
                    fresh = dict(run)
                    fresh["semantic"] = (score_checklist(checklist, run["texts"])
                                         if checklist else None)
                    if fixture:
                        fresh["drift_sec"] = round(predict_drift(
                            fixture["segments"], run["texts"],
                            tts_rate_for(target), fixture["duration"]), 2)
                    runs.append(fresh)
                verdict = grade(runs) if runs else cell.get("verdict")
                out.setdefault(model, {}).setdefault(fid, {})[target] = verdict
    return out


def render_report(data, fixtures) -> str:
    """Turn results.json into the decision matrix, so the doc can't drift."""
    targets = data["targets"]
    by_id = {f["id"]: f for f in fixtures}
    src_of = {fid: by_id[fid]["source_lang"] for fid in data["fixtures"]
              if fid in by_id}
    results = _verdicts(data, fixtures)
    out = []

    out.append("# Translation model decision matrix\n")
    out.append("<!-- Generated by tools/gochidubb_benchmark.py --report. "
               "Do not edit by hand. -->\n")
    out.append("Which local models can be trusted to translate a video into a "
               "given language, unattended. Regenerate with:\n")
    out.append("```bash\npython tools/gochidubb_benchmark.py --targets "
               + ",".join(targets) +
               "\npython tools/gochidubb_benchmark.py --report > "
               "docs/benchmarks/decision-matrix.md\n```\n")
    out.append("Results reflect one machine's installed models at one point in "
               "time. Re-run it on yours — that is the point of shipping the "
               "benchmark rather than only its conclusions. Background, failure "
               "modes and a worked example are in "
               "[choosing a translation model](../choosing-a-translation-model.md).\n")

    out.append("> **This table rules models out. It cannot rule them in.**\n>\n"
               "> Every metric here is mechanical: did the model answer, in the "
               "target language, keeping the numbers, every time. None of it "
               "measures whether the translation is *correct*. `aya-expanse-8b` "
               "scores ✅ across most of this matrix and is the model that "
               "ruined a real dub — it renders Spanish *gracias* as Bulgarian "
               "\"please\" in fluent, correctly-scripted, correctly-sized "
               "output. A ❌ or ⚠️ here is strong evidence against a model. A ✅ "
               "means only that it cleared the bar where machines can judge; "
               "somebody who reads the language still has to check the "
               "output.\n")

    out.append("## Decision matrix\n")
    out.append("Worst verdict across every source language tested, so a model "
               "earns ✅ only if it held up on all of them.\n")
    header = "| Model | " + " | ".join(
        f"{LANG_NAMES.get(t, t)} (`{t}`)" for t in targets) + " |"
    out.append(header)
    out.append("|" + "---|" * (len(targets) + 1))

    order = {"good": 0, "risky": 1, "unusable": 2}
    for model, per_fixture in results.items():
        cells = []
        for t in targets:
            verdicts = [per_fixture[f][t] for f in per_fixture
                        if t in per_fixture[f]]
            if not verdicts:
                cells.append("—")
                continue
            worst = max(verdicts, key=lambda v: order[v["grade"]])
            timed = [v["median_sec"] for v in verdicts
                     if v.get("median_sec") is not None]
            mark = GRADE_MARK[worst["grade"]]
            # No timing at all means every run errored. Showing "0s" there
            # reads as instant when it was usually a timeout.
            cells.append(f"{mark} {max(timed):.0f}s" if timed else f"{mark} n/a")
        out.append(f"| `{model}` | " + " | ".join(cells) + " |")

    out.append("\n✅ clean on every run, and where a semantic checklist exists, "
               "correct  ·  ⚠️ dropped a line, slipped script, lost a number, "
               "or missed some checked meaning  ·  ❌ failed outright, timed "
               "out, narrated the task, or got less than two thirds of the "
               "checked meaning right\n")
    out.append("Times are the slowest median across source languages, after "
               "the model is loaded. A cell shows the worst result that model "
               "had for that target across every source language tested.\n")

    out.append("\n## Per-source detail\n")
    for fid in data["fixtures"]:
        if fid not in by_id:
            continue
        src = src_of[fid]
        pair_targets = [t for t in targets if is_translation_pair(src, t)]
        scored = [t for t in pair_targets if load_checklist(fid, t)]
        out.append(f"\n### `{fid}` — {LANG_NAMES.get(src, src)} source\n")
        head = "| Model | " + " | ".join(f"→ `{t}`" for t in pair_targets)
        if scored:
            head += " | " + " | ".join(f"`{t}` meaning" for t in scored)
        head += " | Length vs source | Est. drift |"
        out.append(head)
        out.append("|" + "---|" * (len(pair_targets) + len(scored) + 3))
        for model, per_fixture in results.items():
            if fid not in per_fixture:
                continue
            row, ratios, drifts = [], [], []
            for t in pair_targets:
                v = per_fixture[fid].get(t)
                if not v:
                    row.append("—")
                    continue
                row.append(GRADE_MARK[v["grade"]])
                if v.get("len_ratio"):
                    ratios.append(v["len_ratio"])
                if v.get("drift_sec") is not None:
                    drifts.append((v["drift_sec"], drift_is_measured(t)))
            for t in scored:
                sem = (per_fixture[fid].get(t) or {}).get("semantic")
                if not sem:
                    row.append("—")
                    continue
                cell = f"{sem['passed']}/{sem['total']}"
                if sem.get("penalty_hits"):
                    cell += f" +{sem['penalty_hits']}⚑"
                row.append(cell)
            ratio = f"{sum(ratios) / len(ratios):.2f}x" if ratios else "—"
            if drifts:
                worst, measured = max(drifts, key=lambda d: d[0])
                drift = f"{worst:.1f}s" + ("" if measured else "~")
            else:
                drift = "—"
            out.append(f"| `{model}` | " + " | ".join(row) +
                       f" | {ratio} | {drift} |")

    out.append("\n**meaning** columns are hand-written semantic checklists — "
               "the count of things the source says that the translation "
               "actually got right, with `⚑` marking grammar penalties. These "
               "*do* decide verdicts, and they only exist for language pairs "
               "somebody has written a checklist for "
               "([how to add one](../../tests/fixtures/benchmark/checklists/README.md)).\n")
    out.append("\nLength is a character count, comparable only within a script "
               "— Arabic output is shorter than its Latin source in characters "
               "while taking about as long to say. Drift marked `~` rests on an "
               "estimated speech rate for that script rather than a measured "
               "one; neither column affects a verdict.\n")
    return "\n".join(out) + "\n"


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="",
                    help="comma-separated; default = every chat model LM Studio lists")
    ap.add_argument("--targets", default="es,pt,ru,hi,ar",
                    help="comma-separated target language codes")
    ap.add_argument("--fixtures", default="", help="comma-separated fixture ids")
    ap.add_argument("--runs", type=int, default=2,
                    help="repeats per combination; 2+ is what exposes a model "
                         "that only sometimes translates")
    ap.add_argument("--timeout", type=int, default=420, help="seconds per run")
    ap.add_argument("--out", default=str(DEFAULT_OUT / "results.json"))
    ap.add_argument("--list", action="store_true",
                    help="show fixtures and available models, then exit")
    ap.add_argument("--report", action="store_true",
                    help="render the decision matrix from an existing "
                         "results.json instead of running anything")
    ap.add_argument("--resume", action="store_true",
                    help="keep combinations already in results.json and only "
                         "run the missing ones — a full sweep can take hours, "
                         "and one slow model should not cost you the rest")
    args = ap.parse_args()

    if args.report:
        data = json.loads(Path(args.out).read_text(encoding="utf-8"))
        print(render_report(data, load_fixtures()))
        return

    fixtures = load_fixtures(
        set(f.strip() for f in args.fixtures.split(",") if f.strip()) or None)
    available, models = [], []
    try:
        _, available = await T.check_lm_studio()
    except Exception as e:
        print(f"Could not reach LM Studio: {e}", file=sys.stderr)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [m for m in available if "embed" not in m.lower()]

    if args.list:
        print("Fixtures:")
        for f in fixtures:
            secs = f["duration"]
            chars = sum(len(s["text"]) for s in f["segments"])
            print(f"  {f['id']:<22} {f['source_lang']}  {len(f['segments'])} segments, "
                  f"{secs:.0f}s, {chars / secs:.1f} src chars/sec  [{f['origin']}]")

        print("\nSemantic checklists (these decide verdicts; everything else "
              "can only rule a model out):")
        found = sorted(CHECKLIST_DIR.glob("*.json"))
        for path in found:
            fid, lang = path.stem.rsplit(".", 1)
            doc = load_checklist(fid, lang)
            n = len(doc["checks"]) if doc else 0
            src = next((f["source_lang"] for f in fixtures if f["id"] == fid), "?")
            print(f"  {src}->{lang:<4} {n:>2} checks   {path.name}")
        if not found:
            print("  (none)")
        print(f"  Add one for your language: "
              f"{CHECKLIST_DIR.relative_to(ROOT)}/README.md")

        rates = _measured_rates()
        have = ", ".join(sorted(rates)) if rates else (
            "(none — run tools/gochidubb_speech_rate.py)")
        print(f"\nMeasured speech rates: {have}")

        print("\nModels LM Studio lists:")
        for m in available:
            print(f"  {m}")
        if not available:
            print("  (none — is LM Studio running?)")
        return

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not models:
        print("No models to test.", file=sys.stderr)
        sys.exit(1)

    # A fixture translated into its own language is not a translation: the
    # correct answer is the source text, which every metric here reads as
    # "the model failed and the segment fell back". Skip the pair rather than
    # score it, or Spanish→Spanish reports as a total failure.
    pairs = [(f, t) for f in fixtures for t in targets
             if is_translation_pair(f["source_lang"], t)]
    total = len(models) * len(pairs) * args.runs
    skipped = len(fixtures) * len(targets) - len(pairs)
    print(f"{len(models)} models x {len(pairs)} source/target pairs x "
          f"{args.runs} runs = {total} translations"
          + (f"  ({skipped} same-language pair(s) skipped)" if skipped else "")
          + "\n", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    prior_fixtures, prior_targets = [], []
    if args.resume and out_path.exists():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        results = prior.get("results", {})
        prior_fixtures = prior.get("fixtures", [])
        prior_targets = prior.get("targets", [])
        done = sum(len(t) for m in results.values() for t in m.values())
        print(f"Resuming: {done} combination(s) already recorded\n", flush=True)

    # Models outermost: LM Studio serves one model at a time, so this loads
    # each one once instead of once per language.
    for model in models:
        for fixture, target in pairs:
            if args.resume and target in results.get(model, {}).get(fixture["id"], {}):
                continue
            runs = [await run_one(model, fixture, target, args.timeout)
                    for _ in range(args.runs)]
            g = grade(runs)
            results.setdefault(model, {}).setdefault(fixture["id"], {})[target] = {
                "verdict": g, "runs": runs,
            }
            mark = {"good": "ok", "risky": "RISKY", "unusable": "UNUSABLE"}[g["grade"]]
            print(f"  {model:<28} {fixture['source_lang']}->{target}  "
                  f"{mark:<9} {g.get('median_sec', '?')}s "
                  f"fallback={g.get('fallback', '?')}/{g.get('segments', '?')} "
                  f"drift={g.get('drift_sec', '?')}s", flush=True)
            # Union with whatever a resumed file already covered, so the
            # report still knows about fixtures/targets from the earlier pass.
            out_path.write_text(json.dumps(
                {"fixtures": sorted(set(prior_fixtures)
                                    | {f["id"] for f in fixtures}),
                 "targets": prior_targets + [t for t in targets
                                             if t not in prior_targets],
                 "results": results},
                ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
