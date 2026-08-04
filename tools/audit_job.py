#!/usr/bin/env python3
"""Audit a job's artifacts for information lost between pipeline stages.

Why this exists: a dub came out missing spoken lines with nothing in the logs.
Every stage reported success. The loss was at a stage *boundary* —
`pipeline/segment_post` split one segment into two copies that both kept the
parent's `idx`, and the translation merge (`by_idx = {s["idx"]: s ...}`)
overwrote one with the other. Two lines of dialogue gone, no error anywhere.

Counting segments per stage is not enough to catch that class of bug: the
pipeline *legitimately* changes the count (it merges sentence fragments and
splits over-long segments), so 190 → 152 is healthy and 152 → 150 is a
disaster, and they look the same from a distance.

So the primary check here is **word coverage**: every word transcribed from the
source should still be accounted for in the segment list that reached the
final render. Merges and splits preserve words; dropped segments do not.

Usage:
    python tools/audit_job.py 947ca81d
    python tools/audit_job.py outputs/947ca81d --json
    python tools/audit_job.py --all          # every job in outputs/

Exit code is 1 when anything was lost, so this works as a CI check or a
post-run assertion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "outputs"

# Stage order, and the checkpoint each one writes. Mirrors PIPELINE_STAGES in
# server.py; kept as a literal so this tool can run against an output
# directory without importing the server.
STAGE_CHECKPOINTS = [
    ("download", "download_done"),
    ("extract", "extract_done"),
    ("transcribe", "transcribe_done"),
    ("diarize", "transcription_done"),
    ("translate", "translation_done"),
    ("tts", "tts_done"),
    ("assemble", "assemble_done"),
    ("merge", "merge_done"),
]

# Stages that restructure the segment list on purpose — `diarize` runs
# pipeline/segment_post, which merges sentence fragments and splits over-long
# segments. Arriving *at* one of these, a changed count is expected and word
# coverage is the real check; anywhere else a changed count means loss.
RESHAPING = {"diarize"}

# Mirrors pipeline/tts_qa.QA_THRESHOLD — kept as a literal (like
# STAGE_CHECKPOINTS above) so this tool can audit an output directory
# without importing the pipeline.
QA_BAD_SCORE = 0.4

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class Finding:
    """One problem worth a human's attention.

    `severity`: "loss" (content disappeared), "warn" (suspicious but may be
    intentional), "info" (context that explains what you are seeing).
    """

    def __init__(self, severity: str, stage: str, summary: str,
                 detail: str = "", items: Optional[List[Any]] = None) -> None:
        self.severity = severity
        self.stage = stage
        self.summary = summary
        self.detail = detail
        self.items = items or []

    def to_dict(self) -> dict:
        return {"severity": self.severity, "stage": self.stage,
                "summary": self.summary, "detail": self.detail,
                "items": self.items}


def audit_job(job_dir: os.PathLike | str) -> dict:
    """Analyse one output directory. Returns a report dict."""
    work = Path(job_dir)
    report: Dict[str, Any] = {
        "job": work.name,
        "path": str(work),
        "stages": [],
        "findings": [],
        "ok": True,
    }
    findings: List[Finding] = []

    if not work.is_dir():
        findings.append(Finding("loss", "-", f"No such directory: {work}"))
        return _finish(report, findings)

    # ── Load every checkpoint that exists ─────────────────────────────
    loaded: Dict[str, dict] = {}
    for stage, cp in STAGE_CHECKPOINTS:
        data = _load(work / f"checkpoint_{cp}.json")
        if data is not None:
            loaded[stage] = data

    if not loaded:
        # A job that failed before its first checkpoint has nothing to lose.
        # Reporting it as loss would make `--all` cry wolf on every dead job
        # in the directory, which is how a check stops being read.
        findings.append(Finding("info", "-", "No checkpoints — job never got past setup"))
        return _finish(report, findings)

    # ── Per-stage summary + idx integrity ─────────────────────────────
    present = [s for s, _ in STAGE_CHECKPOINTS if s in loaded]
    for stage in present:
        segs = loaded[stage].get("segments") or []
        counts = Counter(s.get("idx") for s in segs)
        dups = {k: v for k, v in counts.items() if v > 1 and k is not None}
        row = {
            "stage": stage,
            "segments": len(segs),
            "unique_idx": len(counts),
            "words": sum(len(_words(s.get("text", ""))) for s in segs),
            "duplicate_idx": dups,
        }
        report["stages"].append(row)

        if dups:
            # This is the bug that started it all: downstream code keys dicts
            # by idx, so a duplicate is a segment waiting to be overwritten.
            examples = []
            for idx in sorted(dups)[:5]:
                texts = [s.get("text", "") for s in segs if s.get("idx") == idx]
                examples.append({"idx": idx, "texts": texts})
            findings.append(Finding(
                "loss", stage,
                f"{len(dups)} duplicate idx — downstream keys segments by idx, "
                f"so {sum(dups.values()) - len(dups)} segment(s) will be silently "
                f"overwritten",
                detail="Caused by a stage emitting two segments with the same "
                       "index (e.g. splitting one segment into two copies).",
                items=examples,
            ))

    # ── Boundary checks ───────────────────────────────────────────────
    for a, b in zip(present, present[1:]):
        segs_a = loaded[a].get("segments") or []
        segs_b = loaded[b].get("segments") or []
        if not segs_a and not segs_b:
            continue
        # download/extract carry no segments; skip until the list exists.
        if not segs_a:
            continue
        if not segs_b:
            findings.append(Finding(
                "loss", f"{a}→{b}",
                f"{b} has no segments while {a} had {len(segs_a)}"))
            continue

        _check_boundary(a, b, segs_a, segs_b, findings)

    # ── Content checks on the last segment list ───────────────────────
    last_stage = present[-1]
    final_segs = loaded[last_stage].get("segments") or []
    first_with_segs = next((s for s in present if loaded[s].get("segments")), None)

    if first_with_segs and final_segs:
        _check_word_coverage(loaded[first_with_segs]["segments"], final_segs,
                             first_with_segs, last_stage, findings)

    if "translate" in loaded:
        _check_translation(loaded["translate"].get("segments") or [], findings)
    if "tts" in loaded:
        _check_tts(loaded["tts"].get("segments") or [], work, findings)
    _check_assembly(work, loaded, findings)
    _check_srt(work, loaded, findings)

    return _finish(report, findings)


def _check_boundary(a: str, b: str, segs_a: List[dict], segs_b: List[dict],
                    findings: List[Finding]) -> None:
    """Compare two consecutive stages' segment lists."""
    idx_a = [s.get("idx") for s in segs_a]
    idx_b = [s.get("idx") for s in segs_b]
    lost = [i for i in idx_a if i not in set(idx_b)]

    if b in RESHAPING:
        # Merges and splits are the point of the destination stage; counts are
        # expected to move. Word coverage (checked separately) is what matters
        # across this boundary. Duplicate-idx checks still apply — those are
        # never legitimate.
        return

    if len(segs_b) < len(segs_a):
        # Name the actual lines, not just the count — "2 segments lost" sends
        # you hunting; the sentence that vanished tells you where to look.
        dropped = []
        by_idx_b = {s.get("idx") for s in segs_b}
        for s in segs_a:
            if s.get("idx") not in by_idx_b:
                dropped.append({
                    "idx": s.get("idx"),
                    "start": round(float(s.get("start", 0)), 2),
                    "end": round(float(s.get("end", 0)), 2),
                    "text": s.get("text", ""),
                })
        detail = ""
        if not dropped and len(idx_a) != len(set(idx_a)):
            detail = (f"No index is missing, so the loss came from duplicate "
                      f"idx in {a} collapsing when keyed by index.")
        findings.append(Finding(
            "loss", f"{a}→{b}",
            f"{len(segs_a)} → {len(segs_b)} segments ({len(segs_a) - len(segs_b)} lost)",
            detail=detail, items=dropped[:20],
        ))
    elif lost:
        findings.append(Finding(
            "loss", f"{a}→{b}",
            f"{len(lost)} segment index(es) present in {a} but missing from {b}",
            items=lost[:20]))


def _check_word_coverage(src: List[dict], final: List[dict],
                         src_stage: str, final_stage: str,
                         findings: List[Finding]) -> None:
    """The check that catches losses counting cannot.

    Merging two segments and splitting one in half both change the segment
    count while preserving every word. Dropping a segment does not. Comparing
    word multisets across the whole job therefore separates "restructured" from
    "lost" without needing to model each pass.
    """
    src_words = Counter(w for s in src for w in _words(s.get("text", "")))
    fin_words = Counter(w for s in final for w in _words(s.get("text", "")))
    missing = src_words - fin_words
    total = sum(src_words.values())
    lost = sum(missing.values())
    if not total:
        return
    if lost:
        pct = 100.0 * lost / total
        # Locate the source segments that account for the missing words, so
        # the report points at timestamps rather than a bag of words.
        fin_texts = {(s.get("text") or "").strip() for s in final}
        suspects = []
        for s in src:
            text = (s.get("text") or "").strip()
            if not text or text in fin_texts:
                continue
            seg_words = Counter(_words(text))
            if seg_words and (seg_words & missing):
                overlap = sum((seg_words & missing).values())
                if overlap >= max(2, 0.6 * sum(seg_words.values())):
                    suspects.append({
                        "idx": s.get("idx"),
                        "start": round(float(s.get("start", 0)), 2),
                        "end": round(float(s.get("end", 0)), 2),
                        "text": text,
                    })
        findings.append(Finding(
            "loss" if pct >= 0.5 else "warn",
            f"{src_stage}→{final_stage}",
            f"{lost}/{total} words ({pct:.1f}%) present at {src_stage} are absent "
            f"from {final_stage}",
            detail="Segment counts can change legitimately (merge/split); words "
                   "cannot disappear. Segments below are the likely source.",
            items=suspects[:20],
        ))


def _check_translation(segs: List[dict], findings: List[Finding]) -> None:
    untranslated = []
    for s in segs:
        src = (s.get("text") or "").strip()
        tgt = (s.get("translated_text") or "").strip()
        if not src:
            continue
        if not tgt or tgt == src:
            untranslated.append({"idx": s.get("idx"),
                                 "start": round(float(s.get("start", 0)), 2),
                                 "text": src[:80]})
    if untranslated:
        findings.append(Finding(
            "loss", "translate",
            f"{len(untranslated)}/{len(segs)} segments have no translation "
            f"(empty, or identical to the source)",
            detail="These are spoken in the source language — or silent — in "
                   "the final dub.",
            items=untranslated[:20]))


def _check_tts(segs: List[dict], work: Path, findings: List[Finding]) -> None:
    no_path, missing_file, empty = [], [], []
    for s in segs:
        ap = s.get("audio_path")
        entry = {"idx": s.get("idx"), "start": round(float(s.get("start", 0)), 2),
                 "text": (s.get("translated_text") or s.get("text") or "")[:80]}
        if not ap:
            no_path.append(entry)
            continue
        p = Path(ap)
        if not p.is_absolute():
            p = work / ap
        if not p.exists():
            missing_file.append({**entry, "path": str(ap)})
        elif p.stat().st_size < 1024:
            empty.append({**entry, "bytes": p.stat().st_size})

    if no_path:
        findings.append(Finding(
            "loss", "tts",
            f"{len(no_path)}/{len(segs)} segments were never synthesized",
            detail="The assembler skips these silently — they are gaps of "
                   "silence in the dub.",
            items=no_path[:20]))
    if missing_file:
        findings.append(Finding(
            "loss", "tts",
            f"{len(missing_file)} segments reference a wav that no longer exists",
            detail="Assembly skips a missing file without raising.",
            items=missing_file[:20]))
    if empty:
        findings.append(Finding(
            "warn", "tts",
            f"{len(empty)} synthesized wavs are suspiciously small (<1 KB)",
            items=empty[:20]))

    # ── Roundtrip-QA verdicts persisted by the TTS worker ─────────────
    # seg["qa"] = {score, measured, cer, detected_lang, lang_match,
    # attempts, tier}. A segment can have audio on disk and still be
    # garbled — QA already caught it; surface that instead of making the
    # user listen for it.
    qa_degraded, qa_bad = [], []
    qa_present = qa_unmeasured = 0
    for s in segs:
        qa = s.get("qa")
        if not isinstance(qa, dict):
            continue
        qa_present += 1
        entry = {"idx": s.get("idx"),
                 "start": round(float(s.get("start", 0)), 2),
                 "text": (s.get("translated_text") or s.get("text") or "")[:80],
                 "cer": qa.get("cer"), "detected_lang": qa.get("detected_lang"),
                 "tier": qa.get("tier")}
        if qa.get("tier") == 0:
            # tier 0 = every tier and retry failed QA; the worker kept the
            # last take anyway because silence would be worse.
            qa_degraded.append(entry)
            continue
        if not qa.get("measured"):
            qa_unmeasured += 1
            continue
        score = qa.get("score")
        if (score is not None and score > QA_BAD_SCORE) or qa.get("lang_match") is False:
            qa_bad.append(entry)

    if qa_degraded:
        findings.append(Finding(
            "warn", "tts",
            f"{len(qa_degraded)}/{len(segs)} segments kept a take that failed "
            f"QA on every retry (degraded, tier 0)",
            detail="These are the segments most likely to sound garbled or in "
                   "the wrong language. Regenerate them individually or retry "
                   "TTS with a different seed.",
            items=qa_degraded[:20]))
    if qa_bad:
        findings.append(Finding(
            "warn", "tts",
            f"{len(qa_bad)} segments measured bad on ASR roundtrip "
            f"(QA score > {QA_BAD_SCORE} or wrong detected language) but were "
            f"accepted",
            items=qa_bad[:20]))
    if qa_present and qa_unmeasured:
        findings.append(Finding(
            "info", "tts",
            f"{qa_unmeasured}/{qa_present} segments have no measured QA score "
            f"(whisper unavailable or ASR failed) — their quality is unknown, "
            f"not perfect"))

    # Stale audio from earlier runs is not a loss, but it is the single most
    # confusing thing to find when hand-checking a job directory — 188 wavs
    # sitting next to a 150-segment dub sends you looking in the wrong place.
    seg_dir = work / "tts_segments"
    if seg_dir.is_dir():
        wavs = [f for f in os.listdir(seg_dir)
                if f.startswith("seg_") and f.endswith(".wav")]
        referenced = {os.path.basename(str(s.get("audio_path") or "")) for s in segs}
        stale = sorted(set(wavs) - referenced)
        if stale:
            findings.append(Finding(
                "info", "tts",
                f"{len(stale)} wav files in tts_segments/ are not referenced by "
                f"the current run (leftovers from an earlier attempt)",
                detail="Harmless, but it makes the directory look like it "
                       "contains more segments than the dub actually used.",
                items=stale[:10]))


def _check_assembly(work: Path, loaded: Dict[str, dict],
                    findings: List[Finding]) -> None:
    """Compare what TTS produced with what the assembler actually placed."""
    placements = _load(work / "tts_placements.json")
    tts_segs = (loaded.get("tts") or {}).get("segments") or []
    if placements is None:
        if tts_segs:
            findings.append(Finding(
                "info", "assemble",
                "tts_placements.json missing — cannot verify what was placed"))
        return
    if not isinstance(placements, list):
        return

    placed_idx = {p.get("idx") for p in placements}
    unplaced = [
        {"idx": s.get("idx"), "start": round(float(s.get("start", 0)), 2),
         "text": (s.get("translated_text") or s.get("text") or "")[:80]}
        for s in tts_segs if s.get("idx") not in placed_idx
    ]
    if unplaced:
        findings.append(Finding(
            "loss", "tts→assemble",
            f"{len(unplaced)}/{len(tts_segs)} synthesized segments were not "
            f"placed in the dubbed track",
            detail="assemble_dubbed_audio() skips segments whose audio is "
                   "missing or unreadable, and logs only 'Skipped segment'.",
            items=unplaced[:20]))


def _check_srt(work: Path, loaded: Dict[str, dict],
               findings: List[Finding]) -> None:
    srt = work / "subtitles.srt"
    if not srt.exists():
        return
    try:
        blocks = [b for b in srt.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    except Exception:
        return
    tr = (loaded.get("translate") or {}).get("segments") or []
    if tr and len(blocks) != len(tr):
        findings.append(Finding(
            "warn", "translate",
            f"subtitles.srt has {len(blocks)} entries but the translate "
            f"checkpoint has {len(tr)} segments"))


def _finish(report: dict, findings: List[Finding]) -> dict:
    report["findings"] = [f.to_dict() for f in findings]
    report["ok"] = not any(f.severity == "loss" for f in findings)
    report["counts"] = {
        s: sum(1 for f in findings if f.severity == s)
        for s in ("loss", "warn", "info")
    }
    return report


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════
_SEV_MARK = {"loss": "✗ LOSS", "warn": "! WARN", "info": "· info"}


def print_report(report: dict, verbose: bool = False) -> None:
    print(f"\n=== {report['job']} ===")
    if report["stages"]:
        print(f"{'stage':<12}{'segments':>10}{'unique idx':>12}{'words':>9}")
        for row in report["stages"]:
            flag = "  <-- DUPLICATE IDX" if row["duplicate_idx"] else ""
            print(f"{row['stage']:<12}{row['segments']:>10}"
                  f"{row['unique_idx']:>12}{row['words']:>9}{flag}")

    if not report["findings"]:
        print("\nNo information loss detected.")
        return

    print()
    for f in report["findings"]:
        if f["severity"] == "info" and not verbose:
            continue
        print(f"{_SEV_MARK.get(f['severity'], f['severity'])}  [{f['stage']}] {f['summary']}")
        if f["detail"]:
            for line in f["detail"].split("\n"):
                print(f"        {line}")
        for item in f["items"]:
            if isinstance(item, dict) and "text" in item:
                ts = f"{item.get('start', 0):.2f}-{item.get('end', 0):.2f}s" \
                    if "end" in item else f"{item.get('start', 0):.2f}s"
                print(f"        idx={item.get('idx')} [{ts}] {item['text']!r}")
            elif isinstance(item, dict) and "texts" in item:
                print(f"        idx={item['idx']}:")
                for t in item["texts"]:
                    print(f"            {t!r}")
            else:
                print(f"        {item}")
        print()

    c = report["counts"]
    verdict = "LOSS DETECTED" if not report["ok"] else "no loss (warnings only)"
    print(f"{verdict} — {c['loss']} loss, {c['warn']} warn, {c['info']} info")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", nargs="?", help="job id, or a path to an output dir")
    ap.add_argument("--all", action="store_true", help="audit every job in outputs/")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-v", "--verbose", action="store_true", help="include info findings")
    args = ap.parse_args()

    if args.all:
        targets = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir())
    elif args.job:
        p = Path(args.job)
        targets = [p if p.is_dir() else OUTPUT_DIR / args.job]
    else:
        ap.error("give a job id, a directory, or --all")

    reports = [audit_job(t) for t in targets]
    if args.json:
        print(json.dumps(reports if args.all else reports[0], indent=2, ensure_ascii=False))
    else:
        for r in reports:
            if args.all and not r["findings"]:
                print(f"{r['job']}: clean")
                continue
            print_report(r, verbose=args.verbose)

    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
