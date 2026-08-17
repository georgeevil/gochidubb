#!/usr/bin/env python3
"""Measure how fast TTS actually speaks each language, and record it.

The benchmark predicts timing drift from a translation's character count. That
needs a characters-per-second figure per language, and character counts are not
comparable across scripts: a Chinese line is a third the length of its Spanish
source and takes about as long to say.

Those figures used to be guesses hard-coded in gochidubb_benchmark.py. This
tool replaces them with measurements: it synthesizes real translated lines --
the ones the benchmark already collected -- and divides characters by the
length of the audio that came out.

    python tools/gochidubb_speech_rate.py                 # every language in results.json
    python tools/gochidubb_speech_rate.py --langs zh,ja,ko
    python tools/gochidubb_speech_rate.py --show          # print what is recorded

Writes tests/fixtures/benchmark/speech_rates.json, which the benchmark reads.

Engine note: this measures edge-tts, which needs no GPU and covers all 65
languages. VoxCPM2 is the pipeline's default engine and speaks at its own pace,
so treat these as a calibrated cross-language *ratio* rather than as VoxCPM's
absolute rate. The ratio between languages is what the drift estimate needs;
the absolute figure is anchored separately from a real VoxCPM dub.
"""
import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RATES_FILE = ROOT / "tests" / "fixtures" / "benchmark" / "speech_rates.json"
RESULTS_FILE = ROOT / "docs" / "benchmarks" / "results.json"

# Anchor: a real VoxCPM2 dub of job c5b12e16 into Bulgarian spoke 10.6
# characters per second. Every measured language is expressed relative to
# whatever the SAME engine does for Bulgarian, then scaled to this, so the
# numbers stay meaningful for the engine the pipeline actually uses.
#
# Both halves of that ratio must come from the same engine. The anchor's
# edge-tts rate therefore has to be measured like any other language — it
# cannot be filled in with the VoxCPM figure, which would divide edge-tts
# measurements by a VoxCPM number and inflate every language by the gap
# between the two engines (about 10%).
VOXCPM_ANCHOR_LANG = "bg"
VOXCPM_ANCHOR_RATE = 10.6

# Used only when a sweep contains no lines in the anchor language, so the
# anchor can still be measured with the same engine as everything else.
ANCHOR_REFERENCE_LINES = [
    "Здравей, мамо. Благодаря ти за всичко, което направи за мен през тези "
    "години, и се надявам да имаш прекрасен ден.",
    "Искам да знаеш, че те обичам много и че много ми липсваш, когато не сме "
    "заедно у дома.",
    "Тази работилница започва във вторник и продължава около три часа всяка "
    "седмица до края на месеца.",
]


def _voice_for(lang: str) -> str:
    from pipeline.synthesizer import EdgeTTSFallback
    return EdgeTTSFallback.VOICE_MAP.get((lang or "")[:2].lower(), "")


async def _speak(text: str, voice: str, path: Path) -> float:
    """Synthesize one line; return the audio duration in seconds."""
    import edge_tts
    import soundfile as sf

    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(path))
    # edge-tts writes mp3; soundfile reads it via libsndfile 1.1+.
    info = sf.info(str(path))
    return info.frames / info.samplerate


async def measure_language(lang: str, lines, max_lines: int = 8):
    """Characters per second for one language, from real translated lines."""
    voice = _voice_for(lang)
    if not voice:
        return None, f"no edge-tts voice for {lang!r}"
    usable = [ln for ln in lines if len((ln or "").strip()) >= 12][:max_lines]
    if not usable:
        return None, "no lines long enough to measure"

    # edge-tts is a network service. A transient failure on one line must not
    # discard the lines already measured -- silently falling back to a guessed
    # rate is worse than a slightly smaller sample.
    per_line, errors = [], []
    with tempfile.TemporaryDirectory(prefix="gochidubb_rate_") as tmp:
        for i, line in enumerate(usable):
            out = Path(tmp) / f"{lang}_{i}.mp3"
            try:
                seconds = await _speak(line, voice, out)
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
                continue
            if seconds > 0.2:
                per_line.append(len(line.strip()) / seconds)
    if not per_line:
        return None, errors[0] if errors else "no usable audio"
    return {
        "chars_per_sec": round(statistics.median(per_line), 2),
        "samples": len(per_line),
        "voice": voice,
        "spread": round(max(per_line) - min(per_line), 2),
    }, None


def collect_lines(results, lang):
    """Every translation into `lang` that a benchmark sweep recorded."""
    lines = []
    for per_fixture in results.values():
        for per_target in per_fixture.values():
            cell = per_target.get(lang)
            if not cell:
                continue
            for run in cell.get("runs", []):
                lines.extend(run.get("texts") or [])
    # Longest first: a longer line averages out per-utterance lead-in silence.
    # The text is the tiebreaker because set iteration order varies with
    # PYTHONHASHSEED, and two runs over one results.json must pick the same
    # lines or they record different rates for the same data.
    return sorted({ln.strip() for ln in lines if ln},
                  key=lambda t: (-len(t), t))


async def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="",
                    help="comma-separated; default = every target in results.json")
    ap.add_argument("--results", default=str(RESULTS_FILE))
    ap.add_argument("--out", default=str(RATES_FILE))
    ap.add_argument("--lines", type=int, default=8,
                    help="lines to synthesize per language")
    ap.add_argument("--show", action="store_true",
                    help="print the recorded rates and exit")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.show:
        if not out_path.exists():
            print(f"No rates recorded yet at {out_path}")
            return
        data = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Engine: {data.get('engine')}   anchored to "
              f"{data.get('anchor_lang')} = {data.get('anchor_rate')} chars/sec "
              f"(measured on VoxCPM2)\n")
        print(f"{'lang':<6}{'chars/sec':>11}{'vs anchor':>11}{'samples':>9}  voice")
        for lang, r in sorted(data.get("rates", {}).items()):
            print(f"{lang:<6}{r['chars_per_sec']:>11.2f}{r['scale']:>11.2f}"
                  f"{r['samples']:>9}  {r['voice']}")
        return

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))["results"]
    if args.langs:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    else:
        langs = sorted({t for per_fixture in results.values()
                        for per_target in per_fixture.values()
                        for t in per_target})
    if VOXCPM_ANCHOR_LANG not in langs:
        langs.append(VOXCPM_ANCHOR_LANG)

    measured, failed = {}, {}
    for lang in langs:
        lines = collect_lines(results, lang)
        if not lines and lang == VOXCPM_ANCHOR_LANG:
            # The anchor is not in this sweep. Measure it from built-in
            # reference text rather than substituting the VoxCPM figure — the
            # ratio it anchors is only valid between same-engine measurements.
            lines = ANCHOR_REFERENCE_LINES
            print(f"  {lang:<4} (anchor absent from this sweep; measuring "
                  f"reference lines)", flush=True)
        rate, err = await measure_language(lang, lines, args.lines)
        if err:
            failed[lang] = err
            print(f"  {lang:<4} skipped: {err}", flush=True)
            continue
        if lines is ANCHOR_REFERENCE_LINES:
            # Rate varies with the text measured, so an anchor taken from
            # stock sentences is less comparable to languages measured from
            # this sweep's own translations. Record which it was.
            rate["from_reference_lines"] = True
        measured[lang] = rate
        print(f"  {lang:<4} {rate['chars_per_sec']:>7.2f} chars/sec  "
              f"({rate['samples']} lines, {rate['voice']})", flush=True)

    anchor = measured.get(VOXCPM_ANCHOR_LANG)
    if not anchor:
        print(f"\nNo rate for the anchor language {VOXCPM_ANCHOR_LANG!r}; "
              f"cannot scale. Nothing written.", file=sys.stderr)
        sys.exit(1)

    # Express every language relative to the anchor, then restate it on the
    # VoxCPM scale. What the drift estimate needs is the ratio between
    # languages; the absolute number comes from the real dub.
    base = anchor["chars_per_sec"]
    for lang, r in measured.items():
        r["scale"] = round(r["chars_per_sec"] / base, 3)
        r["voxcpm_chars_per_sec"] = round(r["scale"] * VOXCPM_ANCHOR_RATE, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "engine": "edge-tts",
        "anchor_lang": VOXCPM_ANCHOR_LANG,
        "anchor_rate": VOXCPM_ANCHOR_RATE,
        "anchor_note": ("Anchor measured on a real VoxCPM2 dub (job c5b12e16). "
                        "Other languages measured with edge-tts and scaled to "
                        "it, so cross-language ratios are measured rather than "
                        "guessed."),
        "rates": measured,
        "unmeasured": failed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} ({len(measured)} languages)")


if __name__ == "__main__":
    asyncio.run(main())
