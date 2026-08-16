# Choosing a translation model

Translation is the stage where a dub is most often quietly ruined. TTS failures
are audible; a bad translation is fluent, confident and wrong, and you only find
out by listening to the finished video in a language you may not speak.

This page is the evidence for which local models to use, and a benchmark you can
re-run against whatever you have installed.

- [The short version](#the-short-version)
- [Decision matrix](benchmarks/decision-matrix.md) — per language, generated from the benchmark
- [Case study: how a bad model looks](#case-study-how-a-bad-model-looks)
- [What actually distinguishes a usable model](#what-actually-distinguishes-a-usable-model)
- [Which languages are covered, and why](#which-languages-are-covered-and-why)
- [Run the benchmark yourself](#run-the-benchmark-yourself)
- [Using a glossary](#using-a-glossary)

## The short version

| Situation | Model | Why |
|---|---|---|
| Default, any target | `openai/gpt-oss-20b` | Never once failed to translate across every run of every language pair tested. Best hand-checked quality. 8–24s per clip |
| You need it faster | `qwen/qwen3-8b` | Just as reliable mechanically and about twice as fast, but makes Bulgarian clitic errors ("Те обичам много" for "Обичам те много") |
| **Do not use** | `aya-expanse-8b` | Produces Russian-shaped text for other Slavic targets. Scored 2/13 on hand-checked Bulgarian *while passing every mechanical check* |
| **Do not use** | `google/gemma-4-e4b` | Narrates the task instead of translating on most runs |
| **Do not use** | `qwen/qwen3.5-9b`, `zai-org/glm-4.6v-flash` | Reason past every output budget and never answer |
| Avoid for batch work | `qwen/qwen3.6-27b` | Good output, but exceeded 300s per run on a nine-segment clip. Fine for one video, not for a queue |
| Unproven | `liquid/lfm2-24b-a2b` | Fastest in the field (4–14s) and passes every mechanical check, but invents words in Russian. Check its output before trusting it |

```bash
# .env
LM_STUDIO_MODEL=openai/gpt-oss-20b
```

Nothing here is permanent. Models change monthly, and this table reflects one
machine's installed set at one point in time — which is exactly why the
benchmark ships with the repo rather than only its conclusions.

## Case study: how a bad model looks

A Spanish Mother's Day video, six schoolchildren, dubbed into Bulgarian with
`aya-expanse-8b`. Every stage reported success. The output was fluent-looking
Cyrillic. It was also wrong in ways the pipeline could not see at the time:

| Source | What the model wrote | What it means |
|---|---|---|
| `Te amo mucho` | Ти ме обичаваш много | "**You** love **me** a lot" — subject and object swapped |
| `Gracias` | Моля | "Please" |
| `Día de la Madre` | День на матицата | Russian "День", plus a word that does not exist |
| `te extraño mucho` | скучам по теба | A Russian calque; Bulgarian is "липсваш ми" |
| `Adiós` | До слагание | A mangling of the Russian "До свидания" |

The model was not translating into Bulgarian. It was translating into Russian
and spelling it Bulgarian-ish. On a 13-point checklist of what that script
actually says, it scored 2. `openai/gpt-oss-20b` scored 13.

This is the failure mode to design against: not a model that refuses, but one
that confidently answers in the wrong language.

## What actually distinguishes a usable model

Fluency is what you notice last. These are the things that decide whether a
model can be pointed at a queue of videos and left alone.

**It has to translate every time, not most times.** `google/gemma-4-e4b`
produced flawless Bulgarian on its first run and then, on three consecutive
repeat runs, narrated the task instead — *"The user wants me to translate three
Spanish subtitle lines into Bulgarian…"* — for 21 of 21 lines, taking about 260
seconds each time to fail. A model that is excellent two runs in three is not
usable unattended. This is why the benchmark defaults to `--runs 2`.

**It has to answer in the target language.** The most dangerous reply is the
source text echoed back, or an English answer, because both look like output.
`pipeline/translator.py` now rejects these (`_rejection_reason`), and the
segment keeps its source text so the UI can offer *retry failed segments only* —
but a model that triggers it often is burning your time on retries.

**It has to keep the translation near the source's length.** Speech has to fit
the slot it is dubbed into. A translation 40% longer than its source cannot be
spoken in the same time, and the assembler has to compress it — up to
`cfg.tts_max_stretch`, after which the dub starts running late. Length is a
model property: on the same clip, models ranged from 1.02x to 1.17x the source
length.

**It must not be a reasoning model.** `qwen/qwen3.5-9b` spends a full
chain-of-thought on *"Hola, mamá"* even with `LM_STUDIO_REASONING=off` and
Qwen's `/no_think` token, and never finishes a batch. See
[Thinking models and translation speed](../README.md#thinking-models-and-translation-speed).

**Beware of models that look good because they say less.** `aya-expanse-8b` had
the second-best length ratio in the entire field. It achieved that by dropping
meaning. Short output is only a virtue when the meaning survives.

## Which languages are covered, and why

The default target set is `es,pt,ru,hi,ar`. Two things picked it, and it is
worth being clear that the first is a judgment call and the second is not.

**Likely demand.** Spanish, Portuguese, Hindi, Arabic and Russian are among the
largest audiences for dubbed short-form video by speaker population. That is an
estimate of what people will want, not a measurement of what they use. If your
audience is elsewhere, change `--targets`; nothing in the benchmark is specific
to these five.

**Script coverage, which is the part that actually matters technically.** A
model that handles Spanish tells you nothing about how it handles Hindi,
because the failure modes are per-script:

| Target | Script | What it exposes |
|---|---|---|
| `es`, `pt` | Latin | The baseline. No script check is possible, so failures here are subtle |
| `ru` | Cyrillic | Models trained mostly on Russian will answer in Russian for *any* Slavic target — this is how the Bulgarian dub failed |
| `hi` | Devanagari | Output runs ~1.35x the source in characters. Length, not vocabulary, is the risk |
| `ar` | Arabic | Right-to-left, and output is *shorter* than its Latin source in characters while taking about as long to say |

Those five cover four scripts. Bulgarian is not in the default set but is
covered in depth by the case study above, because it is the one language pair
whose output could be checked by hand.

Two gaps worth naming. **CJK** (`zh`, `ja`, `ko`) is not in the default set and
behaves very differently — a Chinese translation is a fraction of its source's
character count, so every length-based intuition inverts. **Right-to-left
rendering** is checked for script but not for bidirectional text handling in
the burned-in subtitles. Both are open.

## Run the benchmark yourself

The numbers above came from `tools/gochidubb_benchmark.py`, which runs the real
`translate_segments()` path — same prompt, same batching, same validation the
pipeline uses on a live job.

```bash
# What fixtures exist, and what models LM Studio is offering
python tools/gochidubb_benchmark.py --list

# Every installed model, five target languages, two runs each
python tools/gochidubb_benchmark.py --targets es,pt,ru,hi,ar

# One model, three runs, to check it is consistent
python tools/gochidubb_benchmark.py --models openai/gpt-oss-20b --runs 3
```

Results land in `docs/benchmarks/results.json`. Each combination gets a verdict:

| Verdict | Meaning |
|---|---|
| `good` | Translated every line, every run, in the right script, at a length that fits |
| `risky` | Usable but dropped lines, slipped script, or runs long enough to cost sync |
| `unusable` | Failed outright, timed out, or narrated the task |

### It rules models out; it cannot rule them in

Every metric is mechanical: did the model answer, in the target language,
keeping the numbers, every time. None of it measures whether the translation is
*correct*.

`aya-expanse-8b` is the proof. It passes almost this entire matrix — no dropped
lines, right script, sensible length, quick — and it is the model that ruined a
real dub. Fluent, correctly-scripted, correctly-sized, and wrong.

One line makes the gap concrete. Translating *"My daughter was 18 months old"*
into Arabic, it wrote **الثامنة عشر من عمرها** — "eighteen **years** old". The
benchmark noticed something: the digit `18` was spelled out rather than kept.
It could not notice the thing that matters, which is that a baby became a
teenager. No mechanical check catches that; only reading the output does.

So read a `❌` or `⚠️` as strong evidence against a model, and a `✅` as nothing
more than "it cleared the bar where a machine can judge". Somebody who reads
the target language still has to watch the output. The Bulgarian case study
above is the one place in this repo where translations were scored
semantically, because it is the one language pair that could be checked by
hand.

If you are dubbing into a language you do not speak, the cheap safety net is to
dub one clip into a language you *do* speak first, with the same model and the
same context hint. A model that mangles a language you can check is not going
to do better on one you cannot.

### What is not in the matrix, and why

Two models are missing from the generated matrix because they never finish a
batch, and a sweep spends its whole budget waiting for them:

- **`qwen/qwen3.5-9b`** and **`zai-org/glm-4.6v-flash`** both timed out at 900
  seconds on a seven-line clip. qwen3.5-9b emits a full chain-of-thought for
  *"Hola, mamá"* — 14.7 seconds for those two words — with
  `LM_STUDIO_REASONING=off` set and Qwen's `/no_think` token appended. Neither
  is usable for translation at any quality level.

**`qwen/qwen3.6-27b`** is also absent. It produces good output — 12 of 13 on
the hand-scored Bulgarian checklist — but exceeded 300 seconds per run on the
nine-segment German fixture and produced nothing in ten minutes, so it was
pulled from the sweep. Its own numbers on the shorter Spanish clip were 366s on
the first run and ~20s after, so the cost is front-loaded into the model load.
Reasonable for a single video; not for a queue.

`google/gemma-4-e4b` *is* in the matrix, on the Spanish fixture only. Each
language pair takes it about nine minutes to fail, because it burns the whole
split-and-retry ladder before giving up.

### The fixtures

`tests/fixtures/benchmark/` holds short multi-speaker clips chosen to carry the
things that break translation: kinship terms and vocatives, holiday names, set
phrases like farewells, proper nouns that must survive untranslated, numbers
that must survive as numbers, and speech fast enough that the timing is tight.

| Fixture | Source | Speech rate | Origin |
|---|---|---|---|
| `es_mothers_day` | Spanish | 15.7 chars/sec | **Real** — job `c5b12e16`, verbatim, including its whisper transcription errors |
| `en_clinic_thanks` | English | 13.4 chars/sec | Constructed for this benchmark |
| `de_workshop_intro` | German | 13.0 chars/sec | Constructed for this benchmark |

All three leave a similar 6–9% of their runtime as silence between segments.
The Spanish one is nonetheless the timing stress case, because its speakers
talk about 20% faster — which is what leaves no room for a translation that
runs long. Expect near-zero drift on the other two and real drift on this one;
that difference is the source material, not the models.

The Spanish fixture is real production data and keeps its transcription
mistakes on purpose — `Gracias por qué me has hecho` is what whisper actually
heard, and a model's handling of a slightly broken source line is part of what
is being measured.

The other two are written, not scraped: a checked-in benchmark has to be
reproducible offline and free of licensing questions. Their timings mirror the
speech rate and gap structure of real diarized footage. German is included
because it is the hardest common source for length matching — compound nouns and
verb-final clauses make the source unusually dense, so natural translations of
it tend to overrun.

To add a fixture, drop a JSON file in that directory following the same shape:
`source_lang`, `duration`, `context_hint`, `segments` with `start`/`end`/`text`,
and an `origin` of `real` or `constructed`.

## Using a glossary

Recurring term errors are worth fixing once rather than per job.
`openai/gpt-oss-20b` rendered *Día de la Madre* as "весела Деня на майката",
wrong on both gender and case, on every run. One entry in
`presets/user_glossary.json` fixed it, and the output became byte-identical
across repeat runs:

```json
{
  "domains": [
    {
      "target_lang": "bg",
      "terms": { "Día de la Madre": "Ден на майката" }
    }
  ]
}
```

**Keep glossary entries to nouns and names.** A fuller glossary that also
forced `"Te amo mucho": "Обичам те много"` made things worse: the model
inserted the phrase verbatim into subordinate clauses, where Bulgarian grammar
requires "че **те обичам** много", and broke clitic order in three places. The
glossary is a hammer — it makes the model reproduce a string exactly, with no
regard for the grammar around it. That is right for a product name or a holiday
and wrong for anything inflected.
