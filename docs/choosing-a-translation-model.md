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
- [Semantic checklists](#semantic-checklists) — how to add your language
- [Run the benchmark yourself](#run-the-benchmark-yourself)
- [Using a glossary](#using-a-glossary)

## The short version

| Situation | Model | Why |
|---|---|---|
| Default, any target | `openai/gpt-oss-20b` | Never once failed to translate across every run of every language pair tested, and the best semantic scores in the field. 8–24s per clip |
| You need it faster | `qwen/qwen3-8b` | Just as reliable mechanically and about twice as fast, but 9/13 on Bulgarian meaning and it makes clitic errors ("Те обичам много" for "Обичам те много") |
| **Do not use** | `aya-expanse-8b` | 12/12 on Russian meaning, **3/13 on Bulgarian** with 12 Russian intrusions — while passing every mechanical check in both |
| **Do not use** | `liquid/lfm2-24b-a2b` | Fastest in the field (4–14s), mechanically clean everywhere, and 5/13 on Bulgarian meaning |
| **Do not use** | `google/gemma-4-e4b` | Narrates the task instead of translating on most runs |
| **Do not use** | `qwen/qwen3.5-9b`, `zai-org/glm-4.6v-flash` | Reason past every output budget and never answer |
| Avoid for batch work | `qwen/qwen3.6-27b` | Good output, but exceeded 300s per run on a nine-segment clip. Fine for one video, not for a queue |

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
actually says, it scored **3, with 12 Russian intrusions**.
`openai/gpt-oss-20b` scored 13.

This is the failure mode to design against: not a model that refuses, but one
that confidently answers in the wrong language. And note what it is *not* — the
same model scores a clean 12/12 translating the same clip into Russian. It is
not a bad model; it is a Russian model being asked for Bulgarian.

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
model property: translating the same clip into Bulgarian, models ranged from
1.02x to 1.18x the source length.

**It must not be a reasoning model.** `qwen/qwen3.5-9b` spends a full
chain-of-thought on *"Hola, mamá"* even with `LM_STUDIO_REASONING=off` and
Qwen's `/no_think` token, and never finishes a batch. See
[Thinking models and translation speed](../README.md#thinking-models-and-translation-speed).

**Beware of models that look good because they say less.** `aya-expanse-8b` had
the *best* length ratio of any model on the Bulgarian clip, at 1.02x. It earned
that by dropping meaning — the same run scored 3/13 on the semantic checklist.
Short output is only a virtue when the meaning survives.

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
because the failure modes are per-script. The published matrix covers nine
languages across seven scripts:

| Target | Script | What it exposes |
|---|---|---|
| `es`, `pt` | Latin | The baseline. No script check is possible, so failures here are subtle |
| `ru`, `bg` | Cyrillic | Models trained mostly on Russian answer in Russian for *any* Slavic target. This pair is where every model in the field separates |
| `hi` | Devanagari | Output runs ~1.3x the source in characters. Length, not vocabulary, is the risk |
| `ar` | Arabic | Right-to-left, and output is *shorter* than its Latin source in characters while taking about as long to say |
| `zh` | Han | A Chinese line is 0.33x the length of its Spanish source. Every length-based intuition inverts |
| `ja` | Kana + Han | Mixed script; the script check has to accept both |
| `ko` | Hangul | Numbers *can* be written as ordinary Hangul words (십팔), which no pattern counts — but every model tested kept the digits, so the check still runs and a spelled-out number lands in the same tolerance every language gets |

Bulgarian is not in the default `--targets`, because it is a smaller audience
than the other five — but it is the most informative language in the matrix,
and it is where the semantic checklists earn their keep.

**Right-to-left rendering** remains open: Arabic is checked for script, but not
for bidirectional text handling in the burned-in subtitles.

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

### Mechanical checks rule models out; checklists rule them in

Most of the benchmark is mechanical: did the model answer, in the target
language, keeping the numbers, every time. None of that measures whether the
translation is *correct*, and the gap is not small.

`aya-expanse-8b` is the proof. Into Russian it scores a clean **12 of 12** on
the semantic checklist. Into Bulgarian, the same model on the same source
scores **3 of 13 with 12 Russian intrusions**. Every mechanical check passes
in both cases. If you only had the mechanical columns you would conclude it was
fine for both.

One line makes the gap concrete. Translating *"My daughter was 18 months old"*
into Arabic, it wrote **الثامنة عشر من عمرها** — "eighteen **years** old". The
benchmark noticed something: the digit `18` was spelled out rather than kept.
It could not notice the thing that matters, which is that a baby became a
teenager.

That is what [semantic checklists](#semantic-checklists) are for. Where one
exists for your language pair, the verdict means something: a model that gets
less than two thirds of the checked meaning right is marked unusable no matter
how clean its mechanics. Where one does not exist, read a `❌` or `⚠️` as
strong evidence against a model and a `✅` as nothing more than "it cleared the
bar where a machine can judge".

If you are dubbing into a language you do not speak and there is no checklist
for it, the cheap safety net is to dub one clip into a language you *do* speak
first, with the same model and the same context hint. A model that mangles a
language you can check is not going to do better on one you cannot.

### Semantic checklists

A checklist is a list of things the source actually says, each with a pattern
the translation must match — and, where a specific wrong answer is known, one
it must not. Every check is scoped to one segment, so a score traces back to
the line that produced it.

Three ship with the repo, covering Spanish→Bulgarian, Spanish→Russian and
English→Spanish. They are what turns a matrix where everything passes into a
real ranking:

| Model | `es`→`bg` meaning | `es`→`ru` meaning |
|---|---|---|
| `openai/gpt-oss-20b` | 12/13 | 12/12 |
| `qwen/qwen3-8b` | 9/13, 1 penalty | 12/12 |
| `liquid/lfm2-24b-a2b` | 5/13 | 12/12 |
| `aya-expanse-8b` | 3/13, 12 penalties | 12/12 |

These are worst-run figures, the same ones
[the generated matrix](benchmarks/decision-matrix.md) reports — a model is only
as good as its bad run. `openai/gpt-oss-20b` scores 13/13 on a good run and
12/13 on a bad one, which is why it lands ⚠️ rather than ✅ for Bulgarian.

Every one of those models is mechanically clean in both columns. Bulgarian is
where they separate, because most of them have seen far more Russian than
Bulgarian and answer in Russian-shaped text when pushed.

**Writing one needs no Python — only that you read the target language**, and
each one roughly doubles what the benchmark can decide for that language. The
format and the pitfalls are in
[`tests/fixtures/benchmark/checklists/README.md`](../tests/fixtures/benchmark/checklists/README.md).
The main pitfall, learned the hard way twice: make the pattern accept every
*correct* answer, not just the phrasing you happened to think of. An early
Bulgarian checklist rejected "Майчин ден", which is the more idiomatic
rendering of Mother's Day than the one it was looking for.

### Speech rates, and why drift is now trustworthy

The benchmark predicts timing drift from a translation's character count, which
needs a characters-per-second figure per language. Those numbers used to be
guesses hard-coded in the tool. `tools/gochidubb_speech_rate.py` replaces them
with measurements — it synthesizes the real translated lines the benchmark
already collected and divides characters by audio seconds:

```bash
python tools/gochidubb_speech_rate.py            # every language in results.json
python tools/gochidubb_speech_rate.py --show     # what is recorded
```

The guesses were off by up to 25%, and in one case wrong in kind:

| Language | Guessed | Measured | Off by |
|---|---|---|---|
| Korean | 5.83 | 4.39 | −25% |
| Spanish | 10.60 | 12.75 | +20% |
| Arabic | 9.54 | 7.79 | −18% |
| Chinese | 3.18 | 3.73 | +17% |
| Hindi | 9.01 | 9.32 | +3% |

The Spanish row is the instructive one. The old model assumed every
Latin-script target shared one rate; Spanish is actually 20% faster than
Bulgarian. Grouping languages by script was the wrong abstraction.

Rates live in `tests/fixtures/benchmark/speech_rates.json`. Measurement uses
edge-tts, which needs no GPU and covers all 65 languages, then anchors the
whole set to a rate measured from a real VoxCPM2 dub — so what the drift
estimate depends on, the *ratio* between languages, is measured rather than
assumed. Languages with no recorded rate fall back to a per-script guess and
are marked `~` in the matrix.

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
