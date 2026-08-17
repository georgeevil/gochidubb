# Stage reuse (beta)

Re-dubbing a video into a second language currently recomputes the download,
the audio extraction, the transcript and the diarization — none of which
depend on the target language. Measured across 19 real jobs, that is **59.8% of
all pipeline time** spent recomputing work that was already done.

| Stage | Median | Share of pipeline time | Depends on target language? |
|---|---|---|---|
| diarize | 121s | 25.9% | **no** |
| extract | 24s | 22.5% | **no** |
| transcribe | 36s | 10.8% | **no** |
| download | 0.3s | 0.6% | **no** |
| tts | 382s | 24.4% | yes |
| translate | 52s | 12.6% | yes |
| assemble + merge | 8s | 3.2% | yes |

Stage reuse keys each stage on its *inputs* rather than on the job that ran it,
so a second job whose inputs hash the same can copy the first job's output.

**It is off by default.** Turn it on with `GOCHIDUBB_REUSE=1` in `.env`, or
`reuse_enabled` in `config-user.json`, then restart. The beta page at
[`/beta`](http://localhost:8910/beta) shows what is cached, what is being
reused, and why anything is being refused.

## What gets reused, and when

A stage is reused only when **all** of these hold:

1. It is in `reuse_stages` (default: `download,extract,transcribe,diarize`).
2. Its fingerprint matches a previous job's.
3. The files that job wrote still exist.
4. The quality recorded at the time clears the gate.

The fingerprint is a hash of exactly what determines that stage's output:

| Stage | Fingerprint inputs |
|---|---|
| `download` | source URL, or the content hash of the local file |
| `extract` | source content + `auto_denoise` + `keep_bg` |
| `transcribe` | audio content + `whisper_model` + `source_lang` |
| `diarize` | audio content + `speaker_mode` + diarization model |
| `translate` | transcript text + target language + model + context hint + glossary |
| `tts` | translated text + speaker references + engine + voice settings + seed |

`assemble` and `merge` are never cached: together they are 3.2% of runtime and
they are the stages you actually re-tune (`GOCHIDUBB_TTS_MAX_STRETCH`), so
caching them would trade nothing for a stale-output risk.

So, concretely:

- **Re-translate with a different model** — transcribe and diarize are reused;
  translate onward re-runs. You pay ~40% instead of 100%.
- **Dub into another language** — same: everything before `translate` is reused.
- **Change `whisper_model`** — transcribe and diarize re-run, but the extracted
  audio they work from is still reused.
- **Toggle `auto_denoise`** — extract re-runs, and so does everything after it,
  because the audio it produces is different.
- **Two URLs serving byte-identical media** — only the fetch repeats.
  Identity is the content, not the link.

## Quality gates

An artifact's quality is measured **when it is written**, not when it is read —
at reuse time the evidence (the audio, the QA transcripts) is expensive to
recompute or already gone. Reuse then becomes a cheap comparison.

| Gate | Default | Refuses |
|---|---|---|
| `reuse_transcribe_min_word_conf` | 0.45 | A transcript whisper was unsure about |
| `reuse_transcribe_max_no_speech` | 0.35 | A transcript that is mostly silence |
| `reuse_diarize_min_ref_sec` | 1.0 | Speaker references too short to clone from |
| `reuse_translate_max_fallback` | 0.0 | Any segment left untranslated |
| `reuse_translate_min_semantic` | 0.67 | A translation that failed its checklist |
| `reuse_tts_max_failed_qa` | 0.25 | Synthesis that mostly failed QA |

A stage that fails its gate is recomputed. The bad artifact is *not* deleted —
its row stays, and the beta page shows the score, so "why is nothing being
reused" has an answer.

Unmeasured quality passes. Refusing everything a stage cannot score itself
would disable the cache on the first such stage rather than on the bad work it
is meant to catch.

## Changing a stage's behaviour

**If you change what a stage does, bump its version in `app/reuse.py`.**

```python
STAGE_VERSIONS = {
    ...
    "transcribe": 1,   # ← bump this
}
```

The version is part of the fingerprint, so a bump retires every cached entry
for that stage without deleting anything. Skipping it is the one way this
feature fails badly and silently: upgrade faster-whisper without bumping, and
every job keeps serving transcripts from the old one, forever, with nothing in
the log.

Things that need a bump: a new model default, a changed ffmpeg filter chain, a
different prompt, a library upgrade that changes output.

## Failure modes, and what happens

Reuse is an optimisation, and an optimisation that can fail a job is worse than
no optimisation. Every path degrades to "just run the stage":

| Situation | Result |
|---|---|
| The source job's files were deleted | Miss; the dead row is forgotten |
| The store is locked, corrupt, or unreadable | Miss, logged; the stage runs |
| A fingerprint cannot be computed (unreadable audio) | Not cached; the stage runs |
| Recording an artifact fails | The job still succeeds; a future miss |
| Quality below the gate | The stage runs; the row keeps its score |

Every job records what it reused, in `reused_stages` on the job and via
`GET /api/beta/reuse/job/{job_id}`. A run that silently skipped work would be
impossible to debug when its output looks wrong.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/beta/reuse/status` | Config, per-stage cache contents, gate thresholds |
| `GET /api/beta/reuse/entries?stage=&limit=` | Recent artifacts with their quality |
| `GET /api/beta/reuse/job/{job_id}` | What one job reused and contributed |
| `GET /api/beta/reuse/plan/{job_id}?target_lang=` | What re-running this job would reuse, without running it |
| `POST /api/beta/reuse/purge` | Forget cached artifacts (`stage`, `job_id` optional) |

Purging only clears the index. Job files are deleted through the normal
job-deletion path.

## Not done yet

**Per-segment translation reuse.** Short lines recur constantly across videos,
so this is where a shared cache would pay off most. It is deliberately not
built: `pipeline/translator.py` sends neighbouring lines for continuity, so a
segment's translation depends on its context window. Caching per segment would
discard that and reintroduce the pronoun and tense drift the batching exists to
prevent. It would need the neighbour window in the key.

**A shared multi-user store.** The mechanism already works — a content hash is
global, so the table is the only thing that is local. What is unsolved is
policy, not caching: a shared translation cache leaks what other people are
dubbing, and a shared transcript cache leaks their source material.

**`tts` reuse rarely hits.** With the default `auto` voice preset and no
`voice_style`, the voice seed is derived from the job id, so every job gets a
unique tts fingerprint. That is correct — the seed really does change the audio
— but it means enabling `tts` in `reuse_stages` will report zero hits unless a
preset or style pins the seed. It is not in the default set for this reason.

**Cache size management.** Nothing evicts. Artifacts live as long as the jobs
that produced them, so deleting old jobs is what reclaims space today.
