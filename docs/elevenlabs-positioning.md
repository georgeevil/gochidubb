# Covering ElevenLabs, and beating it somewhere it cannot follow

A product-strategy read on where GoChiDUBB stands against ElevenLabs in August 2026:
what of their surface is worth covering, what is worth refusing, and what the
differentiating factor should actually be.

Two kinds of claim appear below and they are kept separate on purpose.
**Repo facts** are read out of this checkout and cite the file. **Market facts**
are from public sources checked 2026-08-21 and cite the source in §11 — they are
vendor and review-site copy, not measurements, and ElevenLabs moves fast enough
that they want re-checking before anything commercial is built on them.

---

## 1. The short version

ElevenLabs sells **finished audio, metered by the minute, from a black box.**
Three pillars — Agents, Creative, API — over one shared credit pool.

Trying to cover that surface feature-for-feature is how this project loses. Music
generation, sound effects and conversational agents are three separate products
against a company with far more capital, and none of them is dubbing.

The winnable framing is narrower and sharper:

> **ElevenLabs gives you a file. GoChiDUBB gives you the pipeline that made it —
> inspectable at every stage, drivable by an agent, portable across engines, and
> priced on capacity rather than credits.**

Three legs hold that up, ranked by how hard they are to copy:

| Leg | Durability | Status |
|---|---|---|
| **Local / open / portable** — runs offline, MIT, swap any engine | **Permanent.** A cloud company with a credit meter structurally cannot ship this | ✅ shipped |
| **Capacity pricing, never credits** — failures free, multilingual genuinely cheaper | **Semi-durable.** They could reprice, but the shared credit pool spans every product | ⚠️ metered as an estimate only (`app/billing.py`) |
| **Agent-native control plane** — 23 MCP tools, quality gates an agent can act on | **~6–12 months.** An MCP wrapper is easy; a gated, resumable, per-stage-inspectable pipeline is not | ✅ mostly shipped |

Lead the *messaging* with the agent-native wedge, because "open-source dubbing
tool" is a crowded and undifferentiated headline. Lead the *moat* with local and
open, because that is the leg nobody can take away.

---

## 2. What ElevenLabs actually sells in 2026

Reorganised into three pillars during the past year:

**ElevenCreative** — TTS (v3), Eleven Music, sound effects, Studio, Dubbing Studio.
**ElevenAgents** — conversational voice agents across phone, chat, WhatsApp, email.
**ElevenAPI** — Scribe v2 STT, low-latency synthesis, Python/TS SDKs, SOC 2 / HIPAA /
GDPR, EU data residency, zero-retention modes.

The dubbing-relevant surface, which is the only part that matters here:

| Capability | ElevenLabs, 2026 |
|---|---|
| Dubbing model | **Dubbing v2** — 90+ languages, sync-aware translation that aligns starts/stops/pacing, emotion and prosody carried across languages |
| Speakers | Up to **32** unique per file (9 recommended); overlapping speech handled |
| Voice similarity | **Cloning strength 0–10**, default 7 — one dial trading similarity against natural target-language phonetics |
| Background | Music/FX/ambience preserved without re-mixing |
| Manual control | **Dubbing Studio** — transcript editing, speaker reassignment, per-clip regeneration. **Currently in maintenance mode, still on the v1 model, critical fixes only** |
| Limits | App: 1 GB / 180 min. Studio: 1 GB / **45 min**. API: 3 GB. Concurrency 3 jobs self-serve, 10 enterprise |
| Watermark | Free tier watermarked; paid not |
| STT | **Scribe v2** — realtime <150 ms; batch with diarization to 48 speakers, PII redaction over 56 categories, keyterm prompting to 1 000 terms |
| Voices | Instant + professional cloning, curated Voice Library collections |
| Price | Free / $6 Starter / $22 Creator / $99 Pro / $299 Scale / $990 Business / Enterprise. Dubbing **$0.33/min** watermarked, **$0.50/min** clean, **$0.50/min** Studio; reported range up to **$2.20/min** |
| Credits | ~2 000 credits/min auto-watermarked → 10 000/min Studio clean. Creator's 121 000 credits ≈ **40 min** of automatic dubbing, or **~12 min** in Studio. One pool shared across TTS, STT, cloning and dubbing; resets monthly |

Two entries in that table are strategic openings rather than trivia, and §5
returns to both: **Dubbing Studio is in maintenance mode on the old model**, and
**45 minutes is the ceiling on the only screen that offers manual control.**

---

## 3. What GoChiDUBB already has

Audited from this checkout, because the gap is much smaller than `README.md`
implies and the strategy depends on knowing that.

**Pipeline** — yt-dlp ingest, ffmpeg, Silero VAD, faster-whisper, pyannote
diarization, demucs / audio-separator background split, LLM translation over
LM Studio or Ollama, four TTS engines (`VoxCPMSynthesizer`, `CosyVoiceEngine`,
`F5TTSEngine`, `EdgeTTSFallback` — `pipeline/synthesizer.py`), whisper-roundtrip
QA, timing-aware assembly, SRT, optional Wav2Lip lip-sync. **66 target languages**
(`static/index.html`).

**Surface** — 92 REST routes, **23 MCP tools** (`tools/gochidubb_mcp.py`), a CLI,
a single-page UI. All three drive one backend.

**Human-in-the-loop, already built** — `edit_transcript`, `edit_translations`,
`edit_speaker_ref`, `regenerate_segment/{i}`, `retry_stage/{stage}`, `continue`,
per-stage checkpoints, `awaiting_translation_review` as a first-class job state.
*This is Dubbing Studio's feature list, on a backend that is not in maintenance mode.*

**Quality as data** — `pipeline/quality.py` scores five dimensions (ASR,
translation, TTS, timing, loudness), emits verdicts with `suggested_action`, and
`gate()` parks a job at `awaiting_review` **before** the expensive stage rather
than reporting the failure afterwards. `pipeline/metrics.py` records per-stage
duration and GPU peak. `tools/audit_job.py` reconstructs what a finished job lost.

**Platform plumbing** — scoped hashed API keys (`app/apikeys.py`, 6 scopes),
HMAC-signed webhooks on 3 events (`app/webhooks.py`), an activity feed
(`app/activity.py`), an append-only audit log (`app/audit.py`), a usage meter
(`app/billing.py`), storage stats and cleanup, content-addressed stage reuse
(`app/reuse.py`, beta).

**Things ElevenLabs has no equivalent for** — YouTube URL → dubbed MP4 in one
step; stitched multilingual showcase reels; trend scout (`app/scout.py`); direct
publish to VK (`pipeline/publisher.py`); a translation **glossary**
(`/api/glossary`); free choice of translation model; no duration ceiling —
the longest job recorded in this repo ran **16.2 hours**.

---

## 4. The gaps that matter

Ranked by value ÷ effort. The estimates lean on how much of each already exists.

### T0 — weeks of work, each unlocks a whole category

**1. Standalone text-to-speech.** There is no `/api/tts` route anywhere in
`server.py`. Four TTS engines are loaded, warmed and QA'd — and the only way to
reach them is to dub a video. This is ElevenLabs' entire core product sitting
behind a door that was never cut. A `POST /api/speak` taking text, a voice preset
and a language, returning WAV, is a few days' work and turns a dubbing tool into
a voice platform. **Do this first.**

**2. Standalone speech-to-text.** Same story: faster-whisper, word timestamps,
VAD and pyannote diarization are all wired, and none of it is reachable without
starting a dub. A `POST /api/transcribe` returning segments, words, speakers and
SRT/VTT is Scribe's batch product, minus the $0.22/hour. Days, not weeks.

**3. Cloning-strength dial.** `voxcpm_cfg`, `voxcpm_xling_cfg` and
`voxcpm_xling_steps` (`app/config.py`) already control exactly the
similarity-versus-naturalness trade-off ElevenLabs exposes as 0–10. Today they
are three expert settings in a Settings tab. Map them onto one per-job slider.

**4. Subtitles as a product.** SRT exists; VTT does not, and there is no
translate-only mode that skips synthesis entirely. A large share of localization
demand is captions, not dubs, and this pipeline already does all the hard parts.

### T1 — the parity gaps worth closing

**5. A real segment editor.** `/api/waveform`, `/api/job/{id}/speaker_ref/{s}/waveform`
and `regenerate_segment/{i}` exist; the UI drives them through forms. A timeline
with per-segment audio, inline translation editing, drag-to-retime and
regenerate-this-clip is 2–3 weeks of front-end over a backend that is already
there. It closes the single biggest experience gap — **against a competitor whose
equivalent screen is frozen on the previous model.**

**6. Speaker naming.** Already on the README roadmap. `/api/job/{id}/speakers`
returns them; they are `SPEAKER_00`. Names, per-speaker voice assignment and
per-speaker cloning strength.

**7. Dialects.** Language handling is two-letter throughout
(`_LANG_SCRIPT`, `translator.py:1030`). ElevenLabs ships `es-MX` vs `es-ES`,
`pt-BR` vs `pt-PT`, four English dialects. The translator prompt and the edge-tts
voice map are both already keyed by language — extending the key to BCP-47 is
mechanical, and it is the difference between "Spanish" and a Spanish a Mexican
audience does not wince at.

**8. Voice library.** `presets/voices/` is a directory. Browsing, tagging,
preview, import/export, and a prompt-to-voice design mode would make it a library.

### T2 — where differentiation actually gets built

**9. Ship the project, not just the file.** The pipeline already produces
isolated `vocals`, isolated `background`, and one WAV per segment — one job in
this repo held **2 885 files in `tts_segments/`** — and then deletes them as
intermediates. `/api/dub/{id}/export` is aspect-ratio presets for social
platforms, nothing more. Package the real thing instead: dialogue stem,
background stem, per-segment clips, timed SRT, and an EDL or OTIO timeline that
imports into Resolve or Premiere. **No cloud dubbing vendor hands you an editable
project, because their business is selling you the render.** This is the cheapest
genuine differentiator on the list — it is packaging work over artifacts that
already exist.

**10. Turn stage reuse on and expose it.** `app/reuse.py` measures that download +
extract + transcribe + diarize are **59.8% of all pipeline time over 19 real jobs**,
and that none of it depends on the target language. It is off by default and
labelled beta. It is also the technical fact that justifies the pricing model in §7.

**11. Translation memory and glossary, properly.** `/api/glossary` exists as a
flat term map. Per-client glossaries, do-not-translate lists, and reuse of
previously approved translations across jobs is what localization *agencies* buy,
and ElevenLabs does not sell it for dubbing at all.

**12. Compliance posture.** Not SOC 2 — that is a hosted problem. The claim to
make is stronger and already true: **the audio never leaves the machine.** For
health, legal, defence and broadcast-embargo work that is not a weaker story than
zero-retention mode, it is the only story that survives procurement. It needs
writing down, not building.

---

## 5. Two openings worth naming

**Dubbing Studio is frozen.** ElevenLabs' own docs describe it as in maintenance
mode, still on the v1 model, receiving critical bug fixes only — while the new v2
model ships to the automatic path. Their message to anyone who needs manual
control is, in effect, *use the old model.* GoChiDUBB's manual-control surface is
on the current pipeline and improving. That is a live opening, and it will not
stay open forever.

**45 minutes.** The Studio ceiling. Anyone dubbing a lecture, a sermon, a
conference talk, a podcast episode or a documentary is outside it. This repo has
a 16.2-hour job on record. Long-form is a segment ElevenLabs is not serving on
its editable path.

Both point at the same customer: **long-form, multi-speaker content where somebody
has to sign off on the output before it ships.** That is the wedge. It is not
"everyone who needs a voice".

---

## 6. What to refuse

Refusing is the strategy, not a gap in it.

- **Music generation.** Different model class, different rights problem, no adjacency.
- **Sound effects.** Same.
- **Conversational agents.** A platform business — telephony, turn-taking, sub-second latency, CRM integrations. It would consume everything and compete on the axis where a cloud vendor is strongest.
- **Realtime streaming TTS.** <150 ms latency is an infrastructure product. Batch dubbing does not need it.
- **A voice marketplace with payouts.** Payments, identity, rights administration and a moderation burden, for a network effect that needs scale this project does not have.
- **Chasing 90+ languages.** 66 is enough. The 24 missing ones are long-tail; being *demonstrably better* on the top 20 is worth more than a bigger number in a table.

The line: **cover the primitives under dubbing (TTS, STT, voices), refuse the
pillars beside it (Music, SFX, Agents).**

---

## 7. The billing model

This is where the clearest differentiation is available, because ElevenLabs'
model has documented, structural, load-bearing problems.

**What is wrong with credits**, from public reviews: one pool shared across TTS,
STT, cloning and dubbing, so a heavy voiceover month silently shrinks dubbing
capacity. Credits reset monthly — unused capacity evaporates. Regenerations and
bad outputs consume credits, with one reviewer reporting an effective rate **2.8×**
the advertised one. And the same minute costs $0.33, $0.50 or $2.20 depending on
watermark and surface, which nobody can forecast.

Every one of those is a promise available to be made in the opposite direction.

### Three lanes

**Lane 1 — Local. Free forever, MIT.** Your GPU, your data, no account. This is
not a loss leader, it is the moat and the funnel: it is the thing ElevenLabs
cannot answer, and it is already shipped.

**Lane 2 — Bring your own compute. Flat software fee.** Point the control plane at
your own GPU box, or your own RunPod/Vast/Lambda account. GoChiDUBB charges a flat
monthly fee per workspace; compute is billed to you, at cost, by your provider.
**Nobody meters your minutes.** Margin comes from software, not GPU arbitrage —
which also removes the trap `docs/cloud-architecture.md` documents, where the
coded tiers do not cover cost in any modelled scenario.

**Lane 3 — Hosted, for people who will not run a GPU.** Metered, but with four
promises credits cannot make:

1. **Failures are never billed.** Already true in code — `app/billing.py` excludes `error`, `cancelled` and `failed`. Put it on the pricing page.
2. **Regenerations are never billed.** Re-running a segment you rejected is quality control, not consumption. This is the complaint that produced the 2.8× figure.
3. **Minutes do not expire.** No monthly reset, no shared pool with unrelated products.
4. **The budget cap is enforced, not advisory.** A hard stop, not an email after the fact.

### The pricing shape that follows from the architecture

The interesting one: **the second language on the same source costs materially
less than the first.** Not a discount — a measurement. `app/reuse.py` establishes
that download, extract, transcribe and diarize are 59.8% of pipeline time and are
language-independent, so the second language genuinely costs roughly 40% of the
first. Price it that way:

```
first language        full rate per source-minute
each additional       ~40% of full rate
```

ElevenLabs charges full freight per language because it has no reuse story to
price against. **"Multilingual is cheaper here because it actually costs us less"**
is a claim that is true, verifiable, derived from the architecture, and awkward
for a credit-pool competitor to match. It is also precisely the multi-language
work this project is built around — batch compare, showcase reels.

Two corrections `docs/cloud-architecture.md` already establishes and this model
must not repeat: **the current $0.050/min third tier is at or below cost**, and
**storage at $0.02/GB is below R2 list price before retention is discussed.** The
tier ladder needs repricing before anything is quoted to anyone, and "100 GB
included" should be redefined as rolling 30-day retention.

---

## 8. Where GoChiDUBB is already better — say so

Most of these are shipped and none of them are in the README's comparison table.

| | GoChiDUBB | ElevenLabs |
|---|---|---|
| Source length | No ceiling (16.2 h on record) | 180 min app, **45 min Studio** |
| Manual editing | Current pipeline, every stage | Studio, **maintenance mode, v1 model** |
| Failed / regenerated output | Never billed | Credits consumed |
| Data | Never leaves the machine | Uploaded; zero-retention is a paid mode |
| Quality | Per-stage scores + verdicts + **gates that stop the job before wasting GPU** | Not exposed |
| Resume | Per-stage checkpoints, retry any stage | Re-run the job |
| Engine choice | 4 TTS engines, any local LLM | Fixed |
| Glossary / terminology | `/api/glossary` | Not offered for dubbing |
| Agent control | 23 MCP tools, first-class | REST/SDK only |
| Cost at volume | Electricity | $0.33–$2.20 / min |
| Source of truth | MIT, auditable, forkable | Closed |

The quality-gate row deserves more weight than it gets. ElevenLabs bills you for a
dub and lets you discover it was bad. GoChiDUBB scores five dimensions, refuses to
spend GPU on synthesis behind a failing transcript, and hands an agent the verdicts
in a `job.awaiting_review` webhook. **"The dub arrives with its own QC report"** is a
differentiator that matters to exactly the buyer §5 identifies — the one who has to
sign off.

---

## 9. Sequence

**Now (2–4 weeks).** `/api/speak` and `/api/transcribe`. Cloning-strength dial.
VTT + translate-only mode. Rewrite the README comparison table around §8 —
several rows in the current one understate what shipped.

**Next (1–2 months).** Segment editor. Speaker naming and per-speaker voices.
BCP-47 dialects. Voice library. Stage reuse out of beta and on by default.

**Then (2–3 months).** Editable project export — stems, per-segment clips, EDL/OTIO.
Translation memory and per-client glossaries. Write the local-processing compliance
posture down properly.

**Only with evidence of demand.** Anything in `docs/cloud-architecture.md` §6 —
three to four months of prerequisite engineering before a hosted lane is safe to
expose, against roughly $1 000/year of infrastructure at low load. Lane 2 needs
almost none of it. **Build Lane 2 first, and let it decide whether Lane 3 is
worth it.**

---

## 10. What would kill this

- **YouTube ingestion from datacenter IPs** (`cloud-architecture.md` §5). The headline feature errors for every hosted user at once, and residential proxy economics are negative on every YouTube job. Lane 2 sidesteps this entirely — another reason it comes first.
- **VoxCPM2 quality falling behind.** The entire dubbing case rests on cloned-voice quality against a competitor with a dedicated research team. Mitigation is already structural: four engines behind one interface, so the bet is on the pipeline, not on any one model.
- **ElevenLabs un-freezing Dubbing Studio.** The §5 opening closes. The long-form ceiling and the local-processing moat survive it; the editing-parity advantage does not.
- **Chasing all three pillars.** The most likely failure, and the cheapest to avoid: it is a decision, not a capability.

---

## 11. Sources

Market facts checked 2026-08-21. Vendor and review-site copy, not measurements.

- [Introducing Dubbing v2 — ElevenLabs](https://elevenlabs.io/blog/introducing-dubbing-v2)
- [Dubbing capabilities — ElevenLabs docs](https://elevenlabs.io/docs/capabilities/dubbing)
- [ElevenLabs pricing](https://elevenlabs.io/pricing) · [ElevenAPI pricing](https://elevenlabs.io/pricing/api)
- [ElevenLabs in 2026: v3, Agents, Music and Scribe — Standout Digital](https://standout.digital/post/elevenlabs-in-2026-the-complete-guide-to-v3-agents-music-and-scribe/)
- [The Complete Guide to ElevenLabs Plans, Overages, and Usage-Based Pricing in 2026 — Flexprice](https://flexprice.io/blog/elevenlabs-pricing-breakdown)
- [ElevenLabs Pricing Explained (2026) — fish.audio](https://fish.audio/vs/pricing/elevenlabs/)
- [ElevenLabs AI Dubbing Pricing: The Full 2026 Breakdown — Geckodub](https://blog.geckodub.com/elevenlabs-ai-dubbing-pricing)
- [ElevenLabs Review 2026 — Brutally Honest Pros, Cons & Hidden Costs — qcall.ai](https://qcall.ai/elevenlabs-review)
- [ElevenLabs Reviews 2026 — G2](https://www.g2.com/products/elevenlabsio/reviews)

Repo facts are cited inline. Related internal docs:
[`saas-redesign-plan.md`](saas-redesign-plan.md) (what the SaaS shell already ships)
and [`cloud-architecture.md`](cloud-architecture.md) (what hosting would cost and
what it would take).
