# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`voxcpm` is now pinned to `>=2.0.3`, and the Python range is 3.10–3.14.**
  The old `>=2.0.0` floor allowed a build where reference clips are trimmed
  twice (voxcpm removed its own auto-trim in 2.0.1; the pipeline already trims
  in `_REF_FILTER`), and missed 2.0.3's fixes for MPS audio quality, CUDA graph
  issues and file-descriptor leaks — the last of which a long-running server
  hits. The Python range was described in four places as a "VoxCPM2 constraint"
  capped at 3.12; VoxCPM declares no upper bound, the real ceiling is `spaces`
  at `<3.15`, and 3.13 resolves the identical dependency set as 3.12 with
  wheels available through 3.14. The installers now accept 3.10–3.14, warning
  rather than blocking above 3.12, and the macOS/MPS recommendation says what
  is actually known rather than repeating a pre-2.0.3 claim.
- **The Studio UI is redesigned around an agent-first workflow** (concept 1a,
  `docs/saas-redesign-plan.md`). The Agent feed — a chronological stream of
  runs, tool calls and system events — is the new home screen; the rail groups
  Work / Develop / Workspace; Jobs and Library consolidate the old
  History/Processing/Review/Batch and Result/Voices/Glossary views behind
  tabs; a ⌘K command bar turns plain language ("dub the last upload into fr,
  es") into dubs. The dark theme is retuned (#c8f542 accent, Geist display
  type) and the hero sphere redrawn to match. Local mode keeps today's
  behavior exactly: no auth, no billing surfaces.

### Fixed
- **`voxcpm_steps` now actually affects synthesis.** It defaulted to `10` but was
  ignored on the batch path, which always recomputed the step count from the
  per-job speed tier — so the setting had no effect on any normal dub. It is now
  an explicit override, with `0` meaning "follow the speed tier" (the new
  default, and exactly what installs experienced before). Existing
  `config-user.json` files carrying the dead `10` are migrated to `0` once on
  startup, so upgrading does not silently change how your dubs are rendered.

- **Long videos no longer die at the final render with `timed out after 600
  seconds`.** The ffmpeg timeout in the merge/extract stages is now *soft*
  (`pipeline/ffmpeg_run.py`): ffmpeg runs with `-progress pipe:1`, and as long
  as the encode position keeps advancing the deadline is extended — a working
  x264 render of a 2-hour source is never killed mid-encode. A genuinely hung
  ffmpeg is still killed once no progress is seen for a stall window past the
  soft deadline. Tunable via `ffmpeg_timeout` / `ffmpeg_stall_timeout` in
  config (env: `GOCHIDUBB_FFMPEG_TIMEOUT`, `GOCHIDUBB_FFMPEG_STALL_TIMEOUT`).

### Added
- **VoxCPM2 settings are editable in the product** — a new **Settings → Voice &
  TTS** tab. Engine tier (`voxcpm` / `f5tts` / `edge-tts`), model id, guidance
  (`cfg`), inference steps, reference denoising, the cross-lingual guidance
  floors, same-language QA and the assembler's time-compression ceiling were all
  previously reachable only by hand-editing `config-user.json` and restarting.
  Saving now releases the loaded voice model so the next job picks the settings
  up — and is refused with a clear message while a job is mid-flight, rather
  than pulling the model out from under it.
- Settings are validated before they are stored (`FIELD_SPECS` in
  `app/config.py`): values are coerced from form strings, out-of-range numbers
  and unknown choices are rejected with a message naming what was wrong, and a
  batch save applies all-or-nothing instead of leaving half of it written. The
  same bounds now apply to the `VOXCPM_*` environment variables.

- **Develop surfaces for driving GoChiDUBB from agents and scripts**: scoped
  API keys (stored hashed, shown once, revocable), webhooks for
  `job.completed` / `job.failed` / `job.awaiting_review` with a delivery log,
  an MCP server onboarding page over the existing tool set, and a merged
  `GET /api/activity` stream. `GOCHIDUBB_MODE=hosted` additionally reveals
  Members, metered-estimate Billing (clearly marked as an estimate), and an
  Audit log; `local` stays the default.
- **37 new target languages — 65 total.** Every candidate was verified live
  against the edge-tts voice catalog and Whisper's language list before being
  registered. New targets: Bengali, Urdu, Persian, Hebrew, Swahili, Filipino
  (Tagalog), Malay, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam,
  Slovak, Croatian, Serbian, Slovenian, Lithuanian, Latvian, Estonian,
  Catalan, Icelandic, Afrikaans, Macedonian, Albanian, Bosnian, Welsh, Kazakh,
  Azerbaijani, Uzbek, Georgian, Mongolian, Nepali, Sinhala, Burmese,
  Khmer and Lao. (Armenian and Punjabi were evaluated but have no edge-tts
  voices in Microsoft's current catalog, so the fallback tier can't cover
  them.)
  - Voice cloning (VoxCPM2) works for `he`, `sw`, `tl`, `ms`, `my`, `km`,
    `lo` (all in its official training coverage); the rest route to edge-tts
    neural voices automatically via `_EDGE_ONLY_TARGET_LANGS`.
  - Filipino uses the new Microsoft `fil-PH` voices (the old `tl-PH` ones
    were retired) while keeping Whisper's `tl` code end-to-end.
  - All new languages are also selectable as *source* languages.
  - Note: the default translation fallback model (`aya-expanse:8b`) doesn't
    cover the South-Asian wave — pick a wider multilingual model
    (e.g. `qwen2.5:14b`) when dubbing into those.
- **Bulgarian is now a first-class dubbing target.** `bg` was already in the
  edge-tts voice map (so `GET /api/languages` advertised it), but the UI
  dropdowns didn't list it and Quick Test / Showcase / Redub rejected it as
  an unknown code.
  - All 28 supported languages are now selectable in the UI — the language
    dropdown was stuck at the original 15.
  - Quick Test / Showcase / Redub validation now derives from the edge-tts
    voice map (the canonical registry), so a registered language can never
    be rejected at submission time again.
  - Targets the cloning engines were not trained on (`bg`, `uk`, `cs`, `ro`,
    `hu` are outside VoxCPM2's official coverage) synthesize via edge-tts
    instead of attempting a cross-lingual clone that can't be reliable —
    generic voice, but intelligible words.
  - Translation prompts now spell the language out ("into Bulgarian" instead
    of "into bg"), which smaller translation models handle more reliably.
  - Bulgarian (and every other supported language) can also be picked as a
    *source* language now — Whisper covers all of them.
- **Setup problems are visible in the UI instead of the console.** A stage could
  finish successfully and still not have done what the user assumed, and nothing
  in the API said so. The case that prompted this: `pyannote/speaker-diarization-3.1`
  fails to download, the loader falls back to `speaker-diarization-community-1`,
  the run completes normally — a multi-speaker video diarized by a weaker model
  with no visible sign. Stages now emit structured **notices**
  (`{code, severity, title, detail, remediation, url}`, `pipeline/notices.py`),
  where `code` is a stable slug used for deduping, dismissal and tests.
  - `GET /api/system` → `notices`, `checks`, `accelerator` (passive, no network,
    rides the poll the UI already makes)
  - `POST /api/diagnostics/run` — the only network probe: validates the HF token,
    asks the Hub whether each gated pyannote repo is actually accessible, pings
    the translation backend. On demand only, so the app still works offline.
  - `GET /api/dub/{id}/stages` → per-stage `notices` + `degraded`, plus a merged
    top-level array. `degraded` is "done, but not the way you think" without
    adding a fourth `state` every existing consumer would have to learn.
  - Four UI surfaces: a banner under the top bar, a chip in the top bar and left
    rail, a `⚠ degraded` marker on the affected stage row, and **System → Setup**
    with remediation steps, action links and an HF-token field.
  - A confirmed deep check retires findings it disproves — if the Hub says access
    is fine, a past download failure was transient and its warning clears.
- **In-app log viewer** (`GET /api/logs`, System → Logs). `logging.basicConfig`
  installed a console handler only, and `./start.sh` runs `serverctl foreground`,
  which writes no file — so in the normal flow nothing was recorded anywhere.
  A 2000-entry ring now captures log records *and* third-party stdout/stderr,
  which matters because pyannote prints its "accept user conditions" banner with
  a bare `print()` that no logging handler would ever see. Credentials are
  scrubbed on the way **in**, not at render time: the server binds `0.0.0.0`, so
  a secret that reaches the buffer is already on the LAN.
- `HF_TOKEN` can be set from **System → Setup** and takes effect immediately.
  `cfg.hf_token` existed in `app/config.py` but nothing read it, so a token
  entered anywhere but `.env` silently did nothing.

### Added
- **`tools/audit_job.py`** — audits a finished job's artifacts for content lost
  between stages. Reports duplicate/missing segment indices, untranslated
  segments, segments never synthesized or whose wav is gone, synthesized
  segments the assembler never placed, and stale `seg_*.wav` from earlier
  attempts. Exit code 1 on loss, `--json` for machines, `--all` for a sweep.
  - The primary check is **word coverage**, not segment counts: the pipeline
    legitimately merges fragments and splits long segments, so 190 → 152 is
    healthy while 152 → 150 is two lines of dialogue gone. Counting cannot tell
    those apart; words can only disappear if something was dropped.

### Fixed
- **Spoken lines silently vanished from finished dubs.**
  `pipeline/segment_post._split_very_long_segments` splits one segment into two
  via `dict(seg)` twice, so **both halves inherited the parent's `idx`**.
  `_stage_translate` then merges results with `by_idx = {s["idx"]: s ...}`,
  where a duplicate key overwrites a real segment. Every stage reported
  success and nothing was logged.
  - Confirmed on job `947ca81d`: 152 segments into translate, 150 out.
    *"Wow bigger than I thought it would be."* and *"These four players decided
    to keep my return a secret"* were both absent from the dub.
  - `postprocess_segments` now renumbers `idx` contiguously after its passes —
    it is the only place that restructures the list, so it owns index
    integrity. Replaying it over that job's real transcription now yields zero
    duplicates and zero words lost.
  - `_serialize_segments` additionally re-keys and loudly logs any duplicate it
    still sees, so no future source can drop a line quietly.

### Fixed
- **The UI was unusable on a phone.** The shell was a fixed 232px rail beside a
  flex column inside `#root { width: 100vw }` with `body { overflow: hidden }`.
  On a 390px screen that left ~158px of content, so every label wrapped one
  word per line, chips turned into circles, and the page could only be read by
  scrolling sideways.
  - Below 860px the rail becomes an off-canvas drawer with a scrim, opened from
    a hamburger in the top bar and closed on selection or on tapping away.
  - View padding routes through a `--pad` custom property so one media query
    retunes every screen; headings use `clamp()` instead of fixed sizes.
  - `100vw` → `100%` (100vw includes the scrollbar gutter, forcing a permanent
    horizontal overflow) and `100vh` → `100dvh` so the app is not cut off by
    the mobile URL bar.
  - `min-width: 0` on the content column — a flex child defaults to
    `min-width: auto`, which is what stopped long URLs from wrapping and pushed
    the page wider than the screen.
  - The floating sphere pill is hidden on narrow screens, where it sat on top
    of the content. Its "1-6 views · ⇧G sphere" hint was removed outright:
    there is no keydown handler anywhere in the app, so it advertised shortcuts
    that never existed.
  - Verified with headless Chrome at 320/360/390/430/768/1024/1440px:
    `scrollWidth == viewport` and zero overflowing elements at every width.

### Changed
- **Quick Test is now "Multi-language" — a primary workflow, not a smoke
  test.** It takes the same options as a single dub (source language,
  translation and Whisper models, speaker mode, voice, context hint,
  background separation, denoise, review pause, lip sync, scheduling), and
  **dubs the whole video by default**. Trimming to a clip is now an opt-in
  toggle for auditioning languages and voices before committing GPU time.
- **Up to 12 target languages** per multi-language or showcase run, raised
  from 6. The bound is served from `GET /api/system` so the picker and the
  API can no longer disagree.
- **The server binds `127.0.0.1` instead of `0.0.0.0`.** It has no
  authentication of any kind, so the old default offered job control, every
  transcript, `/api/logs` and `/api/config` to every device on the network —
  including whatever wifi a laptop happens to be on. `GOCHIDUBB_HOST=0.0.0.0`
  restores the old behaviour deliberately, and startup prints a warning
  whenever the bind is not loopback. `GOCHIDUBB_PORT` is now honoured too.

### Fixed
- **The UI froze for the entire run of a job.** Stage handlers are `async def`
  but called yt-dlp, ffmpeg, WhisperX and pyannote synchronously, pinning the
  one asyncio thread for minutes. Every HTTP request queued behind it, so:
  the Pipeline stages panel sat on "Loading stages…" for the whole job, a newly
  submitted job did not show as running until the in-flight stage finished, and
  the transcribe watchdog (an asyncio task reporting elapsed time) never got
  scheduled. Blocking calls now run via `asyncio.to_thread`; measured on a live
  job, `/api/dub/{id}/stages` went from **timing out at 20s** to **0.01s**, with
  the running stage visible throughout. A test walks `STAGE_HANDLERS` source to
  stop a new stage quietly reintroducing the freeze.
- **"Neither demucs nor audio-separator installed — no separation" said what
  broke but not what to do.** Background separation falls back to a *silent*
  track, which is indistinguishable from success in the output, so a user who
  asked to keep the music just got a quiet video. It now raises
  `audio.separation_unavailable` with install steps, and the Setup checklist
  carries a `Background separation` row. Deliberately a runtime notice rather
  than a passive one — the feature is opt-in per job, and warning everyone who
  never uses it is how a warning area becomes wallpaper.
- **`GET /api/config` returned `hf_token` in plaintext.** It is now masked
  (`hf_a…`); `PATCH` still accepts a real value and ignores a masked one being
  written back. This is reachable from the LAN, not just localhost.
- **The System panel always read "Not connected".** `/api/system` published the
  translation-backend probe as `status["ollama"]` while the UI read
  `system.lm_studio.ok`, a key nothing ever set — so LM Studio showed as down
  while it was serving models. Both spellings are now published from the one probe.
- **A permanent false "Torch broken!" alarm on Apple Silicon.** `check_gpu()` is
  CUDA-only, so `gpu.ok` was `false` on machines where MPS works fine, and the
  left rail showed a red badge that could never clear — training users to ignore
  the warning area. Accelerator status now comes from the MPS-aware probe already
  in `pipeline/metrics.py`, and the rail's warning tracks real notices instead.
- **Diarization discarded its own diagnosis.** `_load_pipeline` collected each
  model's failure into a local list that was only logged when *every* candidate
  failed; a partial failure that fell back successfully threw the explanation
  away. It is now reported as `pyannote.fallback_model`.
  - Deliberately neutral about the cause: pyannote prints *"the repository is
    private or gated"* for **any** `HfHubHTTPError` — a timeout, a 503 and a rate
    limit all produce that text (`pyannote/audio/utils/hf_hub.py:91-104`). It is
    a guess printed as a diagnosis. The deep check asks the Hub directly and can
    tell `pyannote.gated_model` from `pyannote.bad_token` from `hf.offline`.
- `diarize_speakers()` returned `[]` for everything from "no token" to "one
  speaker in the video". It still does, but now fills a caller-supplied
  `notices` list so the two are distinguishable.
- Removed unreachable code after the `return`/`except` in `diarize_speakers()`.

### Fixed
- **`tts_speed="fast"` crashed every TTS segment** with
  `UnboundLocalError: cannot access local variable 'latent_pred'`.
  `speed_retries["fast"]` was `0`, but voxcpm treats `retry_badcase_max_times` as a
  **loop bound**, not a retry count: it binds `latent_pred` only inside
  `while retry_badcase_times < retry_badcase_max_times:` and reads it after the loop
  (`voxcpm/model/voxcpm2.py:923-958`). At `0` the body never ran, so every segment
  failed in ~1 ms with a 100% failure rate for anyone selecting "fast" in the UI, CLI
  or MCP tools. The presets moved to module-level `SPEED_RETRIES` / `SPEED_TIMESTEPS`
  with a minimum of 1, and `tts_worker` now clamps the value so no job spec can
  reintroduce a `0`. "fast" still makes exactly one pass with no re-rolls
  (`retry_badcase=False` breaks the loop after the first attempt).
- **`tts_speed="fast"` silently discarded all voice cloning.** Its tier ladder was
  `[(3, tier3)]` — pure zero-shot voice *design*, with no `reference_wav_path` — so
  extracted speaker references were never used and the dub came out in an arbitrary
  voice. A single-entry ladder also meant one error killed the segment with no
  fallback. Now `[(2, tier2), (3, tier3)]`: tier 2 is already the cheap cloning path.
- **Tier 1 always raised on cross-lingual dubs.** `prompt_wav_path` was set
  unconditionally while `prompt_text` was set only when non-empty, and voxcpm requires
  both or neither (`ValueError: prompt_wav_path and prompt_text must both be provided
  or both be None`). Cross-lingual jobs deliberately clear `speaker_transcripts`, so
  under `tts_speed="quality"` tier 1 burned a guaranteed failure on every segment.
- **TTS failures now name their cause.** `"All TTS synthesis failed - check model/GPU"`
  pointed at the wrong subsystem for what was a bad generation parameter; the first
  per-segment error is now captured and surfaced in the raised error and in the stage
  metrics. The worker's stderr log path (where every real VoxCPM traceback lands) is
  logged when the worker starts.
- **LM Studio translation silently failed on every segment.** `translate_text()`
  used `LM_STUDIO_URL` as a complete endpoint URL, but the configured value ends
  in `/v1` (as our own `.env.example` suggests), so every request went to
  `POST /v1` — which LM Studio answers with *"Unexpected endpoint or method.
  (POST /v1). Returning 200 anyway"*. The 200 has no `choices` key, the
  `KeyError` was swallowed, and the source text was returned as the
  "translation", so the pipeline reported segments as untranslated with no
  visible error. All LM Studio URLs are now normalized to a host root
  (`http://host:port`, with or without `/v1`, `/api/v1` or a full endpoint
  path) and every endpoint is derived from it.
- **Switched to LM Studio's native `POST /api/v1/chat` API** (`input` /
  `system_prompt` / `max_output_tokens`), falling back to the OpenAI-compatible
  `/v1/chat/completions` on older LM Studio builds. The native response
  separates `reasoning` items from `message` items, which is what makes
  thinking models usable — the translation is taken from the message and the
  chain-of-thought is discarded instead of ending up in the subtitle.
- **Thinking models no longer exhaust their token budget before answering.**
  Measured on `qwen/qwen3.6-27b`: one 8-word sentence produced 1606 reasoning
  tokens before 14 tokens of answer, so the old hard-coded 500-token cap
  truncated the response mid-thought and returned nothing at all. The default
  budget is now 4096 (`LM_STUDIO_MAX_OUTPUT_TOKENS`), and `reasoning` is
  requested as `off` by default (`LM_STUDIO_REASONING`), with an automatic
  retry without the field for models that reject it.
- **Qwen thinking models are switched out of reasoning mode in-prompt.**
  `qwen3.6-27b` accepts `reasoning: "off"` and then reasons anyway; appending
  Qwen's `/no_think` token works. Same 4 segments, same model: **171.7s → 8.6s**
  (42.9s → 2.1s per segment). Applied only to Qwen thinking models; a model
  that ignores the reasoning setting now logs one warning instead of being
  quietly slow.
- **Translation errors propagate instead of being swallowed**, so
  `translate_segments()`'s 3-attempt retry actually retries. A bad model name
  now reports itself rather than silently downgrading the endpoint.
- **`.env` is loaded before the pipeline imports.** Those modules read
  `LM_STUDIO_URL`, `LM_STUDIO_MODEL`, timeouts and `HF_TOKEN` into module-level
  constants at import time; the fallback loader ran *after* them, so on any
  machine without `python-dotenv` the entire `.env` was ignored.
- `LM_STUDIO_TIMEOUT` and `LM_STUDIO_MAX_CONCURRENT` are now actually honoured
  (they were documented in `.env` but read by nothing). LM Studio serves one
  model instance at a time, so concurrent requests only queued server-side
  while each request's timeout ran.

### Added
- **Per-stage retry with artifacts.** The pipeline now runs as eight checkpointed
  stages (`download`, `extract`, `transcribe`, `diarize`, `translate`, `tts`,
  `assemble`, `merge`). Each snapshots its full context, so any stage can be
  re-run from the previous stage's checkpoint without recomputing earlier work:
  `POST /api/dub/{id}/retry_stage/{stage}` with a JSON `overrides` payload and
  an optional `stop_after`.
  - `skip_diarization` — recover from a pyannote/HF_TOKEN failure by dropping to
    the single-speaker fallback reference, keeping the existing transcription
  - `translate_failed_only` — re-translate only the segments that came back
    empty, with a different model, keeping the good ones (prior translations are
    discarded if the source text changed since)
  - `tts_keep_existing` — synthesize only the segments missing audio; lines whose
    translation changed are re-rendered rather than kept
- **Stage observability.** New `pipeline/metrics.py` times every stage and samples
  CPU / RAM / GPU utilization on a background thread while it runs, logging a
  `[perf]` line per stage and persisting `outputs/<job>/metrics.json`
  (latest run per stage + capped attempt history).
  - `GET /api/dub/{id}/stages` — per-stage state, artifacts, timings, resources
  - `GET /api/dub/{id}/metrics` — raw per-attempt history
  - `GET /api/system` → `resources` — live CPU/RAM/GPU snapshot
  - GPU probe chain: `pynvml` → `torch.cuda` → `nvidia-smi` → `torch.mps`;
    all optional, stages are still timed without them
- **Pipeline stages UI panel** in the Processing, Result and History views —
  per-stage status, duration breakdown bar, artifact links, CPU/GPU peaks, and a
  Retry control rendering the settings each stage accepts. Stages whose output
  predates a later re-run of an earlier stage are flagged `stale`.
- New `paused` job status for runs stopped early via `stop_after`
- `psutil` dependency (optional at runtime; without it stages are timed but not
  resource-sampled)
- Stitched multilingual showcase reel rendering with per-language `· LL ·` badges
- Resume-from-checkpoint for jobs that errored mid-pipeline
- `gochidubb_rebuild_showcase` (MCP) / `showcase-rebuild` (CLI) — re-stitch without re-dubbing
- `gochidubb_list_models` — query installed Ollama translation models
- `examples/` directory with ready-to-run dub, showcase, and agent scripts
- **Server process manager** (`tools/gochidubb_serverctl.py`) — start, stop, restart,
  status, and logs commands with PID file tracking and orphan detection.
  Prevents lingering server processes after agentic code changes.
- **PID file** (`server.py` writes `.gochidubb.pid` on startup, cleans up on shutdown)
- **Signal handling** (`SIGTERM`/`SIGINT`) for graceful shutdown from external kill commands
- **`--reload` flag** support for development auto-reload
- **macOS launchd integration** (`install-launchd` command) for auto-start on login
- Updated `start.sh` to delegate to the server manager

### Fixed
- **yt-dlp 403 errors now fall back to the `web_embedded` player client.**
  YouTube serves 403 to the default client for videos that download fine
  through `--extractor-args "youtube:player_client=web_embedded"`; both the
  download and the metadata probe now retry that way when — and only when —
  they see a 403. The download keeps its preferred 1080p format on the retry
  rather than degrading, since a 403 is an access problem, not a format one.
- **Multi-language runs now capture source metadata at all.** The fan-out
  never probed the URL, so its jobs carried no title or description and had
  nothing to translate.
- **Voice consistency in cross-lingual cloning** — QA retries were mutating the seed
  per retry attempt in cloning mode, producing audibly different timbres for
  segments that failed-then-retried. Cloning mode now sets `MAX_QA_RETRIES = 0`
  and falls through to the next tier with the original `voice_seed` intact.
- CUDA non-determinism — `torch.backends.cudnn.deterministic=True` plus
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` for reproducible diffusion sampling

## [0.1.0] — initial public release

### Added
- **Video metadata is translated and copyable.** A new **Video metadata**
  panel on a finished result shows the source title, full description and
  chapter marks alongside their translation into that job's target language,
  each with a copy button — plus a one-click "description + chapters" block
  in the format YouTube parses back into real chapters. Chapters and the full
  description are new: descriptions were previously truncated at 2000 chars
  and only the first 500 were ever translated. Served on demand from
  `GET /api/dub/{id}/metadata` so the 2-second job poll stays small.
- One-click installers for Windows (`install.bat`) and Linux/macOS (`install.sh`)
- FastAPI server + React UI
- yt-dlp → faster-whisper → pyannote → Ollama → VoxCPM2 → ffmpeg pipeline
- 28 target languages
- Multi-speaker diarization (pyannote, optional)
- Background music preservation (audio-separator, optional)
- Persistent job history
- MCP server (`tools/gochidubb_mcp.py`) — Claude Code / agent integration
- CLI (`tools/gochidubb_cli.py`) — scriptable from any shell
- Claude Code skill (`.claude/skills/gochidubb/SKILL.md`)
- Whisper-roundtrip QA on synthesized segments with seed-mutation retries
- Tiered TTS fallback: VoxCPM2 cloning → VoxCPM2 reference → voice design → edge-tts
