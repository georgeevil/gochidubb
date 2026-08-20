# Running GoChiDUBB in the cloud

Architecture and cost model for a hosted, multi-tenant GoChiDUBB on AWS, Azure or GCP,
under the assumption that **load stays minimal for the first months to years**.

Every number here is either measured from this repository (22 jobs with recorded
metrics in `outputs/*/metrics.json`) or checked against published 2026 rate cards.
Where a figure is an extrapolation, it says so. §9 lists what is still unproven.

**Status:** analysis only. No infrastructure exists, and none of the prerequisite code
work in §6 has been started.

---

## The three findings that matter

1. **At low load the always-on floor dominates, not the GPU.** Roughly $70–120/month
   buys you a product nobody is using. GPU is a rounding error until you have real
   volume.
2. **The pricing already coded in `app/billing.py` does not cover cost** in any
   scenario modelled here. Storage at $0.02/GB is below S3 list price before retention
   is even discussed, and the third volume tier ($0.050/min) is at or below the cost of
   serving it.
3. **The engineering to get there dwarfs the infrastructure bill.** Three to four
   months of work versus roughly $1 000/year of hosting. Infrastructure is not what
   makes this expensive.

---

## 1. The honest precondition

GoChiDUBB cannot be deployed as a multi-tenant SaaS in anything like its current shape.
This is not "needs tuning" — the load-bearing assumptions are single-machine and
single-user, and `docs/saas-redesign-plan.md` §9–§10 says so itself.

| Blocker | Evidence |
| --- | --- |
| **No authentication on any route** | Zero `Depends(...)`, zero `Authorization` reads in `server.py`. `CORSMiddleware(allow_origins=["*"])` at `server.py:1345`. `/outputs` is a `StaticFiles` mount (`server.py:1346`) — every dubbed video is world-readable by job id. |
| **API-key machinery wired to nothing** | `app/apikeys.py` `verify()` / `has_scope()` are written and unit-tested; no route calls them. `GOCHIDUBB_MODE=hosted` today changes two cosmetic things (`server.py:2859`, `server.py:3094`). |
| **No tenancy** | No `tenant_id` on jobs, outputs, artifacts or audit rows. `app/audit.py:39` hard-codes `actor="local"`. |
| **Job state is in RAM** | `jobs: dict` at `server.py:291`, loaded once at startup, 64 access sites. Two processes diverge. `uvicorn.run()` never passes `workers=`. |
| **Queue is in-process** | `asyncio.Queue` + one worker (`server.py:1230`), unbounded, no backpressure, nothing survives restart. |
| **State is a directory** | SQLite `gochidubb.db`, `outputs/`, `uploads/`, plus `apikeys.json`, `webhooks.json`, `audit.jsonl`, `config-user.json` — all under the checkout, all rewritten in place. |
| **No manifest builds a GPU worker** | `requirements.txt` has `demucs` (:64), `pyannote.audio` (:70) and `nvidia-ml-py` (:31) commented out, and torch only as an instruction to install separately (:78). Yet `extract` uses demucs and `diarize` uses pyannote. |

---

## 2. Recommended architecture

Split the monolith along the seam it already has — the job queue — and split again
between CPU and GPU stages.

```
                    ┌──────────────────────────────┐
   browser  ──────▶ │  CONTROL PLANE  (always on)  │
   CLI / MCP        │  FastAPI: UI, API, auth,     │
                    │  job submission, billing     │
                    └───────┬──────────────┬───────┘
                            │              │
                   managed queue      managed Postgres
                            │              │  (jobs, tenants, keys,
                            ▼              │   webhooks, audit)
       ┌────────────────────────────┐      │
       │  CPU WORKER (scale to 0)   │◀─────┘
       │  download, ffmpeg extract, │
       │  translate (hosted API),   │
       │  assemble, x264 merge      │
       └─────────┬──────────────────┘
                 │  hands off only for GPU stages
                 ▼
       ┌────────────────────────────┐
       │  GPU WORKER (scale to 0)   │
       │  whisper, pyannote,        │
       │  VoxCPM2, demucs           │
       │  L4 24 GB + local NVMe     │
       └─────────┬──────────────────┘
                 ▼
        Cloudflare R2  ──▶  signed URLs
        (durable artifacts; scratch stays on the worker's local disk)
```

### Why this shape

**Splitting CPU stages off the GPU is the single biggest cost lever, and it is easy to
miss.** On the 3.2-hour job in `outputs/89d93af7`, download + extract + assemble +
merge total **21 638 of 41 152 seconds — 53% of wall clock with
`gpu_mem_mb_peak: 0.0`**. Run on one GPU container, half the GPU bill buys ffmpeg and
yt-dlp. Two queues, two pools.

**One job per GPU container preserves an existing invariant.** The serial worker exists
because "multiple concurrent dubs would OOM a 12GB GPU" (`server.py:329-336`). A
queue-driven, one-job-per-container worker keeps that guarantee and turns the
constraint into the scaling unit.

**Moving translation to a hosted API frees the GPU.** The LLM is never unloaded under
LM Studio (`translator.py:1608` — `unload_ollama_model()` is a no-op), so
`server.py:2175`'s "free VRAM before TTS" step does nothing today. Off-boxing it is what
makes a 24 GB L4 comfortable.

**Local scratch, object storage for durable artifacts only.**
`artifact_store.copy_artifacts()` uses `os.link()` hard links and checkpoints are read
by absolute path under `OUTPUT_DIR/<job_id>/`. Both break silently on object storage.
Note the limit of this: a scale-to-zero worker that dies mid-job destroys its scratch,
which is precisely the case checkpoints exist for. Either checkpoints and their
referenced artifacts go to object storage, or **resume quietly becomes "start over" and
the customer's GPU bill pays for it**.

### Which cloud wins

**Compute on AWS; storage and delivery on Cloudflare R2.** Do not pick one vendor.

The intuition that GCP wins on serverless GPU does not survive the real rate card.
Cloud Run GPU **requires instance-based billing** and a **minimum 4 vCPU / 16 GiB**, and
the GPU charge sits *on top* of CPU and memory:

| | rate |
| --- | --- |
| Cloud Run L4 GPU | $0.672/hr — the number usually quoted |
| + 4 vCPU (mandatory) | $0.346/hr — billed for the whole instance lifetime |
| + 16 GiB (mandatory) | $0.144/hr |
| **Cloud Run all-in** | **$1.162/hr** |
| **AWS `g6.xlarge`** | **$0.805/hr** — same L4, 4 vCPU, 16 GB, **plus 250 GB NVMe** |

AWS is **31% cheaper per GPU-hour**, and the local NVMe matters more than the price: a
pipeline that writes three ~2 GB uncompressed WAVs needs real scratch disk.

Cloud Run has two further disqualifiers here. Its container filesystem is **RAM-backed
and counted against the memory limit**, below what a long job writes — so the biggest
jobs cannot run there without a mounted volume, and GCS FUSE breaks the `os.link()`
reuse path. And GCP egress is the most expensive of the three ($0.12/GB premium) for a
product whose deliverable is video. Cloud Run's real advantage — near-instant
scale-to-zero — is worth little when minutes of cold start are acceptable.

**Azure loses outright.** Its cheapest serverless GPU profile is a 16 GB T4 at
~$1.26/hr: more expensive than an L4, less VRAM, no bf16, likely *slower* on VoxCPM. Add
Log Analytics at $2.76/GB (5.5× the others) and a container-registry tier upgrade forced
by a ~9 GB image. Its one genuine win is the cheapest managed Postgres (~$12–15/mo).

**Cloudflare R2 for every artifact byte** — $0.015/GB-month and **$0 egress**. Not a
footnote: it turns the two structurally negative lines positive by itself (§4).

AWS's costs are the boring ones and both are avoidable. A NAT Gateway ($32/mo +
$0.045/GB) and an ALB (~$30/mo) are only mandatory if you architect for them; public
subnets with security groups plus App Runner or a Lambda Function URL for the control
plane removes most of it.

### Provider mapping

| Role | AWS (recommended) | GCP | Azure |
| --- | --- | --- | --- |
| Control plane | App Runner / ECS Fargate | Cloud Run | Container Apps |
| Queue | **SQS** | Pub/Sub — *not* Cloud Tasks | Service Bus |
| Database | RDS Postgres `db.t4g.micro` | Cloud SQL | Flexible Server B1ms |
| GPU worker | ECS Managed Instances, scale-to-zero, `g6.xlarge` | Cloud Run GPU (L4) | Container Apps serverless GPU |
| Artifacts | **Cloudflare R2** | R2 | R2 |

**Queue choice is not free.** The longest job in this repo ran **16.2 hours**. SQS caps
visibility timeout at 12 hours, so a long job *will* be redelivered and run twice
without an idempotency key. **Cloud Tasks caps HTTP dispatch at 30 minutes and is simply
the wrong primitive** — use Pub/Sub on GCP.

---

## 3. Cost model

### Fixed floor (idle — zero jobs)

| Item | Monthly |
| --- | --- |
| Control plane, one small always-on container | $15–35 |
| Managed Postgres, smallest burstable tier | $12–35 |
| Object storage (R2), first ~100 GB | $2 |
| Queue, secrets, registry, monitoring | $10–25 |
| AWS only, if architected naively: NAT + ALB | +$60 |
| **Floor** | **≈ $70–120/mo · $850–1 450/yr** |

### Variable cost per job

Measured, from `outputs/*/metrics.json`:

- End-to-end wall clock is **2.05× realtime minimum, median 3.7×** on an M1 Max. Not one
  full-pipeline job ran under 2×.
- Segment density is consistent at **~10 segments per minute** of source.
- `sec_per_segment` has a **median of 7.1 s** across all jobs.
- Therefore **TTS alone is ~1.2× realtime** — and TTS is the one stage that does *not*
  collapse when you move to a GPU.

The two stages that *do* collapse — transcribe and diarize, ~52% of wall clock on a
representative job — are the two that are accidentally CPU-bound today
(`pipeline/transcriber.py:80` defaults `device="cpu"` and `server.py:1846` never passes
one). Fixing that is a large win but cannot take the total below what TTS costs.

Two costs easily omitted:

- **Per-job model loads, ~90–110 s warm.** VoxCPM is 40–60 s by the code's own comment
  (`server.py:1268`), and `pipeline/transcriber.py:120` constructs `WhisperModel(...)`
  **inside** `transcribe()` — a fresh 2.9 GB load on every call, no caching.
- **GPU-idle time billed at GPU rates** — the 53% figure above.

**Plan at 1.5× realtime.** A 10-minute video, one language:

| | rate/hr | billed | **per job** |
| --- | --- | --- | --- |
| AWS `g6.xlarge` | $0.805 | ~32 min incl. scale-in tail | **$0.43** |
| GCP Cloud Run L4 | $1.162 | ~19 min | **$0.36** |
| Azure ACA T4 | $1.26 | ~28 min | **$0.58** |

Plus ~$0.02–0.03 translation API and ~$0.02 egress → **$0.40–0.63 per job** against
$0.80 of revenue: **25–50% gross margin**. At the third pricing tier ($0.050/min → $0.50)
it is roughly break-even, and negative on Azure.

**Translation model choice is load-bearing.** ~100 segments at roughly 600 in / 200 out
tokens is 60k/20k per job — about $0.02 on a cheap fast model, but **~$0.48 on a
frontier model, more than the GPU**. A 3-hour job is 1 815 segments; the model choice
alone can invert margins.

**Failed jobs are unbillable GPU time.** `app/billing.py` excludes
`{"error","cancelled","failed"}`. On this machine 31 of 53 jobs are in `error` — a dev
box, so not a production rate, but the line is real: one job burned 1 530 s in `extract`
with no output, another 3 045 s across two failed `merge` attempts. Budget 15–25%.

### Scenarios

8-minute average source, 1.5 languages, artifacts on R2, retention enabled.

| | Dormant | Early traction | Real usage |
| --- | --- | --- | --- |
| Jobs / month | 20 | 200 | 2 000 |
| Billable minutes | 240 | 2 400 | 24 000 |
| GPU @1.5× | ~$8 | ~$77 | ~$700 |
| Cold-start / scale-in tail | ~$1 | ~$8 | ~$40 |
| Failed jobs (unbillable) | ~$2 | ~$17 | ~$150 |
| Translation API | ~$1 | ~$6 | ~$60 |
| Egress (R2) | $0 | $0 | $0 |
| Storage, month 1 → month 12 | $1 → $6 | $5 → $40 | $40 → $200 |
| Fixed floor | $70–120 | $80–130 | $150–200 |
| **Month-1 cost** | **$83–133** | **$193–243** | **$1 140–1 250** |
| **Revenue at coded tiers** | **$19** | **$164** | **$1 238** |
| **Month-1 margin** | −$64 to −$114 | −$29 to −$79 | −$12 to +$98 |
| **Annual cost** | **$1 000–1 700** | **$2 400–3 400** | **$14 500–17 000** |

**The coded pricing does not cover cost in any of these scenarios.** That is a pricing
problem, not an infrastructure one. The levers, largest first: split CPU stages off the
GPU, reprice the tier ladder, batch VoxCPM, put storage on R2.

**A billing-entity ambiguity worth up to 55%.** `app/billing.py` tiers are cumulative,
but nothing states whether they are cumulative *per account* or *in aggregate*. At
24 000 minutes the aggregate reading yields $1 238; per-account across many small
accounts yields up to $1 920. Define this before quoting anyone.

---

## 4. Storage

- Measured average **~1.2 GB per job** (36 GB across 31 jobs).
- One 5-hour job consumed **16 GB**, including **three ~2 GB uncompressed WAVs**
  (`audio_hq`, `vocals`, `background`) that are pure intermediates.
- **Nothing deletes them.** `POST /api/storage/cleanup` has an `intermediate` mode
  claiming ~90% savings, but there is no cron, no TTL, no sweep.
- `uploads/` (3.9 GB here) is cleaned by *nothing at all*.

The meter is monthly; the data is forever. At 2 000 jobs/month with no retention you add
~1.1 TB every month — about 13 TB by month twelve. With automatic intermediate cleanup
the same workload adds ~240 GB/month, because only the deliverable survives (~14 MB per
minute of source instead of 50–90).

### $0.02/GB is below cost before retention even enters

| | list price | margin at the coded $0.02 |
| --- | --- | --- |
| S3 Standard | $0.023/GB | **−15%** |
| GCS Standard | $0.020/GB | 0% |
| Azure Hot LRS | $0.018/GB | +10% |
| **Cloudflare R2** | **$0.015/GB** | **+25%, and $0 egress** |

Perpetual storage sold once at $0.02/GB is a subscription liability priced as a one-off.
Reprice to $0.04–0.05/GB, or redefine "100 GB included" as **rolling 30-day retention**
and put the bytes on R2.

### Two things that make retention harder than a cron job

**The per-segment layout defeats lifecycle tiering.** One job holds **2 885 files in
`tts_segments/`** (2 999 in the job directory; **6 802 across all of `outputs/`**). S3
will not transition objects under 128 KB to IA at all, Glacier bills 32–40 KB of
metadata per object, and the transition itself costs per thousand objects. Tar the
directory into one object first.

**Three ~2 GB uncompressed WAVs hold the same content.** FLAC is lossless and roughly
halves that — **16 GB → ~9 GB for a one-line ffmpeg change** — which also halves upload
time and PUT volume. Cheaper than any retention policy.

### The polling cost that would exceed the GPU bill

`_job_checkpoint_info()` (`server.py:844`) states in its own docstring: *"Purely
filesystem inspection — cheap enough to do on every /api/jobs poll."* True on local SSD.
It does one directory check plus one `.exists()` per pipeline stage — **~10 stat calls
per job, per poll** — called from `list_jobs` (`server.py:8152`) for every job, against
a continuously polling UI.

Move `outputs/` to object storage without changing this and every stat becomes a billed
HEAD request. At a few hundred jobs and one open tab that is hundreds of dollars a
month. **It must become a stored column before object storage exists.**

### Cleanup bugs to fix on the way

- `_KEEP_ON_INTERMEDIATE_CLEAN` (`server.py:7969`) names `translated.srt`; the pipeline
  writes `subtitles.srt` (all 22 jobs on disk have the latter, none the former). Data is
  not lost — the keep-list retains the checkpoints the SRT regenerates from, and the
  burn-in endpoint does that (`server.py:3962-3970`) — but the `srt_url` the UI links
  (`server.py:2265`) **404s after a clean**.
- It keeps `checkpoint_tts_done.json` while deleting `tts_segments/`, so "resume from
  TTS" on a cleaned job is already broken.
- It iterates the in-RAM `jobs` dict, so orphaned output directories are never
  reclaimed.

---

## 5. The risk that could invalidate the product, not just the budget

**YouTube ingestion from datacenter IPs.** The primary input path is a YouTube URL, and
`pipeline/downloader.py` has **no proxy support at all** — only cookie options, and
`--cookies-from-browser` is meaningless in a container. For a multi-tenant SaaS that
means shipping your own account's cookies (against YouTube's terms, one ban from total
outage) or asking every customer for theirs.

Cloud egress IPs are exactly what bot detection targets. This is a "the main feature
errors for everyone simultaneously" problem, and no architecture fixes it.

Worse, the economics of the workaround do not close: residential proxy is $2.50–8/GB,
and a 3.3 GB source costs **$8–26 of proxy against $15.28 of revenue**. Even a 200 MB
10-minute video is $0.50–1.60 against $0.80 — **negative margin on every YouTube job**.

Options, in order of preference:

1. **Make file upload the primary hosted input**, URL ingestion best-effort. The upload
   path already exists and has no such exposure.
2. **Client-side fetch** — the browser or a local helper downloads and uploads, so the
   request comes from a residential IP. Preserves the UX, adds a component.
3. **Residential proxy pool** — works, but see the arithmetic above.

Decide this before writing infrastructure code: option 1 changes the product, option 2
changes the architecture.

---

## 6. Prerequisite engineering

Tracked in the [GoChiDUBB Linear project](https://linear.app/siteloom/project/gochidubb-6371b03bc9e4/)
as five milestones and 21 issues, each with acceptance criteria and dependencies.

| Milestone | Theme | Issues |
| --- | --- | --- |
| **M0 · Calibrate** | Measure before spending | CLD-190 … CLD-193 |
| **M1 · Deployable** | Runs in a container at all | CLD-194 … CLD-196 |
| **M2 · Safe to expose** | Auth, tenancy, retention, quotas | CLD-197 … CLD-202 |
| **M3 · Distributed** | Postgres, object storage, queue | CLD-203 … CLD-207 |
| **M4 · Margin** | CPU/GPU split, batching, repricing | CLD-208 … CLD-210 |

### Working this in parallel

Nine issues have **no blockers** and can start simultaneously:

- CLD-190 transcribe device · CLD-192 FLAC intermediates · CLD-193 ingest decision
- CLD-194 Dockerfile · CLD-195 data root · CLD-196 translator auth
- CLD-197 API-key auth · CLD-200 retention · CLD-202 CORS + SSRF

Three sequencing rules matter more than the milestone boundaries:

1. **CLD-190 before CLD-191.** Benchmarking with transcription accidentally on CPU
   measures the wrong thing.
2. **CLD-203 before CLD-205.** Moving artifacts to object storage while `/api/jobs`
   still stats every checkpoint ships a four-figure monthly bill.
3. **CLD-197 before CLD-198 and CLD-199.** Signed URLs and tenant scoping both need
   something to authorise against.

**File-collision warning for concurrent agents:** `server.py` is 8 300+ lines and most
issues touch it. CLD-199 (tenant scoping) rewrites 64 `jobs[...]` access sites and
should be sequenced alone rather than run beside other `server.py` work. CLD-192,
CLD-196 and CLD-209 touch only `pipeline/`, and are the safest to parallelise.

### Effort

| Phase | Estimate |
| --- | --- |
| M0 | 2–4 days |
| M1 | 1–2 weeks |
| M2 | 3–5 weeks |
| M3 | 3–5 weeks |
| M4 + Stripe integration | 3–5 weeks |

**Roughly three to four months of focused work** — more than the first three years of
infrastructure at low load.

---

## 7. A cheaper way to sequence this

Load will be minimal for months to years, which argues for not building the multi-tenant
version first.

**Phase 0 + 1 alone — roughly a month — give a real, remotely-accessible, authenticated
single-tenant deployment** at the $70–120/month floor, serving you and a handful of
invited users over API keys the code already supports once `verify()` is called. Phase 2
is deferred until demand justifies it, and it changes the storage and identity layers
rather than replacing Phase 0 and 1's work.

For comparison, the naive lift-and-shift — one always-on GPU VM running the app as it
stands — is roughly **$590–700/month, $7 000–8 400/year**, whether or not a single job
runs. The scale-to-zero split pays for itself at any load below ~60% GPU utilisation,
which at these volumes is every month for years.

---

## 8. Verification

Before committing to a provider, prove the two numbers everything rests on:

1. **VoxCPM `sec_per_segment` on an L4.** Rent one `g6.xlarge` for an hour, run a
   10-minute job with the transcribe device fixed, read `outputs/<job>/metrics.json` —
   `pipeline/metrics.py` already records per-stage `duration_sec` and GPU memory. If it
   does not beat ~4 s/segment, L4 is the wrong SKU (it has *less* memory bandwidth than
   an M1 Max: 300 vs 400 GB/s) and A10G or L40S changes the whole model.
2. **Cold start with a ~9 GB compressed image.** Time first-request-to-first-stage on a
   scaled-to-zero worker, including the scale-in cooldown tail.

Then re-run §3 with real numbers.

---

## 9. What is still unproven

- **VoxCPM speed on an L4 is an extrapolation.** It is the largest GPU stage and drives
  the entire variable cost. The 1.5× realtime planning figure assumes CUDA kernels beat
  MPS by enough to overcome lower memory bandwidth — plausible, unmeasured.
- **Per-hour prices move.** Anchors were checked in August 2026; re-verify in each
  provider's calculator.
- **Egress volume depends on product behaviour nobody has measured** — re-download
  frequency, whether previews stream. R2 makes this moot, which is part of its appeal.
- **The 31-of-53 failure rate is from a development machine**, not production. The
  unbillable-GPU line is real; its size is a guess.
