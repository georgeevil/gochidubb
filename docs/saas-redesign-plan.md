# SaaS Redesign — Design & Implementation Plan

Status: **plan for review** — no UI code has been changed yet.

Source of truth for the visual direction: the Claude Design project
[`SaaS Redesign Options.dc.html`](https://claude.ai/design/p/64a60fdc-6a9e-4868-a1ab-c87966f4696d?file=SaaS+Redesign+Options.dc.html)
(plus the `support.js` it imports). **That project could not be read from this
session** (the design MCP requires an interactive `/design-login` that isn't
available in remote sessions, and the share URL is auth-walled). Every section
below marked **[needs design file]** is a slot that gets filled in mechanically
once the file is available; everything else is grounded in the current codebase
and does not depend on it.

To unblock the design-specific half, any one of these works:

1. In Claude Design, use **"Send to Claude Code Web"** on the project so the
   files are seeded into the session workspace.
2. Commit `SaaS Redesign Options.dc.html` + `support.js` into the repo (e.g.
   `docs/design/`) on any branch.
3. Paste the file contents into the conversation.

---

## 1. Current state (what the redesign starts from)

`static/index.html` is a single 3,710-line file — but it is **not** a legacy
jQuery-style page. It is already a React 18 app:

- React + ReactDOM UMD and Babel standalone from unpkg, JSX compiled in the
  browser (`<script type="text/babel">`, lines 207–3709). three.js powers a
  decorative "sphere" hero. No build step, by design.
- CSS lives in one `<style>` block (lines 10–197): design tokens on `:root`
  (`--bg`/`--bg-1..3`, `--line`, `--ink-1..4`, oklch lime `--accent`,
  `--warn`, `--err`; fonts Instrument Serif / Geist / JetBrains Mono; layout
  vars `--pad`, `--gutter`, `--rail-w`), a handful of utility classes
  (`.btn`, `.btn-primary`, `.btn-ghost`, `.chip`, `.serif`, `.mono`, `.caps`),
  animations, and a film-grain overlay. Most other styling is inline
  `style={{...}}` referencing those CSS variables.
- App shell: `LeftRail` (nav, collapses to a drawer under a narrow
  breakpoint) + `TopBar` + `SetupBanner`, then one of ten views switched in
  `App()` (index.html:3492): **home, processing, review, result, discover,
  history, batch, voices, glossary, system** (with setup/logs tabs).
- ~45 function components in the same file; state is plain `useState` in
  `App`, passed down as props. Data comes from polling: `/api/system` every
  5 s, `/api/jobs` + `/api/publish/pending` every 2 s while a job is active,
  10 s otherwise.
- Endpoints referenced by the UI: `/api/config`, `/api/diagnostics/run`,
  `/api/dub`, `/api/dub/batch`, `/api/dub/{id}/publish[...]`,
  `/api/dub/{id}/quality`, `/api/dub/{id}/retry_stage/{stage}`,
  `/api/dub/{id}/stages`, `/api/jobs`, `/api/lip_sync/status`,
  `/api/lm_studio/models`, `/api/logs`, `/api/publish/pending`,
  `/api/quick_test`, `/api/scout/dub`, `/api/scout/trending`, `/api/secrets`,
  `/api/secrets/status`, `/api/showcase`, `/api/system`, `/api/voices`.
- `static/beta.html` is a separate, standalone status page for the
  stage-reuse beta with its own (different, teal) palette — currently
  off-brand relative to index.html and to any redesign.

Constraints that shape the plan:

- **No build step, no new runtime dependencies** (CONTRIBUTING.md). The
  redesign must stay a static file served by FastAPI.
- Server API and job lifecycle are untouched — this is a frontend-only change.
- Nothing in CI validates the UI, so the plan includes manual/browser
  verification. Note the Python gates do not currently run either:
  `.github/workflows/lint.yml` triggers on `main` while the default branch is
  `master`, so only `claude-review` executes on a PR — and `ruff check .`
  reports 105 pre-existing findings on master, which would have to be cleared
  (or the ruleset pinned) before that workflow could be turned on. Out of
  scope here, but it means "CI is green" says nothing about this change.

## 2. Design approach

**Reskin, don't rewrite.** The component tree, state management, polling, and
API wiring in index.html are recent and sound. The redesign replaces the
*presentation layer*:

1. **Token swap.** All colors, fonts, radii, and spacing already route through
   `:root` custom properties. The chosen design option lands primarily as a
   new token block plus updated utility classes.
2. **Shell restyle.** `LeftRail`, `TopBar`, `NavItem`, `SetupBanner` are the
   components that carry most of the "product identity" — these get the
   heaviest markup/style changes.
3. **Surface components.** `.btn`/`.chip`/`Field`/`Select`/`Toggle`/
   `StatusBadge`/`SectionHeader`/`LongProgress` are the shared vocabulary;
   restyling them propagates through every view.
4. **Views keep their logic.** Each view's JSX is adjusted for layout/spacing
   only where the design demands it; handlers, effects, and data flow are not
   touched.
5. **beta.html** is restyled to the same token set (it is small — 221 lines).

Items the design file must supply — the fill-in checklist **[needs design
file]**:

- Which of the redesign *options* is the chosen direction (the canvas
  presents several).
- Color palette (light or dark base? accent hue?), typography stack, radius/
  elevation/spacing scale.
- App-shell layout: does the rail survive, or does the design move to a top
  nav / different chrome?
- Treatment of the three.js sphere hero (keep, restyle, or drop).
- Any new screens or rearranged information architecture beyond the current
  ten views.
- Whatever behavior `support.js` encodes (theme switching, shared helpers for
  the artboards, interaction specs).

## 3. Implementation plan

Phased so each step leaves the app working and reviewable:

**Phase 0 — capture the design source (blocked on access, see top).**
Check the two design files into `docs/design/` so the mapping from artboard to
code is reviewable in the PR.

**Phase 1 — extract the style layer (mechanical, no visual change).**
Move the `<style>` block out of index.html into `static/theme.css`, linked
with `<link rel="stylesheet">`. This gives the redesign a single file to
iterate on and cuts index.html by ~190 lines. Verify pixel-identical
rendering. (FastAPI already serves all of `static/` — no server change.)

**Phase 2 — apply the chosen option's tokens.** Rewrite the `:root` token
block and utility classes to the design's palette/type/spacing. Because views
style themselves via `var(--…)` inline, most of the app follows automatically.
Sweep for hard-coded values that bypass tokens (e.g. `#0a0a0d` appears
inline in `.btn-primary` text color and a few components) and route them
through tokens.

**Phase 3 — restyle the shell.** LeftRail/TopBar/SetupBanner markup updated
to the design's chrome; keep the narrow-breakpoint drawer behavior, the
awaiting-review badge, and the running-job indicator (these are functional,
not decorative). Restyle the shared controls (`btn`, `chip`, forms, badges,
progress).

**Phase 4 — per-view pass.** Walk the ten views against the artboards:
home (source input + quick test), processing (stage timeline), review,
result, discover/scout, history, batch, voices, glossary, system (setup +
logs). Adjust layout/spacing/hierarchy to match; leave logic intact.

**Phase 5 — beta.html + polish.** Re-token beta.html; check dark/light
consistency, focus states, reduced-motion, and the narrow layout on every
view; remove any dead CSS.

**Verification (each phase).** Serve with
`python tools/gochidubb_serverctl.py foreground --reload`, drive the UI with
Playwright (Chromium is preinstalled in remote sessions) against a backend
with no models installed — every view renders from polling data, so
screenshots of all ten views at desktop + narrow widths are enough to
review without running a dub. `ruff check .` stays green (no Python changes).

**Deliverable.** Commits per phase on `claude/saas-redesign-plan-abe8vf`,
screenshots in the PR description when a PR is requested.

## 4. Risks

- **In-browser Babel + one file** means a syntax slip breaks the whole app
  with only a console error. Mitigation: phase-sized commits, screenshot
  verification after each, and keeping Phase 1 strictly mechanical.
- **Inline styles everywhere** make a *structural* redesign (e.g. rail → top
  nav) more invasive than a token swap; if the chosen option changes the
  shell geometry, Phase 3 grows but the view logic still survives as-is.
- **CDN fonts/libs**: the current page already depends on unpkg + Google
  Fonts at load time. If the design introduces new fonts, they ride the same
  mechanism; going fully offline is possible (vendor the files into
  `static/vendor/`) but is a separate decision worth a maintainer call.
