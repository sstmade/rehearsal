# Rehearsal web: design research and spec (ui2)

Direction (owner-approved, unchanged): **cinematic dark, Apple product-launch style.** Near-black, one modern
sans, huge type, few words, full-bleed footage as the hero, visuals dominate. Rejected: "AI-looking" UI
(purple pills, monospace chips, jargon) and warm-paper serif editorial.

Method: 15 reference pages were loaded in headless Chromium at 1440×900 and screenshotted at three scroll
depths; computed styles of the first `h1`/`h2`/`p` were read from the live DOM, and Apple's built
stylesheets were parsed for their type tokens and easing curves. Numbers below are measured, not
remembered. No proprietary fonts, assets or copy were taken.

## What the references actually do

| reference | measured / observed | pattern worth stealing |
|---|---|---|
| Apple MacBook Pro / iPhone (built CSS) | `page-header-title` 80/64/48 px, line-height 1.05, tracking −0.015em, weight 600; section headers 48/40/32; body 17 px / 1.47 / −0.022em; captions 12 px; dark surfaces `#18181A`, `#272729`, text `#f5f5f7`, secondary `#86868b`; easing `cubic-bezier(0,0,.5,1)` (most used), durations 100–400 ms | strict 3-step responsive type scale; body never above 17–21 px; big-value + small-caption stat tiles ("24 hours / battery life"); numbered footnote markers for every claim |
| Apple Vision Pro | hero = one full-bleed image, product name + one line, two pill CTAs bottom-right; then centred one-sentence statements, one per screen | **one idea per viewport**; statements revealed while scrolling; video frames edge to edge |
| Linear (home) | Inter Variable, h1 64 px weight 510 tracking −0.022em (−1.41 px), line-height 1.0; h2 48 px with the **second clause in grey** `rgb(138,143,152)`; background `#08090a` | **two-tone headline**: claim in white, explanation continues in grey in the same sentence; weight ~500–600, never bold-black |
| 1X (NEO) | ABC Diatype 44 px, tracking −0.03em; "NEO **Home Robot**" two-tone | same two-tone device on a robotics brand |
| Figure AI | full-bleed grayscale video, tiny caps headline bottom-left, outlined pill "Full video ↗" bottom-right | the footage *is* the page; text sits in a corner on a gradient scrim |
| Stripe Sessions | söhne 100 px **weight 250**, tracking −0.04em, line-height .85 | very large numerals/headlines look premium when **light**, not heavy |
| Rabbit OS3 | black + one orange `#ff5c00` accent, 128 px weight 200 display; expo easing `cubic-bezier(.16,1,.3,1)` | a single hot accent on black, used sparingly, validates our orange |
| Runway Gen-4 | centred headline over full-bleed video, then a mosaic of clips with captions; easings `(.4,0,.2,1)` | clip grids with a one-line caption under each |
| Vercel Ship | black, hairline 1 px grid borders; a **chapter list beside the player** ("Session highlights") | selectable list + one large player instead of N small autoplaying videos |
| Vercel (CSS) | Geist; fluid type `clamp(.875rem, … , 1.25rem)`; `--ease-out: cubic-bezier(0,0,.2,1)` | fluid clamp() type instead of breakpoints |
| π0 (Physical Intelligence) | clip carousel, each clip captioned with the task and a speed label ("2× speed"); failures shown next to successes; limitations section | label every clip with *what* and *how fast*; show failures |
| DROID project page | stats inline (76k trajectories, 564 scenes); credits + BibTeX at the end | credit line next to the footage, citation at the end |
| Mobile ALOHA | autonomous vs teleop labels on clips; "funny robot failures" section | honest failure reel builds trust |
| RT-2 | hero video of one instruction ("pick up the extinct animal") with the prompt as caption | put the **instruction in quotes** on the visual |
| MECHA (Awwwards SOTD 31 Oct 2026) | 60 px statements, tiny caps stat labels ("INCREASE IN MATERIAL RECOVERY RATE") under big numbers | stat = huge numeral + 2-line caps-free caption |
| Arc (farewell) | 76 px serif on black, single column letter | (rejected direction: serif/editorial) kept only as a counter-example |

## The 8 patterns adopted (and where)

1. **Footage is the hero** (Figure, Apple VP, Runway). The hero is the DROID real robot and its Rehearsal twin,
   full-bleed, with a draggable wipe between them. They come from the same calibrated camera, so the arms
   line up across the seam. The credit line sits on the footage.
2. **Two-tone headlines** (Linear, 1X). Each act states its claim in white and finishes the sentence in grey:
   "Physics can certify *the wrong goal.*"
3. **One idea per viewport, chaptered** (Apple VP). Four numbered acts (Intent → Geometry → Reality → Try it),
   each opening with a full-height title and one visual.
4. **Light, enormous numerals with a two-line caption** (Stripe, Apple stat tiles, MECHA). The proof numbers
   are set in Inter at weight 300 and up to 160 px, with an orange accent only on the number that matters.
5. **Instruction in quotes on the visual** (RT-2). GoalBench examples show the user's words, what the model
   compiled, what physics said and what the user meant, as a verdict strip on the real photo.
6. **Chapter list beside one big player** (Vercel Ship). The six DROID episodes are a selectable list next to
   one wipe player, so only one video decodes at a time.
7. **Show the failure** (π0, Mobile ALOHA). The cup that real fingers lifted by wedging, which the twin got
   wrong, has its own callout.
8. **Footnoted numbers** (Apple). Every number carries a source path in an HTML comment and a visible
   sources line per act.

## Spec (tokens in `web/app.css :root`)

**Colour.** `--bg #000` (Apple product pages use pure black under footage) · `--bg-2 #0a0a0b` · `--surface
#141416` · `--surface-2 #1d1d20` · `--surface-3 #2a2a2e` · `--text #f5f5f7` · `--text-2 #a1a1a6` (≈8:1 on
black) · `--dim #86868b` (5.8:1, used for the grey half of headlines and captions) (the same value doubles as `--muted` for captions; never on `--surface-3`, where it drops to 3.9:1) · `--accent #ff5a1f` (failures, the decisive number, the primary action) ·
`--good #30d158` · `--warn #ffb020`. Hairlines `rgba(255,255,255,.1)`.

**Type.** Inter (Google Fonts, variable, `opsz` 14–32, so large sizes get Inter Display's tighter
drawing automatically).
- display (hero) `clamp(48px, 7.4vw, 120px)` / 0.95 / −0.045em / 600
- h2 (act title) `clamp(40px, 6vw, 96px)` / 0.98 / −0.04em / 600, grey continuation in `--dim`
- h3 `clamp(24px, 2.2vw, 34px)` / 1.1 / −0.025em / 600
- stat numerals `clamp(72px, 11vw, 168px)` / 0.9 / −0.055em / **300**, `tabular-nums`
- lede `clamp(19px, 1.6vw, 24px)` / 1.4 / −0.015em / 400, `--text-2`
- body 17 px / 1.47 / −0.011em · caption 13 px / 1.4 · eyebrow 13 px / 600 sentence case (no letter-spaced caps
  chips: those read as "AI UI")

**Space.** Gutter `clamp(20px, 5vw, 80px)`; content max 1320 px; act padding `clamp(96px, 16vh, 200px)`;
media can break out to the full width. 8 px base unit.

**Radii.** Media 22 px (Apple tiles 18–28), cards 18 px, pills 999 px.

**Motion.** `--ease-out: cubic-bezier(0, 0, .5, 1)` (Apple) for UI; `--ease-expo: cubic-bezier(.16, 1, .3, 1)`
(Rabbit) for reveals; reveal = 24 px rise + fade, 900 ms, 80 ms stagger; hover 200 ms. Videos and the 3D
view pause when off-screen. `prefers-reduced-motion`: no autoplay, no wipe sweep, no reveals, no count-ups;
stills are shown instead. No stacked backdrop blurs (one blurred nav bar only) to keep scrolling at 60 fps.

## Decisions → inspiration

| decision | inspired by |
|---|---|
| DROID wipe hero instead of a synthetic 3D scene | Figure (footage-first), Apple VP (full-bleed), DROID page (real data credited) |
| "Words don't collide. *The real world does.*" two-tone | Linear, 1X |
| Act index under the hero (01 Intent … 04 Try it) | Apple "index" lists, Vercel Ship chapters |
| GoalBench verdict strip on the real photo | RT-2 instruction-as-caption |
| Big light numerals 40 → 60 / 60 | Stripe Sessions, Apple stat tiles |
| Episode list + one player, verdict per row | Vercel Ship session list, π0 success/fail labels |
| Honest-miss callout | Mobile ALOHA failures, π0 limitations |
| AI-generated test photos grouped last and labelled | DROID/π0 provenance honesty |
