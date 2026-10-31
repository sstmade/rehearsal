# Rehearsal web — UX audit (before the redesign)

Screenshots: `web/ux_shots/before_*` (1440 and 390 px, kitchen_01 and real_ycbv59, every chapter, demo mode).
Owner's verdict that triggered this: "looks way too AI-generated, and a lot of it is content other people can't understand."

## (a) Jargon a non-roboticist will not get

Seen on the page in the first 60 seconds, in the order encountered:

| Where | Term | What a judge reads |
|---|---|---|
| hero eyebrow | "Sim-only rehearsal · hackathon build · Muse Spark · $0.0021" | a cost in a pill, before the product is explained |
| hero lede | "MuJoCo physics twin", "Franka Panda arm", "logs every contact" | three proper nouns, none explained |
| hero stats | "Robustness 40/40 · 37/40 before looking again", "Reachable 6/7 · 1 outside workspace", "Wall time 131 s · agent: muse-spark-1.3-contributor (agent loop)" | numbers with no meaning; the agent id is an internal string |
| hero models | "Muse Spark · goals · plan · diagnosis · tool calls", "hand-authored scene · no perception run", "MuJoCo · ground truth" | "who did what" chip soup |
| chapter nav | "Photo → twin", "Instruction → plan", "Stress test", "Look again" | "twin" is never defined |
| 01 | "homography", "segments", "σ 1.2 cm", "low-confidence", "Contributor / Standard", "public data", "IK ✓ / out", "μ", object ids `cup_1`, raw positions "0.62, 0.18" | the whole inventory table is a data dump |
| 01 map | "reach ≈ 0.86 m", dashed σ rings | unexplained |
| 02 | "compiles … into checkable predicates", `on(cup_1, coaster_1)`, "implicit", "Universes" input, "Look rounds" input, "carry at 0.4 m", "via (0.50, 0.28)", "MuJoCo · IK reachability 6/7" | predicate syntax in monospace |
| 03 | "Agent loop · tool calls", "5 tool calls $0.0021 15153 prompt / 11204 cached / 8188 completion tokens", "feedback: full_log", `simulate(cup_1→coaster_1@0.4 via; …)`, "SUCCESS / FAILED / ACCEPTED", "protective stop", "gripper body (hand) pressed on kettle_1 with 82 N at [0.64, 0.06, 0.25] (tool tip at [0.715, 0.061, 0.211])", "tcp_z", "IK failed", "Sim log", "execution error" | the single most important moment of the story (the failure) is shown as a log line with raw coordinates |
| 04 | "Parallel universes", "friction ×0.6–1.4, mass ×0.7–1.3", "Perception σ 1.2 cm per object, per universe", "Most sensitive cup_1 fails avg 3.1 cm vs 1.2 cm", "Where it breaks" scatter (perception error vs friction scale), "Failed universes" table with seeds | nothing says "we tried it 40 times with small random changes" |
| 05 | "Look again 再看一眼", "Active perception", "σ=1.2 cm → σ=0.4 cm on the same seeds", "paired comparison", "selection seeds 0–39 (optimistic)", "held-out seeds 1000–1039", "SAM 3.1 re-localises", "σ → 0.4 cm, modelled" | a Chinese subtitle and two kinds of seeds on a product page |
| 06 | "Verified in simulation / Estimated under uncertainty / Assumed, not verified", "top-down IK to each object's top face", "Goal predicates: inside, on, stacked, upright, stable, undisturbed", "suction weld" | honest, but it reads like a methods section |
| gallery | scene ids `real_ycbv59`, `kitchen_01` as titles; "physics caught it", "hard scene", "report" badges; "the gripper body comes down on the kettle → protective stop → move the kettle first" | ids instead of names |
| 3D stage | "live attempt 3 ✓", "Universes" button, "Meshes (SAM 3D)" button, "amber = low-confidence perception · red = out of reach", "held-out seeds, re-observed objects" | chips in monospace |
| demo mode | "phone photo → MuJoCo twin → Muse Spark plans → physics judges → model self-corrects → 40 universes", "02 · INSTRUCTION → PLAN", `on(cup_1, coaster_1)` goal cards, "protective stop · t = 6.038 s", "σ 1.2 → 0.4 cm" | same vocabulary at 100 px |

## (b) "Generic AI aesthetic" tells

- Giant two-line slogan in a display face with the second line in accent blue, followed by exactly four stat tiles — the default "AI landing page" layout.
- Pills and badges everywhere: 21 distinct `.badge` variants (spark purple, sam teal, ok, bad, warn, privacy, offline, phys, mock, model…), most in monospace with a coloured dot. The hero alone has 7 pills.
- Purple (`--spark`) and teal (`--sam`) accents on top of the blue accent: three brand colours plus green/red/amber = six hues on one screen.
- Pulsing green "live" dot in the eyebrow; glowing red "hit" tick marks; `box-shadow` glow on the primary button; radial gradient wash behind the whole page; purple→transparent gradient on the diagnosis card.
- Monospace used as decoration: chapter numbers "01", nav numerals, scene picker, table numbers, chips on the 3D stage, time labels, gallery titles, every badge.
- Dark glassy HUD cards (backdrop-filter blur) over the 3D stage; demo mode cards with purple borders and blur; "mission control" tone announced in the CSS comment itself.
- Typewriter animation with blinking cursor on the diagnosis (twice: page and demo).
- Rounded-pill nav with "1 2 3 4 5 6" numerals; "sticky theater" two-column layout that never lets the reader see a photo and the 3D side by side.
- Emoji-ish glyphs: ✓ ✕ ◆ ● ▶ ↗ ↔ ⌀ as text.
- Chinese subtitles ("再看一眼") mixed into English headings.

## (c) Information overload: what a first-time visitor sees, in order

1440 px above the fold (kitchen_01): top bar with 6 chapter links, a budget meter ("Meta API $2.43 / $15"), a scene picker showing `kitchen_01`, Demo, theme. Left: eyebrow pill, slogan, a 60-word lede with 4 product names, two buttons, four stat tiles, two rows of attribution chips. Right: a 3D viewer already playing "attempt 3" with a timeline, three attempt pills and three toolbar buttons. Nothing above the fold shows a photo, a task sentence, or a failure. The 5-second read is: "some robotics simulator dashboard."

390 px: the 3D viewer takes 44 vh and is sticky, so every chapter has only half a screen; the first text is "Sim-only rehearsal for home robots · hackathon build" in monospace; the stat tiles wrap to 2×2; the inventory table scrolls sideways.

Order of information on the page: outcome stats → upload box → photo+map → inventory table → instruction form → goals/plan → tool-call trace → attempts → scatter plot → bar chart → closed-loop chain → three assumption cards → gallery. The failure (the one thing that makes the product interesting) is the 9th block, and the real photo (the strongest proof) is only visible after switching scenes.

## What the redesign does (see the page)

- One narrative in three acts (the confident wrong plan → what we found → the vision), then the scene story as a numbered photo-essay: photo → 3D copy → the task → first try fails (big) → what the model learned (one sentence) → fixed → tested 40 times → closer look → what was checked.
- Plain language by default; every technical thing kept, but moved under one "Technical details" disclosure per step.
- One accent colour, warm paper background, serif display face, no pills except pass/fail marks, no gradients/glow/blur, monospace only inside technical details.
- Human scene titles with one-line stakes in the gallery; scene ids only in technical details.
