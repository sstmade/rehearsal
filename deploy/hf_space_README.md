---
title: Rehearsal
emoji: 🦾
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Rehearse a home-robot task in a MuJoCo twin of your photo
---

# Rehearsal

**Models propose, physics decides.** A photo of a table becomes a MuJoCo digital twin with a Franka Panda arm.
Meta's Muse Spark and SAM 3.1 / SAM 3D see the objects and turn your instruction into checkable goals, a search
tries many candidate motions in physics, and only a plan that physics verified would ever be sent to a robot.

Open **`/app`**, pick one of the 50 AI-generated test tables, run its suggested instruction, and watch the
rehearsal: perception, the 3D twin (Physics view), the futures that were tried and rejected, and the verdict.

The sample tables replay their recorded Muse Spark answers and are simulated live in MuJoCo on this machine.
You can also **upload your own table photo**: uploads call the Meta models live, under a fixed total budget and
per-visitor limits (3 uploads per hour, 10 per day). The original photo is deleted right after it is downscaled, and
derived files are purged after an hour. When the budget is used up, uploads turn off and the demo shows the recorded
tables.

Author: Shengtao Shen. Code: MIT. Bundled assets keep their own licences (see `THIRD_PARTY_NOTICES.md`).
