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

This Space is the **public replay demo**: the Muse Spark answers for the bundled instructions were recorded
ahead of time and ship with the code, and every rehearsal you start here is simulated live in MuJoCo on this
machine. Photo upload and the private "My desk" scene are disabled here; run the app locally with your own
Meta Model API key to upload a photo.

Author: Shengtao Shen. Code: MIT. Bundled assets keep their own licences (see `THIRD_PARTY_NOTICES.md`).
