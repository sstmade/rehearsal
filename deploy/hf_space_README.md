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

Models propose, physics decides. A phone photo of a table becomes a MuJoCo digital twin with a Franka
Panda arm; Meta's Muse Spark and SAM 3.1 propose the objects, goals and plan, physics executes it,
and the model reads the contact log to fix its own mistakes before any real robot moves.

This Space runs the **offline demo**: the bundled scenes were rehearsed at build time with a
deterministic rule-based stand-in for Muse Spark, so everything you see is real simulation output and
no Meta API key is used. Photo upload needs the Meta Model API and is meant to be run locally.

Source, setup and evaluation: {{TODO-USER: GitHub URL}}
