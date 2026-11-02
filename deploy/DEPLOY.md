# Deploying the live demo (Hugging Face Space, Docker)

> **WARNING: the public demo is OFFLINE. It has no API key and must stay that way.**
> Do not add `MODEL_API_KEY` to the public Space. Anyone on the internet could then spend your Meta credits
> (no login; the budget cap is per container and resets on every restart). The image defaults to
> `REHEARSAL_OFFLINE=1` and `REHEARSAL_PUBLIC_DEMO=1` (uploads refused, bundled scenes read-only).

What judges get: the web app on port 7860 with the shipped scene reports (`out/`, the hybrid runs from your
local machine) already in the gallery. Nothing in this repo pushes or deploys by itself; these are the manual
steps, in order. Expected package: about 205 MB (`out/` is 145 MB of it), no network needed to build it.

## Steps

1. **Confirm the license and fill the TODOs** (see `submission/RELEASE_CHECK.md`): edit `LICENSE`
   (`TODO(user): name`) and commit.
2. **Create the GitHub repo and push** (public). Review `git log` first. Then put its URL into
   `deploy/hf_space_README.md` (replace `{{TODO-USER: GitHub URL}}`) and commit.
3. **Create the Space** at https://huggingface.co/new-space: owner/name `<you>/rehearsal`, SDK **Docker**,
   template **Blank**, hardware **CPU basic** (free), visibility **Public**. Add no secrets or variables.
4. **Create a write token** at https://huggingface.co/settings/tokens.
5. **Assemble the package** (from a checkout that has the shipped `out/` folder; it is gitignored, so use the
   main checkout or point `OUT_SRC` at it):
   ```bash
   cd rehearsal-mvp                          # the checkout that holds out/
   scripts/make_space.sh                     # -> out_space/ (gitignored, ~205 MB)
   # from another checkout: OUT_SRC=/path/to/rehearsal-mvp/out scripts/make_space.sh
   ls out_space/out | wc -l                  # expect 10 scene folders
   ```
   It contains `Dockerfile`, `.dockerignore`, `requirements.txt`, `rehearsal/`, `scenes/`, `web/` (with
   `web/media/droid/`), `photos/`, `bench/`, three runtime scripts, `THIRD_PARTY_NOTICES.md`, `out/`, and
   `README.md` = `deploy/hf_space_README.md`. Build-time precompute is switched off because `out/` ships.
   `datasets/` (HOPE is CC BY-NC-SA) is deliberately not included. `out_llm_v3/` (230 MB, LLM-only history)
   is only added if `rehearsal/server.py` has a route for it.
6. **Upload** (stores binaries via HF large-file storage; a plain `git push` would be rejected):
   ```bash
   uvx --from huggingface_hub hf auth login   # paste the write token
   uvx --from huggingface_hub hf upload <you>/rehearsal out_space . --repo-type space \
       --commit-message "Rehearsal demo"
   ```
7. **Watch the build**: Space page -> Logs -> Build. Expect apt + pip (1-2 min), Menagerie clone, then
   `gl_check: MUJOCO_GL=egl works` (or `osmesa`). With `out/` shipped there is no precompute step.
   Then open the **App** tab: the gallery shows the shipped scenes; try the landing page, a scene report
   and the DROID replays.
8. **Verify it is offline**: Settings -> Variables and secrets must show no `MODEL_API_KEY`. Container logs
   should print `rehearsal: offline mode`.
9. **Put the live URL back** into `README.md`, `submission/devpost.md`, `submission/judge_qa.md`, the video
   end card and the X post, commit and push.

To update later: commit, re-run `scripts/make_space.sh`, run the same `hf upload` (add `--delete "*"` to also
remove files that no longer exist).

If the build fails at the GL check, add the Space variable `PRECOMPUTE` = `0`. If you want build-time
offline replays instead of the shipped reports, run `LIVE_OUT=0 scripts/make_space.sh` (adds ~15-20 min of
build time on the free tier and the badges read "Offline reference replay").

## Optional: live Muse Spark (spends real money, not for the public Space)

Only on a **private** duplicate of the Space or locally (`scripts/setup.sh`, key in `.env`). On a private
Space: secret `MODEL_API_KEY`, variable `REHEARSAL_OFFLINE` = `0`, `REHEARSAL_BUDGET_CAP` = e.g. `2`, and
`REHEARSAL_PUBLIC_DEMO` = `0` if you need uploads. To stop spending immediately, delete the secret.

## Run the same image locally (if Docker is available)

```bash
docker build -t rehearsal out_space          # or from the repo root: docker build -t rehearsal .
docker run --rm -p 7860:7860 rehearsal       # http://localhost:7860, offline
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build log: `gl_check: MUJOCO_GL=egl failed` then `osmesa works` | fine, OSMesa is used |
| Both backends fail | rendering GIFs is impossible; set `PRECOMPUTE=0`, report the error |
| App tab: "Starting" forever | check Container logs; the server must listen on 0.0.0.0:7860 (set in the Dockerfile) |
| Gallery empty | `out/` missing from the package (ran `make_space.sh` where `out/` does not exist); use `OUT_SRC` |
| `hf upload` rejected | token needs write scope; Space name must match `<you>/<space>` |
