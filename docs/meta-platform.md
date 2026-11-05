# Meta platform plan for Rehearsal

Owner: metaspec. Written 2026-10-31 from the docs at https://dev.meta.ai/llms.txt (fetched the same day), the
Cerebral Valley event pages, and probes costing $0.0155 in total (raw output in `experiments/meta_probe/`).
In this doc, **[V]** means verified against a primary source or a probe. **[U]** means unverified.

---

## 1. Hackathon rules, prizes, submission

Primary source: the Official Rules, embedded in https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details
(the page needs JavaScript; the text sits in its Next.js payload). Event pages:
https://dev.meta.ai/events/global-hackathon and https://cerebralvalley.ai/e/meta-ai-developer-hackathon

| Item | What the rules say | |
|---|---|---|
| Deadline | Nov 4, 9:00 PM ET. Judging runs Nov 4–8, finals and winners Nov 9 | [V] |
| Where to submit | https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit | [V] |
| Judging criteria (verbatim weights) | **Impact 30%**: real-world potential, who benefits, whether people would use it, fit to a problem statement. **Use of Meta AI Tooling 25%**: "how creatively the team used Meta's AI tooling, whether they went beyond a basic integration, and whether they surfaced unexpected capabilities". **Technical Depth & Execution 20%**: sound, refined engineering, "real craft". **Demo 25%**: "working, impressive demo, whether the product holds up live, and the quality of the demo video" | [V] |
| Indie track | Individuals and students, teams of up to 4. The Representative submits and names every member. Judged on "creativity, technical ambition, and whether it works" | [V] |
| Indie prizes | $150k / $100k / $75k / $50k / $25k (total $400k) | [V] |
| Bonus prizes | "up to $50,000 in the aggregate". Possible categories include "a community vote award, most viral project award, or other unique or notable project categories designated by the Organizers". Announced on or about **Nov 10**. Bonus prizes do not reduce track prizes | [V] |
| "Best agent app" / "best open-model project" prizes | **Not in the Official Rules.** They may be among the discretionary "other" bonus categories or may have come from a Discord announcement; I found no mirror of one | [U] |
| Products in scope | "Muse Spark, Glimmer, SAM 3, and any other Meta AI products identified on the Event page". The event page lists the Muse Spark series, Muse Wonder, Muse Image, SAM 3.1, open-weight models, and Muse Code | [V] |
| Credits | $150 of Meta Model API credit per participant, valid for the 10-day window. Partners: Baseten (2M TPM), Modal, and Fireworks each offer $100 for open-weight models | [V] |
| IP | Entrant keeps ownership and grants the Organizers a perpetual, non-exclusive license, including for marketing | [V] |
| Eligibility | Excludes anyone "organized or located in Belarus, China, Russia, …". **The team should confirm its submitter's residence** | [V] |
| Submission form fields (video length, repo, etc.) | Behind login, so not read. Expect a demo video, since judging explicitly scores "quality of the demo video" | [U] |

What this means for scoring: the 25% tooling score rewards **depth and surprise**, not a long list of models. Each
integration below is ranked by how much it shows Meta models doing something non-obvious that the physics then checks.

---

## 2. What the platform offers (only what matters for Rehearsal)

| Capability | Key facts (from the docs) | Relevance |
|---|---|---|
| Muse Spark 1.3 (`muse-spark-1.3`, `-contributor`) | 1M context; text, image, video, and PDF input; tools; structured output; `reasoning_effort` (`max` is not available on contributor). $1.25 / $0.15 cached / $4.25 per M tokens; contributor $0.10 / $0.002 / $0.20. Contributor is limited to **150 RPM** | Already our core model |
| Muse Wonder (`muse-wonder-1.0`) | Strongest model, Standard mode only, $4 / $20 per M tokens | Possible "hard case" fallback planner; not needed |
| Tool calling | Chat Completions (`tools[].function`) and Responses (flat `tools[]`, `function_call` / `function_call_output`, `previous_response_id`). **`tool_choice` only accepts `"auto"`**; anything else returns 400. `parallel_tool_calls` defaults to true. Arguments are guided by the schema but "not guaranteed to validate" | The agent loop (§3 #1) |
| Structured output | `response_format.json_schema` (Chat) or `text.format` (Responses). Decoding is always constrained. No recursive schemas, depth ≤ 10 | Already used |
| Reasoning summaries | Responses only: `reasoning: {"summary": "auto" \| "concise" \| "detailed"}`. Returned in the `reasoning` output item's `summary[]`. Not guaranteed on every turn | Shows the agent's thinking in the UI |
| Prompt caching | Automatic prefix caching. Reported as `usage.prompt_tokens_details.cached_tokens` (Chat) or `usage.input_tokens_details.cached_tokens` (Responses). Optional `prompt_cache_key`; `prompt_cache_retention: "24h"` on Responses | Cheaper, faster repeated turns |
| Background responses | `background: true` on Responses, then poll `GET /v1/responses/{id}`. Cannot combine with `stream: true`. Body ≤ 1 MiB | Not needed: our calls take seconds |
| Search grounding | Built-in `web_search` tool, $2.50 per 1k queries | Low; see §3 #7 |
| SAM 3.1 (`sam-3.1`) | Responses API, streaming. Concept prompt (one noun phrase) **or** box/point prompts (≤16 objects, ids you choose). Image $2.50 per 1k; **video $0.20 per 1k frames, minimum 50 frames per request**. MP4/MOV with H.264, H.265, or MPEG-4, ≤15,000 frames. `metadata.include_confidence: "true"`, `metadata.mask_encoding: "one_bit"`, `metadata.propagation_direction` | Masks, confidence, tracking (§3 #2, #3) |
| SAM 3D | **Not on the platform.** Nothing in llms.txt or the model list | — |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | `POST /v1/asr/transcribe` (multipart: `request` JSON + `audio` WAV, mono 16-bit PCM at 16 or 24 kHz, ≤10 min, ≤32 MB) or `wss://api.meta.ai/v1/asr/realtime`. $0.18 per audio hour. `keywords` biasing | Spoken instructions (§3 #4) |
| Muse Image (`muse-image-1.0`) | $0.01 per image; generate or edit; does its own web search | Little real use (§3 #8) |
| Muse Spark Open | 816B MoE (42B active). NVFP4 is 537 GB. On Mac it needs **2× Mac Studio M5 Ultra 512 GB** (or 4× 256 GB) over Thunderbolt-5 RDMA. Otherwise use hosted providers (Baseten, Fireworks, Modal, Together, Groq, OpenRouter) | **Cannot run locally.** Hosted only |
| Muse Glimmer | 30B dense, distilled from Muse Spark, Apache-2.0, text and image input, tool calling. GGUF Q4_K_M is 16.8 GB plus a 1.4 GB vision projector, about 19 GiB at 128K context. Runs under llama.cpp ≥ b10353 or `ollama run muse-glimmer`. The MLX Q4 builds list "48 GB+ unified memory" | **Fits this M3 Max (36 GB) via GGUF**; offline planner (§3 #5) |
| Muse Code | Meta's coding agent CLI | Not used |

---

## 3. Ranked integrations

Cost per demo run assumes Standard mode (user photos are private). Contributor costs are about 1/12 to 1/20 of that.

### #1. Muse Spark tool calling: the agent calls the simulator itself (agentic loop)

- **What it adds to the story:** today our code runs plan → sim → diagnose → revise. After this change, **Muse Spark
  decides** when to call `simulate(plan)`, `stress_test(plan, n)` and `look(object)` (re-segment with SAM), reads the
  physics result, and only answers once physics has passed the plan. That is "Models propose, physics decides" as a
  real agent loop, and it puts the model's own reasoning summary on screen next to each tool call. It speaks directly
  to "went beyond a basic integration" and to any agent-themed bonus prize.
- **Verified by probe** (`probe_tools_json.json`, `probe_responses_tool.json`): `muse-spark-1.3-contributor` accepts
  `tools` together with a strict `response_format` json_schema in one Chat Completions call. Turn 1 returned
  `finish_reason: "tool_calls"` with a schema-valid `simulate` argument. After a `tool` message carrying a failing sim
  log, turn 2 returned schema-valid final JSON (`verdict: "revise"`, citing the slip event). The same flow works on the
  **Responses API** with `text.format` plus `reasoning.summary: "concise"`. Turn 2 carried a summary; turn 1's
  `summary` was `[]`.
- **API shape (Chat Completions, closest to the current `meta_client.chat_json`):**
  ```jsonc
  // request
  {"model": "muse-spark-1.3", "reasoning_effort": "low", "prompt_cache_key": "rehearsal-agent",
   "messages": [{"role":"system","content": SYSTEM}, {"role":"user","content": scene_summary + instruction}],
   "tools": [{"type":"function","function":{"name":"simulate","strict":true,
              "description":"Run the plan in the MuJoCo twin; returns success, goals, events",
              "parameters": ACTION_SCHEMA}}],
   "response_format": {"type":"json_schema","json_schema":{"name":"Final","strict":true,"schema": FINAL}}}
  // response turn 1
  {"choices":[{"finish_reason":"tool_calls","message":{"content":"…","tool_calls":[
     {"id":"call_…","type":"function","function":{"name":"simulate","arguments":"{\"actions\":[…]}"}}]}}],
   "usage":{"prompt_tokens":782,"completion_tokens":156,"prompt_tokens_details":{"cached_tokens":0}}}
  // next request: append the assistant message as-is, then
  {"role":"tool","tool_call_id":"call_…","content":"{\"success\":false,\"events\":[…]}"}
  ```
  Responses variant: `tools: [{"type":"function","name":"simulate","parameters":…,"strict":true}]`,
  `text: {"format": {"type":"json_schema","name":"Final","schema":…,"strict":true}}`,
  `reasoning: {"effort":"low","summary":"concise"}`. The output has `reasoning` (`summary[].text`), `message`, and
  `function_call` (`name`, `arguments`, `call_id`) items. Reply with
  `input=[{"type":"function_call_output","call_id":…,"output":"…"}]` and `previous_response_id`.
- **Gotchas:** `tool_choice` cannot force a call, so if the model answers without calling `simulate`, the code
  should simulate anyway and feed the result back (keep the deterministic guard). Cap the loop at about 4 tool turns.
  The model may emit parallel calls; execute each one and reply to each.
- **Effort:** 3–5 h (agent role: a `ToolAgent` in `agents.py`; commander: a `meta_client.chat_tools()` wrapper with
  ledger and cache).
- **Cost per run:** about 4 turns × (≈3k input, ≈1k output incl. reasoning) ≈ **$0.03 Standard**, $0.002 contributor.
  The probe's two turns cost $0.0002 on contributor.
- **Risk:** low to medium. The loop is short and verified. Main risk is the model calling `simulate` with a sloppy
  plan many times; mitigate with the turn cap and by returning a precise sim diagnosis.

### #2. SAM 3.1 masks and confidence → exact footprints, contact points, sigma

- **What it adds:** we already pay for the masks and throw them away. Decoding them gives the object's **true
  footprint** (the lowest mask pixels, projected through the table homography) instead of the bottom-centre of the
  box, plus **yaw** (min-area rectangle of the projected footprint), a better radius/length, and a per-object
  **confidence `c`** that can set `perception.sigma_m` and `confidence` in place of the "point = low" heuristic. The
  stress test then perturbs each object by what SAM itself is unsure about: a model's uncertainty feeding the physics.
  The UI can draw mask overlays on the photo.
- **Verified by probe** (`probe_sam_image_decoded.json`, synthetic 160×120 PNG: red disc r=22 at (50,60), blue
  rectangle 105..140 × 40..85):
  - Each mask raster is the **box crop at native resolution with inclusive corners**: box 28..72 gives `data=45,45`,
    box 105..140 × 40..85 gives `data=46,36` (H, W). Image coordinates = raster index + (x1, y1).
  - Decoded with `meta_sam_parser.decode_mask_to_raster(SegmentationMask(encoding, payload="~"+payload, width, height))`.
    **The marker must stay in the payload.** Rectangle area 1656 px matches the truth of 1656; disc 1581 px vs
    analytic 1521. Centroids were exactly (50.0, 60.0) and (122.5, 62.5).
  - `include_confidence` gave `c=0.9416` on box and mask for the concept "red circle".
  - **Contrary to the docs**, point prompts returned lossless `~` masks, not `one_bit`. Read the marker.
- **API shape:** the request is unchanged except for `metadata: {"include_confidence": "true"}`. The output record is
  `<0f>0<|box;x1=28;y1=38;x2=72;y2=82;w=160;h=120;c=0.94|><|mask;x=0;y=0;c=0.94;data=45,45,~<base85>|>`.
- **Needs:** `uv pip install meta-sam-parser` (0.1.0, pure Python, now in the shared .venv), and `meta_client.sam()`
  must accept a `metadata` dict (interface change below).
- **Effort:** 2–3 h (perception role). **Cost:** $0 extra. **Risk:** low. For tall objects the lowest mask row is the
  front contact edge, not the centre; take the centre as the front edge plus radius along the camera ray.

### #3. SAM 3.1 video tracking: a "reality check" clip after the rehearsal (or video input)

- **What it adds:** after the plan is executed in the real world (by a robot, or by hand for the demo), the user
  films 3–5 s. SAM tracks every object with **box prompts seeded from the twin's `perception.box_px`** (one request,
  ids = our object ids, `propagation_direction: "forward"`). We project the final masks through the homography and
  score predicted vs real final positions. That closes the loop: photo → twin → plan → reality → SAM verifies.
  Alternative: a 3 s handheld clip as input, choosing the sharpest frame and checking the object set is stable
  across frames (smaller story).
- **Verified by probe** (`probe_sam_video_decoded.json`): a 50-frame, 160×120 H.264 MP4 sent **inline as a base64
  data URL** (`input_video.video_url`, no Files upload needed). Concept "red circle" returned all **50 frames with one
  stable id 0**, and the box followed the disc (x1 8 → 86). Usage was `video_frames_processed: 50`. Near the end the
  disc was partly hidden by the blue rectangle and the box shrank to x2=114 (visible edge 104, full disc 130), so
  expect partial boxes under occlusion.
- **API shape:**
  ```jsonc
  {"model":"sam-3.1","stream":true,
   "metadata":{"propagation_direction":"forward","mask_encoding":"one_bit","include_confidence":"true"},
   "input":[{"type":"message","role":"user","content":[
     {"type":"input_text","text":"<0f>1<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H|>,2<|box;…|>"},
     {"type":"input_video","video_url":"data:video/mp4;base64,…"}]}]}   // or "file_id" via POST /v1/files purpose=user_data
  // one line per frame: <Nf>1<|box;…|><|mask;…|>,2<|box;…|><|mask;…|>   (frames can skip; key by <Nf> and id)
  ```
  The inline media limit is 50 MB; phone clips should be re-encoded to ≤960 px on the long edge (also faster).
  `w` and `h` must match the displayed (rotation-corrected) size, otherwise 400.
- **Effort:** 4–6 h (perception + backend + UI). **Cost:** 5 s × 30 fps = 150 frames = **$0.03** per clip, with all
  objects in one request. **Risk:** medium. It needs the same camera pose as the photo for the homography (tripod or
  paper sheet visible), and real execution footage to show.

### #4. Muse Voice Transcribe for spoken instructions

- **Adds:** "say it, don't type it": hold-to-talk in the UI, `keywords` = scene object labels so "the blue mug" is
  recognised. It adds a third hosted Meta model at almost no cost. Breadth, not depth.
- **API:** `POST https://api.meta.ai/v1/asr/transcribe`, multipart `request={"model":"muse-voice-transcribe-1.0","mode":"PUSH_TO_TALK","audioEncoding":"WAV","keywords":[…]}`
  plus `audio=@clip.wav` (mono s16 16/24 kHz). Response `{"sessionId","transcript","audioDurationMs","turns":[]}`.
  The browser records webm/opus, so convert server-side (we now have the ffmpeg binary from imageio-ffmpeg) or
  encode the WAV in JS.
- **Effort:** 1.5–2 h. **Cost:** 10 s = $0.0005. **Risk:** low. Not probed (multipart is not covered by
  `meta_client`; the docs are unambiguous).

### #5. Muse Glimmer as an offline / private planner

- **Adds:** the planner runs entirely on the laptop. User photos never leave the device, and the agent falls back to
  Glimmer when offline. Pattern from the cookbook "local agent that asks before calling Muse Spark": Glimmer plans
  locally and escalates hard cases to Muse Spark. This is the "open-weight" angle. **Muse Spark Open cannot run here**
  (it needs 1+ TB across Mac Studios). Glimmer Q4_K_M (16.8 GB + 1.4 GB mmproj) should fit the 36 GB M3 Max through
  `ollama run muse-glimmer` (Ollama 0.x is installed; the model is not pulled) or llama.cpp ≥ b10353.
- **API:** the OpenAI-compatible local server (`http://localhost:11434/v1` or `:8080/v1`), same chat and tools shape.
  Whether Glimmer honours `response_format` json_schema in Ollama or llama.cpp is unverified (llama.cpp supports
  grammar-constrained `json_schema`).
- **Effort:** 3–4 h, plus a 17 GB download. **Cost:** $0 API. **Risk:** medium to high: download time, tokens/s on
  M3 Max unknown (30B dense at Q4, perhaps 15–25 tok/s, **[U]**), and quality on our predicate compilation unknown.
  Do it only if #1–#3 are done.

### #6. Prompt caching hygiene (cost and latency)

- Put the static system prompt, then the scene summary, first; put the instruction and sim logs last. Add
  `prompt_cache_key: "rehearsal-<role>"`. **Probe:** turn 2 of the tool loop got `cached_tokens: 753` of 940 with no
  other change, so it already works automatically on Chat Completions. On Responses with `previous_response_id`, the
  probe reported `cached_tokens: 0` and input 945 (the full reconstructed prefix, not "only the new turn" as the docs
  imply).
- **Effort:** 0.5 h. **Saves:** about 88% of the prefix cost ($1.25 → $0.15 per M tokens), roughly $0.003 per turn.
  Small money, but faster and a good line for the README. `meta_client` already reads `cached_tokens`.

### #7. Search grounding for physical priors (optional)

- Muse Spark with `tools: [{"type":"web_search"}]` looks up a recognised product ("IKEA 365+ mug") for its mass and
  material, with citations, to narrow `mass_kg.range`. Novel but adds latency and the citations are unverifiable live.
  $2.50 per 1k queries. 2 h. Low priority.

### #8. Muse Image: not recommended

- Generated images have no ground-truth geometry, so they cannot test perception accuracy. Editing the user's photo
  to "show the predicted end state" would be a picture the physics did not decide, which contradicts our tagline. The
  only defensible use is mass-producing varied desk photos for a robustness gallery ($0.01 each). Skip.

---

## 4. Recommendation

| Rank | Integration | Effort | API cost per demo run | Why |
|---|---|---|---|---|
| 1 | **Tool-calling agent loop** (Responses API + reasoning summaries) | 3–5 h | ≈ $0.03 | The largest change to the story, aimed straight at "beyond a basic integration"; verified tonight |
| 2 | **SAM masks + confidence → footprint, yaw, sigma** | 2–3 h | $0 | Uses output we already pay for; physics now uses SAM's own uncertainty; overlays for the demo |
| 3 | **SAM video reality check** (box prompts seeded from the twin) | 4–6 h | ≈ $0.03 | Closes the loop to reality; shows an unexpected SAM capability (tracking) |
| + | Voice instruction (1.5–2 h, ~$0), prompt-cache hygiene (0.5 h) | | | Cheap extras if time allows |

**Tonight: #1, the tool-calling agent loop.** It changes how judges read the whole project: from "a pipeline that
calls an LLM" to "an agent that rehearses in physics before it acts". Its unknowns are retired: tools and the strict
schema coexist, the tool result is consumed correctly, and reasoning summaries arrive. #2 is independent
(perception role, no API cost) and can run in parallel.

### Interface change requests (commander owns `rehearsal/meta_client.py`)

1. `sam(prompt, media_url, *, caller, metadata=None, video=False)`: pass `metadata` through; for video send
   `{"type":"input_video","video_url":…}`. Bill `max(50, usage.video_frames_processed) × $0.0002` instead of the
   fixed image price. Include `metadata` in the cache key.
2. `chat_tools(messages, tools, schema, name, *, caller, data, effort)`: one turn returning `(message, usage)`, with
   the same reserve, settle, and ledger as `chat_json` (do not cache turns that contain tool calls, or key the cache on
   the full message list). Or a `responses_turn(...)` that also returns `reasoning.summary`.
3. Optional `prompt_cache_key` argument on `chat_json`.
4. `transcribe(wav_bytes, *, caller, keywords)` → `POST /v1/asr/transcribe` (multipart via `client()._client` or
   httpx), billed at $0.18 per hour from `audioDurationMs`.

## 5. Probe log

| Probe | Model | Result | Cost (ledger) |
|---|---|---|---|
| `tools_json` (2 turns) | muse-spark-1.3-contributor | tools + strict json_schema OK; tool call, then schema answer; turn 2 had 753 cached tokens | $0.000206 |
| `responses_tool` (2 turns) | muse-spark-1.3-contributor | Responses tools + text.format + summary OK | $0.000315 |
| `sam_image` concept + points | sam-3.1 | masks are box crops, inclusive; decode verified; `c` present; points → `~` | $0.005 |
| `sam_video` 50 frames | sam-3.1 | inline data-URL MP4 OK; 50 frames, stable id; `video_frames_processed: 50` | $0.010 |
| **Total** | | | **$0.0155** |

Packages added to the shared .venv: `meta-sam-parser==0.1.0`, `imageio-ffmpeg==0.7.0` (bundles an ffmpeg binary).
