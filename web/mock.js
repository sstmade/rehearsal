// Mock for backend endpoints that may not exist in this build (CONTRACTS.md §4 / docs/api.md).
// Enabled with ?mock=1. Patches fetch; everything else goes to the real server.
// What it fakes: /api/gallery, /api/scene, POST /api/perceive, POST /api/rehearse (job-based), /api/job.
// It also decorates the real desk_01 report with perception confidence metadata so the UI can show
// low-confidence flags (desk_01 was hand-authored and has none).
const realFetch = window.fetch.bind(window);
const jobs = new Map();
let seq = 0;
const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MOCK_CONF = { book_small: 0.02, plant_pot: 0.018 };   // low-confidence objects and their sigma_m

function decorate(report) {
  if (!report || !report.scene_spec) return report;
  for (const o of report.scene_spec.objects) {
    if (o.perception) continue;
    const low = MOCK_CONF[o.id];
    o.perception = { source: low ? "point" : "concept", confidence: low ? "low" : "high", sigma_m: low || 0.012 };
  }
  report._mock = true;
  return report;
}

async function runJob(id, kind, steps, result) {
  const j = jobs.get(id);
  await sleep(300);
  j.state = "running"; j.started = Date.now() / 1000;
  for (const [phase, fraction, line, ms] of steps) {
    j.progress = { phase, fraction }; if (line) j.log.push(line);
    await sleep(ms);
  }
  j.progress = { phase: "done", fraction: 1 }; j.state = "done"; j.result = result; j.finished = Date.now() / 1000;
}

window.fetch = async function (input, init = {}) {
  const url = typeof input === "string" ? input : input.url;
  const u = new URL(url, location.origin);
  const method = (init.method || "GET").toUpperCase();

  if (u.pathname === "/api/report") {
    const r = await realFetch(input, init);
    if (!r.ok) return r;
    return json(decorate(await r.json()));
  }
  if (u.pathname === "/api/gallery") {
    const real = await realFetch(input, init);
    if (real.ok) return real;
    const scenes = await (await realFetch("/api/scenes")).json();
    return json(scenes.map((s) => ({ scene: s, title: s.replace(/_/g, " "), thumb: null, has_report: true, n_objects: s === "desk_01" ? 7 : null, privacy: "public" })));
  }
  if (u.pathname === "/api/scene" && method === "GET") {
    const r = await (await realFetch(`/api/report?scene=${u.searchParams.get("name")}`)).json();
    return r ? json(r.scene_spec) : json({ error: "no such scene" }, 404);
  }
  if (u.pathname === "/api/perceive" && method === "POST") {
    const id = `mock${++seq}`;
    const photo = init.body && init.body.get ? init.body.get("photo") : null;
    const upload = photo ? URL.createObjectURL(photo) : null;
    jobs.set(id, { id, kind: "perceive", state: "queued", log: [], result: null, error: null, progress: { phase: "queued", fraction: 0 }, created: Date.now() / 1000 });
    const report = await (await realFetch("/api/report?scene=desk_01")).json();
    const spec = decorate(report).scene_spec;
    runJob(id, "perceive", [
      ["upload", 0.05, `photo received (${photo ? Math.round(photo.size / 1024) : 0} kB) · data=private`, 500],
      ["sam", 0.25, "SAM 3.1: concept prompts → 7 masks (mug, bowl, bottle, book ×3, plant)", 1400],
      ["sam", 0.4, "SAM 3.1: point prompt fallback for 2 ambiguous regions (confidence: low)", 900],
      ["spark", 0.65, "Muse Spark: object list with size / mass / friction ranges", 1500],
      ["calib", 0.85, "homography from A4 sheet → table frame (px → m)", 700],
      ["validate", 0.95, "scene valid · 7 objects on a 1.30 × 1.20 m table", 400],
    ], { scene: "desk_01", scene_spec: spec, overlay: upload, valid: true, issues: [], saved: true, mock: true });
    return json({ job: id, upload });
  }
  if (u.pathname === "/api/rehearse" && method === "POST") {
    const body = JSON.parse(init.body || "{}");
    const id = `mock${++seq}`;
    jobs.set(id, { id, kind: "rehearse", state: "queued", log: [], result: null, error: null, progress: { phase: "queued", fraction: 0 }, params: body, queue_ahead: 0, created: Date.now() / 1000 });
    runJob(id, "rehearse", [
      ["starting", 0.02, `agent: offline-stand-in (data=public) · scene ${body.scene}`, 400],
      ["setup", 0.08, "reachable: mug_1 bowl_1 bottle_1 book_large book_medium book_small · plant_pot ✕", 700],
      ["attempt", 0.2, "attempt 1: 3 actions, transit 0.15 m", 1200],
      ["attempt", 0.35, "attempt 1: ✕ inside(mug_1, bowl_1) · ✕ undisturbed(bottle_1, plant_pot) · contact t=3.026s bottle_1 ↔ robot_arm", 900],
      ["attempt", 0.42, "diagnosis → transit_height mug_1 0.15 → 0.42 m", 800],
      ["attempt", 0.55, "attempt 2: ✓ all 5 goals", 1200],
      ["stress", 0.8, `stress test: ${body.stress} universes · 35/${body.stress} succeeded`, 1600],
      ["look", 0.92, "look again: mug_1 → 40/40", 900],
      ["report", 0.98, "report written: out/desk_01/report.json", 400],
    ], { scene: body.scene, report: `/api/report?scene=${body.scene}`, success: true, attempts: 2, agent: "offline-stand-in", stress: { successes: 35, n: body.stress }, wall_time_s: 97.1 });
    return json({ job: id, started: true });
  }
  if (u.pathname === "/api/job") {
    const j = jobs.get(u.searchParams.get("id"));
    return j ? json(j) : json({ error: "unknown job" }, 404);
  }
  return realFetch(input, init);
};
console.info("[rehearsal] mock backend active (?mock=1): gallery, perceive, rehearse jobs are simulated");
