// Mock for backend endpoints (CONTRACTS.md §4 / docs/api.md), enabled with ?mock=1. Patches fetch;
// everything the real server answers is passed through, the rest is simulated here:
//   POST /api/perceive + /api/job  → a perception job whose result is a REAL perception output from the
//                                    synthetic test set (photos/synth/overlays/*.scene.json + overlay png)
//   POST /api/rehearse (old servers) → a job that replays the desk_01 report's log lines
//   POST /api/scene (old servers)   → accepted
//   GET  /api/gallery, /api/budget   → fallbacks when missing
//   ?mockfail=perceive|rehearse      → that job ends in an error (to show the error states)
// It also decorates hand-authored reports with perception confidence metadata (desk_01 has none).
const realFetch = window.fetch.bind(window);
const params = new URLSearchParams(location.search);
const FAIL = params.get("mockfail") || "";
const jobs = new Map();
let seq = 0;
const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MOCK_CONF = { book_small: 0.02, plant_pot: 0.018 };   // low-confidence objects and their sigma_m
const perceived = {};                                        // name -> scene spec returned by mock perception

function decorate(report) {
  if (!report || !report.scene_spec || report.scene_spec.photo) return report;
  for (const o of report.scene_spec.objects) {
    if (o.perception) continue;
    const low = MOCK_CONF[o.id];
    o.perception = { source: low ? "point" : "concept", confidence: low ? "low" : "high", sigma_m: low || 0.012 };
  }
  report._mock = true;
  return report;
}

function startJob(kind, steps, result, failAt = null) {
  const id = `mock${++seq}`;
  const j = { id, kind, state: "queued", log: [], result: null, error: null, progress: { phase: "queued", fraction: 0 }, queue_ahead: 0, created: Date.now() / 1000 };
  jobs.set(id, j);
  (async () => {
    await sleep(350);
    j.state = "running"; j.started = Date.now() / 1000;
    for (let i = 0; i < steps.length; i++) {
      const [phase, fraction, line, ms] = steps[i];
      j.progress = { phase, fraction }; if (line) j.log.push(line);
      await sleep(ms);
      if (failAt === i) { j.state = "error"; j.error = "MockError: simulated failure (?mockfail) — e.g. SAM 3.1 returned no masks for any object"; j.log.push("Traceback (most recent call last): … MockError"); return; }
    }
    try { j.result = await result(); j.state = "done"; j.progress = { phase: "done", fraction: 1 }; }
    catch (e) { j.state = "error"; j.error = String(e.message || e); }
    j.finished = Date.now() / 1000;
  })();
  return id;
}

window.fetch = async function (input, init = {}) {
  const url = typeof input === "string" ? input : input.url;
  const u = new URL(url, location.origin);
  const method = (init.method || "GET").toUpperCase();

  if (u.pathname === "/api/report") {
    const name = u.searchParams.get("scene");
    if (perceived[name]) return json(null);
    const r = await realFetch(input, init);
    if (!r.ok) return r;
    return json(decorate(await r.json()));
  }
  if (u.pathname === "/api/scene" && method === "GET") {
    const name = u.searchParams.get("name");
    if (perceived[name]) return json(perceived[name]);
    const r = await realFetch(input, init);
    if (r.ok) return r;
    const rep = await (await realFetch(`/api/report?scene=${name}`)).json();
    return rep ? json(rep.scene_spec) : json({ error: "no such scene" }, 404);
  }
  if (u.pathname === "/api/scene" && method === "POST") {
    const r = await realFetch(input, init);
    if (r.status !== 404) return r;
    const spec = JSON.parse(init.body); perceived[spec.name] = spec;
    await sleep(300); return json({ name: spec.name });
  }
  if (u.pathname === "/api/gallery") {
    const real = await realFetch(input, init);
    let items = real.ok ? await real.json() : (await (await realFetch("/api/scenes")).json()).map((s) => ({ scene: s, title: s, thumb: null, has_report: true, n_objects: null }));
    for (const [name, spec] of Object.entries(perceived)) if (!items.some((g) => g.scene === name))
      items.push({ scene: name, title: "Your photo (mock perception)", thumb: spec.photo ? "/" + spec.photo.file : null, has_report: false, n_objects: spec.objects.length, privacy: spec.privacy });
    return json(items);
  }
  if (u.pathname === "/api/scenes") {
    const r = await realFetch(input, init); const names = r.ok ? await r.json() : [];
    return json([...names, ...Object.keys(perceived).filter((n) => !names.includes(n))]);
  }
  if (u.pathname === "/api/budget") {
    const r = await realFetch(input, init);
    if (r.ok) return r;
    return json({ spent: 3.41, reserved: 0.05, cap: 15, remaining: 11.54, mock: true });
  }
  if (u.pathname === "/api/perceive" && method === "POST") {
    const photo = init.body && init.body.get ? init.body.get("photo") : null;
    const privacy = (init.body && init.body.get && init.body.get("privacy")) || "private";
    const side = (init.body && init.body.get && init.body.get("robot_side")) || "bottom";
    const id = startJob("perceive", [
      ["upload", 0.05, `photo received (${photo ? Math.round(photo.size / 1024) : 0} kB) · data=${privacy} · robot_side=${side}`, 500],
      ["list", 0.2, "Muse Spark · list objects → 6 candidates with size / mass / friction ranges", 1300],
      ["segment", 0.35, "SAM 3.1 · concept prompts → masks (score ≥ 0.9 for 4 objects)", 1100],
      ["segment", 0.5, "SAM 3.1 · point-prompt fallback for 2 ambiguous regions → confidence: low", 900],
      ["calibrate", 0.7, "A4 sheet found · homography px → table (m) · camera K, R, t recovered", 800],
      ["validate", 0.9, "guarded mask placement · sigma_m calibrated from mask IoU", 600],
      ["validate", 0.96, "[mock] the uploaded image is ignored: returning a real perception output from photos/synth", 300],
    ], async () => {
      const real = await realFetch("/photos/synth/overlays/rand_2__front55.scene.json").then((r) => r.ok ? r.json() : null).catch(() => null);
      if (!real) throw new Error("perceive_photo is not available in this build and photos/synth is missing");
      real.name = "photo_mock_rand2"; real.privacy = privacy;
      perceived[real.name] = real;
      return { scene: real.name, scene_spec: real, overlay: "/photos/synth/overlays/rand_2__front55.scene.overlay.png", valid: true, issues: [], saved: true, mock: true };
    }, FAIL === "perceive" ? 3 : null);
    return json({ job: id, upload: null });
  }
  if (u.pathname === "/api/rehearse" && method === "POST") {
    const body = JSON.parse(init.body || "{}");
    if (FAIL !== "rehearse") { const r = await realFetch(input, init); if (r.status !== 404) return r; }
    const rep = await (await realFetch("/api/report?scene=desk_01")).json();
    const steps = [["starting", 0.02, `agent: offline-stand-in (data=public) · scene ${body.scene}`, 400], ["setup", 0.08, "reachable: " + Object.entries(rep.reachability).map(([k, v]) => `${k}${v ? "" : " ✕"}`).join(" "), 700]];
    rep.attempts.forEach((a, i) => {
      steps.push(["attempt", 0.15 + 0.2 * i, `attempt ${a.attempt}: ${a.plan.actions.length} actions · ${a.success ? "✓ success" : "✕ " + a.goals.filter((g) => !g.ok).map((g) => `${g.pred}(${g.args.join(",")})`).join(", ")}`, 1100]);
      if (a.plan.diagnosis) steps.push(["attempt", 0.2 + 0.2 * i, `Muse Spark diagnosis: ${a.plan.diagnosis_en || a.plan.diagnosis}`, 900]);
    });
    if (rep.stress_test) steps.push(["stress", 0.8, `stress test: ${body.stress} universes · ${rep.stress_test.successes}/${rep.stress_test.n} succeeded`, 1500]);
    if (rep.active_perception && rep.active_perception.steps.length) steps.push(["look", 0.92, `look again: ${rep.active_perception.steps.map((s) => `${s.look_at} ${s.before}→${s.after}`).join(", ")}`, 800]);
    steps.push(["report", 0.98, `report written: out/${body.scene}/report.json`, 400]);
    const id = startJob("rehearse", steps, async () => ({ scene: body.scene, report: `/api/report?scene=${body.scene}`, success: true, attempts: rep.attempts.length, agent: rep.agent, wall_time_s: rep.wall_time_s }), FAIL === "rehearse" ? 2 : null);
    return json({ job: id, started: true });
  }
  if (u.pathname === "/api/job") {
    const j = jobs.get(u.searchParams.get("id"));
    if (j) return json(j);
    return realFetch(input, init);
  }
  return realFetch(input, init);
};
console.info("[rehearsal] mock active (?mock=1): perception, old-server rehearse/scene/gallery/budget are simulated; add &mockfail=perceive|rehearse for error states");
