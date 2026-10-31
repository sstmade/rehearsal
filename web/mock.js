// Mock layer for UI development (`?mock=1`). Uses the real desk_01 report from the server when
// it exists; everything else (gallery, perception jobs, budget) is synthesized here.
// `?mockfail=perceive|rehearse` makes that job fail, to show error states.
const params = new URLSearchParams(location.search);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const FAIL = params.get("mockfail") || "";

// A calibrated camera across the table from the robot, like a person holding a phone.
function lookAtCamera(eye, target, W, H, f) {
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const norm = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const z = norm(sub(target, eye));             // forward
  const x = norm(cross(z, [0, 0, 1]));          // right
  const y = cross(z, x);                        // down
  const R = [x, y, z];
  const t = R.map((row) => -(row[0] * eye[0] + row[1] * eye[1] + row[2] * eye[2]));
  return { K: [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], R, t };
}

function project(cam, X) {
  const c = [0, 1, 2].map((i) => cam.R[i][0] * X[0] + cam.R[i][1] * X[1] + cam.R[i][2] * X[2] + cam.t[i]);
  return [cam.K[0][0] * c[0] / c[2] + cam.K[0][2], cam.K[1][1] * c[1] / c[2] + cam.K[1][2]];
}

function boxPx(cam, o) {
  const s = o.shape, h = s.type === "box" ? s.size[2] : s.height;
  const rx = s.type === "box" ? s.size[0] / 2 : s.radius, ry = s.type === "box" ? s.size[1] / 2 : s.radius;
  const pts = [];
  for (const dx of [-rx, rx]) for (const dy of [-ry, ry]) for (const z of [0, h]) pts.push(project(cam, [o.pos[0] + dx, o.pos[1] + dy, z]));
  const us = pts.map((p) => p[0]), vs = pts.map((p) => p[1]);
  return [Math.min(...us), Math.min(...vs), Math.max(...us), Math.max(...vs)].map(Math.round);
}

const LOW = { bottle_1: 0.022, book_small: 0.018 };   // pretend these came from point prompts

function synthScene(base) {
  const W = 1280, H = 960;
  const cam = lookAtCamera([1.55, 0.55, 0.78], [0.55, 0.0, 0.0], W, H, 1050);
  const objects = base.objects.map((o) => ({
    ...o, category: o.label.split(" ").pop().replace(/[()]/g, ""),
    perception: { box_px: boxPx(cam, o), source: LOW[o.id] ? "point" : "concept",
      confidence: LOW[o.id] ? "low" : "high", sigma_m: LOW[o.id] || 0.008 },
  }));
  return { ...base, name: "synth_desk", privacy: "public", source: "synthetic render (mock) — calibrated camera",
    objects, photo: { file: "mock:synth", width: W, height: H, camera: cam } };
}

export function createMock(live) {
  const jobs = new Map();
  let deskSpec = null;
  const reports = { desk_01: true, synth_desk: false };

  async function desk() {
    if (!deskSpec) {
      const rep = await live.report("desk_01").catch(() => null);
      deskSpec = rep ? rep.scene_spec : null;
    }
    return deskSpec;
  }

  function startJob(lines, result, failAt) {
    const id = "mock" + Math.random().toString(36).slice(2, 8);
    const job = { id, state: "queued", log: [], result: null, error: null, progress: { phase: "queued", fraction: 0 } };
    jobs.set(id, job);
    (async () => {
      await sleep(400);
      job.state = "running";
      for (let i = 0; i < lines.length; i++) {
        const [ms, phase, text] = lines[i];
        await sleep(ms);
        job.log.push(text);
        job.progress = { phase, fraction: (i + 1) / (lines.length + 1) };
        if (failAt != null && i === failAt) {
          job.state = "error";
          job.error = "MockError: simulated failure (?mockfail) — e.g. SAM 3.1 returned no masks for any object";
          return;
        }
      }
      job.result = await result();
      job.state = "done"; job.progress = { phase: "done", fraction: 1 };
    })();
    return { job: id };
  }

  return {
    mock: true,
    scenes: async () => ["desk_01", "synth_desk"],
    async gallery() {
      await sleep(150);
      const g = await live.gallery().catch(() => [{ scene: "desk_01", title: "desk_01", thumb: null, has_report: true }]);
      return [...g.filter((x) => x.scene !== "synth_desk"),
        { scene: "synth_desk", title: "Calibrated render (mock)", thumb: null, has_report: reports.synth_desk }];
    },
    async scene(name) {
      if (name !== "synth_desk") return live.scene(name);
      const d = await desk();
      if (!d) throw new Error("mock needs out/desk_01/report.json — run the desk_01 rehearsal first");
      return name === "synth_desk" ? synthScene(d) : d;
    },
    async saveScene(spec) { await sleep(300); return { name: spec.name }; },
    perceive(file, opts) {
      const fail = FAIL === "perceive" ? 4 : null;
      return startJob([
        [300, "upload", `received ${file?.name || "photo"} (${Math.round((file?.size || 0) / 1024)} KB) · privacy=${opts?.privacy || "private"}`],
        [700, "list", "Muse Spark · list objects → 7 candidates: mug, bowl, bottle, book ×3, plant pot"],
        [600, "segment", "SAM 3.1 · concept prompt 'mug' → 1 mask (score 0.94)"],
        [400, "segment", "SAM 3.1 · concept prompt 'bowl' → 1 mask (score 0.91)"],
        [500, "segment", "SAM 3.1 · concept prompt 'bottle' → 0 masks · retry with point prompt → 1 mask (low confidence)"],
        [500, "segment", "SAM 3.1 · concept prompt 'book' → 3 masks"],
        [500, "segment", "SAM 3.1 · point prompt 'notebook' → 1 mask (low confidence)"],
        [500, "calibrate", "homography from paper sheet · reprojection error 3.1 px"],
        [400, "validate", "scene validated · 7 objects · 2 low-confidence (σ ≥ 1.8 cm)"],
        [300, "validate", "[mock] uploaded image ignored — returning the synthetic calibrated scene"],
      ], async () => ({ scene: "synth_desk", scene_spec: synthScene(await desk()), overlay: null }), fail);
    },
    async job(id) {
      await sleep(60);
      const j = jobs.get(id);
      if (!j) { const e = new Error("unknown job"); e.status = 404; throw e; }
      return JSON.parse(JSON.stringify(j));
    },
    async rehearse(body) {
      const rep = await live.report("desk_01");
      const fail = FAIL === "rehearse" ? 3 : null;
      const A = rep.attempts, st = rep.stress_test, ap = rep.active_perception;
      const lines = [[300, "setup", `agent: ${rep.agent} · scene ${body.scene} · stress ${body.stress} · look ${body.look}`],
        [500, "attempt", `Muse Spark · compile goals → ${rep.goals.map((g) => `${g.pred}(${g.args.join(",")})`).join(", ")}`]];
      for (const a of A) {
        lines.push([700, "attempt", `--- attempt ${a.attempt}: ${a.success ? "SUCCESS" : "FAILED"}`]);
        if (a.plan.diagnosis) lines.push([500, "attempt", `Muse Spark · diagnosis: ${a.plan.diagnosis}`]);
      }
      if (st) for (let k = 10; k <= st.n; k += 10) lines.push([350, "stress", `stress test ${k}/${st.n}`]);
      if (st) lines.push([200, "stress", `stress test: ${st.successes}/${st.n} succeeded`]);
      if (ap) for (const s of ap.steps) lines.push([700, "look", `look at ${s.look_at}: ${s.before}/${ap.n} -> ${s.after}/${ap.n}`]);
      lines.push([300, "report", "report: out/desk_01/report.json"]);
      return startJob(lines, async () => { reports[body.scene] = true; return { scene: body.scene }; }, fail);
    },
    async report(scene) {
      if (scene !== "synth_desk") return live.report(scene);
      if (!reports[scene]) return null;
      return live.report("desk_01");
    },
    async budget() { try { return await live.budget(); } catch { /* fall through */ } return { spent: 3.4127, reserved: 0.05, cap: 15, remaining: 11.5373, mock: true }; },
    legacyStatus: live.legacyStatus,
  };
}
