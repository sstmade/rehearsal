// HTTP client for the contract in CONTRACTS.md §4. `?mock=1` swaps in web/mock.js.
export const params = new URLSearchParams(location.search);
export const MOCK = params.has("mock");

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const b = await r.json(); msg = b.error || msg; } catch { /* not json */ }
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return r.json();
}

const live = {
  scenes: () => j("/api/scenes"),
  // falls back to /api/scenes + /api/report when /api/gallery is not deployed yet
  async gallery() {
    try { return await j("/api/gallery"); } catch (e) {
      if (e.status !== 404) throw e;
      const names = await j("/api/scenes");
      return Promise.all(names.map(async (scene) => ({ scene, title: scene, thumb: null,
        has_report: !!(await j(`/api/report?scene=${encodeURIComponent(scene)}`).catch(() => null)) })));
    }
  },
  scene: (name) => j(`/api/scene?name=${encodeURIComponent(name)}`),
  saveScene: (spec) => j("/api/scene", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }),
  perceive(file, { privacy = "private", paperCorners } = {}) {
    const fd = new FormData();
    fd.append("photo", file);
    fd.append("privacy", privacy);
    if (paperCorners) fd.append("paper_corners", JSON.stringify(paperCorners));
    return j("/api/perceive", { method: "POST", body: fd });
  },
  job: (id) => j(`/api/job?id=${encodeURIComponent(id)}`),
  rehearse: (body) => j("/api/rehearse", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  report: (scene) => j(`/api/report?scene=${encodeURIComponent(scene)}`),
  budget: () => j("/api/budget"),
  legacyStatus: () => j("/api/status"),
};

let impl = live;
if (MOCK) impl = (await import("./mock.js")).createMock(live);
export const api = impl;

// Poll a job until done/error. onUpdate(jobView) on every poll.
export async function waitJob(id, onUpdate, interval = 700) {
  for (;;) {
    const v = await api.job(id);
    onUpdate(v);
    if (v.state === "done") return v.result;
    if (v.state === "error") { const e = new Error(v.error || "job failed"); e.log = v.log; throw e; }
    await new Promise((r) => setTimeout(r, interval));
  }
}

// Start a rehearsal and follow it; works with the job API or the legacy single-job API.
export async function runRehearsal(body, onUpdate) {
  const r = await api.rehearse(body);
  if (r.job) return waitJob(r.job, onUpdate);
  for (;;) {   // legacy: {"started": true} + /api/status
    const st = await api.legacyStatus();
    onUpdate({ state: st.running ? "running" : st.error ? "error" : "done", log: st.log, progress: null });
    if (!st.running) { if (st.error) throw new Error(st.error); return { scene: body.scene }; }
    await new Promise((res) => setTimeout(res, 700));
  }
}
