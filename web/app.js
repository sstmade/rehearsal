// Rehearsal web app: scroll-driven narrative on the left, one persistent 3D stage on the right.
import { Replay } from "./replay.js";

const P = new URLSearchParams(location.search);
if (P.has("mock")) await import("./mock.js");           // patches fetch for endpoints that do not exist yet
const DEMO = P.has("demo");
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const PRED = { inside: "inside", on: "on top of", upright: "upright", stacked: "stacked", stable: "stable", undisturbed: "undisturbed" };
const REL = { in: "into", on: "onto" };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtRange = (r, unit = "") => `${r[0]}–${r[1]}${unit}`;
const col = (rgba) => `rgba(${rgba.slice(0, 3).map((c) => Math.round(c * 255)).join(",")},${rgba[3] ?? 1})`;
const dims = (o) => o.shape.type === "box" ? `${o.shape.size.map((v) => Math.round(v * 100)).join("×")} cm`
  : `⌀${Math.round(o.shape.radius * 200)}×${Math.round(o.shape.height * 100)} cm`;
// English gloss for the offline agent's Chinese diagnosis (kept verbatim above it).
const GLOSS = {
  "t=3.026s 时，吸盘在 0.139 m 高度搬运 mug_1，撞到了 bottle_1（高 0.26 m）。mug_1 挂在吸盘下方 0.1 m，所以搬运高度必须超过 0.26 + 0.1 m，改为 0.42 m。":
    "At t = 3.026 s the suction cup was carrying mug_1 at 0.139 m and hit bottle_1 (0.26 m tall). mug_1 hangs 0.10 m below the cup, so the transit height must clear 0.26 + 0.10 m → set it to 0.42 m.",
};

// ------------------------------------------------------------------ theme
const themeBtn = $("themeBtn");
function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
try { applyTheme(localStorage.getItem("rehearsal-theme")); } catch {}
themeBtn.onclick = () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  applyTheme(next); try { localStorage.setItem("rehearsal-theme", next); } catch {}
};

// ------------------------------------------------------------------ tooltip
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; moveTip(e); }
function moveTip(e) {
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(e.clientX + 14, innerWidth - r.width - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, innerHeight - r.height - 8) + "px";
}
function hideTip() { tip.style.display = "none"; }
function bindTips(root) {
  root.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mouseenter", (e) => showTip(e, el.dataset.tip));
    el.addEventListener("mousemove", moveTip);
    el.addEventListener("mouseleave", hideTip);
  });
}

// ------------------------------------------------------------------ state
const state = { report: null, scene: null, current: 0, bust: Date.now(), stage: null, trailSets: [], trailSet: 0, traj: {}, loaded: null };
let replay = null;
try { replay = new Replay($("stage"), { onTime, onEnd }); } catch (err) { console.warn("WebGL unavailable", err); }
const base = () => `/out/${encodeURIComponent(state.scene)}`;
const getJson = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);

// ------------------------------------------------------------------ init
async function init() {
  if (DEMO) document.body.classList.add("demo");
  const scenes = (await getJson("/api/scenes")) || [];
  $("scene").innerHTML = scenes.map((s) => `<option>${esc(s)}</option>`).join("");
  state.scene = P.get("scene") && scenes.includes(P.get("scene")) ? P.get("scene") : scenes[0];
  $("scene").value = state.scene;
  $("scene").onchange = () => { state.scene = $("scene").value; loadScene(); };
  $("form").onsubmit = runRehearsal;
  $("demoLink").href = `?demo=1&scene=${encodeURIComponent(state.scene || "")}${P.has("mock") ? "&mock=1" : ""}`;
  bindStageControls();
  bindUpload();
  renderGallery();
  await loadScene();
  if (P.has("still") && replay) { replay.renderer.setAnimationLoop(null); replay.noOrbit = true; replay.orbit(false); setInterval(() => replay.tick(), 400); } // software-GL screenshots
  if (DEMO) { const m = await import("./demo.js"); m.run(app); }
  else setupObservers();
}

async function loadScene() {
  state.report = await getJson(`/api/report?scene=${encodeURIComponent(state.scene)}`);
  state.current = state.report ? state.report.attempts.length - 1 : 0;
  state.userPicked = false;
  state.traj = {}; state.trailSets = []; state.trailSet = 0;
  $("viewerEmpty").hidden = !!state.report;
  document.body.classList.toggle("noreport", !state.report);
  $("rehearseJob").hidden = true;
  if (!state.report) {
    // Scene without a rehearsal yet: show the twin inventory / photo and invite a run.
    const spec = await getJson(`/api/scene?name=${encodeURIComponent(state.scene)}`);
    $("loader").classList.add("done");
    if (!spec) return;
    const gal = state.gallery && state.gallery.find((g) => g.scene === state.scene);
    state.report = { scene: state.scene, scene_spec: spec, instruction: (gal && gal.demo_instruction) || "", agent: "—", reachability: {}, goals: [], attempts: [], stress_test: null, active_perception: null, wall_time_s: 0, _empty: true };
    state.loaded = Promise.resolve();
    $("instruction").value = state.report.instruction;
    renderInventory(); renderPhoto();
    $("goals0").innerHTML = `<li class="note">Goals are compiled by Muse Spark when you press Rehearse.</li>`;
    $("plan0").innerHTML = ""; $("reach").innerHTML = "";
    if (!$("s2").querySelector(".noreport-note")) $("s2").querySelector(".ch-head").insertAdjacentHTML("afterend", `<div class="noreport-note">No rehearsal has been run for <code>${esc(state.scene)}</code> yet. Press <b>Rehearse</b> to let physics judge the plan.</div>`);
    return;
  }
  $("s2").querySelector(".noreport-note")?.remove();
  state.loaded = ensureScene();
  render();
  await state.loaded;
  if (!DEMO) setStage(state.stage || "hero", true);
}

function render() {
  const r = state.report;
  $("instruction").value = r.instruction;
  $("reportLink").href = `${base()}/report.md`;
  renderHeroStats(); renderInventory(); renderGoalsPlan(); renderAttempts(); renderStress(); renderLook();
  renderPhoto();
}

// ------------------------------------------------------------------ 3D loading
async function ensureScene() {
  const r = state.report;
  const loader = $("loader"), bar = $("loaderBar"), txt = $("loaderTxt");
  loader.classList.remove("done");
  if (!replay) { loader.classList.add("done"); return; }
  await replay.loadScene(base(), (got, total) => {
    const f = total ? got / total : 0.5;
    bar.style.width = `${Math.round(f * 100)}%`;
    txt.textContent = `Loading twin meshes · ${(got / 1048576).toFixed(1)} MB`;
  });
  txt.textContent = "Loading trajectories…"; bar.style.width = "100%";
  const trajs = await Promise.all(r.attempts.map((a) => getJson(`${base()}/attempt_${a.attempt}.traj.json?v=${state.bust}`)));
  r.attempts.forEach((a, i) => { state.traj[a.attempt] = trajs[i]; });
  const sets = [];
  if (r.stress_test) {
    const before = await getJson(`${base()}/stress_trails.json?v=${state.bust}`);
    if (before) sets.push({ label: "one photo", runs: before, successes: before.filter((x) => x.success).length });
    if (r.active_perception && r.active_perception.steps.length) {
      const after = await getJson(`${base()}/stress_trails_after.json?v=${state.bust}`);
      if (after) sets.push({ label: `after looking at ${r.active_perception.steps.map((s) => s.look_at).join(", ")}`, runs: after, successes: after.filter((x) => x.success).length });
    }
  }
  state.trailSets = sets;
  $("tglTrails").disabled = !sets.length;
  if (sets.length) { replay.setTrails(sets[0].runs); replay.showTrails(false); }
  loadAttempt(state.current, { play: false });
  loader.classList.add("done");
}

const otherOf = (e) => e.kind === "protective_stop" ? e.obj : (e.between.find((x) => x !== "robot_arm" && x !== e.while_holding) || e.between[0]);
function hitsOf(a) {
  return unexpectedContacts(a).map((e) => ({ t: e.t, obj: otherOf(e), text: e.kind === "protective_stop" ? `protective stop · ${esc(e.obj)}` : `hit ${esc(otherOf(e))}` }));
}
// Contacts the plan did not intend: the held object or a robot link touching something that is neither the
// destination nor the object being picked. Protective stops are always unexpected.
function unexpectedContacts(a) {
  const dests = new Set(a.plan.actions.map((x) => x.dest)), picks = new Set(a.plan.actions.map((x) => x.obj));
  return a.events.filter((e) => {
    if (e.kind === "protective_stop") return true;
    if (e.kind !== "contact" || e.between.includes("floor") || e.between.includes("table")) return false;
    const other = otherOf(e);
    if (e.while_holding) return !dests.has(other);
    return e.between.includes("robot_arm") && !dests.has(other) && !picks.has(other);
  });
}

function loadAttempt(i, { play = true } = {}) {
  const r = state.report; if (!r || !replay) return;
  state.current = i;
  const a = r.attempts[i], frames = state.traj[a.attempt];
  if (!frames) return;
  replay.setTrajectory(frames, a.events, hitsOf(a));
  const t0 = frames[0][0], dur = frames[frames.length - 1][0];
  $("scrub").min = t0; $("scrub").max = dur;
  const hitT = new Set(hitsOf(a).map((h) => h.t));
  $("ticks").innerHTML = a.events.filter((e) => e.kind !== "contact" || hitT.has(e.t)).map((e) => {
    const cls = e.kind === "contact" ? "hit" : e.kind;
    const label = e.kind === "contact" ? `collision ${e.between.join(" ↔ ")}` : `${e.kind} ${e.obj}`;
    return `<span class="tick ${cls}" data-t="${e.t}" title="${esc(label)} · ${e.t.toFixed(2)} s" style="left:${(100 * (e.t - t0) / (dur - t0)).toFixed(2)}%"></span>`;
  }).join("");
  $("ticks").querySelectorAll(".tick").forEach((el) => el.onclick = () => jumpTo(+el.dataset.t));
  $("attemptPills").innerHTML = r.attempts.map((x, j) =>
    `<button type="button" aria-selected="${j === i}" data-i="${j}"><span class="ic ${x.success ? "ok" : "bad"}">${x.success ? "✓" : "✕"}</span>attempt ${x.attempt}</button>`).join("");
  $("attemptPills").querySelectorAll("button").forEach((b) => b.onclick = () => { state.userPicked = true; loadAttempt(+b.dataset.i); markActiveAttempt(); });
  state.lastNow = -2; eventRows = [...document.querySelectorAll(`#att-${a.attempt} .events li[data-t]`)].map((el) => ({ el, t: +el.dataset.t }));
  setChip(`<span class="k">attempt ${a.attempt}</span> <span class="${a.success ? "ok" : "bad"}">${a.success ? "✓ success" : "✕ failed"}</span>`);
  replay.seek(t0);
  if (play) replay.play();
  syncPlay();
}

let eventRows = [];
function onTime(t) {
  $("scrub").value = t;
  $("timeLbl").textContent = `${t.toFixed(2)} s`;
  let now = -1;
  eventRows.forEach((r, i) => { if (r.t <= t + 1e-6) now = i; });
  if (now === state.lastNow) return;
  state.lastNow = now;
  eventRows.forEach((r, i) => { r.el.classList.toggle("future", i > now); r.el.classList.toggle("now", i === now); });
}
function onEnd() {
  syncPlay();
  if ((state.stage === "hero" || state.stage === "report") && !DEMO) setTimeout(() => { if (state.stage === "hero" || state.stage === "report") replay.play(); }, 1200);
}
function syncPlay() {
  $("playIcon").setAttribute("d", replay && replay.playing ? "M3 2.5h3.5v11H3zM9.5 2.5H13v11H9.5z" : "M4 2.5v11l9-5.5z");
}
function jumpTo(t, obj) {
  if (!replay) return;
  if (state.stage !== "rehearse" && !DEMO) document.getElementById("s3").scrollIntoView({ behavior: "smooth", block: "start" });
  replay.seek(Math.max(replay.t0, t - 0.7));
  replay.speed = 0.5; $("speed").value = "0.5";
  if (obj && replay.bodyByName[obj]) {
    const p = replay.bodyByName[obj].position;
    replay.flyTo({ pos: [p.x + 0.55, p.y + 0.45, 0.42], target: [p.x, p.y, 0.1] }, 1.1);
  }
  replay.play(); syncPlay();
}

function bindStageControls() {
  $("playBtn").onclick = () => { if (!replay) return; replay.playing ? replay.pause() : replay.play(); syncPlay(); };
  $("scrub").oninput = (e) => { if (!replay) return; replay.pause(); replay.seek(+e.target.value); syncPlay(); };
  $("speed").onchange = (e) => { if (replay) replay.speed = +e.target.value; };
  $("camReset").onclick = () => replay && replay.flyTo(state.stageView || "home", 1.2);
  $("tglRobot").onclick = (e) => toggle(e.currentTarget, (on) => replay && replay.showRobot(on));
  $("tglTrails").onclick = (e) => toggle(e.currentTarget, (on) => { if (!replay) return; replay.revealTrails(Infinity); replay.showTrails(on); $("legend3d").hidden = !on; if (on) renderTrailLegend(); });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) { e.preventDefault(); $("playBtn").click(); }
  });
}
function toggle(btn, fn) { const on = btn.getAttribute("aria-pressed") !== "true"; btn.setAttribute("aria-pressed", on); fn(on); }
function setChip(html) { $("stageChip").innerHTML = html; }
function setCaption(h, s) { $("caption").innerHTML = h ? `<div class="h">${h}</div>${s ? `<div class="s">${s}</div>` : ""}` : ""; }

function renderTrailLegend() {
  const sets = state.trailSets; if (!sets.length) return;
  const i = state.trailSet, runs = sets[i].runs, ok = sets[i].successes;
  $("legend3d").innerHTML = (sets.length > 1 ? `<div class="seg">${sets.map((s, j) =>
    `<button type="button" data-i="${j}" aria-pressed="${i === j}">${esc(s.label)} · ${s.successes}/${s.runs.length}</button>`).join("")}</div>` : "") +
    `<div>Paths of moved objects in ${runs.length} universes · <span class="ok">● ${ok} succeed</span> · <span class="bad">● ${runs.length - ok} fail</span></div>`;
  $("legend3d").querySelectorAll(".seg button").forEach((b) => b.onclick = () => showTrailSet(+b.dataset.i));
}
function showTrailSet(i, reveal = Infinity) {
  state.trailSet = i;
  replay.setTrails(state.trailSets[i].runs);
  replay.revealTrails(reveal);
  replay.showTrails(true);
  renderTrailLegend();
}
async function fanOut(i, ms = 2600) {
  const runs = state.trailSets[i].runs;
  showTrailSet(i, 0);
  const t0 = performance.now();
  const token = state.fanToken = Symbol();
  for (;;) {
    const u = Math.min(1, (performance.now() - t0) / ms);
    replay.revealTrails(Math.ceil(u * runs.length));
    if (u >= 1 || state.fanToken !== token) break;
    await new Promise(requestAnimationFrame);
  }
}

// ------------------------------------------------------------------ stage modes (driven by scroll)
function setStage(mode, force = false) {
  if (!replay || !state.report || (mode === state.stage && !force)) return;
  const prev = state.stage; state.stage = mode;
  const r = state.report, last = r.attempts.length - 1, a = r.attempts[state.current];
  const tl = $("timeline");
  const trailsBtn = $("tglTrails"), userTrails = trailsBtn.getAttribute("aria-pressed") === "true";
  // leave universes/look: restore robot + trails preference
  if (prev === "universes" || prev === "look") {
    state.fanToken = null;
    replay.showRobot($("tglRobot").getAttribute("aria-pressed") === "true");
    if (!userTrails) { replay.showTrails(false); $("legend3d").hidden = true; } else replay.revealTrails(Infinity);
  }
  replay.setFocus({});
  tl.classList.toggle("off", !(mode === "rehearse" || mode === "hero"));
  const fly = (v, d = 1.6) => { state.stageView = v; replay.flyTo(v, prev ? d : 0); };
  switch (mode) {
    case "hero":
    case "report": {
      if (state.current !== last) loadAttempt(last, { play: false });
      fly("home"); replay.orbit(true, 0.45);
      replay.speed = 1; $("speed").value = "1";
      replay.play(); syncPlay();
      setChip(`<span class="k">live</span> attempt ${r.attempts[last].attempt} <span class="${r.attempts[last].success ? "ok" : "bad"}">${r.attempts[last].success ? "✓" : "✕"}</span>`);
      setCaption("", "");
      break;
    }
    case "twin": {
      replay.pause(); replay.seek(replay.t0); syncPlay();
      fly("twin");
      const low = {};
      for (const o of r.scene_spec.objects) if (o.perception && o.perception.confidence === "low") low[o.id] = "warn";
      for (const [id, ok] of Object.entries(r.reachability)) if (!ok) low[id] = "bad";
      replay.setFocus(low);
      setChip(`<span class="k">twin</span> ${r.scene_spec.objects.length} objects · primitives`);
      setCaption("Digital twin, as the models saw it", `${Object.keys(low).length ? "amber = low-confidence perception · red = out of reach" : "all objects reachable"}`);
      break;
    }
    case "plan": {
      replay.pause(); replay.seek(replay.t0); syncPlay();
      fly("top");
      const f = {};
      for (const x of r.attempts[0].plan.actions) { f[x.obj] = "accent"; if (!f[x.dest]) f[x.dest] = "good"; }
      replay.setFocus(f);
      setChip(`<span class="k">plan</span> ${r.attempts[0].plan.actions.length} actions`);
      setCaption("Plan, before physics has a say", "blue = picked · green = destination");
      break;
    }
    case "rehearse": {
      fly("cinema");
      const idx = state.userPicked ? state.current : 0;
      loadAttempt(idx, { play: true });
      markActiveAttempt();
      setCaption("", "");
      break;
    }
    case "universes":
    case "look": {
      if (state.current !== last) { loadAttempt(last, { play: false }); markActiveAttempt(); }
      replay.pause(); replay.seek(replay.duration); replay.orbit(false); syncPlay();
      fly("universe");
      replay.showRobot(false);
      if (!state.trailSets.length) { setCaption("No stress test", "the final plan did not succeed"); break; }
      const i = mode === "look" && state.trailSets.length > 1 ? 1 : 0;
      fanOut(i);
      $("legend3d").hidden = false;
      const s = state.trailSets[i];
      if (mode === "look") replay.setFocus(Object.fromEntries(r.active_perception.steps.map((st) => [st.look_at, "accent"])));
      setChip(`<span class="k">${mode === "look" ? "look again" : "universes"}</span> ${s.successes}/${s.runs.length}`);
      setCaption(mode === "look" ? `Look again at ${r.active_perception.steps.map((st) => st.look_at).join(", ")}` : `${s.runs.length} parallel universes`,
        mode === "look" ? "same seeds, one object re-observed" : "paths of every moved object · red = a universe where the plan failed");
      break;
    }
  }
}

function focusFor(mode) {
  const r = state.report, f = {};
  if (!r) return f;
  if (mode === "twin") {
    for (const o of r.scene_spec.objects) if (o.perception && o.perception.confidence === "low") f[o.id] = "warn";
    for (const [id, ok] of Object.entries(r.reachability)) if (!ok) f[id] = "bad";
  } else if (mode === "plan") {
    for (const x of r.attempts[0].plan.actions) { f[x.obj] = "accent"; if (!f[x.dest]) f[x.dest] = "good"; }
  } else if (mode === "look" && r.active_perception) {
    for (const st of r.active_perception.steps) f[st.look_at] = "accent";
  }
  return f;
}

function setupObservers() {
  const sections = [...document.querySelectorAll("section[data-stage]")];
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      setStage(e.target.dataset.stage);
      document.querySelectorAll(".chapters a").forEach((a) => a.setAttribute("aria-current", a.dataset.nav === e.target.id));
    }
  }, { rootMargin: "-38% 0px -52% 0px", threshold: 0 });
  sections.forEach((s) => io.observe(s));
  // typewriter for the diagnosis when it scrolls into view
  const tw = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { typewrite(e.target); tw.unobserve(e.target); }
  }, { threshold: 0.4 });
  document.querySelectorAll("[data-typewrite]").forEach((el) => tw.observe(el));
}
async function typewrite(el) {
  const full = el.dataset.typewrite; el.innerHTML = `<span class="cursor"></span>`;
  for (let i = 1; i <= full.length; i++) { el.innerHTML = `${esc(full.slice(0, i))}<span class="cursor"></span>`; await sleep(full.charCodeAt(i - 1) > 255 ? 34 : 14); }
  el.textContent = full;
}

// ------------------------------------------------------------------ renderers
function stat(k, v, s) { return `<div><dt>${k}</dt><dd>${v}${s ? `<small>${s}</small>` : ""}</dd></div>`; }
function renderHeroStats() {
  const r = state.report, A = r.attempts, last = A[A.length - 1], st = r.stress_test, ap = r.active_perception;
  const reach = Object.values(r.reachability), nReach = reach.filter(Boolean).length;
  const after = ap && ap.steps.length ? ap.final.successes : null;
  $("heroStats").innerHTML =
    stat("Outcome", last.success ? `<span class="ok">✓</span> done` : `<span class="bad">✕</span> failed`, `${A.length} attempt${A.length > 1 ? "s" : ""} · ${A.length - 1} self-correction${A.length - 1 === 1 ? "" : "s"}`) +
    stat("Robustness", st ? (after != null ? `<span class="ok">${after}</span>/${st.n}` : `${st.successes}/${st.n}`) : "—", st ? (after != null ? `${st.successes}/${st.n} before looking again` : "randomised universes") : "no stress test") +
    stat("Reachable", `${nReach}/${reach.length}`, nReach < reach.length ? `${reach.length - nReach} outside workspace` : "all objects") +
    stat("Wall time", `${Math.round(r.wall_time_s)} s`, `agent: ${esc(r.agent)}`);
}

function renderInventory() {
  const r = state.report, objs = r.scene_spec.objects, hasPerc = objs.some((o) => o.perception);
  $("inv").innerHTML = `<tr><th>Object</th><th>Size</th><th>Mass</th><th>μ</th>${hasPerc ? "<th>Conf.</th>" : ""}<th>Reach</th></tr>` + objs.map((o) => {
    const ok = r.reachability[o.id], low = o.perception && o.perception.confidence === "low";
    return `<tr data-id="${esc(o.id)}" class="${low ? "low" : ""}"><td><span class="sw" style="background:${col(o.rgba)}"></span><span class="lbl">${esc(o.label)}</span><br><span class="id">${esc(o.id)}</span></td>
      <td class="num">${dims(o)}</td><td class="num">${fmtRange(o.mass_kg.range, " kg")}</td><td class="num">${fmtRange(o.friction.range)}</td>
      ${hasPerc ? `<td>${o.perception ? `<span class="badge ${low ? "warn" : "ok"}">${low ? "low" : "high"}${o.perception.sigma_m ? ` · σ${(o.perception.sigma_m * 100).toFixed(1)}` : ""}</span>` : "—"}</td>` : ""}
      <td><span class="badge ${ok ? "ok" : "bad"}">${ok ? "✓ IK" : "✕ out"}</span></td></tr>`;
  }).join("");
  $("inv").querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.onmouseenter = () => replay && replay.setFocus({ [tr.dataset.id]: "accent" });
    tr.onmouseleave = () => replay && replay.setFocus(focusFor(state.stage));
    tr.onclick = () => { if (!replay) return; const p = replay.bodyByName[tr.dataset.id]?.position; if (p) replay.flyTo({ pos: [p.x + 0.5, p.y + 0.4, 0.4], target: [p.x, p.y, 0.08] }, 1); };
  });
}

async function renderPhoto() {
  const r = state.report, spec = r.scene_spec, img = $("photoImg"), svg = $("photoBoxes"), empty = $("photoEmpty");
  let W, H, boxes = {}, synthetic = false;
  img.hidden = true; empty.hidden = false; svg.innerHTML = ""; img.classList.remove("synthetic");
  img.onload = () => { img.hidden = false; empty.hidden = true; };
  if (spec.photo && spec.photo.file) {
    img.src = state.perceivedPhoto || `/${spec.photo.file.replace(/^\/+/, "")}`;
    W = spec.photo.width; H = spec.photo.height;
    for (const o of spec.objects) if (o.perception && o.perception.box_px) boxes[o.id] = o.perception.box_px;
    $("photoCap").innerHTML = `<span class="badge model sam">SAM 3.1 boxes</span> ${esc(spec.photo.file)} · ${W}×${H}`;
  } else {
    // Hand-authored scene: prefer the perception test-set render of it (photos/synth), with the twin's
    // boxes projected through that render's known camera; else render a frame from the twin itself.
    const synthUrl = `/photos/synth/${encodeURIComponent(spec.name)}__front55`;
    const gt = await getJson(`${synthUrl}.gt.json`);
    if (gt && gt.camera) {
      img.src = `${synthUrl}.jpg`; W = 1280; H = 960; synthetic = true;
      boxes = projectBoxes(spec, gt.camera, W, H);
      $("photoCap").innerHTML = `<span class="badge warn">synthetic photo</span> <code>${esc(spec.name)}</code> is a hand-authored scene; this is its render from the perception test set, with the twin's boxes projected through the known camera. A real photo gets SAM 3.1 masks instead.`;
    } else {
      if (!replay) { empty.innerHTML = "No photo for this scene."; return; }
      await state.loaded;
      const snap = replay.snapshotFrom({ pos: [1.25, 0.62, 0.68], target: [0.6, -0.02, 0.03], fov: 50 }, 1280, 800);
      img.src = snap.dataUrl; W = snap.width; H = snap.height; boxes = snap.boxes; synthetic = true;
      img.classList.add("synthetic");
      $("photoCap").innerHTML = `<span class="badge warn">stand-in photo</span> <code>${esc(spec.name)}</code> is a hand-authored scene, so this frame is rendered from the twin. Boxes show what SAM 3.1 returns for a real photo.`;
    }
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const k = W / 640;
  svg.innerHTML = spec.objects.map((o) => {
    const b = boxes[o.id]; if (!b) return "";
    const [x1, y1, x2, y2] = b, low = o.perception && o.perception.confidence === "low", w = x2 - x1, h = y2 - y1, c = Math.min(14 * k, w / 3, h / 3);
    const tag = low ? `low${o.perception.sigma_m ? ` · σ ${(o.perception.sigma_m * 100).toFixed(1)} cm` : ""}` : "";
    return `<g class="${low ? "low" : ""}" data-id="${esc(o.id)}"><rect x="${x1}" y="${y1}" width="${w}" height="${h}" rx="${3 * k}"/>
      <path class="corner" d="M${x1} ${y1 + c}V${y1}H${x1 + c}M${x2 - c} ${y1}H${x2}V${y1 + c}M${x2} ${y2 - c}V${y2}H${x2 - c}M${x1 + c} ${y2}H${x1}V${y2 - c}"/>
      <text x="${x1 + 4 * k}" y="${y1 - 6 * k}" style="font-size:${12 * k}px">${esc(o.category || o.label)}</text>
      ${tag ? `<text class="tag" x="${x1 + 4 * k}" y="${y2 + 14 * k}" style="font-size:${10.5 * k}px">${esc(tag)}</text>` : ""}</g>`;
  }).join("");
  svg.querySelectorAll("g[data-id]").forEach((g) => {
    g.onmouseenter = () => { replay && replay.setFocus({ [g.dataset.id]: "accent" }); $("inv").querySelector(`tr[data-id="${g.dataset.id}"]`)?.classList.add("sel"); };
    g.onmouseleave = () => { replay && replay.setFocus(focusFor(state.stage)); $("inv").querySelector(`tr[data-id="${g.dataset.id}"]`)?.classList.remove("sel"); };
  });
  if (synthetic && !$("photoFrame").querySelector(".grain")) { const gr = document.createElement("div"); gr.className = "grain"; $("photoFrame").insertBefore(gr, svg); }
}

// Project each object's bounding box (table frame, z-up) through a pinhole camera {K, R, t} -> [x1,y1,x2,y2] px.
function projectBoxes(spec, cam, W, H) {
  const { K, R, t } = cam, out = {};
  const proj = ([x, y, z]) => {
    const c = [0, 1, 2].map((i) => R[i][0] * x + R[i][1] * y + R[i][2] * z + t[i]);
    if (c[2] <= 1e-6) return null;
    return [(K[0][0] * c[0] + K[0][2] * c[2]) / c[2], (K[1][1] * c[1] + K[1][2] * c[2]) / c[2]];
  };
  for (const o of spec.objects) {
    const s = o.shape, [cx, cy] = o.pos, yaw = o.yaw || 0, pts = [];
    if (s.type === "box") {
      const [sx, sy, sz] = s.size, cs = Math.cos(yaw), sn = Math.sin(yaw);
      for (const dx of [-sx / 2, sx / 2]) for (const dy of [-sy / 2, sy / 2]) for (const z of [0, sz]) pts.push([cx + dx * cs - dy * sn, cy + dx * sn + dy * cs, z]);
    } else {
      for (let k = 0; k < 12; k++) { const a = (k / 12) * Math.PI * 2; for (const z of [0, s.height]) pts.push([cx + s.radius * Math.cos(a), cy + s.radius * Math.sin(a), z]); }
    }
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const p of pts) { const q = proj(p); if (!q) continue; x1 = Math.min(x1, q[0]); y1 = Math.min(y1, q[1]); x2 = Math.max(x2, q[0]); y2 = Math.max(y2, q[1]); }
    if (x2 > 0 && y2 > 0 && x1 < W && y1 < H) out[o.id] = [Math.max(0, x1), Math.max(0, y1), Math.min(W, x2), Math.min(H, y2)].map(Math.round);
  }
  return out;
}

function goalHtml(g, showStatus = true) {
  const ic = !showStatus ? `<span class="ic na">○</span>` : `<span class="ic ${g.ok ? "ok" : "bad"}">${g.ok ? "✓" : "✕"}</span>`;
  return `<li>${ic}<span><code>${esc(g.pred)}(${g.args.map(esc).join(", ")})</code><span class="imp">${PRED[g.pred] || ""}${g.implicit ? " · implicit" : ""}</span>
    ${showStatus && g.why ? `<div class="why">${esc(g.why)}</div>` : ""}</span></li>`;
}
const viaTxt = (v) => (v || []).map((p) => `(${p.map((n) => n.toFixed(2)).join(", ")})`).join(" → ");
function planHtml(a, prev) {
  return a.plan.actions.map((x, i) => {
    const p = prev && prev.plan.actions[i];
    const reordered = prev && p && (p.obj !== x.obj || p.dest !== x.dest);
    let chg = "";
    if (reordered) chg += ` <span class="chg">reordered</span>`;
    else if (p && p.transit_height !== x.transit_height) chg += ` <span class="chg">transit <s>${p.transit_height}</s> → ${x.transit_height} m</span>`;
    if (p && !reordered && viaTxt(p.via) !== viaTxt(x.via)) chg += ` <span class="chg">${x.via && x.via.length ? "detour added" : "detour removed"}</span>`;
    const via = x.via && x.via.length ? ` <span class="via">via ${esc(viaTxt(x.via))}</span>` : "";
    return `<li><span>pick <code>${esc(x.obj)}</code> ${REL[x.relation] || x.relation} <code>${esc(x.dest)}</code><span class="meta"> · carry at ${x.transit_height} m</span>${via}${chg}</span></li>`;
  }).join("");
}
function planChanges(a, prev) {
  if (!prev) return [];
  const out = [];
  const sameOrder = a.plan.actions.every((x, j) => prev.plan.actions[j] && prev.plan.actions[j].obj === x.obj && prev.plan.actions[j].dest === x.dest);
  if (!sameOrder) out.push(`actions reordered: ${a.plan.actions.map((x) => `<b>${esc(x.obj)}</b>`).join(" → ")}`);
  a.plan.actions.forEach((x, j) => {
    const p = prev.plan.actions[j]; if (!p || !sameOrder) return;
    if (p.transit_height !== x.transit_height) out.push(`<code>${esc(x.obj)}</code> transit_height <b><s>${p.transit_height}</s> → ${x.transit_height} m</b>`);
    if (viaTxt(p.via) !== viaTxt(x.via)) out.push(`<code>${esc(x.obj)}</code> ${x.via && x.via.length ? `detour via <b>${esc(viaTxt(x.via))}</b>` : "straight carry"}`);
  });
  return out;
}
function renderGoalsPlan() {
  const r = state.report;
  $("goals0").innerHTML = r.goals.map((g) => goalHtml(g, false)).join("");
  $("plan0").innerHTML = planHtml(r.attempts[0]);
  const un = Object.entries(r.reachability).filter(([, ok]) => !ok).map(([k]) => k);
  $("reach").innerHTML = `<span class="badge phys">MuJoCo · IK</span> reachability ${Object.keys(r.reachability).length - un.length}/${Object.keys(r.reachability).length}` +
    (un.length ? ` · ${un.map((k) => `<code>${esc(k)}</code>`).join(", ")} outside the workspace` : " · all objects reachable");
  $("nUni").textContent = r.stress_test ? r.stress_test.n : $("stress").value;
}

function renderAttempts() {
  const r = state.report;
  let html = "";
  r.attempts.forEach((a, i) => {
    const prev = r.attempts[i - 1];
    if (a.plan.diagnosis) {
      const gloss = GLOSS[a.plan.diagnosis];
      const ev = unexpectedContacts(prev || a);
      const changes = planChanges(a, prev);
      html += `<div class="att diag"><div class="node">◆</div><div class="box diag" tabindex="0">
        <div class="head"><h3>Diagnosis</h3><span class="badge model spark">Muse Spark reads the sim log</span></div>
        <blockquote data-typewrite="${esc(a.plan.diagnosis)}">${esc(a.plan.diagnosis)}</blockquote>
        ${gloss ? `<p class="note" style="margin-top:6px">${esc(gloss)}</p>` : ""}
        <div class="evidence">${ev.map((e) => `<code data-t="${e.t}" data-obj="${esc(otherOf(e))}">t=${e.t.toFixed(3)}s ${e.kind === "protective_stop" ? `protective stop · ${esc(e.link_role || "robot")} pressed ${esc(e.obj)} with ${Math.round(e.force_n || 0)} N` : `${esc(e.between.join(" ↔ "))}${e.link_role ? ` (${esc(e.link_role)})` : ""}${e.while_holding ? ` · holding ${esc(e.while_holding)}` : ""}${e.tcp_z != null ? ` · tcp_z ${e.tcp_z} m` : ""}`}</code>`).join("")}${prev && prev.execution_error ? `<code data-t="${prev.events.length ? prev.events[prev.events.length - 1].t : 0}">${esc(prev.execution_error.error)}</code>` : ""}</div>
        ${changes.length ? `<div class="fix">Fix: ${changes.join(" · ")}</div>` : ""}
      </div></div>`;
    }
    const hitSet = new Set(unexpectedContacts(a));
    const ev = a.events.filter((e) => e.kind !== "contact" || !e.between.includes("table") || e.between.includes("robot_arm"));
    html += `<div class="att ${a.success ? "ok" : "bad"}" id="att-${a.attempt}" data-i="${i}"><div class="node">${a.attempt}</div><div class="box" tabindex="0" role="button" aria-label="Replay attempt ${a.attempt}">
      <div class="head"><h3>Attempt ${a.attempt}</h3><span class="badge ${a.success ? "ok" : "bad"}">${a.success ? "✓ all goals met" : `✕ ${a.goals.filter((g) => !g.ok).length} goal${a.goals.filter((g) => !g.ok).length > 1 ? "s" : ""} failed`}</span><span class="badge phys">MuJoCo</span><span class="t">${a.sim_time ? `${a.sim_time.toFixed(1)} s sim` : ""}</span></div>
      <div class="cols"><div><p class="sub">Goals · judged by physics</p><ul class="goals">${a.goals.map((g) => goalHtml(g)).join("")}</ul>
        <p class="sub" style="margin-top:12px">Plan</p><ol class="plan">${planHtml(a, prev)}</ol></div>
      <div><p class="sub">Sim log · click to replay</p><ul class="events">${ev.map((e) => {
          let text, hit = false, obj = "", cls = "";
          if (e.kind === "contact") { const [p, q] = e.between; hit = hitSet.has(e); obj = otherOf(e);
            text = `<code>${esc(p)}</code> ↔ <code>${esc(q)}</code>${e.link_role ? ` <span class="role">${esc(e.link_role)}</span>` : ""}${e.while_holding ? ` · holding <code>${esc(e.while_holding)}</code>` : ""}${e.tcp_z != null ? ` · z ${e.tcp_z}` : ""}`; }
          else if (e.kind === "protective_stop") { hit = true; cls = "stop"; obj = e.obj;
            text = `<b>protective stop</b> · ${esc(e.link_role || "robot")} pressed <code>${esc(e.obj)}</code> with ${Math.round(e.force_n || 0)} N${e.phase ? ` during “${esc(e.phase)}”` : ""}`; }
          else if (e.kind === "grasp") { text = `grasp <code>${esc(e.obj)}</code>`; obj = e.obj; }
          else if (e.kind === "release") { text = `release <code>${esc(e.obj)}</code>`; obj = e.obj; }
          else text = `${esc(e.kind)} ${esc(e.msg || e.obj || "")}`;
          return `<li class="${hit ? "hit" : ""} ${cls}" data-t="${e.t}" data-obj="${esc(obj)}"><span class="t">${e.t.toFixed(2)}s</span><span>${text}</span></li>`;
        }).join("")}</ul>${a.execution_error ? `<div class="exec"><b>execution error</b> · ${esc(a.execution_error.error)}</div>` : ""}</div></div></div></div>`;
  });
  $("attempts").innerHTML = html;
  $("attempts").querySelectorAll(".att[data-i] .box").forEach((box) => {
    const i = +box.parentElement.dataset.i;
    const go = () => { state.userPicked = true; loadAttempt(i); markActiveAttempt(); if (state.stage !== "rehearse") setStage("rehearse"); };
    box.onclick = (e) => { if (!e.target.closest("li[data-t]")) go(); };
    box.onkeydown = (e) => { if (e.key === "Enter") go(); };
  });
  $("attempts").querySelectorAll("li[data-t], code[data-t]").forEach((el) => el.onclick = (e) => {
    e.stopPropagation();
    const att = el.closest(".att[data-i]"); if (att && +att.dataset.i !== state.current) { loadAttempt(+att.dataset.i, { play: false }); markActiveAttempt(); }
    if (!att) { const prevI = Math.max(0, state.report.attempts.findIndex((a) => a.plan.diagnosis) - 1); if (state.current !== prevI) { loadAttempt(prevI, { play: false }); markActiveAttempt(); } }
    jumpTo(+el.dataset.t, el.dataset.obj);
  });
  markActiveAttempt();
}
function markActiveAttempt() {
  document.querySelectorAll(".att[data-i]").forEach((el) => el.classList.toggle("active", +el.dataset.i === state.current));
  document.querySelectorAll("#attemptPills button").forEach((b) => b.setAttribute("aria-selected", +b.dataset.i === state.current));
}

function renderStress() {
  const r = state.report, st = r.stress_test, objs = r.scene_spec.objects;
  if (!st) { $("stressStats").innerHTML = ""; $("scatter").innerHTML = `<p class="note">The final plan did not succeed, so no stress test was run.</p>`; $("fails").innerHTML = ""; return; }
  const runs = st.runs, fails = runs.filter((x) => !x.success), oks = runs.filter((x) => x.success);
  const ids = objs.map((o) => o.id);
  const mean = (rs, id) => rs.reduce((s, x) => s + x.perception_error_cm[id], 0) / Math.max(rs.length, 1);
  const best = fails.length ? ids.slice().sort((a, b) => (mean(fails, b) - mean(oks, b)) - (mean(fails, a) - mean(oks, a)))[0] : ids[0];
  const sig = Object.values(st.sigma_cm || {})[0];
  $("stressStats").innerHTML = stat("Succeeded", `<span class="${st.successes === st.n ? "ok" : ""}">${st.successes}</span>/${st.n}`, `${Math.round(100 * st.successes / st.n)}% robust`) +
    stat("Perception σ", sig != null ? `${sig.toFixed(1)} cm` : "—", "per object, per universe") +
    stat("Most sensitive", `<code>${esc(best)}</code>`, fails.length ? `fails avg ${mean(fails, best).toFixed(1)} cm vs ${mean(oks, best).toFixed(1)} cm` : "no failures") +
    stat("Failures", `<span class="${fails.length ? "bad" : "ok"}">${fails.length}</span>`, fails.length ? "see table" : "none");
  const sel = $("xobj");
  sel.innerHTML = ids.map((id) => `<option ${id === best ? "selected" : ""}>${esc(id)}</option>`).join("");
  sel.onchange = () => { drawScatter(runs, sel.value); failsTable(fails, sel.value); };
  drawScatter(runs, sel.value); failsTable(fails, sel.value);
}
function failsTable(fails, id) {
  $("fails").innerHTML = fails.length ? `<tr><th>Seed</th><th>${esc(id)} err</th><th>μ</th><th>Why</th></tr>` +
    fails.map((x) => `<tr><td class="seed">${x.seed}</td><td class="num">${x.perception_error_cm[id].toFixed(1)} cm</td><td class="num">×${x.friction_scale.toFixed(2)}</td><td class="reason">${x.failed.map(esc).join("<br>")}</td></tr>`).join("")
    : `<tr><td class="note">All universes succeeded.</td></tr>`;
}
function drawScatter(runs, id) {
  const W = 560, H = 300, m = { l: 46, r: 12, t: 14, b: 38 };
  const xmax = Math.max(4, Math.ceil(Math.max(...runs.map((x) => x.perception_error_cm[id]))));
  const X = (v) => m.l + (v / xmax) * (W - m.l - m.r), Y = (v) => m.t + (1 - (v - 0.6) / 0.8) * (H - m.t - m.b);
  let g = "";
  for (let v = 0; v <= xmax; v++) g += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--border)"/><text x="${X(v)}" y="${H - m.b + 16}" text-anchor="middle">${v}</text>`;
  for (const v of [0.6, 0.8, 1.0, 1.2, 1.4]) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--border)"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">×${v.toFixed(1)}</text>`;
  g += `<text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">perception error of ${esc(id)} (cm)</text>`;
  g += `<text transform="translate(11 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">friction scale</text>`;
  const failErr = runs.filter((x) => !x.success).map((x) => x.perception_error_cm[id]);
  if (failErr.length) {
    const mn = Math.min(...failErr);
    g += `<line x1="${X(mn)}" x2="${X(mn)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--bad)" stroke-dasharray="4 4"/><text x="${X(mn) + 6}" y="${m.t + 10}" style="fill:var(--bad)">${mn.toFixed(1)} cm</text>`;
  }
  for (const x of runs) {
    const px = X(x.perception_error_cm[id]), py = Y(x.friction_scale);
    const t = esc(`<b>seed ${x.seed} · ${x.success ? "✓ success" : "✕ failure"}</b>${id} error ${x.perception_error_cm[id].toFixed(1)} cm<br>friction ×${x.friction_scale.toFixed(2)} · mass ×${x.mass_scale.toFixed(2)}` + (x.failed.length ? "<br>" + x.failed.map(esc).join("<br>") : ""));
    g += `<g data-tip="${t}"><circle cx="${px}" cy="${py}" r="10" fill="transparent"/>` + (x.success
      ? `<circle cx="${px}" cy="${py}" r="4.5" fill="var(--good)" stroke="var(--surface)" stroke-width="1.5"/>`
      : `<path d="M${px - 5} ${py - 5}L${px + 5} ${py + 5}M${px + 5} ${py - 5}L${px - 5} ${py + 5}" stroke="var(--bad)" stroke-width="2.5" stroke-linecap="round"/>`) + `</g>`;
  }
  $("scatter").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Stress test: perception error vs friction, coloured by outcome">${g}</svg>`;
  bindTips($("scatter"));
}

function renderLook() {
  const r = state.report, ap = r.active_perception, st = r.stress_test;
  $("s5").hidden = !ap;
  if (!ap) return;
  $("sig0").textContent = ap.sigma_cm.toFixed(1); $("sig1").textContent = ap.refined_sigma_cm.toFixed(1);
  const n = ap.n;
  if (!ap.steps.length) {
    $("lookBars").innerHTML = `<p class="note">${st && st.successes === n ? "Every universe already succeeds; nothing to look at again." : "Re-observing any single object does not raise the success rate: the bottleneck is not perception."}</p>`;
    $("lookChain").innerHTML = ""; $("lookNote").textContent = ""; return;
  }
  const first = ap.steps[0], vals = first.value_by_object;
  const ranked = Object.keys(vals).sort((a, b) => vals[b] - vals[a]);
  const W = 560, rowH = 30, m = { l: 104, r: 70, t: 6 }, H = m.t + ranked.length * rowH + 6;
  const maxGain = Math.max(1, ...ranked.map((k) => vals[k] - first.before));
  const X = (v) => m.l + (v / maxGain) * (W - m.l - m.r);
  let g = `<line x1="${X(0)}" x2="${X(0)}" y1="${m.t - 2}" y2="${H - 4}" stroke="var(--border-strong)"/>`;
  ranked.forEach((k, i) => {
    const gain = vals[k] - first.before, y = m.t + i * rowH, chosen = k === first.look_at;
    g += `<g data-tip="${esc(`<b>look again at ${k}</b>${first.before}/${n} → ${vals[k]}/${n}`)}"><rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>
      <text x="${m.l - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" style="${chosen ? "fill:var(--text);font-weight:600" : ""}">${esc(k)}</text>`;
    if (gain > 0) { const w = X(gain) - X(0); g += `<rect x="${X(0)}" y="${y + 7}" width="${w}" height="${rowH - 14}" rx="4" fill="${chosen ? "var(--accent)" : "var(--muted)"}"/>`; }
    g += `<text x="${X(Math.max(gain, 0)) + 8}" y="${y + rowH / 2 + 4}" style="fill:var(--text-2)">${gain > 0 ? "+" + gain : "0"} · ${vals[k]}/${n}</text></g>`;
  });
  $("lookBars").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Value of re-observing each object">${g}</svg>`;
  bindTips($("lookBars"));
  let chain = `<div class="step"><div class="lbl">One photo · every object σ = ${ap.sigma_cm.toFixed(1)} cm</div><div class="big">${first.before}/${n}</div></div>`;
  ap.steps.forEach((s, i) => {
    chain += `<div class="arrow">↓ look again at <code>${esc(s.look_at)}</code> <span class="badge model sam">SAM 3.1 re-localises</span></div>
      <div class="step ${i === ap.steps.length - 1 ? "final" : ""}"><div class="lbl"><code>${esc(s.look_at)}</code> σ = ${ap.refined_sigma_cm.toFixed(1)} cm, same seeds</div><div class="big">${s.after}/${n}</div></div>`;
  });
  $("lookChain").innerHTML = chain;
  const useless = ranked.filter((k) => vals[k] <= first.before);
  $("lookNote").innerHTML = (useless.length ? `Re-observing ${useless.map((k) => `<code>${esc(k)}</code>`).join(", ")} would not help. ` : "") +
    "In this build, a second look is modelled as a smaller σ for that object; with live perception it becomes a prompt to photograph it from another angle.";
}

async function renderGallery() {
  let items = await getJson("/api/gallery");
  if (!items) items = ((await getJson("/api/scenes")) || []).map((s) => ({ scene: s, title: s, thumb: null, has_report: true }));
  state.gallery = items;
  $("gallery").innerHTML = items.map((it) => `<button type="button" class="gal ${it.scene === state.scene ? "sel" : ""}" data-scene="${esc(it.scene)}">
    <div class="th">${it.thumb ? `<img src="${esc(it.thumb)}" alt="">` : `<span>${it.n_objects ? `${it.n_objects} objects` : "twin"}</span>`}</div>
    <div class="cap"><span><b>${esc(it.scene)}</b>${it.title && it.title !== it.scene ? `<br><span class="note" style="margin:0">${esc(it.title)}</span>` : ""}</span>${it.has_report ? `<span class="badge ok">report</span>` : `<span class="badge warn">no run</span>`}</div></button>`).join("");
  $("gallery").querySelectorAll(".gal").forEach((b) => b.onclick = () => { $("scene").value = b.dataset.scene; state.scene = b.dataset.scene; loadScene(); renderGallery(); scrollTo({ top: 0, behavior: "smooth" }); });
}

// ------------------------------------------------------------------ jobs (perceive / rehearse)
function jobBox(el) {
  el.hidden = false;
  el.innerHTML = `<div class="phase"><span class="badge model spark jp">queued</span><div class="pbar"><i></i></div><span class="mono jf">0%</span></div><pre></pre>`;
  return {
    update(j) {
      const ph = j.progress ? j.progress.phase : j.state, f = j.progress ? j.progress.fraction : (j.state === "done" ? 1 : 0);
      el.querySelector(".jp").textContent = ph + (j.queue_ahead ? ` · ${j.queue_ahead} ahead` : "");
      el.querySelector(".pbar i").style.width = `${Math.round(f * 100)}%`;
      el.querySelector(".jf").textContent = `${Math.round(f * 100)}%`;
      const pre = el.querySelector("pre"); pre.textContent = (j.log || []).join("\n") + (j.state === "running" ? "\n▍" : ""); pre.scrollTop = pre.scrollHeight;
      if (j.error) pre.innerHTML += `\n<span class="err">${esc(j.error)}</span>`;
    },
  };
}
async function pollJob(id, box) {
  for (;;) {
    const j = await getJson(`/api/job?id=${encodeURIComponent(id)}`);
    if (!j) throw new Error("job vanished");
    box.update(j);
    if (j.state === "done") return j.result;
    if (j.state === "error") throw new Error(j.error || "job failed");
    await sleep(500);
  }
}
async function runRehearsal(e) {
  e.preventDefault();
  const btn = $("run"); btn.disabled = true; btn.textContent = "Rehearsing…";
  const box = jobBox($("rehearseJob"));
  try {
    const res = await fetch("/api/rehearse", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene: state.scene, instruction: $("instruction").value, stress: +$("stress").value, look: +$("look").value }) });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.statusText);
    if (j.job) await pollJob(j.job, box);
    else for (;;) { // legacy /api/status
      const st = await getJson("/api/status"); box.update({ state: st.running ? "running" : (st.error ? "error" : "done"), log: st.log, error: st.error, progress: st.progress });
      if (!st.running) { if (st.error) throw new Error(st.error); break; }
      await sleep(600);
    }
    state.bust = Date.now();
    await loadScene();
    $("rehearseJob").hidden = true;
  } catch (err) { box.update({ state: "error", error: String(err.message || err), log: [] }); }
  btn.disabled = false; btn.textContent = "Rehearse";
}
function bindUpload() {
  const drop = $("drop"), file = $("file");
  drop.onclick = () => file.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } };
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) perceive(e.dataTransfer.files[0]); };
  file.onchange = () => file.files[0] && perceive(file.files[0]);
}
async function perceive(f) {
  const box = jobBox($("perceiveJob"));
  try {
    const fd = new FormData(); fd.append("photo", f); fd.append("privacy", "private");
    const res = await fetch("/api/perceive", { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.statusText);
    const result = await pollJob(j.job, box);
    state.perceivedPhoto = result.overlay || j.upload || URL.createObjectURL(f);
    if (result.scene_spec && result.scene_spec.photo) {
      const r = state.report; r.scene_spec = result.scene_spec; renderInventory(); renderPhoto();
    }
    if (result.scene && result.scene !== state.scene && [...$("scene").options].some((o) => o.value === result.scene)) { $("scene").value = result.scene; state.scene = result.scene; await loadScene(); }
  } catch (err) { box.update({ state: "error", error: String(err.message || err), log: [] }); }
}

// ------------------------------------------------------------------ public surface for demo.js
const app = {
  state, replay, $, esc, GLOSS, sleep,
  loadAttempt, setChip, setCaption, showTrailSet, fanOut, unexpectedContacts,
  ready: () => state.loaded,
};
window.app = app;
init();
