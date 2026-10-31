// Rehearsal web app: scroll-driven narrative on the left, one persistent 3D stage on the right.
import { Replay, shapeBBox } from "./replay.js";

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
const isRound = (s) => (s.type === "cylinder" || s.type === "bowl") && (s.axis || "z") === "z";
const dims = (o) => isRound(o.shape) ? `⌀${Math.round(o.shape.radius * 200)}×${Math.round(o.shape.height * 100)} cm`
  : `${shapeBBox(o.shape).map((v) => Math.round(v * 100)).join("×")} cm${o.shape.type === "compound" ? ` · ${o.shape.parts.length} parts` : o.shape.axis ? " · lying" : ""}`;
// English gloss for the offline agent's Chinese diagnosis (kept verbatim above it).
const GLOSS = {
  "t=3.026s 时，吸盘在 0.139 m 高度搬运 mug_1，撞到了 bottle_1（高 0.26 m）。mug_1 挂在吸盘下方 0.1 m，所以搬运高度必须超过 0.26 + 0.1 m，改为 0.42 m。":
    "At t = 3.026 s the suction cup was carrying mug_1 at 0.139 m and hit bottle_1 (0.26 m tall). mug_1 hangs 0.10 m below the cup, so the transit height must clear 0.26 + 0.10 m → set it to 0.42 m.",
};

// ------------------------------------------------------------------ plain language
// Human titles and one-line stakes for the precomputed tables (ids stay in technical details).
const SCENE_META = {
  real_ycbv59: { title: "A real photo: the soup can onto the cracker box", stakes: "From a lab dataset photo. The plan passed once, then luck ran out.", real: true },
  kitchen_01: { title: "The kettle in the way", stakes: "Lowering the cup, the hand presses on the kettle. The model changes the order." },
  lamp_01: { title: "A lamp in the way", stakes: "The wrist sweeps the lamp over. Two fixes before it works." },
  windowsill_01: { title: "A vase and a far tray", stakes: "Too far to reach in one go; the arm has to relay the mug." },
  storage_01: { title: "Stacking boxes", stakes: "The stack the robot builds becomes its own obstacle." },
  desk_02: { title: "Pen cup and stapler", stakes: "The stapler sits at the edge of what the arm can reach." },
  desk_01: { title: "Mug into bowl, books stacked", stakes: "Worked first time. Tested 40 times anyway." },
};
const SCENE_ORDER = Object.keys(SCENE_META);
const sceneTitle = (name) => (SCENE_META[name] && SCENE_META[name].title) || ((state.gallery || []).find((g) => g.scene === name) || {}).title || name;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const cm = (m) => `${Math.round(m * 100)} cm`;
const ORD = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"];
const tryName = (i) => `${ORD[i] || `${i + 1}th`} try`;
function nameOf(id) {
  const spec = state.spec || (state.report && state.report.scene_spec);
  const o = spec && spec.objects.find((x) => x.id === id);
  return o ? (o.label || id) : String(id).replace(/_\d+$/, "").replace(/_/g, " ");
}
const the = (id) => `the ${nameOf(id)}`;
const list = (xs) => xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
function goalPlain(g) {
  const [a, b] = g.args;
  switch (g.pred) {
    case "on": return `${cap(the(a))} ends up on ${the(b)}`;
    case "inside": return `${cap(the(a))} ends up inside ${the(b)}`;
    case "stacked": return `${cap(list(g.args.map(the)))} end up stacked`;
    case "upright": return `${cap(the(a))} stays upright`;
    case "stable": return `${cap(the(a))} comes to rest`;
    case "undisturbed": return `Nothing else is disturbed (${g.args.map(nameOf).join(", ")})`;
    default: return `${g.pred}(${g.args.join(", ")})`;
  }
}
// "gap -1.0 cm, xy offset 36.1 cm" -> "it ended up 36 cm away"
function whyPlain(g) {
  const w = g.why || "";
  const off = w.match(/xy offset ([\d.]+) cm/);
  if (off && +off[1] >= 2) return `it ended up ${Math.round(+off[1])} cm away`;
  const gap = w.match(/gap (-?[\d.]+) cm/);
  if (gap && +gap[1] > 1) return `it was left ${Math.round(+gap[1])} cm above`;
  return w === "ok" ? "" : w;
}
const idRe = /\b([a-z]+(?:_[a-z]+)*_\d+)\b/g;
function phasePlain(p) {
  if (!p) return "";
  return p.replace(/^lower\b/, "lowering").replace(/^carry\b/, "carrying").replace(/^approach\b/, "approaching").replace(/^lift\b/, "lifting")
    .replace(/^place\b/, "placing").replace(/^grasp\b/, "grasping").replace(/ via waypoint \d+ \[[^\]]*\]/, " along its detour").replace(/ on /, " onto ").replace(/ to /, " toward ")
    .replace(idRe, (m) => the(m));
}
function actionPlain(x) {
  return `Pick up ${the(x.obj)}, carry it ${cm(x.transit_height)} above the table${x.via && x.via.length ? ", taking a detour around the clutter" : ""}, and set it ${x.relation === "in" ? "inside" : "on"} ${the(x.dest)}.`;
}
// One plain sentence for why an attempt failed, from the sim events.
function failurePlain(a) {
  const ps = a.events.find((e) => e.kind === "protective_stop");
  if (ps) return `The arm stopped itself: its ${ps.link_role || "hand"} pressed on ${the(ps.obj)} with ${Math.round(ps.force_n || 0)} N while ${phasePlain(ps.phase)}.`;
  const err = a.execution_error;
  if (a.events.some((e) => e.kind === "ik_failure") || (err && /IK/i.test(err.error))) {
    const act = err && err.action;
    return `The arm could not reach that far: ${act ? `carrying ${the(act.obj)} ${cm(act.transit_height)} high toward ${the(act.dest)}` : "the target"} is beyond its stretch at that height.`;
  }
  const hits = unexpectedContacts(a);
  if (hits.length) {
    const e = hits[0], other = otherOf(e);
    const fell = a.events.some((x) => x.kind === "contact" && x.between.includes("floor") && x.between.includes(other));
    return `${e.while_holding ? `While carrying ${the(e.while_holding)}, the arm` : `The arm's ${e.link_role || "wrist"}`} hit ${the(other)}${fell ? " and knocked it off the table" : ""}.`;
  }
  const bad = a.goals.filter((g) => !g.ok);
  if (bad.length) { const w = whyPlain(bad[0]); return `${goalPlain(bad[0])}? It didn't${w ? `: ${w}` : ""}.`; }
  if (err) return err.error;
  return "Something went wrong.";
}
function successPlain(a) {
  const done = a.goals.filter((g) => g.pred !== "undisturbed").map((g) => goalPlain(g).replace(/^(.)/, (c) => c.toLowerCase()).replace("ends up", "is").replace("end up", "are").replace("stays", "stayed").replace("comes to rest", "came to rest"));
  const quiet = a.goals.some((g) => g.pred === "undisturbed" && g.ok);
  return cap(`${list(done)}${quiet ? ", and nothing else was disturbed" : ""}.`);
}
// What changed between two plans, in plain words.
function planChangesPlain(a, prev) {
  if (!prev) return [];
  const out = [];
  const sameOrder = a.plan.actions.length === prev.plan.actions.length && a.plan.actions.every((x, j) => prev.plan.actions[j] && prev.plan.actions[j].obj === x.obj && prev.plan.actions[j].dest === x.dest);
  if (!sameOrder) out.push(`Change the order: ${list(a.plan.actions.map((x) => the(x.obj)))}.`);
  a.plan.actions.forEach((x, j) => {
    const p = prev.plan.actions[j]; if (!p || !sameOrder) return;
    if (p.transit_height !== x.transit_height) out.push(`Carry ${the(x.obj)} ${x.transit_height > p.transit_height ? "higher" : "lower"}: ${cm(p.transit_height)} → ${cm(x.transit_height)}.`);
    if (viaTxt(p.via) !== viaTxt(x.via)) out.push(x.via && x.via.length ? `Take a detour with ${the(x.obj)}.` : `Carry ${the(x.obj)} straight instead of detouring.`);
  });
  return out;
}

// ------------------------------------------------------------------ theme
// Dark only. (A light toggle was dropped: the page is built around full-bleed footage.)

// ------------------------------------------------------------------ landing motion: reveals, count-ups, photo↔3D slider
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
function setupLanding() {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); e.target.querySelectorAll("[data-count]").forEach(countUp); if (e.target.dataset.count) countUp(e.target); }
  }, { threshold: 0.18 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  const range = $("compareRange");
  if (range) {
    const b = $("compareB"), la = $("lblA"), lb = $("lblB");
    const set = (v) => { b.style.opacity = v / 100; la.style.opacity = 1 - v / 100 * 0.7; lb.style.opacity = 0.3 + v / 100 * 0.7; };
    range.oninput = () => set(+range.value);
    set(0);
    // first time it scrolls into view, glide from photo to 3D so the morph is seen without touching anything
    const once = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { once.disconnect(); if (reduceMotion) { range.value = 100; set(100); return; } const t0 = performance.now(); const f = () => { const u = Math.min(1, (performance.now() - t0 - 600) / 1800); const v = Math.round(100 * Math.max(0, (1 - Math.cos(Math.PI * Math.max(0, u))) / 2)); range.value = v; set(v); if (u < 1) requestAnimationFrame(f); }; requestAnimationFrame(f); } }, { threshold: 0.5 });
    once.observe($("compareFig"));
  }
}
function countUp(el) {
  const target = +el.dataset.count; if (!(target > 0) || el.dataset.done) return;
  el.dataset.done = "1";
  if (reduceMotion) { el.textContent = target; return; }
  const t0 = performance.now(), dur = 900;
  const f = () => { const u = Math.min(1, (performance.now() - t0) / dur); el.textContent = Math.round(target * (1 - Math.pow(1 - u, 3))); if (u < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}
// Full-viewport live hero: lamp_01's first try (the wrist sweeps the lamp), looping. Falls back to the GIF.
let hero = null;
async function setupHero() {
  const host = $("heroStage"), fb = $("heroFallback");
  const fallback = () => { fb.hidden = false; host.hidden = true; };
  try {
    hero = new Replay(host, { onEnd: () => { if (hero) { hero.seek(hero.t0); hero.play(); } } });
    hero.noOrbit = true; hero.orbit(false); hero.controls.enabled = false;
    await hero.loadScene("/out/lamp_01", () => {});
    const frames = await getJson("/out/lamp_01/attempt_1.traj.json");
    const rep = await getJson("/api/report?scene=lamp_01");
    if (!frames || !rep) throw new Error("lamp_01 footage missing");
    const a = rep.attempts[0];
    hero.setTrajectory(frames, a.events, hitsOfFor(a, rep));
    // framed so the arm and the lamp sit right of the headline; 4.4 s = the lamp mid-fall
    hero.flyTo({ pos: [0.95, -1.8, 0.78], target: [0.12, 0.02, 0.25] }, 0);
    hero.seek(hero.t0); hero.play();
    if (P.has("still")) { hero.renderer.setAnimationLoop(null); hero.seek(4.4); setInterval(() => hero.tick(), 400); }
    else if (reduceMotion) { hero.seek(4.4); hero.pause(); }
  } catch (err) { console.warn("hero fallback", err); hero = null; fallback(); }
}
// hitsOf() for a report that is not the current one (the hero uses lamp_01 whatever table is selected).
function hitsOfFor(a, rep) {
  const name = (id) => { const o = rep.scene_spec.objects.find((x) => x.id === id); return o ? o.label : id; };
  const dests = new Set(a.plan.actions.map((x) => x.dest)), picks = new Set(a.plan.actions.map((x) => x.obj));
  return a.events.filter((e) => e.kind === "protective_stop" || (e.kind === "contact" && !e.between.includes("floor") && !e.between.includes("table") && (e.while_holding ? !dests.has(otherOf(e)) : e.between.includes("robot_arm") && !dests.has(otherOf(e)) && !picks.has(otherOf(e)))))
    .map((e) => ({ t: e.t, obj: otherOf(e), text: e.kind === "protective_stop" ? `the arm stops: pressing on ${esc(name(e.obj))}` : `hits ${esc(name(otherOf(e)))}` }));
}

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
const state = { report: null, scene: null, current: 0, bust: Date.now(), stage: null, trailSets: [], trailSet: 0, traj: {}, loaded: null,
  spec: null, saved: null, overlay: null, uploadUrl: null, editing: false, robotBase: null, homeFrame: null, busy: false, hard: {} };
let replay = null;
try { replay = new Replay($("stage"), { onTime, onEnd }); } catch (err) { console.warn("WebGL unavailable", err); }
const base = () => `/out/${encodeURIComponent(state.scene)}`;
const getJson = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
// fetch that turns HTTP errors into Error(message) with .status (server errors are {"error": msg})
async function apiJson(url, opts) {
  let r;
  try { r = await fetch(url, opts); } catch (e) { throw new Error(`network error: ${e.message}`); }
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const b = await r.json(); msg = b.error || msg; } catch {}
    const e = new Error(r.status === 404 ? `${msg} — ${url.split("?")[0]} is not available on this server` : msg); e.status = r.status; throw e;
  }
  return r.json();
}
let toastTimer;
function toast(msg, ms = 4200) {
  let el = document.querySelector(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; el.setAttribute("role", "status"); document.body.appendChild(el); }
  el.textContent = msg; el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}
const hasCJK = (s) => /[㐀-鿿]/.test(s || "");
// Primary (English) diagnosis text and the original when it differs (Chinese offline agent).
function diagText(a) {
  const d = a.plan.diagnosis || "", en = a.plan.diagnosis_en || (hasCJK(d) ? GLOSS[d] : null);
  return en ? { primary: en, original: d } : { primary: d, original: null };
}

// ------------------------------------------------------------------ attribution (data-driven: who actually did what in this report)
function agentInfo(r = state.report) {
  const a = (r && r.agent) || "";
  if (/^muse-spark/i.test(a)) return { kind: "muse", label: "Muse Spark", sub: /-contributor$/i.test(a) ? "Contributor" : "", name: a, cost: r && r.agent_trace ? r.agent_trace.cost_usd : null };
  if (/offline|stand-in/i.test(a)) return { kind: "offline", label: "Offline reference replay", sub: "no model call", name: a };
  if (!a || a === "—") return { kind: "none", label: "no agent yet", sub: "", name: a };
  return { kind: "other", label: a, sub: "", name: a };
}
// Who planned, in words (no badges).
function plannerName() {
  const ag = agentInfo();
  if (ag.kind === "muse") return "Meta Muse Spark";
  if (ag.kind === "offline") return "a rule-based stand-in (no model call)";
  if (ag.kind === "none") return "Meta Muse Spark";
  return ag.label;
}
const hasPerception = (spec) => !!spec && spec.objects.some((o) => o.perception && (o.perception.box_px || o.perception.mask_px || o.perception.source));
function percText(spec) {
  return hasPerception(spec) ? "objects listed by Meta Muse Spark and found in the photo by Meta SAM 3.1" : "a table written by hand (no photo was processed)";
}

// ------------------------------------------------------------------ budget + attribution line
async function loadBudget() {
  const el = $("budget");
  try {
    const b = await apiJson("/api/budget");
    el.textContent = `Meta API spend so far: $${b.spent.toFixed(2)} of a $${b.cap.toFixed(0)} cap${b.mock ? " (mock)" : ""}`;
    el.hidden = false;
  } catch { el.hidden = true; }
}
function renderAgentLine() {
  const r = state.report, el = $("agentLine");
  if (!r) { el.textContent = ""; return; }
  const ag = agentInfo(r), spec = state.spec || r.scene_spec;
  const parts = [];
  if (r._empty) parts.push(`Planned by Meta Muse Spark when you press Rehearse`);
  else parts.push(ag.kind === "muse" ? `Planned by Meta Muse Spark${ag.cost != null ? ` (API cost $${ag.cost.toFixed(4)})` : ""}` : ag.kind === "offline" ? "Planned by a rule-based stand-in, no model call" : `Planned by ${ag.name}`);
  parts.push(hasPerception(spec) ? "objects found by Meta SAM 3.1" : "table written by hand");
  parts.push("physics by MuJoCo");
  el.textContent = parts.join(" · ");
  $("plannerName").textContent = cap(plannerName());
}

// ------------------------------------------------------------------ init
async function init() {
  if (DEMO) document.body.classList.add("demo");
  let scenes = (await getJson("/api/scenes")) || [];
  scenes = [...SCENE_ORDER.filter((s) => scenes.includes(s)), ...scenes.filter((s) => !SCENE_ORDER.includes(s))];
  state.scene = P.get("scene") && scenes.includes(P.get("scene")) ? P.get("scene") : (scenes.includes("kitchen_01") ? "kitchen_01" : scenes[0]);
  await renderGallery();
  $("scene").innerHTML = scenes.map((s) => `<option value="${esc(s)}">${esc(sceneTitle(s))}</option>`).join("");
  $("scene").value = state.scene;
  $("scene").onchange = () => selectScene($("scene").value, false);
  document.querySelectorAll("[data-go]").forEach((a) => a.onclick = (e) => { e.preventDefault(); selectScene(a.dataset.go); });
  $("form").onsubmit = runRehearsal;
  $("demoLink").href = `?demo=1&scene=${encodeURIComponent(state.scene || "")}${P.has("mock") ? "&mock=1" : ""}`;
  bindStageControls();
  bindUpload();
  $("saveTwin").onclick = saveTwin;
  $("undoEdits").onclick = undoEdits;
  loadBudget();
  await loadScene();
  if (P.has("still") && replay) { replay.renderer.setAnimationLoop(null); replay.noOrbit = true; replay.orbit(false); setInterval(() => replay.tick(), 400); } // software-GL screenshots
  if (DEMO) { const m = await import("./demo.js"); m.run(app); }
  else { setupObservers(); setupLanding(); setupHero(); }
}

// Switch tables (from the gallery, the landing buttons or the picker) and jump to the rehearsal.
async function selectScene(name, scroll = true) {
  if (!name) return;
  state.scene = name;
  if (![...$("scene").options].some((o) => o.value === name)) $("scene").add(new Option(sceneTitle(name), name));
  $("scene").value = name;
  renderGallery();
  if (scroll) $("story").scrollIntoView({ behavior: "smooth", block: "start" });
  await loadScene();
}

async function loadScene({ spec: givenSpec = null, overlay = null, uploadUrl = null, saved = true } = {}) {
  state.report = await getJson(`/api/report?scene=${encodeURIComponent(state.scene)}`);
  state.current = state.report ? state.report.attempts.length - 1 : 0;
  state.userPicked = false; state.editing = false;
  state.traj = {}; state.trailSets = []; state.trailSet = 0;
  state.overlay = overlay; state.uploadUrl = uploadUrl;
  $("viewerEmpty").hidden = true;
  document.body.classList.toggle("noreport", !state.report);
  $("rehearseJob").hidden = true; $("saveBar").hidden = true;
  $("demoLink").href = `?demo=1&scene=${encodeURIComponent(state.scene || "")}${P.has("mock") ? "&mock=1" : ""}`;
  if (!state.report) {
    // Scene without a rehearsal yet (fresh upload or hand-written): editable twin, invite a run.
    const spec = givenSpec || await getJson(`/api/scene?name=${encodeURIComponent(state.scene)}`);
    if (!spec) {
      $("loader").classList.add("done"); $("viewerEmpty").hidden = false;
      $("viewerEmpty").textContent = `Scene "${state.scene}" could not be loaded.`;
      toast(`Scene ${state.scene} is missing on the server.`); return;
    }
    const gal = state.gallery && state.gallery.find((g) => g.scene === state.scene);
    state.report = { scene: state.scene, scene_spec: spec, instruction: (gal && gal.demo_instruction) || "", agent: "—", reachability: {}, goals: [], attempts: [], stress_test: null, active_perception: null, wall_time_s: 0, _empty: true };
    state.spec = structuredClone(spec); state.saved = saved ? JSON.stringify(spec) : null;
    $("instruction").value = state.report.instruction;
    renderStoryHead(); renderAgentLine(); renderInventory(); renderPhoto(); renderMap();
    $("goals0").innerHTML = `<li><span class="ic na">○</span><span>Written by Meta Muse Spark when you press Rehearse.</span></li>`;
    $("plan0").innerHTML = ""; $("reach").innerHTML = ""; $("goals0tech").innerHTML = ""; $("plan0tech").innerHTML = ""; $("reachTech").innerHTML = "";
    if (!$("s2").querySelector(".noreport-note")) $("s2").querySelector(".step-lede").insertAdjacentHTML("afterend", `<div class="noreport-note">No rehearsal has been run for <span class="scene-name">${esc(sceneTitle(state.scene))}</span> yet. Open the technical details below and press <b>Rehearse</b> to let physics judge the plan.</div>`);
    else $("s2").querySelector(".noreport-note .scene-name").textContent = sceneTitle(state.scene);
    state.loaded = showSpecStage();
    await state.loaded;
    return;
  }
  $("s2").querySelector(".noreport-note")?.remove();
  state.spec = structuredClone(state.report.scene_spec); state.saved = JSON.stringify(state.spec);
  state.loaded = ensureScene();
  render();
  await state.loaded;
  if (!DEMO) setStage(state.stage || "hero", true);
}

// Stage for a scene with no recorded rollout: table + objects from the spec, robot borrowed from a recorded scene.
async function showSpecStage() {
  const loader = $("loader");
  if (!replay) { loader.classList.add("done"); return; }
  loader.classList.remove("done"); $("loaderTxt").textContent = "Building twin from the scene…"; $("loaderBar").style.width = "60%";
  await ensureRobotBase();
  await replay.loadSpec(state.spec, state.robotBase, state.homeFrame);
  replay.frames = []; replay.duration = 0;
  $("timeline").classList.add("off"); $("tglTrails").disabled = true;
  replay.showRobot(true); replay.showTrails(false); replay.orbit(false);
  replay.flyTo("twin", state.stage ? 1.2 : 0); state.stageView = "twin"; state.stage = "spec";
  replay.setFocus(focusFor("twin"));
  setChip(`${state.spec.objects.length} objects · no rehearsal yet`);
  setCaption("The 3D copy, before any rehearsal", "drag objects in the map (technical details, step 1) to correct it, then Rehearse");
  loader.classList.add("done");
}
// Robot meshes + home pose come from any scene that has a recorded rollout.
async function ensureRobotBase() {
  if (state.homeFrame) return;
  const withReport = (state.gallery || []).find((g) => g.has_report && g.scene !== state.scene) || (state.gallery || []).find((g) => g.has_report);
  if (!withReport) return;
  state.robotBase = `/out/${encodeURIComponent(withReport.scene)}`;
  const traj = await getJson(`${state.robotBase}/attempt_1.traj.json`);
  if (traj && traj.length) state.homeFrame = traj[0];
}

function render() {
  const r = state.report;
  $("instruction").value = r.instruction;
  $("reportLink").href = `${base()}/report.md`;
  renderStoryHead(); renderAgentLine(); renderInventory(); renderGoalsPlan(); renderAttempts(); renderStress(); renderLook();
  renderPhoto(); renderMap();
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
    txt.textContent = `Loading the 3D copy · ${(got / 1048576).toFixed(1)} MB`;
  });
  syncMeshToggle();
  txt.textContent = "Loading the arm's movements…"; bar.style.width = "100%";
  const trajs = await Promise.all(r.attempts.map((a) => getJson(`${base()}/attempt_${a.attempt}.traj.json?v=${state.bust}`)));
  r.attempts.forEach((a, i) => { state.traj[a.attempt] = trajs[i]; });
  if (trajs[0] && trajs[0].length) { state.robotBase = base(); state.homeFrame = trajs[0][0]; }
  const sets = [];
  if (r.stress_test) {
    const before = await getJson(`${base()}/stress_trails.json?v=${state.bust}`);
    if (before) sets.push({ label: "from the photo", runs: before, successes: before.filter((x) => x.success).length });
    if (r.active_perception && r.active_perception.steps.length) {
      const after = await getJson(`${base()}/stress_trails_after.json?v=${state.bust}`);
      if (after) sets.push({ label: `after a closer look at ${list(r.active_perception.steps.map((s) => the(s.look_at)))}`, runs: after, successes: after.filter((x) => x.success).length });
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
  return unexpectedContacts(a).map((e) => ({ t: e.t, obj: otherOf(e), text: e.kind === "protective_stop" ? `the arm stops: pressing on ${esc(nameOf(e.obj))}` : `hits ${esc(nameOf(otherOf(e)))}` }));
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
    const label = e.kind === "contact" ? `hits ${nameOf(otherOf(e))}` : e.kind === "protective_stop" ? `the arm stops at ${nameOf(e.obj)}` : `${e.kind === "grasp" ? "picks up" : e.kind === "release" ? "lets go of" : e.kind} ${nameOf(e.obj)}`;
    return `<span class="tick ${cls}" data-t="${e.t}" title="${esc(label)} · ${e.t.toFixed(1)} s" style="left:${(100 * (e.t - t0) / (dur - t0)).toFixed(2)}%"></span>`;
  }).join("");
  $("ticks").querySelectorAll(".tick").forEach((el) => el.onclick = () => jumpTo(+el.dataset.t));
  $("attemptPills").innerHTML = r.attempts.map((x, j) =>
    `<button type="button" aria-selected="${j === i}" data-i="${j}"><span class="ic ${x.success ? "ok" : "bad"}">${x.success ? "✓" : "✕"}</span>${tryName(j)}</button>`).join("");
  $("attemptPills").querySelectorAll("button").forEach((b) => b.onclick = () => { state.userPicked = true; loadAttempt(+b.dataset.i); markActiveAttempt(); });
  state.lastNow = -2; eventRows = [...document.querySelectorAll(`#att-${a.attempt} .events li[data-t]`)].map((el) => ({ el, t: +el.dataset.t }));
  setChip(`${tryName(i)} · <span class="${a.success ? "ok" : "bad"}">${a.success ? "worked" : "failed"}</span>`);
  replay.seek(t0);
  if (play) replay.play();
  syncPlay();
}

let eventRows = [];
function onTime(t) {
  $("scrub").value = t;
  $("timeLbl").textContent = `${t.toFixed(1)} s`;
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
  $("tglMeshes").onclick = (e) => toggle(e.currentTarget, (on) => {
    if (!replay) return;
    replay.setMeshMode(on); e.currentTarget.textContent = on ? "Real look" : "Simple shapes";
    if (state.stage === "twin") setStage("twin", true);
  });
  $("tglRobot").onclick = (e) => toggle(e.currentTarget, (on) => replay && replay.showRobot(on));
  $("tglTrails").onclick = (e) => toggle(e.currentTarget, (on) => { if (!replay) return; replay.revealTrails(Infinity); replay.showTrails(on); $("legend3d").hidden = !on; if (on) renderTrailLegend(); });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) { e.preventDefault(); $("playBtn").click(); }
  });
}
function toggle(btn, fn) { const on = btn.getAttribute("aria-pressed") !== "true"; btn.setAttribute("aria-pressed", on); fn(on); }
// SAM 3D look-alike meshes: toggle visible only when the scene has some; default on.
function syncMeshToggle() {
  const b = $("tglMeshes"), n = replay ? replay.visualCount : 0;
  b.hidden = !n;
  if (n) { const on = b.getAttribute("aria-pressed") === "true"; replay.setMeshMode(on); b.textContent = on ? "Real look" : "Simple shapes"; }
}
const meshNote = () => replay && replay.visualCount ? (replay.meshMode ? "looks rebuilt by Meta SAM 3D Objects; physics uses simple shapes underneath" : "the simple shapes physics actually uses") : "";
function setChip(html) { $("stageChip").innerHTML = html; }
function setCaption(h, s) { $("caption").innerHTML = h ? `<div class="h">${h}</div>${s ? `<div class="s">${s}</div>` : ""}` : ""; }

function renderTrailLegend() {
  const sets = state.trailSets; if (!sets.length) return;
  const i = state.trailSet, runs = sets[i].runs, ok = sets[i].successes;
  $("legend3d").innerHTML = (sets.length > 1 ? `<div class="seg">${sets.map((s, j) =>
    `<button type="button" data-i="${j}" aria-pressed="${i === j}">${esc(s.label)} · ${s.successes}/${s.runs.length}</button>`).join("")}</div>` : "") +
    `<div>Paths of the moved objects across ${runs.length} test runs · <span class="ok">${ok} worked</span> · <span class="bad">${runs.length - ok} failed</span></div>`;
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
    if (state.fanToken !== token) break;
    replay.revealTrails(Math.ceil(u * runs.length));
    if (u >= 1) break;
    await new Promise(requestAnimationFrame);
  }
}

// ------------------------------------------------------------------ stage modes (driven by scroll)
function setStage(mode, force = false) {
  if (!replay || !state.report || (mode === state.stage && !force)) return;
  if (!state.report.attempts.length || state.editing) {   // spec-mode twin: only the camera follows the chapters
    if (state.stage === "spec" && mode === "twin") return;
    state.stage = mode === "twin" ? "spec" : mode;
    replay.flyTo(mode === "plan" ? "top" : mode === "twin" || mode === "hero" ? "twin" : "home", 1.4);
    return;
  }
  const prev = state.stage; state.stage = mode;
  const r = state.report, last = r.attempts.length - 1;
  const tl = $("timeline");
  const userTrails = $("tglTrails").getAttribute("aria-pressed") === "true";
  // reset everything a mode may have changed, then apply the new mode
  state.fanToken = null;
  replay.orbit(false);
  replay.showRobot($("tglRobot").getAttribute("aria-pressed") === "true");
  if (userTrails) { replay.revealTrails(Infinity); replay.showTrails(true); } else { replay.showTrails(false); $("legend3d").hidden = true; }
  replay.setFocus(focusFor(mode));
  tl.classList.toggle("off", !(mode === "rehearse" || mode === "hero"));
  const fly = (v, d = 1.6) => { state.stageView = v; replay.flyTo(v, prev ? d : 0); };
  switch (mode) {
    case "hero":
    case "report": {
      if (state.current !== last) loadAttempt(last, { play: false });
      fly("home"); replay.orbit(true, 0.45);
      replay.speed = 1; $("speed").value = "1";
      replay.play(); syncPlay();
      setChip(`${tryName(last)} · <span class="${r.attempts[last].success ? "ok" : "bad"}">${r.attempts[last].success ? "worked" : "failed"}</span>`);
      setCaption("", "");
      break;
    }
    case "twin": {
      replay.pause(); replay.seek(replay.t0); syncPlay();
      fly("twin");
      const low = focusFor("twin"), perc = hasPerception(r.scene_spec), nLow = Object.values(low).filter((v) => v === "warn").length, nBad = Object.values(low).filter((v) => v === "bad").length;
      setChip(`${r.scene_spec.objects.length} objects · ${perc ? "from the photo" : "built by hand"}`);
      const notes = [meshNote(), nLow ? `amber ring = ${nLow} uncertain` : "", nBad ? `red ring = ${nBad} out of the arm's reach` : (perc ? "every object within reach" : "")].filter(Boolean);
      setCaption(perc ? "The 3D copy of the photo" : "The 3D copy of a hand-built table", notes.join(" · "));
      break;
    }
    case "plan": {
      replay.pause(); replay.seek(replay.t0); syncPlay();
      fly("top"); replay.showRobot(false);
      setChip(`the plan · ${r.attempts[0].plan.actions.length} move${r.attempts[0].plan.actions.length === 1 ? "" : "s"}`);
      setCaption("The plan, before physics has a say", "blue ring = what gets picked up · green ring = where it goes");
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
      if (!state.trailSets.length) { setCaption("No repeated test", "the final plan did not succeed, so it was not tested again"); break; }
      const i = mode === "look" && state.trailSets.length > 1 ? 1 : 0;
      fanOut(i);
      $("legend3d").hidden = false;
      const s = state.trailSets[i];
      setChip(`${mode === "look" ? "after a closer look" : `${s.runs.length} test runs`} · ${s.successes} of ${s.runs.length} worked`);
      setCaption(mode === "look" ? `After a closer look at ${list(r.active_perception.steps.map((st) => the(st.look_at)))}: ${s.successes} of ${s.runs.length} worked` : `Tried ${s.runs.length} times: ${s.successes} worked`,
        mode === "look" ? "same kind of random nudges, fresh runs" : "each line is the path of a moved object in one run · red = that run failed");
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
  let ticking = false;
  const update = () => {
    ticking = false;
    const probe = innerHeight * 0.42;
    let active = sections[0];
    for (const sec of sections) if (sec.getBoundingClientRect().top <= probe) active = sec;
    if (active !== state.activeSection) {
      state.activeSection = active;
      setStage(active.dataset.stage);
    }
    const inStory = $("story").getBoundingClientRect().top <= probe;
    document.querySelectorAll(".nav a[data-nav]").forEach((a) => a.setAttribute("aria-current", inStory));
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  addEventListener("resize", update);
  update();
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
// Title, task and a one-sentence outcome for the table being shown.
function renderStoryHead() {
  const r = state.report;
  $("storyTitle").textContent = sceneTitle(state.scene);
  $("storyTask").textContent = r.instruction || "(no task yet)";
  $("reportLink").href = `${base()}/report.md`;
  if (r._empty || !r.attempts.length) { $("storyOutcome").textContent = "No rehearsal has been run for this table yet."; return; }
  const A = r.attempts, st = r.stress_test, ap = r.active_perception;
  const fails = A.filter((a) => !a.success).length, lastOk = A.map((a, i) => (a.success ? i : -1)).filter((i) => i >= 0)[0];
  let s;
  if (!A[A.length - 1].success) s = `None of the ${A.length} tries worked.`;
  else if (fails === 0) s = A.length > 1 ? `Worked on the first try; the model still revised the plan ${A.length - 1} time${A.length > 2 ? "s" : ""} after testing it.` : "Worked on the first try.";
  else s = `The first try failed${fails > 1 ? `, and so did the next ${fails - 1}` : ""}. Shown what physics saw, the model fixed its own plan: the ${tryName(lastOk).toLowerCase()} worked.`;
  if (st) {
    s += ` Tried ${st.n} more times with small random changes, it worked ${st.successes} of ${st.n} times`;
    const after = ap && ap.steps.length ? (ap.heldout ? ap.heldout.after : ap.final.successes) : null;
    const before = ap && ap.steps.length && ap.heldout ? ap.heldout.before : st.successes;
    s += after != null ? `; after a closer look at ${list(ap.steps.map((x) => the(x.look_at)))}, ${after} of ${st.n}${before !== st.successes ? ` (${before} before, on the same fresh runs)` : ""}.` : ".";
  }
  $("storyOutcome").textContent = s;
}

function renderInventory() {
  const r = state.report, objs = (state.spec || r.scene_spec).objects, hasPerc = objs.some((o) => o.perception);
  $("percBadges").textContent = percText(state.spec || r.scene_spec);
  $("inv").innerHTML = `<tr><th>Object</th><th>Position (m)</th><th>Size</th><th>Weight</th><th>Grip (friction)</th>${hasPerc ? "<th>Found</th>" : ""}<th>In reach</th></tr>` + objs.map((o) => {
    const ok = r.reachability[o.id], low = o.perception && o.perception.confidence === "low";
    return `<tr data-id="${esc(o.id)}" class="${low ? "low" : ""}"><td><span class="sw" style="background:${col(o.rgba || [.7,.7,.7,1])}"></span><span class="lbl">${esc(o.label)}</span><br><span class="id">${esc(o.id)}</span></td>
      <td class="num">${o.pos[0].toFixed(2)}, ${o.pos[1].toFixed(2)}</td><td class="num">${dims(o)}</td><td class="num">${fmtRange(o.mass_kg.range, " kg")}</td><td class="num">${o.friction ? fmtRange(o.friction.range) : "—"}</td>
      ${hasPerc ? `<td>${o.perception ? `<span class="tag ${low ? "warn" : "ok"}" title="SAM 3.1 ${esc(o.perception.source || "")} prompt${o.perception.sam_confidence != null ? ` · score ${(+o.perception.sam_confidence).toFixed(2)}` : ""}">${low ? "uncertain" : "confident"}${o.perception.sigma_m ? ` · ±${(o.perception.sigma_m * 100).toFixed(1)} cm` : ""}</span>` : "—"}</td>` : ""}
      <td>${ok === undefined ? `<span class="tag">—</span>` : `<span class="tag ${ok ? "ok" : "bad"}">${ok ? "yes" : "no, too far"}</span>`}</td></tr>`;
  }).join("");
  $("inv").querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.onmouseenter = () => hl(tr.dataset.id);
    tr.onmouseleave = () => hl(null);
    tr.onclick = () => { if (!replay) return; const p = replay.bodyByName[tr.dataset.id]?.position; if (p) replay.flyTo({ pos: [p.x + 0.5, p.y + 0.4, 0.4], target: [p.x, p.y, 0.08] }, 1); };
  });
}
// Highlight one object everywhere: 3D halo, photo box, map glyph, inventory row.
function hl(id) {
  if (replay) replay.setFocus(id ? { [id]: "accent" } : focusFor(state.stage === "spec" ? "twin" : state.stage));
  document.querySelectorAll("#photoBoxes g[data-id], #map .obj[data-id]").forEach((el) => el.classList.toggle("hl", !!id && el.dataset.id === id));
  document.querySelectorAll("#inv tr[data-id]").forEach((el) => el.classList.toggle("sel", !!id && el.dataset.id === id));
}

async function renderPhoto() {
  const r = state.report, spec = state.spec || r.scene_spec, img = $("photoImg"), svg = $("photoBoxes"), empty = $("photoEmpty"), frame = $("photoFrame"), tools = $("photoTools");
  let W, H, boxes = {}, synthetic = false, masks = {};
  img.hidden = true; empty.hidden = false; svg.innerHTML = ""; img.classList.remove("synthetic"); tools.innerHTML = "";
  $("maskImg").hidden = true; $("twinImg").hidden = true; frame.classList.remove("no-boxes", "no-masks");
  empty.innerHTML = `<div class="spinner"></div><span>Loading photo…</span>`;
  img.onload = () => { img.hidden = false; empty.hidden = true; };
  img.onerror = () => { empty.innerHTML = `<span>Photo not available on this server (${esc(img.getAttribute("src") || "")}).</span>`; };
  if (spec.photo && spec.photo.file) {
    img.src = state.uploadUrl || (spec.photo.file.startsWith("blob:") || spec.photo.file.startsWith("data:") ? spec.photo.file : `/${spec.photo.file.replace(/^\/+/, "")}`);
    W = spec.photo.width; H = spec.photo.height;
    for (const o of spec.objects) if (o.perception && o.perception.box_px) boxes[o.id] = o.perception.box_px;
    for (const o of spec.objects) if (o.perception && o.perception.mask_px && o.perception.mask_px.length > 2) masks[o.id] = o.perception.mask_px;
    const lows = spec.objects.filter((o) => o.perception && o.perception.confidence === "low").length;
    const nB = Object.keys(boxes).length;
    $("photoCap").innerHTML = `Meta SAM 3.1 found ${nB} object${nB === 1 ? "" : "s"} in the photo${lows ? `; ${lows} ${lows > 1 ? "are" : "is"} uncertain (dashed)` : ""}. ${W}×${H} px, ${esc(spec.privacy || "private")} data${!spec.photo.camera ? "" : spec.photo.camera_info?.f_source === "given_camera" ? `; camera known from the dataset${spec.photo.camera_info.dataset ? ` (${esc(spec.photo.camera_info.dataset)})` : ""}` : "; camera worked out from the A4 sheet"}.`;
    // tools: boxes / masks / perception overlay png / twin ghost blend (when the camera is known)
    tools.innerHTML = `<button type="button" id="tglBoxes" aria-pressed="true">Boxes</button>` +
      (Object.keys(masks).length ? `<button type="button" id="tglMasks" aria-pressed="true">Masks</button>` : "") +
      (state.overlay ? `<button type="button" id="tglOverlay" aria-pressed="false">Perception overlay</button>` : "") +
      (spec.photo.camera ? `<label class="blend"><span>twin ghost</span><input type="range" id="blend" min="0" max="1" step="0.02" value="0" aria-label="Twin overlay opacity"></label>` : "");
    $("tglBoxes").onclick = (e) => toggle(e.currentTarget, (on) => frame.classList.toggle("no-boxes", !on));
    if ($("tglMasks")) $("tglMasks").onclick = (e) => toggle(e.currentTarget, (on) => frame.classList.toggle("no-masks", !on));
    if ($("tglOverlay")) { $("maskImg").src = state.overlay; $("tglOverlay").onclick = (e) => toggle(e.currentTarget, (on) => { $("maskImg").hidden = !on; }); }
    if ($("blend")) $("blend").oninput = (e) => setTwinBlend(+e.target.value);
  } else {
    // Hand-authored scene: prefer the perception test-set render of it (photos/synth), with the twin's
    // boxes projected through that render's known camera; else render a frame from the twin itself.
    // Probe for a test-set render only where one is known to exist (no 404 on every other scene).
    const synthUrl = `/photos/synth/${encodeURIComponent(spec.name)}__front55`;
    const gt = SYNTH_RENDERS.has(spec.name) ? await getJson(`${synthUrl}.gt.json`) : null;
    if (gt && gt.camera) {
      img.src = `${synthUrl}.jpg`; W = 1280; H = 960; synthetic = true;
      boxes = projectBoxes(spec, gt.camera, W, H);
      $("photoCap").innerHTML = `This table was built by hand to be tricky, so there is no photo: this is a rendering of it from the perception test set, with each object's box drawn in. A real photo gets Meta SAM 3.1 masks instead.`;
    } else {
      if (!replay) { empty.innerHTML = "No photo for this scene."; return; }
      await state.loaded;
      const snap = replay.snapshotFrom({ pos: [1.25, 0.62, 0.68], target: [0.6, -0.02, 0.03], fov: 50 }, 1280, 800);
      img.src = snap.dataUrl; W = snap.width; H = snap.height; boxes = snap.boxes; synthetic = true;
      img.classList.add("synthetic");
      $("photoCap").innerHTML = `This table was built by hand to be tricky, so there is no photo: this frame is rendered from the 3D copy. On a real photo, the boxes come from Meta SAM 3.1.`;
    }
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const k = W / 640;
  svg.innerHTML = spec.objects.map((o) => {
    const b = boxes[o.id]; if (!b) return "";
    const [x1, y1, x2, y2] = b, low = o.perception && o.perception.confidence === "low", w = x2 - x1, h = y2 - y1, c = Math.min(14 * k, w / 3, h / 3);
    const tag = low ? `uncertain${o.perception.sigma_m ? ` · ±${(o.perception.sigma_m * 100).toFixed(1)} cm` : ""}` : "";
    const poly = masks[o.id] ? `<polygon class="mask" points="${masks[o.id].map((p) => p.map((v) => Math.round(v)).join(",")).join(" ")}"/>` : "";
    return `<g class="${low ? "low" : ""}" data-id="${esc(o.id)}">${poly}<rect x="${x1}" y="${y1}" width="${w}" height="${h}" rx="${3 * k}"/>
      <path class="corner" d="M${x1} ${y1 + c}V${y1}H${x1 + c}M${x2 - c} ${y1}H${x2}V${y1 + c}M${x2} ${y2 - c}V${y2}H${x2 - c}M${x1 + c} ${y2}H${x1}V${y2 - c}"/>
      <text x="${x1 + 4 * k}" y="${y1 - 6 * k}" style="font-size:${12 * k}px">${esc(o.category || o.label)}</text>
      ${tag ? `<text class="tag" x="${x1 + 4 * k}" y="${y2 + 14 * k}" style="font-size:${10.5 * k}px">${esc(tag)}</text>` : ""}</g>`;
  }).join("");
  svg.querySelectorAll("g[data-id]").forEach((g) => { g.onmouseenter = () => hl(g.dataset.id); g.onmouseleave = () => hl(null); });
  const grain = $("photoFrame").querySelector(".grain");
  if (synthetic && !grain) { const gr = document.createElement("div"); gr.className = "grain"; $("photoFrame").insertBefore(gr, svg); }
  if (!synthetic && grain) grain.remove();
}

// Ghost of the twin rendered through the photo's calibrated camera (OpenCV K, R, t), blended over the photo.
let ghost = null;
async function twinGhostUrl(spec) {
  const p = spec.photo, W = 960, H = Math.round(960 * p.height / p.width);
  if (!ghost) {
    const host = document.createElement("div"); host.style.cssText = `position:fixed;left:-9999px;top:0;width:${W}px;height:${H}px;`;
    document.body.appendChild(host);
    ghost = new Replay(host, {}); ghost.renderer.setAnimationLoop(null); ghost.renderer.setPixelRatio(1); ghost.host = host;
  }
  ghost.host.style.width = `${W}px`; ghost.host.style.height = `${H}px`; ghost.resize();
  await ghost.loadSpec(spec, null, null);
  ghost.grid.visible = false;
  for (const b of ghost.bodies) if (b) for (const m of b.userData.materials) {
    const isTable = b.userData.name === "table";
    m.color.setHex(isTable ? 0x1b2a55 : 0x3b63ff); m.emissive.setHex(0x3b63ff); m.emissiveIntensity = isTable ? 0.15 : 0.7;
    m.transparent = true; m.opacity = isTable ? 0.25 : 0.95;
  }
  ghost.setBackground(0x000000, true);
  ghost.setCameraKRt(p.camera.K, p.camera.R, p.camera.t, p.width, p.height);
  ghost.renderOnce();
  return ghost.renderer.domElement.toDataURL("image/png");
}
async function setTwinBlend(v) {
  const img = $("twinImg");
  if (v <= 0) { img.hidden = true; return; }
  if (!img.dataset.for || img.dataset.for !== JSON.stringify(state.spec.objects.map((o) => [o.id, o.pos, o.yaw]))) {
    img.src = await twinGhostUrl(state.spec); img.dataset.for = JSON.stringify(state.spec.objects.map((o) => [o.id, o.pos, o.yaw]));
  }
  img.style.opacity = v; img.hidden = false;
}

// ------------------------------------------------------------------ top-down map editor (drag / arrow keys → twin updates live → Save twin)
const MAP_S = 300;
const objRadius = (o) => isRound(o.shape) ? o.shape.radius : Math.hypot(...shapeBBox(o.shape).slice(0, 2)) / 2;
function renderMap() {
  const spec = state.spec, r = state.report, reach = (r && r.reachability) || {};
  if (!spec) { $("map").innerHTML = ""; return; }
  const t = spec.table, Sc = MAP_S;
  const sx = (y) => -y * Sc, sy = (x) => -x * Sc;
  const tx0 = t.center[0] - t.size[0] / 2, ty0 = t.center[1] - t.size[1] / 2;
  let g = `<rect x="${sx(ty0 + t.size[1])}" y="${sy(tx0 + t.size[0])}" width="${t.size[1] * Sc}" height="${t.size[0] * Sc}" rx="6" fill="var(--surface-2)" stroke="var(--border-strong)"/>`;
  g += `<circle cx="0" cy="0" r="${0.855 * Sc}" fill="var(--accent-soft)" stroke="var(--accent)" stroke-dasharray="4 4" stroke-width="1"/>`;
  g += `<text x="${sx(-0.6)}" y="${sy(0.1)}" text-anchor="end" style="fill:var(--accent)">arm's reach ≈ 0.86 m</text>`;
  for (const o of spec.objects) {
    const ok = reach[o.id] !== false, s = o.shape, p = o.perception;
    const stroke = ok ? "rgba(0,0,0,.35)" : "var(--bad)", dash = ok ? "" : `stroke-dasharray="4 3"`, fill = col(o.rgba || [.7,.7,.7,1]);
    let shape;
    if (!isRound(s)) { const bb = shapeBBox(s), w = bb[1] * Sc, h = bb[0] * Sc; shape = `<rect x="${-w / 2}" y="${-h / 2}" width="${w}" height="${h}" rx="2" fill="${fill}" stroke="${stroke}" stroke-width="1.5" ${dash} transform="rotate(${-(o.yaw || 0) * 180 / Math.PI})"/>`; }
    else if (s.type === "bowl") shape = `<circle r="${s.radius * Sc}" fill="${fill}" fill-opacity=".35" stroke="${fill}" stroke-width="4"/>`;
    else shape = `<circle r="${s.radius * Sc}" fill="${fill}" stroke="${stroke}" stroke-width="1.5" ${dash}/>`;
    const ring = p ? `<circle r="${(objRadius(o) + 2 * (p.sigma_m || 0.012)) * Sc}" fill="none" stroke="${p.confidence === "low" ? "var(--warn)" : "var(--accent)"}" stroke-opacity="${p.confidence === "low" ? 1 : .4}" stroke-width="1.5" stroke-dasharray="3 3"/>` : "";
    const tipTxt = esc(`<b>${esc(o.label)}</b><code>${esc(o.id)}</code> · ${dims(o)}<br>x ${o.pos[0].toFixed(3)} m · y ${o.pos[1].toFixed(3)} m${p ? `<br>position ±${(p.sigma_m * 100).toFixed(1)} cm · ${p.confidence === "low" ? "uncertain" : "confident"}` : ""}<br>${reach[o.id] === undefined ? "reach unknown until rehearsed" : ok ? "within the arm's reach" : "too far for the arm"}<br><i>drag or use arrow keys</i>`);
    g += `<g class="obj" data-id="${esc(o.id)}" transform="translate(${sx(o.pos[1])} ${sy(o.pos[0])})" tabindex="0" role="button" aria-label="${esc(o.label)}, drag to move" data-tip="${tipTxt}">${shape}${ring}<text y="${objRadius(o) * Sc + 13}" text-anchor="middle">${esc(o.id)}${ok ? "" : " ✕"}</text></g>`;
  }
  g += `<rect x="-14" y="-6" width="28" height="18" rx="4" fill="var(--text-2)"/><text x="0" y="26" text-anchor="middle">robot</text>`;
  const vb = [sx(ty0 + t.size[1]) - 10, sy(tx0 + t.size[0]) - 10, t.size[1] * Sc + 20, (tx0 + t.size[0]) * Sc + 44];
  $("map").innerHTML = `<svg viewBox="${vb.join(" ")}" role="img" aria-label="Top-down map of the table; objects can be dragged">${g}</svg>`;
  bindTips($("map"));
  wireMapDrag();
}
function wireMapDrag() {
  const svg = $("map").querySelector("svg"), spec = state.spec, t = spec.table;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const move = (o, g, x, y) => {
    o.pos = [+clamp(x, t.center[0] - t.size[0] / 2, t.center[0] + t.size[0] / 2).toFixed(4), +clamp(y, t.center[1] - t.size[1] / 2, t.center[1] + t.size[1] / 2).toFixed(4)];
    g.setAttribute("transform", `translate(${-o.pos[1] * MAP_S} ${-o.pos[0] * MAP_S})`);
    if (replay && replay.specMode) replay.placeObject(o.id, o.pos, o.shape, o.yaw);
  };
  const done = () => { renderInventory(); markDirty(); if ($("blend") && +$("blend").value > 0) setTwinBlend(+$("blend").value); };
  svg.querySelectorAll(".obj").forEach((g) => {
    const o = spec.objects.find((x) => x.id === g.dataset.id);
    g.addEventListener("mouseenter", () => hl(o.id)); g.addEventListener("mouseleave", () => hl(null));
    g.onpointerdown = async (e) => {
      e.preventDefault(); g.setPointerCapture(e.pointerId); hideTip();
      await enterEditMode();
      const m = svg.getScreenCTM().inverse();
      const p0 = new DOMPoint(e.clientX, e.clientY).matrixTransform(m), start = [...o.pos];
      g.onpointermove = (ev) => { const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m); move(o, g, start[0] - (p.y - p0.y) / MAP_S, start[1] - (p.x - p0.x) / MAP_S); };
      g.onpointerup = g.onpointercancel = () => { g.onpointermove = null; done(); };
    };
    g.onkeydown = async (e) => {
      const d = e.shiftKey ? 0.05 : 0.01, k = { ArrowUp: [d, 0], ArrowDown: [-d, 0], ArrowLeft: [0, d], ArrowRight: [0, -d] }[e.key];
      if (!k) return;
      e.preventDefault(); await enterEditMode(); move(o, g, o.pos[0] + k[0], o.pos[1] + k[1]); done();
    };
  });
}
// Editing a scene that has a recorded rollout: swap the stage to the editable primitive twin.
async function enterEditMode() {
  if (!replay || replay.specMode || state.editing) return;
  state.editing = true;
  replay.pause(); syncPlay();
  await ensureRobotBase();
  await replay.loadSpec(state.spec, state.robotBase, state.homeFrame);
  replay.frames = []; replay.duration = 0; $("timeline").classList.add("off"); $("tglTrails").disabled = true;
  replay.showRobot(true); replay.showTrails(false); replay.orbit(false);
  replay.flyTo("twin", 1.2); state.stage = "spec";
  setChip(`editing · ${state.spec.objects.length} objects`);
  setCaption("Editing the 3D copy", "the recorded rehearsal no longer matches: save, then rehearse again");
}
function markDirty() {
  const dirty = state.saved === null || JSON.stringify(state.spec) !== state.saved;
  $("saveBar").hidden = !dirty;
  if (dirty) {
    $("saveMsg").textContent = state.saved === null ? "Perception needs a correction before it can be saved" : "Edited, not saved";
    $("saveBar").classList.toggle("issue", state.saved === null);
    if (!$("saveName").value) $("saveName").value = (state.report && !state.report._empty) ? `${state.scene}_edit` : state.scene;
  }
}
async function undoEdits() {
  if (state.saved) state.spec = JSON.parse(state.saved);
  state.editing = false; $("saveBar").hidden = true; $("saveName").value = "";
  if (state.report && !state.report._empty) { state.loaded = ensureScene(); renderMap(); renderInventory(); await state.loaded; setStage("twin", true); }
  else { renderMap(); renderInventory(); await showSpecStage(); }
}
async function saveTwin() {
  const btn = $("saveTwin"), name = ($("saveName").value || state.scene).trim();
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) { toast("Scene name: letters, digits, _ and - only."); return; }
  btn.disabled = true;
  try {
    const spec = structuredClone(state.spec); spec.name = name;
    const r = await apiJson("/api/scene", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) });
    state.saved = JSON.stringify(spec); state.spec = spec;
    toast(`Saved scenes/${r.name}.json — rehearsals now use your corrections.`);
    await renderGallery();
    state.scene = r.name;
    if (![...$("scene").options].some((o) => o.value === r.name)) $("scene").add(new Option(sceneTitle(r.name), r.name));
    $("scene").value = r.name;
    $("saveName").value = "";
    await loadScene({ spec, overlay: state.overlay, uploadUrl: state.uploadUrl });
    $("saveMsg").textContent = "Saved";
    $("s2").scrollIntoView({ behavior: "smooth", block: "start" });
    $("instruction").focus();
  } catch (e) { toast(`Could not save the twin: ${e.message}`); }
  finally { btn.disabled = false; }
}

const SYNTH_RENDERS = new Set(["desk_01"]);   // scenes with photos/synth/<scene>__front55.{jpg,gt.json}
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
    if (!isRound(s)) {
      const [sx, sy, sz] = shapeBBox(s), cs = Math.cos(yaw), sn = Math.sin(yaw);
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
  if (!a.plan.actions.length) return `<li><span class="note" style="margin:0">The agent produced an empty plan — nothing is moved, only the implicit goals are checked.</span></li>`;
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
const planPlainHtml = (a) => a.plan.actions.length ? a.plan.actions.map((x) => `<li><span>${esc(actionPlain(x))}</span></li>`).join("") : `<li><span>An empty plan: nothing is moved.</span></li>`;
function renderGoalsPlan() {
  const r = state.report;
  $("goals0").innerHTML = r.goals.map((g) => `<li><span class="ic na">○</span><span>${esc(goalPlain(g))}${g.implicit ? ` <span class="imp">added by the model</span>` : ""}</span></li>`).join("");
  $("plan0").innerHTML = planPlainHtml(r.attempts[0]);
  $("goals0tech").innerHTML = r.goals.map((g) => goalHtml(g, false)).join("");
  $("plan0tech").innerHTML = planHtml(r.attempts[0]);
  const un = Object.entries(r.reachability).filter(([, ok]) => !ok).map(([k]) => k), n = Object.keys(r.reachability).length;
  $("reach").textContent = un.length ? `${cap(list(un.map(the)))} ${un.length > 1 ? "are" : "is"} too far for the arm to reach from where it stands.` : n ? "Every object is within the arm's reach." : "";
  $("reachTech").innerHTML = `Reachability (top-down IK to each object's top face): ${n - un.length}/${n}${un.length ? ` · outside the workspace: ${un.map((k) => `<code>${esc(k)}</code>`).join(", ")}` : ""}`;
  $("nUni").textContent = r.stress_test ? r.stress_test.n : $("stress").value;
}

function renderAttempts() {
  const r = state.report;
  let html = "";
  // Agent-loop reports: attempt 1's "diagnosis" is the model's opening narration (its intent), and the submitted
  // attempt (the last one with a diagnosis) carries the real diagnosis of the failure it fixed.
  const loop = !!r.agent_trace, hasDiag = (x) => x && (x.plan.diagnosis || x.plan.diagnosis_en);
  const submitted = [...r.attempts].reverse().find(hasDiag);
  let shownDiag = null;
  const who = plannerName();
  const nFail = r.attempts.filter((a) => !a.success).length;
  $("s3Title").textContent = nFail ? (r.attempts[r.attempts.length - 1].success ? "The first try fails. The model learns, and fixes it." : "The tries, and what went wrong") : "The first try works";
  r.attempts.forEach((a, i) => {
    const prev = r.attempts[i - 1];
    if (loop && i === 0 && hasDiag(a)) {
      html += `<p class="fine">Before trying, ${esc(cap(who))} said: “${esc(diagText(a).primary)}”</p>`;
    } else if (hasDiag(a) || (loop && prev && !prev.success && submitted && submitted !== shownDiag)) {
      const src = hasDiag(a) ? a : submitted;
      if (src !== shownDiag) {
        shownDiag = src;
        const { primary, original } = diagText(src);
        const ev = unexpectedContacts(prev || a);
        const changes = planChangesPlain(a, prev);
        html += `<div class="learned">
          <p class="who">What ${esc(who)} learned from the log</p>
          <blockquote>${esc(primary)}</blockquote>
          ${original ? `<p class="fine" style="margin-top:6px" lang="zh">${esc(original)}</p>` : ""}
          ${changes.length ? `<p class="fix"><b>The fix:</b> ${changes.map(esc).join(" ")}</p>` : ""}
          ${ev.length || (prev && prev.execution_error) ? `<div class="evidence">${ev.map((e) => `<code data-t="${e.t}" data-obj="${esc(otherOf(e))}" title="Replay this moment">${e.t.toFixed(1)} s · ${e.kind === "protective_stop" ? `the ${esc(e.link_role || "hand")} pressed on ${esc(nameOf(e.obj))} with ${Math.round(e.force_n || 0)} N` : `hit ${esc(nameOf(otherOf(e)))}${e.while_holding ? ` while carrying ${esc(nameOf(e.while_holding))}` : ""}`}</code>`).join("")}${prev && prev.execution_error && !ev.length ? `<code data-t="${prev.events.length ? prev.events[prev.events.length - 1].t : 0}">${esc(prev.execution_error.error)}</code>` : ""}</div>` : ""}
        </div>`;
      }
    }
    const hitSet = new Set(unexpectedContacts(a));
    const ev = a.events.filter((e) => e.kind !== "contact" || !e.between.includes("table") || e.between.includes("robot_arm"));
    const nBad = a.goals.filter((g) => !g.ok).length;
    html += `<div class="att ${a.success ? "ok" : "bad"}" id="att-${a.attempt}" data-i="${i}"><div class="box" tabindex="0" role="button" aria-label="Replay the ${tryName(i).toLowerCase()}">
      <div class="head"><h4>${tryName(i)}</h4><span class="verdict">${a.success ? "Worked" : "Failed"}</span><span class="t">${a.sim_time ? `${a.sim_time.toFixed(0)} s of simulated time` : ""}</span></div>
      <p class="what">${esc(a.success ? successPlain(a) : failurePlain(a))}</p>
      <p class="goalsum">${a.goals.map((g) => `<span class="${g.ok ? "ok" : "bad"}"><i>${g.ok ? "✓" : "✕"}</i>${esc(goalPlain(g))}</span>`).join("")}</p>
      <p class="replay">Watch this try in the 3D view ▸</p>
      <details class="tech"><summary>Technical details <span class="s">goals as checked, plan, simulator log</span></summary><div class="tech-body">
      <div class="cols"><div><h4>Goals, judged by physics</h4><ul class="goals">${a.goals.map((g) => goalHtml(g)).join("")}</ul>
        <h4>Plan</h4><ol class="plan">${planHtml(a, prev)}</ol></div>
      <div><h4>Simulator log <span class="hint">click a line to replay it</span></h4><ul class="events">${ev.map((e) => {
          let text, hit = false, obj = "", cls = "";
          if (e.kind === "contact") { const [p, q] = e.between; hit = hitSet.has(e); obj = otherOf(e);
            text = `<code>${esc(p)}</code> ↔ <code>${esc(q)}</code>${e.link_role ? ` <span class="role">${esc(e.link_role)}</span>` : ""}${e.while_holding ? ` · holding <code>${esc(e.while_holding)}</code>` : ""}${e.tcp_z != null ? ` · z ${e.tcp_z}` : ""}`; }
          else if (e.kind === "protective_stop") { hit = true; cls = "stop"; obj = e.obj;
            text = `<b>protective stop</b> · ${esc(e.link_role || "robot")} pressed <code>${esc(e.obj)}</code> with ${Math.round(e.force_n || 0)} N${e.phase ? ` during “${esc(e.phase)}”` : ""}`; }
          else if (e.kind === "grasp") { text = `grasp <code>${esc(e.obj)}</code>`; obj = e.obj; }
          else if (e.kind === "release") { text = `release <code>${esc(e.obj)}</code>`; obj = e.obj; }
          else text = `${esc(e.kind)} ${esc(e.msg || e.obj || "")}`;
          return `<li class="${hit ? "hit" : ""} ${cls}" data-t="${e.t}" data-obj="${esc(obj)}"><span class="t">${e.t.toFixed(2)}s</span><span>${text}</span></li>`;
        }).join("")}</ul>${a.execution_error ? `<div class="exec"><b>execution error</b> · ${esc(a.execution_error.error)}</div>` : ""}</div></div></div></details></div></div>`;
  });
  $("attempts").innerHTML = html;
  renderTrace();
  $("attempts").querySelectorAll(".att[data-i] .box").forEach((box) => {
    const i = +box.parentElement.dataset.i;
    const go = () => { state.userPicked = true; loadAttempt(i); markActiveAttempt(); if (state.stage !== "rehearse") setStage("rehearse"); };
    box.onclick = (e) => { if (!e.target.closest("li[data-t], details, summary")) go(); };
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
// Agentic tool-call trace (rehearsal/agent_loop.py): model narration + simulate / stress_test / look_again / submit.
function renderTrace() {
  const r = state.report, tr = r && r.agent_trace, el = $("trace");
  if (!tr || !(tr.turns || tr.story)) { $("traceWrap").hidden = true; el.innerHTML = ""; return; }
  const ag = agentInfo();
  const count = (n) => (tr.turns || []).reduce((s, t) => s + (t.tool_calls || []).filter((c) => c.name === n).length, 0);
  const nSim = count("simulate"), nStress = count("stress_test"), nLook = count("look_again");
  $("traceSummary").textContent = `${cap(plannerName())} ran ${nSim} simulation${nSim === 1 ? "" : "s"}${nStress ? `, ${nStress} repeated test${nStress === 1 ? "" : "s"}` : ""}${nLook ? ` and ${nLook} closer look${nLook === 1 ? "" : "s"}` : ""} on its own before submitting the plan`;
  $("traceWrap").hidden = false;
  let simIdx = -1, rows = "";
  const short = (name, a) => name === "look_again" ? esc(a.object_id || "") : (a.actions || []).map((x) => `${esc(x.obj)}→${esc(x.dest)}@${x.transit_height}${x.via && x.via.length ? " via" : ""}`).join("; ") + (name === "stress_test" && a.n ? `, n=${a.n}` : "");
  const resText = (name, res) => {
    if (!res) return ""; if (res.error) return "error";
    if (name === "submit") return res.accepted ? "ACCEPTED" : "REJECTED";
    if (name === "simulate") return (res.physics || res).success ? "SUCCESS" : "FAILED";
    if (name === "stress_test") return `${res.successes}/${res.n}`;
    if (name === "look_again") return `${res.before} → ${res.after}`;
    return "";
  };
  const turns = tr.turns || [{ tool_calls: (tr.story || []).map((line) => ({ story: line })) }];
  for (const t of turns) {
    if (t.text) rows += `<li class="say"><i></i><span>${esc(t.text)}</span></li>`;
    for (const c of t.tool_calls || []) {
      if (c.story) { rows += `<li><i></i><span class="call">${esc(c.story)}</span></li>`; continue; }
      const name = c.name, res = c.result || {}, ph = res.physics || res;
      const ok = name === "simulate" ? !!ph.success : name === "submit" ? !!res.accepted : null;
      if (name === "simulate" || name === "submit") simIdx++;
      const cls = `${name === "simulate" ? "sim" : name === "stress_test" ? "stress" : name === "look_again" ? "look" : name === "submit" ? "submit" : ""} ${ok === true ? "ok" : ok === false ? "bad" : ""}`;
      const why = ok === false ? ((ph.execution_error || {}).error || (ph.failed_goals || []).slice(0, 2).join("; ") || (ph.evidence || [])[0] || "") : "";
      const idx = (name === "simulate" || name === "submit") && simIdx < r.attempts.length ? ` data-i="${simIdx}" title="replay attempt ${simIdx + 1}"` : "";
      rows += `<li class="${cls}"${idx}><i></i><span class="call"><b>${esc(name)}</b>(${short(name, c.args || {})})</span><span class="res">${esc(resText(name, res))}${c.seconds ? ` · ${c.seconds}s` : ""}</span>${why ? `<span class="why">${esc(String(why).slice(0, 160))}</span>` : ""}</li>`;
    }
  }
  const sub = tr.submitted;
  el.innerHTML = `<h4>Agent loop · tool calls <span class="hint">${esc(tr.stopped || "")}</span></h4>
    <div class="stats-line"><span>${tr.tool_calls ?? "?"} tool calls</span><span>$${(tr.cost_usd || 0).toFixed(4)}</span><span>${tr.prompt_tokens ?? 0} prompt / ${tr.cached_tokens ?? 0} cached / ${tr.completion_tokens ?? 0} completion tokens</span><span>${tr.wall_time_s ?? "?"} s</span>${tr.feedback ? `<span>feedback: ${esc(tr.feedback)}</span>` : ""}${ag.kind === "muse" ? `<span>${esc(tr.model || ag.name)} · ${esc(tr.data || "")} data</span>` : ""}</div>
    <ol>${rows}</ol>${sub && sub.rationale ? `<p class="fine">Submitted: ${esc(sub.rationale)}${sub.verified ? " · verified in simulation" : ""}</p>` : ""}`;
  el.querySelectorAll("li[data-i]").forEach((li) => li.onclick = () => { state.userPicked = true; loadAttempt(+li.dataset.i); markActiveAttempt(); if (state.stage !== "rehearse") setStage("rehearse"); });
}
function markActiveAttempt() {
  document.querySelectorAll(".att[data-i]").forEach((el) => el.classList.toggle("active", +el.dataset.i === state.current));
  document.querySelectorAll("#attemptPills button").forEach((b) => b.setAttribute("aria-selected", +b.dataset.i === state.current));
}

function renderStress() {
  const r = state.report, st = r.stress_test, objs = r.scene_spec.objects;
  if (!st) { $("stressLine").textContent = "Not tested again: the final plan did not succeed."; $("stressNote").textContent = ""; $("stressStats").innerHTML = ""; $("scatter").innerHTML = ""; $("fails").innerHTML = ""; return; }
  const runs = st.runs, fails = runs.filter((x) => !x.success), oks = runs.filter((x) => x.success);
  const ids = objs.map((o) => o.id);
  const mean = (rs, id) => rs.reduce((s, x) => s + x.perception_error_cm[id], 0) / Math.max(rs.length, 1);
  const best = fails.length ? ids.slice().sort((a, b) => (mean(fails, b) - mean(oks, b)) - (mean(fails, a) - mean(oks, a)))[0] : ids[0];
  const sig = Object.values(st.sigma_cm || {})[0];
  $("stressLine").innerHTML = `It worked <span class="num">${st.successes} of ${st.n}</span> times.`;
  const sensitive = fails.length && oks.length && mean(fails, best) > mean(oks, best) + 0.3;
  $("stressNote").textContent = !fails.length ? "Every run worked: this plan is not fragile to small errors."
    : st.successes === 0 ? `Every run failed: a plan that passes once but survives no nudges at all.`
    : fails.length > oks.length ? `Most runs failed, so passing the first time was luck. ${sensitive ? `The plan is most sensitive to where ${the(best)} is: in the failed runs it was off by about ${mean(fails, best).toFixed(1)} cm, against ${mean(oks, best).toFixed(1)} cm in the runs that worked.` : "No single object explains the failures."}`
    : sensitive ? `${fails.length === 1 ? "The one run that failed" : "The runs that failed"} had ${the(best)} out of place by about ${mean(fails, best).toFixed(1)} cm, against ${mean(oks, best).toFixed(1)} cm in the runs that worked. That is where this plan is fragile.`
    : `${fails.length === 1 ? "One run failed" : `${fails.length} runs failed`}; no single object explains it.`;
  $("stressStats").innerHTML = stat("Worked", `<span class="${st.successes === st.n ? "ok" : ""}">${st.successes}</span>/${st.n}`, `${Math.round(100 * st.successes / st.n)}% of runs`) +
    stat("Position noise σ", sig != null ? `${sig.toFixed(1)} cm` : "—", "per object, per run") +
    stat("Most sensitive object", `<code>${esc(best)}</code>`, fails.length ? `failed runs avg ${mean(fails, best).toFixed(1)} cm off vs ${mean(oks, best).toFixed(1)} cm` : "no failures") +
    stat("Failed runs", `<span class="${fails.length ? "bad" : "ok"}">${fails.length}</span>`, fails.length ? "listed below" : "none");
  const sel = $("xobj");
  sel.innerHTML = ids.map((id) => `<option value="${esc(id)}" ${id === best ? "selected" : ""}>${esc(nameOf(id))}</option>`).join("");
  sel.onchange = () => { drawScatter(runs, sel.value); failsTable(fails, sel.value); };
  drawScatter(runs, sel.value); failsTable(fails, sel.value);
}
function failsTable(fails, id) {
  $("fails").innerHTML = fails.length ? `<tr><th>Run (seed)</th><th>${esc(nameOf(id))} off by</th><th>friction ×</th><th>Why it failed</th></tr>` +
    fails.map((x) => `<tr><td class="seed">${x.seed}</td><td class="num">${x.perception_error_cm[id].toFixed(1)} cm</td><td class="num">×${x.friction_scale.toFixed(2)}</td><td class="reason">${x.failed.map(esc).join("<br>")}</td></tr>`).join("")
    : `<tr><td class="fine">Every run worked.</td></tr>`;
}
function drawScatter(runs, id) {
  const W = 560, H = 300, m = { l: 46, r: 12, t: 14, b: 38 };
  const xmax = Math.max(4, Math.ceil(Math.max(...runs.map((x) => x.perception_error_cm[id]))));
  const X = (v) => m.l + (v / xmax) * (W - m.l - m.r), Y = (v) => m.t + (1 - (v - 0.6) / 0.8) * (H - m.t - m.b);
  let g = "";
  for (let v = 0; v <= xmax; v++) g += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--border)"/><text x="${X(v)}" y="${H - m.b + 16}" text-anchor="middle">${v}</text>`;
  for (const v of [0.6, 0.8, 1.0, 1.2, 1.4]) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--border)"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">×${v.toFixed(1)}</text>`;
  g += `<text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">how far ${esc(nameOf(id))} was from where the photo said (cm)</text>`;
  g += `<text transform="translate(11 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">friction ×</text>`;
  const failErr = runs.filter((x) => !x.success).map((x) => x.perception_error_cm[id]);
  if (failErr.length) {
    const mn = Math.min(...failErr);
    g += `<line x1="${X(mn)}" x2="${X(mn)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--bad)" stroke-dasharray="4 4"/><text x="${X(mn) + 6}" y="${m.t + 10}" style="fill:var(--bad)">${mn.toFixed(1)} cm</text>`;
  }
  for (const x of runs) {
    const px = X(x.perception_error_cm[id]), py = Y(x.friction_scale);
    const t = esc(`<b>run ${x.seed} · ${x.success ? "worked" : "failed"}</b>${esc(nameOf(id))} off by ${x.perception_error_cm[id].toFixed(1)} cm<br>friction ×${x.friction_scale.toFixed(2)} · weight ×${x.mass_scale.toFixed(2)}` + (x.failed.length ? "<br>" + x.failed.map(esc).join("<br>") : ""));
    g += `<g data-tip="${t}"><circle cx="${px}" cy="${py}" r="10" fill="transparent"/>` + (x.success
      ? `<circle cx="${px}" cy="${py}" r="4.5" fill="var(--good)" stroke="var(--surface)" stroke-width="1.5"/>`
      : `<path d="M${px - 5} ${py - 5}L${px + 5} ${py + 5}M${px + 5} ${py - 5}L${px - 5} ${py + 5}" stroke="var(--bad)" stroke-width="2.5" stroke-linecap="round"/>`) + `</g>`;
  }
  $("scatter").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Stress test: perception error vs friction, coloured by outcome">${g}</svg>`;
  bindTips($("scatter"));
}

// sigma_cm is a per-object dict in newer reports, a number in older ones
const sigNum = (ap) => typeof ap.sigma_cm === "number" ? ap.sigma_cm : (ap.default_sigma_cm ?? ap.sigma_cm_mean ?? 1.2);

function renderLook() {
  const r = state.report, ap = r.active_perception, st = r.stress_test;
  $("s5").hidden = !ap;
  if (!ap) return;
  $("sig0").textContent = sigNum(ap).toFixed(1); $("sig1").textContent = ap.refined_sigma_cm.toFixed(1);
  const n = ap.n;
  if (!ap.steps.length) {
    $("lookLine").textContent = st && st.successes === n ? "Nothing to look at again: every run already worked." : "No single object is worth a second look here.";
    $("lookNote").textContent = st && st.successes === n ? "" : "Re-measuring any one object does not raise the success rate, so the weak point is not where things are.";
    $("lookBars").innerHTML = ""; $("lookChain").innerHTML = ""; return;
  }
  const first = ap.steps[0], vals = first.value_by_object, ho = ap.heldout, sel = ap.selection;
  const ranked = Object.keys(vals).sort((a, b) => vals[b] - vals[a]);
  {
    const looked = ap.steps.map((s) => s.look_at), lastStep = ap.steps[ap.steps.length - 1];
    const before = ho ? ho.before : first.before, after = ho ? ho.after : lastStep.after;
    $("lookLine").innerHTML = `A closer look at ${esc(list(looked.map(the)))} takes it from ${before} to <span class="num">${after} of ${n}</span>.`;
    const useless = ranked.filter((k) => vals[k] <= first.before);
    $("lookNote").textContent = (useless.length ? `Re-measuring ${list(useless.map(the))} would not help. ` : "") +
      (ho ? "The gain is measured on fresh test runs, not the ones used to choose. " : "") +
      "In this build a closer look means a tighter position estimate; with a live camera it becomes a prompt to photograph that object from another angle.";
  }
  const W = 560, rowH = 30, m = { l: 124, r: 98, t: 6 }, H = m.t + ranked.length * rowH + 6;
  const maxGain = Math.max(1, ...ranked.map((k) => vals[k] - first.before));
  const X = (v) => m.l + (v / maxGain) * (W - m.l - m.r);
  let g = `<line x1="${X(0)}" x2="${X(0)}" y1="${m.t - 2}" y2="${H - 4}" stroke="var(--border-strong)"/>`;
  ranked.forEach((k, i) => {
    const gain = vals[k] - first.before, y = m.t + i * rowH, chosen = k === first.look_at;
    g += `<g data-tip="${esc(`<b>a closer look at ${esc(nameOf(k))}</b>${first.before}/${n} → ${vals[k]}/${n}`)}"><rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>
      <text x="${m.l - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" style="${chosen ? "fill:var(--text);font-weight:600" : ""}">${esc(nameOf(k).length > 18 ? k : nameOf(k))}</text>`;
    if (gain > 0) { const w = X(gain) - X(0); g += `<rect x="${X(0)}" y="${y + 7}" width="${w}" height="${rowH - 14}" rx="4" fill="${chosen ? "var(--accent)" : "var(--muted)"}"/>`; }
    g += `<text x="${X(Math.max(gain, 0)) + 8}" y="${y + rowH / 2 + 4}" style="fill:var(--text-2)">${gain > 0 ? "+" + gain : "0"} · ${vals[k]}/${n}</text></g>`;
  });
  $("lookBars").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Value of re-observing each object (selection runs)">${g}</svg>` +
    (ho ? `<p class="fine">Bars: selection runs, seeds ${sel ? `${sel.seed0}–${sel.seed0 + sel.n - 1}` : ""} (the choice was made on these, so they are optimistic). The headline gain and the 3D paths are measured on held-out runs, seeds ${ho.seed0}–${ho.seed0 + ho.n - 1}.</p>` : "");
  bindTips($("lookBars"));
  const lastStep = ap.steps[ap.steps.length - 1], looked = ap.steps.map((s) => s.look_at);
  const before = ho ? ho.before : first.before, after = ho ? ho.after : lastStep.after;
  let chain = `<div class="stepc"><div class="lbl">From one photo · σ ≈ ${sigNum(ap).toFixed(1)} cm per object${ho ? ` · held-out runs` : ""}</div><div class="big">${before}/${n}</div></div>`;
  chain += `<div class="arrow">↓ closer look at ${looked.map((k) => `<code>${esc(k)}</code>`).join(", ")} · σ → ${ap.refined_sigma_cm.toFixed(1)} cm for that object</div>`;
  chain += `<div class="stepc final"><div class="lbl">σ = ${ap.refined_sigma_cm.toFixed(1)} cm for ${looked.map(esc).join(", ")}${ho ? " · held-out runs" : " · same runs"}</div><div class="big">${after}/${n}</div></div>`;
  if (ho) chain += `<p class="fine">Selection runs (optimistic): ${first.before}/${n} → ${ap.steps.map((s) => `${esc(s.look_at)} ${s.after}/${n}`).join(" → ")}.</p>`;
  $("lookChain").innerHTML = chain;
}

async function renderGallery() {
  let items = await getJson("/api/gallery");
  if (!items) items = ((await getJson("/api/scenes")) || []).map((s) => ({ scene: s, title: s, thumb: null, has_report: true }));
  state.gallery = items;
  // "physics caught the model": a correction, a protective stop or an IK failure in the recorded rehearsal
  await Promise.all(items.filter((it) => it.has_report && state.hard[it.scene] === undefined).map(async (it) => {
    const rep = await getJson(`/api/report?scene=${encodeURIComponent(it.scene)}`);
    state.hard[it.scene] = !!rep && (rep.attempts.length > 1 || rep.attempts.some((a) => a.execution_error || a.events.some((e) => e.kind === "protective_stop")));
  }));
  items = [...SCENE_ORDER.map((s) => items.find((it) => it.scene === s)).filter(Boolean), ...items.filter((it) => !SCENE_ORDER.includes(it.scene))];
  const empty = !items.length ? `<p class="fine">No tables on this server yet. Upload a photo in step 1's technical details.</p>` : "";
  $("gallery").innerHTML = empty + items.map((it) => {
    const meta = SCENE_META[it.scene] || {};
    const stakes = meta.stakes || (it.demo_instruction ? `“${it.demo_instruction}”` : "");
    return `<button type="button" class="gal ${it.scene === state.scene ? "sel" : ""}" data-scene="${esc(it.scene)}">
    <div class="th">${it.thumb ? `<img src="${esc(it.thumb)}" alt="" loading="lazy">` : `<span>${it.n_objects ? `${it.n_objects} objects` : "no picture yet"}</span>`}</div>
    <div class="cap">${meta.real ? `<span class="real">Real photo</span>` : ""}<b>${esc(sceneTitle(it.scene))}</b><span>${esc(stakes)}</span>${!it.has_report ? `<span class="fine">No rehearsal run yet.</span>` : state.hard[it.scene] ? "" : ""}</div></button>`;
  }).join("");
  $("gallery").querySelectorAll(".gal").forEach((b) => b.onclick = () => selectScene(b.dataset.scene));
}

// ------------------------------------------------------------------ jobs (perceive / rehearse)
// Job card: title, state badge, queue position, phase stepper, progress bar, live log, error card.
const PERCEIVE_STEPS = [["upload", /upload|receiv/i], ["Muse Spark lists objects", /list|muse|spark|attribute/i], ["SAM 3.1 finds them", /segment|sam|mask/i], ["scale", /calib|homograph|paper|sheet/i], ["build the copy", /valid|twin|scene|save/i]];
const REHEARSE_STEPS = [["setup", /setup|start|reach/i], ["tries", /attempt|diagnos|plan/i], ["repeated test", /stress/i], ["closer look", /look|active/i], ["report", /report|done/i]];
function jobBox(el, title, steps, badges = "") {
  el.hidden = false;
  el.innerHTML = `<div class="jobhead"><b>${title}</b><span class="tag run jstate">queued</span><span class="grow"></span>${badges}</div>
    <div class="errcard" hidden></div>
    <div class="stepper">${steps.map(([l]) => `<div>${esc(l)}</div>`).join("")}</div>
    <div class="phase"><span class="tag jp">queued</span><div class="pbar"><i></i></div><span class="mono jf">0%</span></div><pre aria-live="polite"></pre>`;
  const stepEls = [...el.querySelectorAll(".stepper div")];
  return {
    update(j) {
      const ph = j.progress ? j.progress.phase : j.state, logs = j.log || [];
      let idx = steps.findIndex(([, re]) => re.test(ph));
      if (idx < 0) for (let i = logs.length - 1; i >= 0 && idx < 0; i--) idx = steps.findIndex(([, re]) => re.test(logs[i]));
      if (j.state === "done") idx = steps.length;
      const f = j.state === "done" ? 1 : (j.progress && j.progress.fraction != null ? j.progress.fraction : Math.max(0, idx) / steps.length);
      stepEls.forEach((s, i) => { s.className = i < idx ? "done" : i === idx ? "now" : ""; });
      const st = el.querySelector(".jstate");
      st.textContent = j.state === "error" ? "error" : j.state === "done" ? "done" : j.state === "queued" ? `queued${j.queue_ahead ? ` · ${j.queue_ahead} ahead` : ""}` : "running";
      st.className = `tag jstate ${j.state === "error" ? "bad" : j.state === "done" ? "ok" : "run"}`;
      el.querySelector(".jp").textContent = ph + (j.queue_ahead ? ` · ${j.queue_ahead} ahead` : "");
      el.querySelector(".pbar i").style.width = `${Math.round(f * 100)}%`;
      el.querySelector(".jf").textContent = `${Math.round(f * 100)}%`;
      const pre = el.querySelector("pre");
      pre.innerHTML = logs.slice(-200).map((l) => /error|fail|traceback/i.test(l) ? `<span class="err">${esc(l)}</span>` : /muse spark|sam 3/i.test(l) ? `<span class="mm">${esc(l)}</span>` : /success|✓/i.test(l) ? `<span class="ok">${esc(l)}</span>` : esc(l)).join("\n") + (j.state === "running" ? "\n▍" : "") || "waiting for the first log line…";
      pre.scrollTop = pre.scrollHeight;
      const err = el.querySelector(".errcard"); err.hidden = !j.error; if (j.error) err.textContent = j.error;
    },
  };
}
async function pollJob(id, box) {
  for (;;) {
    const j = await apiJson(`/api/job?id=${encodeURIComponent(id)}`);
    box.update(j);
    if (j.state === "done") return j.result;
    if (j.state === "error") { const e = new Error(j.error || "job failed"); e.log = j.log; throw e; }
    await sleep(500);
  }
}
async function runRehearsal(e) {
  e.preventDefault();
  if (state.busy) { toast("A job is already running."); return; }
  if (!$("instruction").value.trim()) { toast("Write an instruction first."); $("instruction").focus(); return; }
  if (!$("saveBar").hidden && state.saved !== null) { toast("Save the edited twin first (or undo the edits)."); return; }
  state.busy = true;
  const btn = $("run"); btn.disabled = true; btn.textContent = "Rehearsing…";
  const box = jobBox($("rehearseJob"), `Rehearsing “${esc(sceneTitle(state.scene))}”`, REHEARSE_STEPS, `<span class="tag">Meta Muse Spark + MuJoCo</span>`);
  box.update({ state: "queued", log: [], progress: { phase: "queued", fraction: 0 } });
  try {
    const j = await apiJson("/api/rehearse", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene: state.scene, instruction: $("instruction").value, stress: +$("stress").value, look: +$("look").value }) });
    if (j.job) await pollJob(j.job, box);
    else for (;;) { // legacy /api/status
      const st = await getJson("/api/status"); box.update({ state: st.running ? "running" : (st.error ? "error" : "done"), log: st.log, error: st.error, progress: st.progress });
      if (!st.running) { if (st.error) throw new Error(st.error); break; }
      await sleep(600);
    }
    state.bust = Date.now();
    await loadScene();
    await renderGallery();
    $("rehearseJob").hidden = true;
    toast(`Rehearsal of ${state.scene} finished.`);
    $("s3").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) { box.update({ state: "error", error: String(err.message || err), log: err.log || [] }); }
  finally { state.busy = false; btn.disabled = false; btn.textContent = "Rehearse"; loadBudget(); }
}
function bindUpload() {
  const drop = $("drop"), file = $("file");
  drop.onclick = (e) => { if (!e.target.closest(".drop-opts")) file.click(); };
  drop.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === drop) { e.preventDefault(); file.click(); } };
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) perceive(e.dataTransfer.files[0]); };
  file.onchange = () => { if (file.files[0]) perceive(file.files[0]); file.value = ""; };
}
async function perceive(f) {
  if (state.busy) { toast("A job is already running."); return; }
  if (!f.type.startsWith("image/")) { toast("Please choose an image (JPEG, PNG or WebP)."); return; }
  if (f.size > 15 * 1048576) { toast("The photo must be 15 MB or smaller."); return; }
  state.busy = true;
  const privacy = $("privacy").value, robotSide = $("robotSide").value;
  const box = jobBox($("perceiveJob"), `Reading ${esc(f.name)}`, PERCEIVE_STEPS, `<span class="tag">${esc(privacy)} data · Meta Muse Spark + SAM 3.1</span>`);
  box.update({ state: "queued", log: [], progress: { phase: "upload", fraction: 0.02 } });
  const localUrl = URL.createObjectURL(f);
  try {
    const fd = new FormData(); fd.append("photo", f); fd.append("privacy", privacy); fd.append("robot_side", robotSide);
    const j = await apiJson("/api/perceive", { method: "POST", body: fd });
    const res = await pollJob(j.job, box);
    if (!res || !res.scene_spec) throw new Error("perception returned no scene");
    const spec = res.scene_spec;
    state.scene = res.scene || spec.name;
    if (![...$("scene").options].some((o) => o.value === state.scene)) $("scene").add(new Option(sceneTitle(state.scene), state.scene));
    $("scene").value = state.scene;
    await renderGallery();
    await loadScene({ spec, overlay: res.overlay || null, uploadUrl: j.upload || localUrl, saved: res.valid !== false });
    if (res.valid === false) {
      $("saveBar").hidden = false; $("saveBar").classList.add("issue");
      $("saveMsg").textContent = `Perception needs a correction before it can be saved: ${(res.issues || []).join("; ")}. Drag objects to fix it, then Save twin.`;
      if (!$("saveName").value) $("saveName").value = state.scene;
    } else setTimeout(() => { $("perceiveJob").hidden = true; }, 1800);
    $("s1").querySelector(".photo-twin").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) { box.update({ state: "error", error: String(err.message || err), log: err.log || [] }); }
  finally { state.busy = false; loadBudget(); }
}

// ------------------------------------------------------------------ public surface for demo.js
const app = {
  state, replay, $, esc, GLOSS, sleep, diagText, hitsOf, otherOf, planChanges, planChangesPlain, agentInfo, hasPerception,
  loadAttempt, setChip, setCaption, showTrailSet, fanOut, unexpectedContacts, nameOf, the, list, cap, goalPlain, failurePlain, successPlain, actionPlain, tryName, sceneTitle, plannerName,
  ready: () => state.loaded,
  get hero() { return hero; },
};
window.app = app;
init();
