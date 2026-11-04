// Rehearsal product app: photo -> perceive -> 3D twin -> instruct -> rehearse -> replay.
// Every number and verdict on screen comes from the server's real pipeline (rehearsal/app_api.py).
import { PerceptionHUD } from "/app/hud.js";
import { Studio } from "/app/viewer.js";
import { AttemptWall } from "/app/tiles.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const STEPS = ["photo", "perceive", "twin", "instruct", "rehearse"];
const money = (v) => "$" + (v || 0).toFixed(v && v < 0.01 ? 4 : 3);

const S = { cfg: null, samples: [], group: "tables", sid: null, info: null, twin: null, twinP: null, step: "photo",
  maxStep: 0, runs: [], sessionCost: 0, intent: null, instruction: "", turns: [], busy: false, view: "pick", runSel: null };
window.__rehearsal = S;   // read-only handle for the headless smoke/capture scripts

async function api(path, body) {
  const opt = body instanceof FormData ? { method: "POST", body } :
    body !== undefined ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
async function poll(jid, onEvents, every = 350, maxS = 900) {
  let since = 0, fails = 0;
  const t0 = performance.now();
  for (;;) {
    if ((performance.now() - t0) / 1000 > maxS) throw new Error(`no result after ${maxS} s; the server may be overloaded. Try again.`);
    let e;
    try { e = await api(`/api/app/events?job=${jid}&since=${since}`); fails = 0; }
    catch (err) {                       // a dropped connection is not a failed job: back off and keep polling
      if (++fails > 8) throw new Error("lost the connection to the server (" + err.message + ")");
      await sleep(Math.min(4000, 500 * fails)); continue;
    }
    if (e.events.length) { since += e.events.length; onEvents && onEvents(e.events); }
    const st = e.job && e.job.state;
    if (st === "done" || st === "error") {
      const j = await api(`/api/job?id=${jid}`);
      if (j.state === "error") throw new Error(j.error || "job failed");
      return j.result;
    }
    if (!e.job) throw new Error("job lost (server restarted?)");
    await sleep(every);
  }
}
function toast(msg, ms = 4200) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}
function addCost(v) { if (v) { S.sessionCost += v; } $("#costPill").textContent = `${money(S.sessionCost)} session`; }

// ------------------------------------------------------------------------------------------- stepper + views
function setStep(step) {
  S.step = step;
  S.maxStep = Math.max(S.maxStep, STEPS.indexOf(step));
  $$("#stepper button").forEach((b) => {
    const i = STEPS.indexOf(b.dataset.step);
    b.classList.toggle("cur", b.dataset.step === step);
    b.classList.toggle("done", i < STEPS.indexOf(step) || (i <= S.maxStep && b.dataset.step !== step));
  });
}
function showView(v) {
  S.view = v;
  const map = { pick: "#viewPick", photo: "#viewPhoto", twin: "#viewTwin" };
  for (const [k, sel] of Object.entries(map)) {
    const el = $(sel);
    if (k === v) { el.classList.remove("hidden"); requestAnimationFrame(() => el.classList.remove("fade")); }
    else { el.classList.add("fade"); setTimeout(() => { if (S.view !== k) el.classList.add("hidden"); }, 600); }
  }
  $$("#stageTabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("#stageTabs").classList.toggle("hidden", v === "pick" || !S.twin || !S.info?.photo);
}
$("#stepper").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b || !b.classList.contains("done")) return;
  const s = b.dataset.step;
  if (s === "photo") return resetToPick();
  if (s === "perceive") { if (S.info) showView("photo"); return; }
  if (S.twin) { showView("twin"); const c = { twin: "#cardTwin", instruct: "#cardInstruct", rehearse: "#cardRehearse" }[s]; const el = c && $(c); el && el.scrollIntoView({ behavior: "smooth", block: "start" }); }
});
$("#stageTabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) showView(b.dataset.view); });

// ------------------------------------------------------------------------------------------- 01 photo
async function init() {
  const [cfg, samples] = await Promise.all([api("/api/app/config"), api("/api/app/samples")]);
  S.cfg = cfg; S.samples = samples;
  const mp = $("#modePill");
  mp.textContent = cfg.live ? "LIVE · MUSE SPARK" : "RECORDED · NO MODEL CALLS";
  mp.classList.toggle("live", cfg.live);
  mp.title = cfg.live ? "This server calls Muse Spark and SAM 3.1 through the Meta API (cached requests are free)"
    : "This server makes no model calls: Muse Spark answers come from recorded responses; planning and physics run live";
  addCost(0);
  const drop = $("#drop");
  if (!cfg.uploads) {
    drop.classList.add("disabled");
    $("#dropSub").textContent = cfg.public_demo ? "Upload is off on the public demo. Every sample below runs live planning and physics."
      : "Upload needs a Meta API key on this server. Samples below still run live planning and physics.";
  } else {
    $("#dropNotice").innerHTML = cfg.upload_data === "public"
      ? "<b>Heads up:</b> photos are processed by Meta AI models in <b>contributor mode</b>. Don't upload private or personal images."
      : "Photos are processed by Meta AI models in Standard (private) mode.";
  }
  if (cfg.my_desk) $("#tabLocal").classList.remove("hidden");
  $("#railFoot").innerHTML = cfg.budget ? `Meta API ledger ${money(cfg.budget.spent)} of ${money(cfg.budget.cap)} cap` : "";
  renderGrid();
  const want = new URLSearchParams(location.search).get("scene");
  if (want && samples.some((s) => s.id === want)) selectScene(want);
}
function renderGrid() {
  $$("#sampleTabs button").forEach((b) => b.classList.toggle("on", b.dataset.g === S.group));
  const items = S.samples.filter((s) => s.group === S.group);
  $("#sampleGrid").innerHTML = items.map((s) => `
    <button class="card" data-id="${esc(s.id)}">
      <div class="th" style="background-image:url('${esc(s.thumb || "")}')"></div>
      <div class="flag">${s.local_only ? '<span class="badge warn">Local only</span>' : s.ai_generated ? '<span class="badge">AI-generated photo</span>' : s.group === "authored" ? '<span class="badge">Hand-authored</span>' : '<span class="badge">Photo</span>'}</div>
      <div class="cap"><div class="ct">${esc(s.title)}</div>
      <div class="cm">${s.n_objects} OBJECTS${s.sam3d ? " · SAM 3D" : ""}</div></div>
    </button>`).join("");
}
$("#sampleTabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { S.group = b.dataset.g; renderGrid(); } });
$("#sampleGrid").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) selectScene(c.dataset.id); });
$("#changeTable").addEventListener("click", resetToPick);

function resetToPick() {
  if (S.busy) return toast("A rehearsal is running; wait for its verdict.");
  S.sid = null; S.info = null; S.twin = null; S.maxStep = 0;
  $("#tableCard").classList.add("hidden"); $("#runsBox").classList.add("hidden"); $("#railEmpty").classList.remove("hidden");
  $("#panelScroll").innerHTML = "";
  setStep("photo"); showView("pick");
}

// upload
const drop = $("#drop"), fileIn = $("#fileIn");
drop.addEventListener("click", (e) => { if (drop.classList.contains("disabled")) { e.preventDefault(); toast($("#dropSub").textContent); } });
["dragenter", "dragover"].forEach((k) => drop.addEventListener(k, (e) => { e.preventDefault(); if (!drop.classList.contains("disabled")) drop.classList.add("over"); }));
["dragleave", "drop"].forEach((k) => drop.addEventListener(k, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f && !drop.classList.contains("disabled")) upload(f); });
fileIn.addEventListener("change", () => { if (fileIn.files[0]) upload(fileIn.files[0]); fileIn.value = ""; });

async function upload(file) {
  if (file.size > 15 * 1024 * 1024) return toast("That photo is over 15 MB.");
  S.busy = true;
  setStep("perceive"); showView("photo");
  $("#noPhoto").classList.add("hidden"); $("#photoWrap").classList.remove("hidden");
  const img = $("#photoImg"); img.src = URL.createObjectURL(file);
  hud.set({ width: 1, height: 1, objects: [] });
  $("#photoWrap").classList.add("dim"); $("#scan").classList.remove("hidden");
  $("#percSub").textContent = "UPLOADING"; $("#percSrc").innerHTML = "<small>MUSE SPARK · SAM 3.1</small>";
  $("#railEmpty").classList.add("hidden");
  const panel = $("#panelScroll"); panel.innerHTML = "";
  const card = stepCard("cardPerceive", "02", "Perceive", "LIVE");
  panel.appendChild(card);
  const body = $(".step-b", card);
  body.innerHTML = `<p class="muted">Muse Spark names the objects and the tabletop; SAM 3.1 outlines each one. Then camera, table and
    object primitives are fitted (no reference object needed).</p><div class="feed" id="upFeed"></div>
    <div class="meta">Meta API data mode: ${esc(S.cfg.upload_data === "public" ? "contributor" : "standard (private)")}</div>`;
  const feed = $("#upFeed", card);
  const t0 = performance.now();
  const tick = setInterval(() => { $("#percSub").textContent = `PERCEIVING · ${((performance.now() - t0) / 1000).toFixed(0)} S`; }, 500);
  try {
    const fd = new FormData(); fd.append("photo", file);
    const { job } = await api("/api/app/upload", fd);
    const res = await poll(job, (evs) => {
      for (const ev of evs) {
        if (ev.type === "phase") feed.insertAdjacentHTML("beforeend", `<div class="cand note">▸ ${esc(ev.text)}</div>`);
        if (ev.type === "objects") feed.insertAdjacentHTML("beforeend", `<div class="cand note">▸ ${ev.n} objects: ${esc(ev.labels.slice(0, 12).join(", "))}</div>`);
      }
    }, 600);
    addCost(res.cost_usd);
    clearInterval(tick);
    S.busy = false;
    $("#scan").classList.add("hidden");
    toast(`Perceived ${res.n_objects} objects · ${money(res.cost_usd)}`);
    const s = { id: res.scene, title: "Your photo", group: "upload", n_objects: res.n_objects };
    S.samples.push(s);
    await selectScene(res.scene, { cost: res.cost_usd });
  } catch (e) {
    clearInterval(tick); S.busy = false; $("#scan").classList.add("hidden");
    feed.insertAdjacentHTML("beforeend", `<div class="cand no"><span></span><span class="ck">✕</span><span class="ct">${esc(e.message)}</span></div>
      <p class="muted">Nothing was invented. Retry the photo, or continue with a sample table: those carry recorded Muse Spark + SAM 3.1 perception and work offline.</p>
      <div class="row-btns"><button class="primary" id="upRetry">Retry this photo</button><button class="ghost" id="upSample">Use a sample table</button></div>`);
    $("#upRetry", card).onclick = () => upload(file);
    $("#upSample", card).onclick = () => { location.href = "/app"; };
    toast("Perception failed: " + e.message, 7000);
  }
}

// ------------------------------------------------------------------------------------------- 02 perceive
const hud = new PerceptionHUD($("#hudCanvas"), $("#photoImg"));

async function selectScene(id, { cost } = {}) {
  S.sid = id; S.twin = null; S.runs = loadRuns(id); S.turns = []; S.intent = null; S.runSel = null;
  S.info = await api(`/api/app/scene?id=${encodeURIComponent(id)}`);
  const info = S.info;
  S.twinP = api("/api/app/twin", { scene: id });             // build the twin while the HUD plays
  S.twinP.catch(() => {});
  history.replaceState(null, "", `/app?scene=${encodeURIComponent(id)}`);
  // rail
  $("#railEmpty").classList.add("hidden"); $("#tableCard").classList.remove("hidden");
  const thumb = info.photo || (S.samples.find((s) => s.id === id) || {}).thumb || "";
  $("#tcThumb").style.backgroundImage = thumb ? `url('${thumb}')` : "";
  $("#tcTitle").textContent = info.title;
  const badges = [];
  if (info.local_only) badges.push('<span class="badge warn">Local only</span>');
  if ((info.provenance || "").includes("AI-generated")) badges.push('<span class="badge">AI-generated photo</span>');
  badges.push(`<span class="badge">${info.privacy === "public" ? "Contributor mode" : "Standard mode"}</span>`);
  $("#tcMeta").innerHTML = `${info.objects.length} objects · ${(info.table.size[0] * 100).toFixed(0)} × ${(info.table.size[1] * 100).toFixed(0)} cm table<br>${badges.join("")}`;
  renderRuns();
  // stage
  setStep("perceive");
  const panel = $("#panelScroll"); panel.innerHTML = "";
  const card = stepCard("cardPerceive", "02", "Perceive", info.photo ? "SAM 3.1 · MUSE SPARK" : "AUTHORED");
  panel.appendChild(card);
  const body = $(".step-b", card);
  const src = info.local_only ? "Cached perception of your desk (no new Meta calls; never uploaded)."
    : id.startsWith("photo_") ? `Live perception of your photo${cost ? ` · ${money(cost)}` : ""}.`
      : info.photo ? "Recorded perception (Muse Spark labels + SAM 3.1 masks stored with this scene) · $0." : "Hand-authored scene: no photo, no perception.";
  const nm = info.objects.filter((o) => o.mask_px).length;
  const models = info.models ? info.models.map((m) => ({ ...m, role: m.role + (m.model === "SAM 3.1" ? ` · ${nm} masks` : ""), state: m.calls ? "live" : "cached" }))
    : info.photo ? [{ model: "Muse Spark", role: `object list · ${info.objects.length}`, state: info.local_only ? "cached" : "recorded", usd: info.local_only ? undefined : 0 },
                    { model: "SAM 3.1", role: `${nm} masks`, state: info.local_only ? "cached" : "recorded", usd: info.local_only ? undefined : 0 }] : [];
  body.innerHTML = `
    ${models.length ? modelBadges(models) : ""}
    <p class="muted">${esc(src)}</p>
    <div class="objlist" id="objList">${info.objects.map((o, i) => `
      <div class="obj ${info.photo ? "pending" : ""}" data-id="${esc(o.id)}"><span class="num">${String(i + 1).padStart(2, "0")}</span>
      <span class="sw" style="background:${rgbaCss(o.rgba)}"></span><span class="lb">${esc(o.label)}</span>
      <span class="cf">${typeof o.conf === "number" ? o.conf.toFixed(2) : ""}</span></div>`).join("")}</div>
    <button class="primary" id="buildBtn">Build the 3D twin <span class="k">↵</span></button>`;
  card.querySelector(".sum").textContent = `${info.objects.length} objects outlined`;
  $("#buildBtn").onclick = buildTwin;
  $$(".obj", body).forEach((el) => {
    el.onmouseenter = () => hud.setHover(el.dataset.id);
    el.onmouseleave = () => hud.setHover(null);
  });
  if (info.photo) {
    showView("photo");
    $("#noPhoto").classList.add("hidden"); $("#photoWrap").classList.remove("hidden");
    $("#photoWrap").classList.remove("dim"); $("#scan").classList.add("hidden");
    const img = $("#photoImg");
    await new Promise((res) => { img.onload = res; img.onerror = res; img.src = info.photo; });
    $("#percSrc").innerHTML = `<small>${info.local_only ? "CACHED · LOCAL ONLY" : id.startsWith("photo_") ? "LIVE · META API" : "RECORDED · $0"}</small>`;
    hud.set(info);
    const n = hud.objs.length;
    $("#percSub").textContent = `OBJECTS 00/${String(n).padStart(2, "0")}`;
    hud.onReveal = (k, objs) => {
      $("#percSub").textContent = `OBJECTS ${String(k).padStart(2, "0")}/${String(n).padStart(2, "0")}`;
      const ids = new Set(objs.slice(0, k).map((o) => o.id));
      $$("#objList .obj").forEach((el) => { if (ids.has(el.dataset.id)) el.classList.remove("pending"); });
      if (k === n) { $$("#objList .obj").forEach((el) => el.classList.remove("pending")); $("#photoWrap").classList.add("dim"); }
    };
    setTimeout(() => $("#photoWrap").classList.add("dim"), 150);
    hud.play();
  } else {
    showView("photo");
    $("#photoWrap").classList.add("hidden"); $("#noPhoto").classList.remove("hidden");
    $("#percSub").textContent = `OBJECTS ${String(info.objects.length).padStart(2, "0")}`;
    $("#percSrc").innerHTML = "<small>HAND-AUTHORED</small>";
  }
}
function rgbaCss(c) { if (!c) return "#666"; return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`; }

function stepCard(id, n, title, sub) {
  const el = document.createElement("section");
  el.className = "step"; el.id = id;
  el.innerHTML = `<div class="step-h"><span class="n">${n}</span><span class="t">${esc(title)}</span><span class="s">${esc(sub || "")}</span></div>
    <div class="sum"></div><div class="step-b"></div>`;
  $(".step-h", el).onclick = () => { if (el.classList.contains("done")) el.classList.toggle("collapsed"); };
  return el;
}
function collapse(id, summary) {
  const el = $("#" + id); if (!el) return;
  el.classList.add("done", "collapsed");
  if (summary) $(".sum", el).innerHTML = summary;
}

// ------------------------------------------------------------------------------------------- 03 twin
let studio = null;
function ensureStudio() {
  if (studio) return studio;
  studio = new Studio($("#viewer"), {
    onTime: (t, d) => {
      const f = d > 0 ? (t - studio.rp.t0) / (d - studio.rp.t0) : 0;
      $("#scrubFill").style.width = `${Math.max(0, Math.min(1, f)) * 100}%`;
      $("#timeLbl").textContent = `${t.toFixed(1)} s`;
    },
    onEnd: () => setPlayIcon(false),
  });
  return studio;
}
async function buildTwin() {
  if (S.view === "twin" && S.twin) return;
  hud.finish();
  setStep("twin");
  ensureStudio();
  showView("twin");
  $("#twinLoading").classList.remove("hidden");
  $("#verdict").classList.add("hidden"); $("#replayBar").classList.add("hidden");
  $("#simSub").textContent = "BUILDING TWIN";
  collapse("cardPerceive", `${S.info.objects.length} objects outlined`);
  let card = $("#cardTwin");
  if (!card) { card = stepCard("cardTwin", "03", "3D twin", "MUJOCO"); $("#panelScroll").appendChild(card); }
  $(".step-b", card).innerHTML = `<div class="thinking"><div class="spin"></div>Exporting the MuJoCo model and settling the scene…</div>`;
  try {
    const twin = await S.twinP.catch(() => api("/api/app/twin", { scene: S.sid }));
    S.twin = twin;
    studio._instant = true; studio.look = "physics";
    await studio.load(twin, null, S.info);
    studio._instant = false;
    setLookBtns("physics");
    wallR().load(twin);
    $("#twinLoading").classList.add("hidden");
    $("#simSub").textContent = `PRIMITIVES · ${String(twin.objects.length).padStart(2, "0")} RIGID BODIES`;
    // hero moment: primitives settle under gravity, then the refined, photo-textured twin sweeps in
    if (studio.proj) {
      clearTimeout(S.morphT);
      S.morphT = setTimeout(() => {
        if (S.twin !== twin || studio.look !== "physics") return;
        $("#simSub").textContent = "REFINING · PHOTO PROJECTED THROUGH THE CALIBRATED CAMERA";
        setLook("visual");
        setTimeout(() => { if (S.twin === twin && !S.rehearsing) $("#simSub").textContent = `TWIN READY · ${twin.objects.length} OBJECTS · PHYSICS + PHOTO`; }, 2600);
      }, 1700);
    }
    showView("twin");
    renderTwinCard(card);
    collapse("cardTwin", `${twin.objects.length} bodies + Franka arm · built in ${twin.build_s ?? "<1"} s · settled under gravity`);
    openInstruct();
  } catch (e) {
    $("#twinLoading").classList.add("hidden");
    $(".step-b", card).innerHTML = `<div class="bub no"><span class="who">TWIN FAILED</span>${esc(e.message)}</div>`;
  }
}
function renderTwinCard(card) {
  const t = S.twin;
  const vis = t.visual_meshes.length;
  $(".step-b", card).innerHTML = `
    ${modelBadges([{ model: "MuJoCo", role: `${t.objects.length} rigid bodies + Franka`, usd: 0, state: "local" },
                   ...(vis || S.cfg.sam3d ? [{ model: "SAM 3D", role: vis ? `${vis} refined meshes` : "refine on demand", state: vis ? "cached" : "local" }] : [])])}
    <div class="stats"><div class="stat"><b>${t.objects.length}</b><span>Rigid bodies</span></div>
      <div class="stat"><b>${t.build_s ?? "–"}s</b><span>Build</span></div>
      <div class="stat"><b>${vis}</b><span>SAM 3D meshes</span></div></div>
    <p class="muted">Each object is a collision primitive with mass and friction from Muse Spark's estimate. The robot is a Franka Panda
      (MuJoCo Menagerie). <b>Physics</b> shows what is simulated; <b>Visual</b> adds photo colours${vis ? " and SAM 3D look-alike meshes (visual only)" : ""}.</p>
    ${S.cfg.sam3d ? `<div class="sam3d"><div class="eyebrow">Refine with SAM 3D · local · 5–8 min per object</div>
      <div id="samList">${t.objects.map((o) => `<div class="obj" data-id="${esc(o.id)}"><span class="lb">${esc(o.label)}</span>
        ${t.visual_meshes.includes(o.id) ? '<span class="st">MESH READY</span>' : `<button data-sam="${esc(o.id)}">Refine</button>`}</div>`).join("")}</div></div>` : ""}`;
  $$("[data-sam]", card).forEach((b) => b.onclick = () => startSam(b.dataset.sam, b));
}
async function startSam(obj, btn) {
  try {
    await api("/api/app/sam3d", { scene: S.sid, obj });
    btn.outerHTML = `<span class="st" data-samst="${esc(obj)}">RUNNING…</span>`;
    toast("SAM 3D reconstruction started in the background (5–8 min). Keep working; the mesh appears when it is done.");
    pollSam();
  } catch (e) { toast(e.message); }
}
async function pollSam() {
  if (pollSam.on) return; pollSam.on = true;
  for (;;) {
    await sleep(5000);
    const st = await api(`/api/app/sam3d?scene=${encodeURIComponent(S.sid)}`).catch(() => []);
    let running = 0;
    for (const j of st) {
      const el = $(`[data-samst="${j.obj}"]`);
      if (j.state === "running") { running++; if (el) el.textContent = `RUNNING · ${Math.floor(j.elapsed_s / 60)}:${String(j.elapsed_s % 60).padStart(2, "0")}`; }
      if (el && j.state === "done" && !el.dataset.fin) { el.dataset.fin = 1; el.textContent = "MESH READY"; reloadTwin(); toast(`SAM 3D mesh ready: ${j.obj}. Switch to Visual.`); }
      if (el && j.state === "error") el.textContent = "FAILED";
    }
    if (!running) break;
  }
  pollSam.on = false;
}
async function reloadTwin() { S.twin = await api("/api/app/twin", { scene: S.sid }); await studio.load(S.twin); }

$("#lookSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || !studio) return; studio.setLook(b.dataset.look); $$("#lookSeg button").forEach((x) => x.classList.toggle("on", x === b)); });
$("#camSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || !studio) return; studio.camera(b.dataset.cam); $$("#camSeg button").forEach((x) => x.classList.toggle("on", x === b)); });

// ------------------------------------------------------------------------------------------- 04 instruct
function openInstruct(prefill) {
  setStep("instruct");
  let card = $("#cardInstruct");
  if (!card) { card = stepCard("cardInstruct", "04", "Instruct", S.cfg.live ? "MUSE SPARK" : "RECORDED"); $("#panelScroll").appendChild(card); }
  card.classList.remove("collapsed", "done");
  const sug = S.info.suggestions || [];
  $(".step-b", card).innerHTML = `
    <div class="ask"><textarea id="instr" placeholder="Tell the robot what you want." maxlength="1000">${esc(prefill || "")}</textarea>
      <span class="hint">⌘↵</span><button class="send" id="sendBtn">Compile</button></div>
    ${sug.length ? `<div class="sugg">${sug.map((s) => `<button>${esc(s)}</button>`).join("")}</div>` : ""}
    <div id="intentOut"></div>`;
  const ta = $("#instr");
  $$(".sugg button", card).forEach((b) => b.onclick = () => { ta.value = b.textContent; ta.focus(); });
  $("#sendBtn").onclick = () => compile(ta.value);
  ta.onkeydown = (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); compile(ta.value); } };
  $("#cardRehearse")?.remove();
  setTimeout(() => card.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  if (!prefill) setTimeout(() => ta.focus(), 400);
}

async function compile(text, turns = []) {
  text = (text || "").trim();
  if (!text || S.busy) return;
  S.instruction = text; S.turns = turns;
  const out = $("#intentOut");
  $("#cardRehearse")?.remove();
  const chat = turns.map((t) => `<div class="bub ai"><span class="who">MUSE SPARK ASKS</span>${esc(t.q)}</div><div class="bub me">${esc(t.a)}</div>`).join("");
  out.innerHTML = `<div class="chat"><div class="bub me">${esc(text)}</div>${chat}</div>
    <div class="thinking" style="margin-top:10px"><div class="spin"></div>Muse Spark is compiling goals; the intent gate checks them…</div>`;
  $("#sendBtn").disabled = true; S.busy = true;
  try {
    const { job } = await api("/api/app/intent", { scene: S.sid, instruction: text, turns });
    const r = await poll(job);
    addCost(r.cost_usd);
    S.intent = r;
    renderIntent(r, text, turns);
  } catch (e) {
    out.innerHTML = `<div class="bub no"><span class="who">ERROR</span>${esc(e.message)}</div>`;
  } finally { S.busy = false; const sb = $("#sendBtn"); if (sb) sb.disabled = false; }
}

const PRED = { on: "On", inside: "Inside", upright: "Upright", stable: "Stable", stacked: "Stacked", near: "Next to", clear: "Clear", undisturbed: "Keep still" };
function goalCard(g, labels, res) {
  const L = (id) => esc(labels[id] || id);
  const a = g.args || [];
  let txt;
  if (["on", "inside", "near"].includes(g.pred)) txt = `${L(a[0])}<span class="arr">→</span>${L(a[1])}`;
  else if (g.pred === "stacked") txt = a.map(L).join('<span class="arr">/</span>');
  else if (g.pred === "undisturbed") txt = a.length > 4 ? `${a.slice(0, 3).map(L).join(", ")} <span class="dim">+${a.length - 3} more</span>` : a.map(L).join(", ");
  else txt = a.map(L).join(", ");
  const cls = ["goal", g.pred === "undisturbed" ? "keep" : "", g.implicit ? "implicit" : "", res ? (res.ok ? "ok" : "no") : ""].join(" ");
  const formal = `${g.pred}(${a.join(", ")})`;
  return `<div class="${cls}"><span class="gp">${PRED[g.pred] || g.pred}</span><span class="ga">${txt}</span>
    <span class="gf" title="${esc(res && !res.ok ? res.why : formal)}">${esc(res && !res.ok ? res.why : (g.implicit ? "implicit · " : "") + formal)}</span></div>`;
}
function costLine(r) {
  const st = r.intent?.status, role = st === "refused" ? "safety refusal" : st === "needs_clarification" ? "clarifying question" : "intent → goals";
  return modelBadges([{ model: "Muse Spark", role, s: r.wall_s, usd: r.cost_usd, state: r.cached ? "recorded" : "live" }]) +
    (r.fallback ? `<p class="muted" style="color:var(--warn)">Fallback: ${esc(r.fallback)}.</p>` : "");
}
// Which Meta model did what, with measured latency and cost (state: live | recorded | cached | local).
const MODEL_CLS = { "Muse Spark": "muse", "SAM 3.1": "sam", "SAM 3D": "sam3d", "MuJoCo": "mj" };
function modelBadges(list) {
  return `<div class="models">${list.map((m) => {
    const bits = [m.role];
    if (m.s != null) bits.push(`${(+m.s).toFixed(m.s < 10 ? 1 : 0)} s`);
    if (m.usd != null) bits.push(m.state === "recorded" && !m.usd ? "$0 · recorded" : money(m.usd));
    else if (m.state) bits.push(m.state);
    return `<span class="mb ${MODEL_CLS[m.model] || ""}" title="${esc(m.title || "")}"><i></i><b>${esc(m.model)}</b><small>${esc(bits.join(" · "))}</small></span>`;
  }).join("")}</div>`;
}
function renderIntent(r, text, turns) {
  const out = $("#intentOut");
  const it = r.intent || {};
  const labels = r.labels || {};
  const chat = turns.map((t) => `<div class="bub ai"><span class="who">MUSE SPARK ASKS</span>${esc(t.q)}</div><div class="bub me">${esc(t.a)}</div>`).join("");
  const head = `<div class="chat"><div class="bub me">${esc(text)}</div>${chat}</div>`;
  if (r.status === "no_recording" || r.status === "api_error") {
    out.innerHTML = head + `<div class="bub no"><span class="who">${r.status === "api_error" ? "MUSE SPARK UNAVAILABLE" : "NO RECORDING"}</span>${esc(r.message)}</div>
      ${r.status === "api_error" ? '<button class="ghost small" id="retryBtn">Try again</button>' : ""}
      <div class="sugg">${(r.suggestions || []).map((s) => `<button>${esc(s)}</button>`).join("")}</div>`;
    $$(".sugg button", out).forEach((b) => b.onclick = () => { $("#instr").value = b.textContent; compile(b.textContent); });
    if ($("#retryBtn", out)) $("#retryBtn", out).onclick = () => compile(text, turns);
    return;
  }
  if (r.status === "refused") {
    out.innerHTML = head + `<div class="bub no"><span class="who">REFUSED · BY ${it.decided_by === "gate" ? "THE INTENT GATE" : "MUSE SPARK"}</span>${esc(it.reason)}</div>
      <p class="muted">Nothing was planned or simulated. Rephrase the instruction to try again.</p>` + costLine(r);
    pushRun({ instruction: text, verdict: "refused", reason: it.reason, cost: r.cost_usd });
    return;
  }
  if (r.status === "needs_clarification") {
    const q = it.question || "Could you clarify?";
    const gateOk = it.decided_by === "gate" && (it.goals || []).length;
    out.innerHTML = head + `<div class="chat"><div class="bub ai"><span class="who">MUSE SPARK ASKS${it.decided_by === "gate" ? " · INTENT GATE" : ""}</span>${esc(q)}</div></div>
      <div class="ask"><textarea id="reply" placeholder="Your answer…" style="min-height:64px"></textarea><span class="hint">⌘↵</span><button class="send" id="replyBtn">Reply</button></div>
      ${gateOk ? `<button class="ghost small" id="confirmBtn">Confirm the goals as compiled</button>` : ""}` + costLine(r);
    const send = () => { const a = $("#reply").value.trim(); if (a) compile(text, [...turns, { q, a }]); };
    $("#replyBtn").onclick = send;
    $("#reply").onkeydown = (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); } };
    if (gateOk) $("#confirmBtn").onclick = () => showGoals(r, text, turns, true);
    setTimeout(() => $("#reply").focus(), 100);
    pushRun({ instruction: text, verdict: "ask", reason: q, cost: r.cost_usd });
    return;
  }
  showGoals(r, text, turns);
}
function showGoals(r, text, turns, confirmed) {
  const out = $("#intentOut");
  const it = r.intent || {};
  const labels = r.labels || {};
  const goals = it.goals || [];
  const moved = new Set();
  goals.forEach((g) => { if (["on", "inside", "near"].includes(g.pred)) moved.add(g.args[0]); if (g.pred === "stacked") g.args.slice(1).forEach((x) => moved.add(x)); });
  const mentioned = new Set(goals.flatMap((g) => g.args));
  const rest = Object.keys(labels).filter((id) => !mentioned.has(id));
  const implicit = rest.length ? [{ pred: "undisturbed", args: rest, implicit: true }] : [];
  const chat = turns.map((t) => `<div class="bub ai"><span class="who">MUSE SPARK ASKS</span>${esc(t.q)}</div><div class="bub me">${esc(t.a)}</div>`).join("");
  const flags = ((it.gate || {}).flags || []);
  const hasNear = goals.some((g) => g.pred === "near");
  out.innerHTML = `<div class="chat"><div class="bub me">${esc(text)}</div>${chat}</div>
    <div class="eyebrow" style="margin-top:4px">Goals physics will check</div>
    <div class="goals">${[...goals, ...implicit].map((g, i) => goalCard(g, labels).replace('class="goal', `style="animation-delay:${i * 70}ms" class="goal`)).join("")}</div>
    ${(it.assumptions || []).length ? `<details class="assume"><summary>${it.assumptions.length} assumptions Muse Spark made</summary><ul>${it.assumptions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>` : ""}
    ${flags.length ? `<p class="muted">Intent gate: ${flags.map((f) => esc(f.message)).join("; ")}${confirmed ? " (confirmed)" : ""}</p>` : ""}
    ${costLine(r)}
    <div class="budget"><span class="lbl">Search budget</span>
      <div class="seg small" id="budgetSeg"><button data-b="60">60 s</button><button data-b="90" class="on">90 s</button><button data-b="120">120 s</button></div></div>
    ${hasNear ? `<label class="toggle"><input type="checkbox" id="nearBeta" ${S.cfg.public_demo ? "disabled" : "checked"}> Near-placement <span class="badge warn">beta</span>
      <span class="dim">free-spot search for “next to” goals</span></label>` : ""}
    <button class="primary acc" id="rehearseBtn">Rehearse in physics <span class="k">⌘↵</span></button>`;
  $$("#budgetSeg button").forEach((b) => b.onclick = () => $$("#budgetSeg button").forEach((x) => x.classList.toggle("on", x === b)));
  $("#rehearseBtn").onclick = () => rehearse(goals, text);
  const ids = [...mentioned].filter((x) => labels[x]);
  studio.setFocus(Object.fromEntries(ids.map((id) => [id, moved.has(id) ? "accent" : "warn"])));
  studio.setTags(ids, labels);
  document.onkeydown = (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && $("#rehearseBtn") && document.activeElement?.id !== "instr") { e.preventDefault(); $("#rehearseBtn").click(); } };
}

// ------------------------------------------------------------------------------------------- 05 rehearse
async function rehearse(goals, text) {
  S.rehearsing = true; setLook("physics");
  if (S.busy) return;
  S.busy = true;
  const budget = +($("#budgetSeg .on")?.dataset.b || 90);
  const near = !!$("#nearBeta")?.checked;
  setStep("rehearse");
  collapse("cardInstruct", `“${esc(text)}” → ${goals.length} goal${goals.length === 1 ? "" : "s"}`);
  $("#cardRehearse")?.remove();
  const card = stepCard("cardRehearse", "05", "Rehearse", `BUDGET ${budget} S`);
  $("#panelScroll").appendChild(card);
  const body = $(".step-b", card);
  body.innerHTML = `
    <div class="counters"><div class="stat"><b id="cC">0</b><span>Candidates</span></div><div class="stat"><b id="cR">0</b><span>Rejected</span></div>
      <div class="stat"><b id="cK">0</b><span>Dry runs</span></div><div class="stat"><b id="cT">0s</b><span>Elapsed</span></div></div>
    <div class="bar"><i id="cBar"></i></div>
    <div class="phase" id="cPhase">Search: order × carry height × detour, every candidate simulated</div>
    <div class="feed" id="feed"></div><div id="verdictCard"></div>`;
  card.scrollIntoView({ behavior: "smooth", block: "start" });
  $("#verdict").classList.add("hidden"); $("#replayBar").classList.add("hidden");
  $("#simSub").textContent = "SEARCH · STARTING";
  const cands = [];
  resetWall();
  const t0 = performance.now();
  let nC = 0, nR = 0, nDry = 0, kin = 0, lastSims = 0;
  const tick = setInterval(() => {
    const el = (performance.now() - t0) / 1000;
    $("#cT").textContent = `${el.toFixed(0)}s`;
    $("#cBar").style.width = `${Math.min(100, (el / budget) * 100)}%`;
  }, 250);
  const feed = $("#feed");
  const addFeed = (html) => { feed.insertAdjacentHTML("beforeend", html); feed.scrollTop = feed.scrollHeight; while (feed.children.length > 160) feed.firstChild.remove(); };
  try {
    const { job } = await api("/api/app/rehearse", { scene: S.sid, instruction: text, goals, budget_s: budget, near_beta: near });
    const res = await poll(job, (evs) => {
      for (const ev of evs) {
        if (ev.type === "candidate") {
          nC++; if (!ev.ok) nR++;
          lastSims = ev.sims;
          $("#cC").textContent = nC; $("#cR").textContent = nR;
          const h = ev.height ? `carry ${(+ev.height).toFixed(2)} m${ev.via ? " · detour" : ""}` : "";
          addFeed(`<div class="cand ${ev.ok ? "ok" : "no"}"><span class="cn">#${ev.n}</span><span class="ck">${ev.ok ? "✓" : "✕"}</span>
            <span class="ct">${ev.prefix}/${ev.of} ${esc(ev.action)}<small>${esc(h)}</small>${ev.ok ? "" : `<span class="cr">${esc((ev.reasons || []).join(" · ") || "goal check failed")}</span>`}</span></div>`);
          $("#simSub").textContent = `SEARCH  ·  SIM ${ev.sims}  ·  ${ev.ok ? "PASSED" : "REJECTED"}${ev.ok ? "" : " · " + ((ev.reasons || [])[0] || "").toUpperCase().slice(0, 48)}`;
          if (ev.traj) { const c = { n: ev.n, traj: ev.traj, hits: ev.hits || [], ok: ev.ok, reasons: ev.reasons || [], plan: ev.plan || [], prefix: ev.prefix, of: ev.of }; cands.push(c); addTile(c); }
          wallHeader(cands, nDry);
        } else if (ev.type === "dry") {
          nDry++;
          kin = Math.max(kin, ev.kin_checks); $("#cK").textContent = kin;
          if (nDry <= 60) addFeed(`<div class="cand dry"><span class="cn">dry</span><span class="ck">✕</span>
            <span class="ct">${esc(ev.action)} <small>${(+ev.height).toFixed(2)} m${ev.via ? " · detour" : ""}</small><span class="cr">dry run: ${esc(ev.why)}</span></span></div>`);
          if (!lastSims) $("#simSub").textContent = `SEARCH  ·  DRY RUN ${ev.kin_checks}  ·  ${ev.why.toUpperCase().slice(0, 44)}`;
        } else if (ev.type === "kin") {
          kin = Math.max(kin, ev.kin_checks); $("#cK").textContent = kin;
          if (!lastSims) $("#simSub").textContent = `SEARCH  ·  KINEMATIC DRY RUNS ${ev.kin_checks}`;
        } else if (ev.type === "order") {
          addFeed(`<div class="cand note">▸ order ${esc(ev.order.replace(/[\[\]']/g, "").replace(/, /g, " → "))}${ev.level ? ` · clearance level ${ev.level}` : ""}</div>`);
        } else if (ev.type === "phase") {
          $("#cPhase").textContent = ev.text;
          addFeed(`<div class="cand note">▸ ${esc(ev.text)}</div>`);
          $("#simSub").textContent = ev.text.toUpperCase().slice(0, 60);
        } else if (ev.type === "stress") {
          $("#cPhase").textContent = `Stress test · ${ev.k} of ${ev.n} held-out variants run`;
        } else if (ev.type === "note") {
          addFeed(`<div class="cand note">${esc(ev.text)}</div>`);
        }
      }
    }, 300);
    clearInterval(tick);
    $("#cBar").style.width = "100%";
    S.busy = false;
    const run = { instruction: text, goals, result: res, verdict: res.verdict, job, cost: (S.intent?.cost_usd) || 0, when: Date.now(), cands, dry: nDry };
    pushRun(run);
    showResult(run);
  } catch (e) {
    clearInterval(tick); S.busy = false;
    $("#verdictCard").innerHTML = `<div class="bub no"><span class="who">ERROR</span>${esc(e.message)}</div>`;
  }
}

function showResult(run) {
  S.rehearsing = false;
  const r = run.result;
  const ok = r.verdict === "sent";
  const labels = S.intent?.labels || Object.fromEntries((S.twin?.objects || []).map((o) => [o.id, o.label]));
  const st = r.search?.stats || {};
  const ns = r.budget.used_sims ?? st.sims ?? 0;
  const used = `${ns} simulation${ns === 1 ? "" : "s"} · ${st.kin_checks ?? 0} dry runs · ${r.budget.used_s} s of ${r.budget.time_s} s budget`;
  if ($("#cK") && st.kin_checks != null) $("#cK").textContent = st.kin_checks;
  const robust = r.stress && r.stress.n ? r.stress : null;
  const reason = (r.reasons || [])[0] || r.search?.reason || "";
  const direct = r.direct;
  // stage banner
  const v = $("#verdict");
  v.className = "verdict " + (ok ? "ok" : "bad");
  v.innerHTML = `<div class="vmark">${ok ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>' : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M6 6l12 12M18 6L6 18"/></svg>'}</div>
    <div><div class="vt">${ok ? "PLAN VERIFIED · SENT" : "NOT SENT"}</div>
    <div class="vs">${ok ? `${esc(used)}${robust ? ` · held-out stress ${robust.successes}/${robust.n}` : ""}`
      : `${esc(reason)}${direct ? ` · replaying the direct attempt${(direct.reasons || [])[0] ? ": " + esc(direct.reasons[0]) : ""}` : ""}<br>${esc(used)}`}</div></div>`;
  v.classList.remove("hidden");
  $("#simSub").textContent = ok ? "PLAN VERIFIED · REPLAY" : direct ? "NOT SENT · DIRECT ATTEMPT REPLAY" : "NOT SENT";
  $("#simChip .blink").className = "blink " + (ok ? "green" : "red");
  // panel
  const acts = r.actions || (direct && direct.actions) || [];
  const gres = r.goal_results || (direct && direct.goal_results) || [];
  const gmap = (g) => gres.find((x) => x.pred === g.pred && JSON.stringify(x.args) === JSON.stringify(g.args));
  $("#verdictCard").innerHTML = `
    <div class="vcard ${ok ? "ok" : "bad"}"><div class="vt">${ok ? "PLAN VERIFIED · SENT" : "NOT SENT"}</div>
      <div class="vs">${ok ? (r.via_direct ? "The search ran out of budget, but the direct carry passed every goal in physics on its own. Only a verified plan is sent." : "Every goal held in physics. Only this verified plan would be sent to the robot.")
        : esc(reason) + (direct ? ". Nothing was sent. The replay shows what the direct attempt would have done." : ". Nothing was sent.")}</div>
      ${modelBadges([{ model: "MuJoCo", role: `${ns} sim${ns === 1 ? "" : "s"} · ${st.kin_checks ?? 0} dry runs · ${r.budget.used_s} s of ${r.budget.time_s}`, usd: 0, state: "local" }])}</div>
    ${acts.length ? `<div class="eyebrow">${ok ? "Verified plan" : "Direct attempt (rejected)"}</div><div class="plan">${acts.map((a) => `<div>${esc(a.text)}<small>${(+a.transit_height).toFixed(2)} m${a.via && a.via.length ? " · detour" : ""}</small></div>`).join("")}</div>` : ""}
    ${robust ? `<div class="eyebrow">Held-out stress test · friction, mass, perception error</div>
      <div class="robust">${Array.from({ length: robust.n }, (_, i) => `<i class="${i < robust.successes ? "ok" : "no"}"></i>`).join("")}</div>
      <p class="muted">${robust.successes}/${robust.n} variants still succeed${robust.successes < robust.n * 0.5 ? ". Fragile under perception error: worth a second look before running it for real." : "."}</p>` : ""}
    ${gres.length ? `<div class="eyebrow">Goal check</div><div class="goals">${(r.goals || []).filter((g) => gmap(g)).map((g) => goalCard(g, labels, gmap(g))).join("")}</div>` : ""}
    <button class="ghost small" id="againBtn">New instruction on this twin</button>`;
  $("#againBtn").onclick = () => { $("#verdict").classList.add("hidden"); openInstruct(); };
  // the final tile (verified plan, or the direct attempt), first in the wall
  if (run.cands) { resetWall(); run.cands.forEach((c) => addTile(c)); wallHeader(run.cands, run.dry || 0, r); }
  if (r.traj) addTile({ final: true, ok, traj: r.traj, hits: r.hits || [], reasons: ok ? ["plan verified · sent"] : (direct?.reasons || r.reasons || []),
    plan: acts.map((a) => a.text) }, true);
  // replay in the viewer: a verified plan on the textured twin, a failure on the physics primitives
  setLook(ok ? "visual" : "physics");
  if (r.traj) {
    const hits = r.hits || [];
    $("#replayBar").classList.remove("hidden");
    $("#replayTag").textContent = ok ? "VERIFIED PLAN" : "DIRECT ATTEMPT";
    const mv = new Set(acts.map((a) => a.obj));
    studio.setTags([...new Set(acts.flatMap((a) => [a.obj, a.dest]))], labels);
    studio.playRun(r.traj, hits, Object.fromEntries([...mv].map((id) => [id, ok ? "good" : "accent"]))).then((dur) => {
      $("#scrubMarks").innerHTML = hits.map((h) => `<i style="left:${(h.t / dur) * 100}%" title="${esc(h.text)}"></i>`).join("");
      setPlayIcon(true);
    }).catch(() => toast("This replay is no longer on the server (older runs are pruned). Rehearse again to regenerate it."));
  }
}

function setPlayIcon(playing) { $("#playIcon").setAttribute("d", playing ? "M6 4h4v16H6zM14 4h4v16h-4z" : "M7 4l13 8-13 8z"); }
$("#playBtn").onclick = () => { const rp = studio.rp; if (rp.playing) { rp.pause(); setPlayIcon(false); } else { rp.play(); setPlayIcon(true); } };
$("#speedBtn").onclick = () => { const rp = studio.rp; rp.speed = rp.speed === 1 ? 2 : rp.speed === 2 ? 0.5 : 1; studio.speed = rp.speed; $("#speedBtn").textContent = `${rp.speed}×`; };
const scrub = $("#scrub");
function scrubTo(e) { const r = scrub.getBoundingClientRect(), u = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); const rp = studio.rp; rp.pause(); setPlayIcon(false); rp.seek(rp.t0 + u * (rp.duration - rp.t0)); }
scrub.addEventListener("pointerdown", (e) => { scrubTo(e); const mv = (ev) => scrubTo(ev); window.addEventListener("pointermove", mv); window.addEventListener("pointerup", () => window.removeEventListener("pointermove", mv), { once: true }); });
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !/TEXTAREA|INPUT/.test(document.activeElement?.tagName) && !$("#replayBar").classList.contains("hidden")) { e.preventDefault(); $("#playBtn").click(); }
  if (e.key === "Enter" && !e.metaKey && !e.ctrlKey && S.step === "perceive" && $("#buildBtn") && !/TEXTAREA|INPUT/.test(document.activeElement?.tagName)) buildTwin();
});

// ------------------------------------------------------------------------------------------- 06 runs history
function storeKey(id) { return `rehearsal.runs.${id}`; }
function loadRuns(id) { try { return JSON.parse(localStorage.getItem(storeKey(id)) || "[]"); } catch { return []; } }
function saveRuns() { try { localStorage.setItem(storeKey(S.sid), JSON.stringify(S.runs.slice(0, 30))); } catch { /* private mode */ } }
function pushRun(run) {
  run.when = run.when || Date.now(); run.id = run.id || Math.random().toString(36).slice(2, 9);
  S.runs.unshift(run); S.runSel = run.id; saveRuns(); renderRuns();
}
function renderRuns() {
  $("#runsBox").classList.toggle("hidden", !S.runs.length);
  $("#runCount").textContent = S.runs.length ? `· ${S.runs.length}` : "";
  const B = { sent: '<span class="badge ok">Sent</span>', not_sent: '<span class="badge bad">Not sent</span>', ask: '<span class="badge acc">Asked</span>', refused: '<span class="badge warn">Refused</span>' };
  $("#runList").innerHTML = S.runs.map((r) => `<li data-id="${r.id}" class="${r.id === S.runSel ? "sel" : ""}"><div class="ri">${esc(r.instruction)}</div>
    <div class="rm">${B[r.verdict] || ""}${r.result ? `<span>${r.result.budget.used_s}s · ${r.result.budget.used_sims} sims</span>` : ""}<span>${new Date(r.when).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></div></li>`).join("");
}
$("#runList").addEventListener("click", (e) => {
  const li = e.target.closest("li"); if (!li || S.busy) return;
  const run = S.runs.find((r) => r.id === li.dataset.id); if (!run) return;
  S.runSel = run.id; renderRuns();
  if (!S.twin) return;
  showView("twin");
  if (run.result) {
    setStep("rehearse");
    collapse("cardInstruct", `“${esc(run.instruction)}” → ${run.goals.length} goals`);
    $("#cardRehearse")?.remove();
    const card = stepCard("cardRehearse", "05", "Rehearse", `BUDGET ${run.result.budget.time_s} S · HISTORY`);
    $("#panelScroll").appendChild(card);
    $(".step-b", card).innerHTML = `<div id="feed" class="hidden"></div><div id="verdictCard"></div>`;
    showResult(run);
  } else openInstruct(run.instruction);
});

init().catch((e) => toast("Failed to start: " + e.message, 8000));

// ------------------------------------------------------------------------------------------- attempt wall
let wall = null;
function wallR() { if (!wall) wall = new AttemptWall(); return wall; }
function resetWall() {
  wallR().clear();
  $("#wallRow").innerHTML = "";
  $("#wall").classList.remove("hidden", "open");
  $("#viewTwin").classList.add("has-wall");
  $("#wallExpand").textContent = "Grid view";
  wallHeader([], 0);
}
function wallHeader(cands, dry, r) {
  const ok = cands.filter((c) => c.ok).length;
  $("#wallTitle").textContent = `FUTURES TESTED ${String(cands.length).padStart(2, "0")}`;
  $("#wallSub").textContent = `${ok} PASSED · ${cands.length - ok} REJECTED IN PHYSICS${dry ? ` · ${dry} PRUNED BY DRY RUN` : ""}${r ? (r.verdict === "sent" ? " · PLAN SENT" : " · NOTHING SENT") : ""}`;
  $("#wallBlink").className = "blink " + (r ? (r.verdict === "sent" ? "green" : "red") : "");
}
function setLook(look) {
  if (!studio) return;
  if (studio.look !== look) studio.setLook(look);
  setLookBtns(look);
}
function setLookBtns(look) { $$("#lookSeg button").forEach((x) => x.classList.toggle("on", x.dataset.look === look)); }
async function addTile(c, first) {
  const el = document.createElement("div");
  el.className = `tile ${c.ok ? "ok" : "no"}${c.final ? " final" + (c.ok ? "" : " bad") : ""}`;
  const label = c.final ? (c.ok ? "VERIFIED PLAN" : "DIRECT ATTEMPT") : `#${String(c.n).padStart(2, "0")}${c.of > 1 ? ` · STEP ${c.prefix}/${c.of}` : ""}`;
  el.innerHTML = `<canvas></canvas><span class="tn">${label}</span><div class="tr" title="${esc((c.plan || []).join(" → "))}">${c.ok ? "✓ " : "✕ "}${esc((c.reasons || [])[0] || (c.ok ? (c.of > 1 && c.prefix < c.of ? `step ${c.prefix} holds · ${(c.plan || []).slice(-1)[0] || ""}` : "every goal holds") : "goal check failed"))}</div>`;
  const row = $("#wallRow");
  if (first) row.prepend(el); else row.appendChild(el);
  if (!first) row.scrollLeft = row.scrollWidth;
  el.onclick = () => playAttempt(c, el);
  try {
    await wallR().load(S.twin);
    const frames = await fetch(c.traj).then((x) => x.json());
    wallR().add($("canvas", el), frames, c.hits);
  } catch (e) { console.warn("tile", e); }
}
function playAttempt(c, el) {
  $$(".tile").forEach((t) => t.classList.toggle("sel", t === el));
  $("#wall").classList.remove("open"); $("#wallExpand").textContent = "Grid view";
  setLook(c.final && c.ok ? "visual" : "physics");
  $("#replayBar").classList.remove("hidden");
  $("#replayTag").textContent = c.final ? (c.ok ? "VERIFIED PLAN" : "DIRECT ATTEMPT") : `ATTEMPT #${c.n} · ${c.ok ? "PASSED" : "REJECTED"}`;
  studio.playRun(c.traj, c.hits).then((dur) => {
    $("#scrubMarks").innerHTML = (c.hits || []).map((h) => `<i style="left:${(h.t / dur) * 100}%" title="${esc(h.text)}"></i>`).join("");
    setPlayIcon(true);
  }).catch(() => toast("This replay is no longer on the server (older runs are pruned). Rehearse again to regenerate it."));
  $("#simSub").textContent = c.final ? (c.ok ? "PLAN VERIFIED · REPLAY" : "NOT SENT · DIRECT ATTEMPT") : `ATTEMPT #${c.n} · ${c.ok ? "PASSED" : "REJECTED · " + ((c.reasons || [])[0] || "").toUpperCase().slice(0, 40)}`;
}
$("#wallExpand").onclick = () => { const w = $("#wall"); w.classList.toggle("open"); $("#wallExpand").textContent = w.classList.contains("open") ? "Close grid" : "Grid view"; };
