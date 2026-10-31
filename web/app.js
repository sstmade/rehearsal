import { api, MOCK, params, waitJob, runRehearsal } from "./api.js";
import { Replay } from "./replay.js";
import { $, esc, sleep, PRED, REL, GLOSS, unexpectedContacts, hitList, col, dims, goalText, eventText, MB } from "./common.js";

const S = {
  gallery: [], scene: null, spec: null, saved: null, report: null, current: 0,
  robotBase: null, homeFrame: null, overlay: null, bust: Date.now(), busy: false,
};
const base = () => `/out/${encodeURIComponent(S.report.scene)}`;
const PLAY = `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>`;
const PAUSE = `<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2.5" width="3.5" height="11" rx="1"/><rect x="9.5" y="2.5" width="3.5" height="11" rx="1"/></svg>`;

// ------------------------------------------------------------------ tooltips / toast
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; moveTip(e); }
function moveTip(e) {
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.max(8, Math.min(e.clientX + 14, innerWidth - r.width - 8)) + "px";
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
let toastTimer;
function toast(msg) {
  let el = document.querySelector(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; el.setAttribute("role", "status"); document.body.appendChild(el); }
  el.textContent = msg; el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (el.hidden = true), 3800);
}

// ------------------------------------------------------------------ theme + chrome
$("themeBtn").onclick = () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("rehearsal-theme", dark ? "light" : "dark"); } catch { /* private mode */ }
};
if (MOCK) document.body.insertAdjacentHTML("beforeend", `<span class="chip unc mockflag" title="?mock=1: gallery, perception jobs and budget are synthesized in web/mock.js">MOCK DATA</span>`);

async function loadBudget() {
  try {
    const b = await api.budget();
    const pct = Math.min(100, 100 * (b.spent + (b.reserved || 0)) / b.cap);
    $("budget").innerHTML = `<span class="lbl">Meta API</span> $${b.spent.toFixed(2)} / $${b.cap.toFixed(0)} <span class="bar"><span style="width:${pct.toFixed(1)}%"></span></span>`;
    $("budget").hidden = false;
  } catch { $("budget").hidden = true; }
}

// ------------------------------------------------------------------ viewers
function makeReplay(el, opts) {
  try { return new Replay(el, opts); } catch (err) { console.warn("WebGL unavailable", err); return null; }
}
const replay = makeReplay($("stage"), { onTime });
const hero = makeReplay($("heroStage"), {});
const twin = makeReplay($("twinStage"), {});
if (hero) { hero.controls.enableZoom = false; }
if (twin) { twin.controls.enableZoom = false; }
let camR = null;   // offscreen renderer for calibrated-camera views (synthetic photo, overlay)
function cameraRenderer(w, h) {
  if (!camR) {
    const box = document.createElement("div");
    box.style.cssText = `position:fixed;left:-10000px;top:0;width:${w}px;height:${h}px`;
    document.body.appendChild(box);
    camR = makeReplay(box, { interactive: false, photoLook: true });
    if (camR) camR.renderer.setAnimationLoop(null);
  }
  return camR;
}

// ------------------------------------------------------------------ boot
async function main() {
  loadBudget();
  wireUpload();
  wireViewer();
  $("form").onsubmit = run;
  $("instruction").oninput = () => { $("gloss").textContent = GLOSS[$("instruction").value] ? `“${GLOSS[$("instruction").value]}”` : ""; };
  try {
    S.gallery = await api.gallery();
  } catch (e) {
    $("galleryGrid").innerHTML = `<div class="card empty"><b>Could not reach the Rehearsal server</b>${esc(e.message)}</div>`;
    return;
  }
  const withReport = S.gallery.find((g) => g.has_report);
  if (withReport) {
    S.robotBase = `/out/${encodeURIComponent(withReport.scene)}`;
    S.homeFrame = await fetch(`${S.robotBase}/attempt_1.traj.json?v=${S.bust}`).then((r) => r.ok ? r.json() : null).then((f) => f && f[0]).catch(() => null);
  }
  renderGallery();
  startHero(withReport);
  const want = params.get("scene");
  const first = S.gallery.find((g) => g.scene === want) || withReport || S.gallery[0];
  if (first) await openScene(first.scene);
}

// ------------------------------------------------------------------ hero: live replay of a precomputed scene
async function startHero(g) {
  if (!hero || !g) { $("heroLoader").innerHTML = `<div class="viewer-empty">No precomputed rehearsal yet.</div>`; return; }
  try {
    const b = `/out/${encodeURIComponent(g.scene)}`;
    const rep = await api.report(g.scene);
    await hero.loadScene(b, (f) => { $("heroLoader").querySelector(".ring span").style.width = `${(f * 100).toFixed(0)}%`; });
    const last = rep.attempts[rep.attempts.length - 1];
    const frames = await fetch(`${b}/attempt_${last.attempt}.traj.json?v=${S.bust}`).then((r) => r.json());
    hero.setTrajectory(frames, last.events, hitList(last));
    const trails = rep.stress_test ? await fetch(`${b}/stress_trails.json?v=${S.bust}`).then((r) => r.ok ? r.json() : null).catch(() => null) : null;
    if (trails) hero.setTrails(trails);
    hero.showTrails(false);
    hero.camera.position.set(1.95, 1.2, 1.25); hero.controls.target.set(0.38, 0, 0.22);
    hero.orbit(true, 0.35);
    hero.play();
    $("heroLoader").hidden = true;
    const st = rep.stress_test;
    const cap = (showingTrails) => `<span class="glass"><span class="live-dot"></span>MuJoCo replay · ${esc(g.title || g.scene)}</span>` +
      (showingTrails && st ? `<span class="glass"><span class="ok">●</span> ${st.successes}/${st.n} universes succeed</span>`
        : `<span class="glass">attempt ${last.attempt} · <span class="${last.success ? "ok" : "bad"}">${last.success ? "✓ all goals met" : "✕ failed"}</span></span>`);
    $("heroCaption").innerHTML = cap(false);
    // loop: plan replay, then the stress-test universes fan out, then again
    let phase = 0;
    hero.onTime = (t, dur) => {
      if (t >= dur && phase === 0) {
        phase = 1;
        if (trails) { hero.showRobot(false); hero.setTrails(trails, { animate: 3.5 }); hero.showTrails(true); $("heroCaption").innerHTML = cap(true); }
        setTimeout(() => { hero.showTrails(false); hero.showRobot(true); $("heroCaption").innerHTML = cap(false); hero.seek(hero.t0); hero.play(); phase = 0; }, trails ? 6500 : 1500);
      }
    };
  } catch (e) {
    console.warn(e);
    $("heroLoader").innerHTML = `<div class="viewer-empty">Could not load the 3D scene.<br>${esc(e.message)}</div>`;
  }
}

// ------------------------------------------------------------------ gallery
function thumbSvg(spec) {
  if (!spec) return "";
  const t = spec.table, S2 = 100;
  const sx = (y) => -y * S2, sy = (x) => -x * S2;
  const x0 = t.center[0] - t.size[0] / 2, y0 = t.center[1] - t.size[1] / 2;
  let g = `<rect x="${sx(y0 + t.size[1])}" y="${sy(x0 + t.size[0])}" width="${t.size[1] * S2}" height="${t.size[0] * S2}" rx="3" fill="var(--table)"/>`;
  for (const o of spec.objects) {
    const s = o.shape, cx = sx(o.pos[1]), cy = sy(o.pos[0]);
    g += s.type === "box" ? `<rect x="${cx - s.size[1] * S2 / 2}" y="${cy - s.size[0] * S2 / 2}" width="${s.size[1] * S2}" height="${s.size[0] * S2}" rx="1" fill="${col(o.rgba)}"/>`
      : `<circle cx="${cx}" cy="${cy}" r="${s.radius * S2}" fill="${col(o.rgba)}" ${s.type === "bowl" ? `fill-opacity=".5" stroke="${col(o.rgba)}" stroke-width="1.5"` : ""}/>`;
  }
  const w = t.size[1] * S2, h = t.size[0] * S2;
  return `<svg viewBox="${sx(y0 + t.size[1]) - w * 0.25} ${sy(x0 + t.size[0]) - 8} ${w * 1.5} ${h + 16}" preserveAspectRatio="xMidYMid meet" aria-hidden="true" style="background:var(--surface-2)">${g}</svg>`;
}

async function renderGallery() {
  const cards = S.gallery.map((g) => `
    <button class="gcard" type="button" data-scene="${esc(g.scene)}" aria-current="${g.scene === S.scene}">
      <div class="thumb">${g.thumb ? `<img src="${esc(g.thumb)}" alt="" loading="lazy">` : `<div class="skeleton" style="width:100%;height:100%"></div>`}</div>
      <div class="meta"><div class="t">${esc(g.title || g.scene)}</div>
        <div class="row" style="gap:6px"><span class="s">${esc(g.scene)}</span><span class="grow" style="flex:1"></span>
        ${g.has_report ? `<span class="chip good"><span class="dot"></span>report ready</span>` : `<span class="chip">twin only</span>`}</div></div>
    </button>`).join("");
  $("galleryGrid").innerHTML = cards + `
    <label class="gcard upload" id="dropZone" for="fileInput" tabindex="0">
      <span class="big"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/></svg></span>
      <span class="t" style="font-weight:600">Upload a desk photo</span>
      <span class="note" style="margin:0">Drop an image or click · processed as private data</span>
      <span class="mbs" style="margin-top:6px;justify-content:center">${MB.spark("list")}${MB.sam("segment")}</span>
    </label>`;
  $("galleryGrid").querySelectorAll(".gcard[data-scene]").forEach((b) => b.onclick = () => openScene(b.dataset.scene, true));
  const dz = $("dropZone");
  dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); } };
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove("drag")));
  dz.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) perceive(f); });
  // thumbnails drawn from each scene's spec when the server has none
  for (const g of S.gallery) {
    if (g.thumb) continue;
    const spec = await sceneSpec(g.scene).catch(() => null);
    const el = $("galleryGrid").querySelector(`.gcard[data-scene="${CSS.escape(g.scene)}"] .thumb`);
    if (el) el.innerHTML = spec ? thumbSvg(spec) : "";
  }
}

const specCache = new Map();
async function sceneSpec(name) {
  if (!specCache.has(name)) {
    specCache.set(name, api.scene(name).catch(async (e) => {
      if (e.status !== 404) throw e;
      const rep = await api.report(name);   // legacy server: the report carries the spec
      if (!rep) throw new Error(`no spec for ${name}`);
      return rep.scene_spec;
    }));
    specCache.get(name).catch(() => specCache.delete(name));
  }
  return structuredClone(await specCache.get(name));
}

async function openScene(name, scroll = false) {
  S.scene = name; S.overlay = null;
  $("galleryGrid").querySelectorAll(".gcard[data-scene]").forEach((b) => b.setAttribute("aria-current", b.dataset.scene === name));
  const [spec, report] = await Promise.all([sceneSpec(name).catch(() => null), api.report(name).catch(() => null)]);
  S.spec = spec || (report && structuredClone(report.scene_spec));
  S.saved = JSON.stringify(S.spec);
  S.report = report;
  S.current = report ? report.attempts.length - 1 : 0;
  renderTwin();
  renderAll();
  if (scroll) $("twin").scrollIntoView({ behavior: "smooth" });
}

// ------------------------------------------------------------------ 01 photo → twin
function renderTwin() {
  const spec = S.spec;
  if (!spec) { $("photoPane").innerHTML = `<div class="empty">No scene loaded</div>`; return; }
  $("objCount").textContent = `${spec.objects.length} objects`;
  renderPhoto();
  renderMap();
  renderObjs();
  $("dirtyBar").hidden = S.saved !== null;
  if (twin) {
    $("twinLoader").hidden = false;
    twin.loadSpec(spec, S.robotBase, S.homeFrame).then(() => { $("twinLoader").hidden = true; twin.camera.position.set(1.7, 0.9, 1.05); twin.controls.target.set(0.55, 0, 0.05); });
  }
}

const synthCache = new Map();
async function photoSrc(spec) {
  const p = spec.photo;
  if (!p.file.startsWith("mock:")) return "/" + p.file.replace(/^\//, "");
  if (synthCache.has(spec.name)) return synthCache.get(spec.name);
  // mock: render the "photo" from the calibrated camera (stand-in for perception's synthetic renders)
  const r = cameraRenderer(960, Math.round(960 * p.height / p.width));
  if (!r) return "";
  await r.loadSpec(JSON.parse(S.saved), null, null);
  r.markGroup.clear();
  r.setBackground(0xcfc8bd);
  r.setCameraKRt(p.camera.K, p.camera.R, p.camera.t, p.width, p.height);
  const url = r.snapshot();
  synthCache.set(spec.name, url);
  return url;
}

async function overlaySrc(spec) {
  const p = spec.photo;
  const r = cameraRenderer(960, Math.round(960 * p.height / p.width));
  if (!r) return "";
  await r.loadSpec(spec, null, null);
  // ghost the twin in the accent colour so misalignment with the photo is obvious
  for (const b of r.bodies) if (b) for (const m of b.userData.materials) {
    const isTable = b.userData.name === "table";
    m.color.setHex(isTable ? 0x7eaaff : 0x2f6bff); m.emissive.setHex(0x1d4fd6); m.emissiveIntensity = isTable ? 0.2 : 0.6;
    m.transparent = true; m.opacity = isTable ? 0.18 : 0.9;
  }
  r.markGroup.clear();
  r.setBackground(0x000000, true);
  r.setCameraKRt(p.camera.K, p.camera.R, p.camera.t, p.width, p.height);
  return r.snapshot();
}

let blendMode = "photo";
async function renderPhoto() {
  const spec = S.spec, p = spec.photo, pane = $("photoPane");
  $("photoMeta").hidden = !p;
  if (!p) {
    pane.innerHTML = `<div class="empty" style="position:absolute;inset:0">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="color:var(--muted)"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="11" r="2"/><path d="M21 16l-5-5-8 8"/></svg>
      <b>No photo for this scene</b><span>It was written by hand in the same schema perception produces.</span>
      <label class="btn sm" for="fileInput" tabindex="0" style="margin-top:6px">Upload a photo</label></div>`;
    return;
  }
  const W = p.width, H = p.height, objs = spec.objects.filter((o) => o.perception && o.perception.box_px);
  const lows = objs.filter((o) => o.perception.confidence === "low").length;
  $("photoMeta").textContent = `${W}×${H} · ${objs.length} boxes${lows ? ` · ${lows} low` : ""}`;
  $("photoMeta").className = `chip ${lows ? "unc" : ""}`;
  const hasCam = !!p.camera;
  pane.innerHTML = `<svg class="photo-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="position:absolute;inset:0;width:100%;height:100%" role="img" aria-label="Photo with SAM 3.1 boxes">
      <image id="photoImg" href="" width="${W}" height="${H}"/>
      ${S.overlay ? `<image id="maskImg" href="${esc(S.overlay)}" width="${W}" height="${H}" opacity="0"/>` : ""}
      <image id="ovlImg" href="" width="${W}" height="${H}" opacity="0"/>
      <g id="boxes">${objs.map((o) => {
        const [x1, y1, x2, y2] = o.perception.box_px, low = o.perception.confidence === "low";
        const t = esc(`<b>${esc(o.label)}</b>SAM 3.1 · ${o.perception.source} prompt · ${low ? "low" : "high"} confidence<br>σ = ${(o.perception.sigma_m * 100).toFixed(1)} cm in the stress test`);
        return `<g data-id="${esc(o.id)}" data-tip="${t}"><rect class="sam-box ${low ? "low" : ""}" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" rx="4"/>
          <text class="sam-tag" x="${x1 + 6}" y="${Math.max(24, y1 - 8)}">${esc(o.id)}${low ? " ?" : ""}</text></g>`;
      }).join("")}</g>
    </svg>
    <div class="tl"><span class="glass">${esc(spec.privacy === "public" ? "public data" : "private data")}</span></div>
    <div class="toolbar">
      <button class="tbtn" type="button" id="tglBoxes" aria-pressed="true">Boxes</button>
      ${S.overlay ? `<button class="tbtn" type="button" id="tglMasks" aria-pressed="false">Masks</button>` : ""}
    </div>
    ${hasCam ? `<div class="blend"><div class="seg" role="group" aria-label="Photo or twin overlay" style="margin:0">
        <button type="button" data-m="photo" aria-pressed="${blendMode === "photo"}">Photo</button>
        <button type="button" data-m="overlay" aria-pressed="${blendMode === "overlay"}">Overlay twin</button></div>
        <input type="range" id="blend" min="0" max="1" step="0.01" value="${blendMode === "overlay" ? 0.6 : 0}" aria-label="Twin overlay opacity">
        <span class="glass" id="blendLbl">camera-calibrated</span></div>` : ""}`;
  photoSrc(spec).then((u) => pane.querySelector("#photoImg")?.setAttribute("href", u));
  bindTips(pane);
  pane.querySelectorAll("#boxes g").forEach((g) => { g.onmouseenter = () => hl(g.dataset.id); g.onmouseleave = () => hl(null); });
  $("tglBoxes").onclick = (e) => toggle(e.currentTarget, (on) => (pane.querySelector("#boxes").style.display = on ? "" : "none"));
  if (S.overlay) $("tglMasks").onclick = (e) => toggle(e.currentTarget, (on) => pane.querySelector("#maskImg").setAttribute("opacity", on ? 1 : 0));
  if (hasCam) {
    const setBlend = async (v) => {
      const img = pane.querySelector("#ovlImg");
      if (v > 0 && !img.getAttribute("href")) img.setAttribute("href", await overlaySrc(S.spec));
      img.setAttribute("opacity", v);
    };
    $("blend").oninput = (e) => { setBlend(+e.target.value); blendMode = +e.target.value > 0 ? "overlay" : "photo"; syncSeg(); };
    const syncSeg = () => pane.querySelectorAll(".blend .seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.m === blendMode));
    pane.querySelectorAll(".blend .seg button").forEach((b) => b.onclick = () => {
      blendMode = b.dataset.m; const v = blendMode === "overlay" ? 0.6 : 0; $("blend").value = v; setBlend(v); syncSeg();
    });
    if (blendMode === "overlay") setBlend(0.6);
  }
}

async function refreshOverlay() {
  const img = $("photoPane").querySelector("#ovlImg");
  if (!img || !S.spec.photo?.camera) return;
  if (+img.getAttribute("opacity") > 0) img.setAttribute("href", await overlaySrc(S.spec));
  else img.setAttribute("href", "");
}

function hl(id) {
  document.querySelectorAll("[data-id]").forEach((el) => {
    const on = id && el.dataset.id === id;
    el.classList.toggle("hl", !!on);
    el.querySelector?.(".sam-box")?.classList.toggle("hl", !!on);
  });
}

const MAP_S = 300;
function objRadius(o) { return o.shape.radius || Math.hypot(o.shape.size[0], o.shape.size[1]) / 2; }
function renderMap() {
  const spec = S.spec, t = spec.table, Sc = MAP_S, reach = S.report?.reachability || {};
  const sx = (y) => -y * Sc, sy = (x) => -x * Sc;
  const tx0 = t.center[0] - t.size[0] / 2, ty0 = t.center[1] - t.size[1] / 2;
  let g = `<rect x="${sx(ty0 + t.size[1])}" y="${sy(tx0 + t.size[0])}" width="${t.size[1] * Sc}" height="${t.size[0] * Sc}" rx="6" fill="var(--table)" stroke="var(--table-edge)"/>`;
  g += `<circle cx="0" cy="0" r="${0.855 * Sc}" fill="var(--reach)" stroke="var(--accent)" stroke-dasharray="4 4" stroke-width="1"/>`;
  g += `<text x="${sx(-0.62)}" y="${sy(0.08)}" text-anchor="end" style="fill:var(--accent-text)">arm reach ≈ 0.86 m</text>`;
  for (const o of spec.objects) {
    const ok = reach[o.id] !== false, s = o.shape, p = o.perception;
    const stroke = ok ? "rgba(0,0,0,.35)" : "var(--bad)", dash = ok ? "" : `stroke-dasharray="4 3"`;
    let shape;
    if (s.type === "box") {
      const w = s.size[1] * Sc, h = s.size[0] * Sc;
      shape = `<rect x="${-w / 2}" y="${-h / 2}" width="${w}" height="${h}" rx="2" fill="${col(o.rgba)}" stroke="${stroke}" stroke-width="1.5" ${dash} transform="rotate(${-(o.yaw || 0) * 180 / Math.PI})"/>`;
    } else if (s.type === "bowl") {
      shape = `<circle r="${s.radius * Sc}" fill="${col(o.rgba)}" fill-opacity=".35" stroke="${col(o.rgba)}" stroke-width="4"/>`;
    } else shape = `<circle r="${s.radius * Sc}" fill="${col(o.rgba)}" stroke="${stroke}" stroke-width="1.5" ${dash}/>`;
    const ring = p ? `<circle r="${(objRadius(o) + 2 * (p.sigma_m || 0.012)) * Sc}" fill="none" stroke="${p.confidence === "low" ? "var(--unc)" : "var(--accent)"}" stroke-opacity="${p.confidence === "low" ? 1 : .35}" stroke-width="1.5" stroke-dasharray="3 3"/>` : "";
    const tipTxt = esc(`<b>${esc(o.label)}</b><code>${esc(o.id)}</code> · ${dims(o)}<br>x ${o.pos[0].toFixed(3)} m · y ${o.pos[1].toFixed(3)} m` +
      (p ? `<br>σ ${(p.sigma_m * 100).toFixed(1)} cm · ${p.confidence} confidence` : "") + `<br>${ok ? "✓ reachable" : "✕ outside the workspace"}<br><i>drag or use arrow keys</i>`);
    g += `<g class="obj" data-id="${esc(o.id)}" transform="translate(${sx(o.pos[1])} ${sy(o.pos[0])})" tabindex="0" role="button" aria-label="${esc(o.label)}, drag to move" data-tip="${tipTxt}">
      ${shape}${ring}<text y="${objRadius(o) * Sc + 13}" text-anchor="middle">${esc(o.id)}${ok ? "" : " ✕"}</text></g>`;
  }
  g += `<rect x="-14" y="-6" width="28" height="18" rx="4" fill="var(--text-2)"/><text x="0" y="26" text-anchor="middle">robot base</text>`;
  const vb = [sx(ty0 + t.size[1]) - 10, sy(tx0 + t.size[0]) - 10, t.size[1] * Sc + 20, (tx0 + t.size[0]) * Sc + 44];
  $("map").innerHTML = `<svg viewBox="${vb.join(" ")}" role="img" aria-label="Top-down map of the table">${g}</svg>`;
  bindTips($("map"));
  wireMapDrag();
}

function wireMapDrag() {
  const svg = $("map").querySelector("svg"), spec = S.spec, t = spec.table;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const move = (o, g, x, y) => {
    o.pos = [+clamp(x, t.center[0] - t.size[0] / 2, t.center[0] + t.size[0] / 2).toFixed(4), +clamp(y, t.center[1] - t.size[1] / 2, t.center[1] + t.size[1] / 2).toFixed(4)];
    g.setAttribute("transform", `translate(${-o.pos[1] * MAP_S} ${-o.pos[0] * MAP_S})`);
    twin && twin.placeObject(o.id, o.pos, o.shape, o.yaw);
  };
  const done = () => { renderObjs(); markDirty(); refreshOverlay(); };
  svg.querySelectorAll(".obj").forEach((g) => {
    const o = spec.objects.find((x) => x.id === g.dataset.id);
    g.onmouseenter = () => hl(o.id); g.onmouseleave = () => hl(null);
    g.onpointerdown = (e) => {
      e.preventDefault(); g.setPointerCapture(e.pointerId); hideTip();
      const m = svg.getScreenCTM().inverse();
      const p0 = new DOMPoint(e.clientX, e.clientY).matrixTransform(m), start = [...o.pos];
      g.onpointermove = (ev) => {
        const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m);
        move(o, g, start[0] - (p.y - p0.y) / MAP_S, start[1] - (p.x - p0.x) / MAP_S);
      };
      g.onpointerup = g.onpointercancel = () => { g.onpointermove = null; done(); };
    };
    g.onkeydown = (e) => {
      const d = e.shiftKey ? 0.05 : 0.01, k = { ArrowUp: [d, 0], ArrowDown: [-d, 0], ArrowLeft: [0, d], ArrowRight: [0, -d] }[e.key];
      if (!k) return;
      e.preventDefault(); move(o, g, o.pos[0] + k[0], o.pos[1] + k[1]); done();
    };
  });
}

function markDirty() { $("dirtyBar").hidden = JSON.stringify(S.spec) === S.saved; }
$("undoEdits").onclick = () => { if (S.saved) { S.spec = JSON.parse(S.saved); renderTwin(); } };
$("saveTwin").onclick = () => saveTwin();
async function saveTwin() {
  const btn = $("saveTwin");
  btn.disabled = true;
  try {
    const r = await api.saveScene(S.spec);
    S.saved = JSON.stringify(S.spec);
    specCache.delete(r.name);
    if (r.name !== S.scene) { S.scene = r.name; S.gallery = await api.gallery().catch(() => S.gallery); renderGallery(); }
    markDirty();
    toast(`Saved as scenes/${r.name}.json — rehearsals now use your corrections`);
    return true;
  } catch (e) {
    toast(`Could not save the twin: ${e.status === 404 ? "POST /api/scene is not available on this server" : e.message}`);
    return false;
  } finally { btn.disabled = false; }
}

function renderObjs() {
  const reach = S.report?.reachability || {};
  $("objs").innerHTML = `<thead><tr><th>Object</th><th>Size · mass</th><th>Detection</th><th>Reach</th></tr></thead><tbody>` +
    S.spec.objects.map((o) => {
      const p = o.perception, ok = reach[o.id];
      const det = p ? `<span class="chip ${p.confidence === "low" ? "unc" : "accent"}">${p.confidence === "low" ? "low" : "high"} · σ ${(p.sigma_m * 100).toFixed(1)} cm</span><div class="meta" style="margin-top:3px">SAM 3.1 ${esc(p.source)} prompt</div>`
        : `<span class="meta">hand-written</span>`;
      return `<tr data-id="${esc(o.id)}"><td><span class="sw" style="background:${col(o.rgba)}"></span>${esc(o.label)}<div class="meta mono" style="margin-left:20px">${esc(o.id)} · (${o.pos[0].toFixed(2)}, ${o.pos[1].toFixed(2)}) m</div></td>
        <td class="meta">${dims(o)}<br>${o.mass_kg.range.join("–")} kg</td><td>${det}</td>
        <td>${ok === undefined ? `<span class="meta">—</span>` : ok ? `<span class="chip good">✓ reachable</span>` : `<span class="chip bad">✕ out of reach</span>`}</td></tr>`;
    }).join("") + "</tbody>";
  $("objs").querySelectorAll("tr[data-id]").forEach((r) => { r.onmouseenter = () => hl(r.dataset.id); r.onmouseleave = () => hl(null); });
}

// ------------------------------------------------------------------ upload → perceive job
function wireUpload() {
  $("fileInput").onchange = (e) => { const f = e.target.files[0]; if (f) perceive(f); e.target.value = ""; };
  $("heroUpload").onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); } };
}

const PERCEIVE_STEPS = [["upload", "Upload"], ["list", "Muse Spark · list"], ["segment", "SAM 3.1 · segment"], ["calibrate", "Calibrate"], ["validate", "Build twin"]];
async function perceive(file) {
  if (S.busy) { toast("A job is already running"); return; }
  if (!file.type.startsWith("image/")) { toast("Please choose an image file"); return; }
  S.busy = true;
  const box = $("perceiveProgress");
  box.hidden = false;
  $("twin").scrollIntoView({ behavior: "smooth" });
  const draw = (v, err) => progressCard(box, `Perceiving ${esc(file.name)}`, PERCEIVE_STEPS, v, err,
    `<span class="mbs">${MB.spark("list objects")}${MB.sam("segmentation")}</span>`);
  draw({ state: "queued", log: [], progress: { phase: "upload", fraction: 0.02 } });
  try {
    const { job } = await api.perceive(file, { privacy: "private" });
    const res = await waitJob(job, (v) => draw(v));
    S.scene = res.scene; S.spec = res.scene_spec; S.overlay = res.overlay || null;
    S.saved = res.valid === false ? null : JSON.stringify(S.spec);
    specCache.set(res.scene, Promise.resolve(structuredClone(res.scene_spec)));
    S.report = await api.report(res.scene).catch(() => null);
    S.current = S.report ? S.report.attempts.length - 1 : 0;
    S.gallery = await api.gallery().catch(() => S.gallery);
    renderGallery(); renderTwin(); renderAll();
    if (res.valid === false) {
      $("dirtyBar").hidden = false;
      $("dirtyBar").querySelector(".grow").textContent = `Perception needs a correction before it can be saved: ${(res.issues || []).join("; ")}. Drag objects to fix, then Save twin.`;
    } else setTimeout(() => (box.hidden = true), 1600);
  } catch (e) {
    draw({ state: "error", log: e.log || [], progress: null }, e.status === 404 ? "POST /api/perceive is not available on this server yet." : e.message);
  } finally { S.busy = false; loadBudget(); }
}

function progressCard(box, title, steps, v, err, extra = "") {
  const phase = v.progress?.phase || inferPhase(v.log || []);
  let idx = steps.findIndex(([k]) => k === phase);
  if (v.state === "done" || phase === "done") idx = steps.length;
  const frac = v.state === "done" ? 1 : v.progress?.fraction ?? Math.max(0, idx) / steps.length;
  const log = (v.log || []).slice(-200).map((l) => {
    const cls = /error|fail|traceback/i.test(l) ? "err" : /success|✓|ok\b/i.test(l) ? "ok" : /muse spark|sam 3/i.test(l) ? "mm" : "";
    return cls ? `<span class="${cls}">${esc(l)}</span>` : esc(l);
  }).join("\n");
  box.innerHTML = `<div class="card-h"><h3>${title}</h3>
      <span class="chip ${err ? "bad" : v.state === "done" ? "good" : "accent"}">${err ? "error" : v.state === "done" ? "done" : v.state === "queued" ? `queued${v.queue_ahead ? ` · ${v.queue_ahead} ahead` : ""}` : `<span class="spin" style="width:10px;height:10px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:spin .8s linear infinite;display:inline-block"></span> running`}</span>
      <span class="grow"></span>${extra}</div>
    ${err ? `<div class="error-card">${esc(err)}</div>` : ""}
    <div class="stepper">${steps.map(([, label], i) => `<div class="${i < idx ? "done" : i === idx ? "now" : ""}">${esc(label)}</div>`).join("")}</div>
    <div class="pbar"><span style="width:${(frac * 100).toFixed(1)}%"></span></div>
    <pre class="logbox" aria-live="polite">${log || "waiting for the first log line…"}</pre>`;
  const lb = box.querySelector(".logbox"); lb.scrollTop = lb.scrollHeight;
}

function inferPhase(log) {
  const s = log.join("\n");
  if (/report:/.test(s)) return "report";
  if (/active perception|look again/.test(s)) return "look";
  if (/stress test/.test(s)) return "stress";
  if (/attempt/.test(s)) return "attempt";
  return "setup";
}

// ------------------------------------------------------------------ 02/03 rehearse job
const RUN_STEPS = [["setup", "Compile goals"], ["attempt", "Attempts"], ["stress", "Stress test"], ["look", "Look again"], ["report", "Report"]];
async function run(e) {
  e.preventDefault();
  if (S.busy) { toast("A job is already running"); return; }
  if (!S.scene) return;
  if (JSON.stringify(S.spec) !== S.saved && !(await saveTwin())) return;
  S.busy = true;
  const btn = $("run");
  btn.disabled = true; btn.innerHTML = `<span class="spin"></span>Rehearsing…`;
  const box = $("runProgress");
  box.hidden = false;
  $("rehearse").scrollIntoView({ behavior: "smooth" });
  replay && replay.pause();
  const t0 = Date.now();
  const draw = (v, err) => progressCard(box, `Rehearsing <code>${esc(S.scene)}</code>`, RUN_STEPS, v, err,
    `<span class="mbs">${MB.spark("goals · plan · diagnose")}${MB.phys("physics")}</span>`);
  draw({ state: "queued", log: [], progress: { phase: "setup", fraction: 0 } });
  try {
    const res = await runRehearsal({ scene: S.scene, instruction: $("instruction").value, stress: +$("stress").value, look: +$("look").value }, (v) => draw(v));
    S.bust = Date.now();
    const name = res?.scene || S.scene;
    S.report = await api.report(name);
    S.current = S.report ? S.report.attempts.length - 1 : 0;
    sceneLoaded = null;
    renderAll();
    toast(`Rehearsal finished in ${Math.round((Date.now() - t0) / 1000)} s`);
    setTimeout(() => (box.hidden = true), 1500);
    S.gallery = await api.gallery().catch(() => S.gallery);
    renderGallery();
  } catch (err) {
    draw({ state: "error", log: err.log || [], progress: null }, err.status === 409 ? "A rehearsal is already running on the server." : err.message);
  } finally {
    S.busy = false; btn.disabled = false; btn.textContent = "Rehearse"; loadBudget();
  }
}

// ------------------------------------------------------------------ 3D replay
let sceneLoaded = null, trailsRuns = null, eventRows = [];
function wireViewer() {
  $("playBtn").innerHTML = PLAY;
  $("playBtn").onclick = () => { if (!replay) return; replay.playing ? replay.pause() : replay.play(); syncPlay(); };
  $("scrub").oninput = (e) => { replay && (replay.pause(), replay.seek(+e.target.value)); syncPlay(); };
  $("speed").onchange = (e) => { if (replay) replay.speed = +e.target.value; };
  $("camReset").onclick = () => replay && replay.resetCamera();
  $("twinReset").onclick = () => { if (twin) { twin.resetCamera(); twin.camera.position.set(1.7, 0.9, 1.05); twin.controls.target.set(0.55, 0, 0.05); } };
  $("tglRobot").onclick = (e) => toggle(e.currentTarget, (on) => replay && replay.showRobot(on));
  $("tglTrails").onclick = (e) => toggle(e.currentTarget, (on) => {
    if (!replay) return;
    replay.showTrails(on);
    $("legend3d").hidden = !on;
    if (on) { replay.trailGroup.children.length && replay.setTrails(currentTrails, { animate: 3 }); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName) && document.activeElement.getAttribute("role") !== "button") {
      const r = $("viewer").getBoundingClientRect();
      if (r.bottom < 0 || r.top > innerHeight) return;
      e.preventDefault(); $("playBtn").click();
    }
  });
}
let currentTrails = [];
function toggle(btn, fn) { const on = btn.getAttribute("aria-pressed") !== "true"; btn.setAttribute("aria-pressed", on); fn(on); }
function syncPlay() {
  const p = replay && replay.playing;
  $("playBtn").innerHTML = p ? PAUSE : PLAY;
  $("playBtn").setAttribute("aria-label", p ? "Pause" : "Play");
}

let lastNow = -2;
function onTime(t) {
  $("scrub").value = t;
  $("timeLbl").textContent = `${t.toFixed(2)} s`;
  if (!replay.playing) syncPlay();
  let now = -1;
  eventRows.forEach((r, i) => { if (r.t <= t + 1e-6) now = i; });
  if (now === lastNow) return;
  lastNow = now;
  eventRows.forEach((r, i) => {
    r.el.classList.toggle("future", i > now);
    r.el.classList.toggle("now", i === now);
  });
}

async function loadReplay(a) {
  if (!replay) {
    $("viewer").innerHTML = `<img alt="Simulation of attempt ${a.attempt}" src="${base()}/attempt_${a.attempt}.gif?v=${S.bust}">`;
    return;
  }
  $("viewerEmpty").hidden = true;
  const key = report_key();
  if (sceneLoaded !== key) {
    $("mainLoader").hidden = false;
    try {
      await replay.loadScene(base(), (f) => { $("mainLoader").querySelector(".ring span").style.width = `${(f * 100).toFixed(0)}%`; $("mainPct").textContent = `${(f * 100).toFixed(0)}%`; });
    } catch (e) {
      $("mainLoader").innerHTML = `<div>Could not load the 3D scene: ${esc(e.message)}</div>`; return;
    }
    $("mainLoader").hidden = true;
    sceneLoaded = key;
    const getJson = (name) => fetch(`${base()}/${name}?v=${S.bust}`).then((r) => r.ok ? r.json() : null).catch(() => null);
    trailsRuns = S.report.stress_test ? await getJson("stress_trails.json") : null;
    const ap = S.report.active_perception;
    const after = ap && ap.steps.length ? await getJson("stress_trails_after.json") : null;
    if (trailsRuns) {
      const sets = [{ label: "One photo", runs: trailsRuns }];
      if (after) sets.push({ label: `After looking again at ${ap.steps.map((s) => s.look_at).join(", ")}`, runs: after });
      const show = (i, animate = 0) => {
        currentTrails = sets[i].runs;
        replay.setTrails(sets[i].runs, { animate });
        const runs = sets[i].runs, ok = runs.filter((r) => r.success).length;
        $("legend3d").innerHTML = (sets.length > 1 ? `<div class="seg">${sets.map((s, j) =>
            `<button type="button" data-i="${j}" aria-pressed="${i === j}">${esc(s.label)} · ${s.runs.filter((r) => r.success).length}/${s.runs.length}</button>`).join("")}</div>` : "") +
          `Moved objects in ${runs.length} randomized runs: <span class="ok">● ${ok} succeed</span> · <span class="bad">● ${runs.length - ok} fail</span>`;
        $("legend3d").querySelectorAll(".seg button").forEach((b) => b.onclick = () => show(+b.dataset.i, 2.5));
      };
      show(0);
    }
    $("tglTrails").disabled = !trailsRuns;
    replay.showTrails($("tglTrails").getAttribute("aria-pressed") === "true");
  }
  const frames = await fetch(`${base()}/attempt_${a.attempt}.traj.json?v=${S.bust}`).then((r) => r.json());
  const hits = hitList(a);
  replay.setTrajectory(frames, a.events, hits);
  const t0 = frames[0][0], dur = frames[frames.length - 1][0];
  $("scrub").min = t0; $("scrub").max = dur;
  $("ticks").innerHTML = a.events.filter((e) => e.kind !== "contact" || hits.some((h) => h.t === e.t)).map((e) => {
    const cls = e.kind === "contact" || e.kind === "protective_stop" ? "hit" : e.kind;
    const label = e.kind === "contact" ? `collision ${e.between.join(" ↔ ")}` : e.kind === "protective_stop" ? `protective stop on ${e.obj}` : `${e.kind === "grasp" ? "pick up" : "release"} ${e.obj}`;
    return `<span class="tick ${cls}" data-t="${e.t}" title="${esc(label)} · ${e.t.toFixed(2)} s" style="left:${(100 * (e.t - t0) / (dur - t0)).toFixed(2)}%"></span>`;
  }).join("");
  $("ticks").querySelectorAll(".tick").forEach((el) => el.onclick = () => jumpTo(+el.dataset.t));
  lastNow = -2;
  replay.seek(t0);
  // autoplay only when the viewer is on screen
  const r = $("viewer").getBoundingClientRect();
  if (r.top < innerHeight && r.bottom > 0) replay.play();
  syncPlay();
}
function report_key() { return `${S.report.scene}@${S.bust}`; }

function jumpTo(t) {
  if (!replay) return;
  replay.seek(t - 0.6);
  replay.speed = 0.5; $("speed").value = "0.5";
  replay.play(); syncPlay();
}

// ------------------------------------------------------------------ render report
function renderAll() {
  const r = S.report;
  $("agentChip").hidden = !r;
  if (r) {
    const offline = /offline/i.test(r.agent);
    $("agentChip").className = `chip ${offline ? "unc" : "accent"}`;
    $("agentChip").innerHTML = `<span class="dot"></span>${offline ? "offline stand-in agent" : esc(r.agent)}`;
    $("agentChip").title = offline ? "This report was produced by the rule-based OfflineAgent, not Muse Spark" : "Planner/diagnoser used for this report";
    $("instruction").value = r.instruction;
  }
  if (!r) { const g = S.gallery.find((x) => x.scene === S.scene); if (g?.demo_instruction) $("instruction").value = g.demo_instruction; }
  $("instruction").oninput();
  renderPlanStep();
  if (!r) {
    $("tiles").innerHTML = ""; $("tabs").innerHTML = ""; $("planN").innerHTML = ""; $("diag").innerHTML = ""; $("verdict").innerHTML = "";
    $("goals").innerHTML = `<li class="empty" style="display:block">No rehearsal yet for this scene.</li>`; $("events").innerHTML = "";
    $("viewerEmpty").hidden = false; replay && replay.pause();
    $("stressSummary").innerHTML = ""; $("scatter").innerHTML = `<div class="empty"><b>No report yet</b>Press Rehearse to run the plan in physics.</div>`; $("fails").innerHTML = "";
    $("lookCard").hidden = true; $("reportLinks").innerHTML = "";
    return;
  }
  $("reportLinks").innerHTML = `<a class="btn sm" href="${base()}/report.md" target="_blank" rel="noopener">report.md</a><a class="btn sm" href="${base()}/report.json" target="_blank" rel="noopener">report.json</a>`;
  renderTiles(); renderTabs(); renderAttempt(); renderStress(r.scene_spec.objects); renderLook();
}

function goalsWithWhy(r) {
  const g0 = r.attempts[0]?.goals || [];
  return r.goals.map((g) => ({ ...g, ...(g0.find((x) => goalText(x) === goalText(g)) || {}), implicit: g.implicit }));
}

function renderPlanStep() {
  const r = S.report;
  if (!r) {
    $("goals0").innerHTML = `<li class="empty" style="display:block">Goals appear after the first rehearsal — Muse Spark compiles them from the instruction.</li>`;
    $("plan0").innerHTML = `<li class="empty" style="display:block;border:0">No plan yet.</li>`;
    return;
  }
  $("goals0").innerHTML = goalsWithWhy(r).map((g) => `<li><span class="ic" style="color:var(--accent-text)">◇</span>
    <span><code>${esc(goalText(g))}</code>${g.implicit ? `<span class="tag-imp">implicit</span>` : `<span class="imp">${PRED[g.pred] || ""}</span>`}
    ${g.why ? `<div class="why">${esc(g.why)}</div>` : ""}</span></li>`).join("");
  $("plan0").innerHTML = r.attempts[0].plan.actions.map((x) =>
    `<li><span><code>${esc(x.obj)}</code> ${REL[x.relation] || x.relation} <code>${esc(x.dest)}</code></span><span class="meta">carry height ${x.transit_height} m${x.via ? ` · ${x.via.length} waypoints` : ""}</span></li>`).join("");
}

function renderTiles() {
  const r = S.report, A = r.attempts, last = A[A.length - 1], st = r.stress_test, ap = r.active_perception;
  const unreach = Object.values(r.reachability).filter((v) => !v).length, nObj = Object.keys(r.reachability).length;
  const tiles = [
    ["Outcome", last.success ? `<span class="ic ok">✓</span> Done` : `<span class="ic bad">✕</span> Not done`,
      last.success ? `succeeded on attempt ${A.length}` : `still failing after ${A.length} attempts`, MB.phys("verdict")],
    ["Self-corrections", `${A.length - 1}`, A.length > 1 ? "diagnosed from the sim log, re-planned" : "first plan worked", MB.spark("diagnose")],
    ["Stress test", st ? `${st.successes}/${st.n}` : "—", !st ? "final plan failed — not run"
      : ap && ap.steps.length ? `${ap.final.successes}/${st.n} after looking again at ${ap.steps.map((s) => s.look_at).join(", ")}`
        : `${Math.round(100 * st.successes / st.n)}% robust success`, ""],
    ["Reachability", `${nObj - unreach}/${nObj}`, unreach ? `${unreach} object${unreach > 1 ? "s" : ""} outside the arm's workspace` : `all reachable · ${r.wall_time_s} s`, ""],
  ];
  $("tiles").innerHTML = tiles.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
}

function renderTabs() {
  $("tabs").innerHTML = S.report.attempts.map((a, i) =>
    `<button type="button" role="tab" aria-selected="${i === S.current}" data-i="${i}">
      <span class="ic ${a.success ? "ok" : "bad"}">${a.success ? "✓" : "✕"}</span>Attempt ${a.attempt}</button>`).join("");
  $("tabs").querySelectorAll("button").forEach((b) => b.onclick = () => { S.current = +b.dataset.i; renderTabs(); renderAttempt(); });
}

function renderAttempt() {
  const a = S.report.attempts[S.current], prev = S.report.attempts[S.current - 1];
  loadReplay(a);
  $("planN").innerHTML = a.plan.actions.map((x, i) => {
    const p = prev && prev.plan.actions[i];
    const chg = p && p.transit_height !== x.transit_height ? `<span class="chg">height ${p.transit_height} → ${x.transit_height} m</span>` : "";
    return `<li><span><code>${esc(x.obj)}</code> ${REL[x.relation] || x.relation} <code>${esc(x.dest)}</code></span><span class="meta">carry height ${x.transit_height} m</span>${chg}</li>`;
  }).join("");
  const failed = a.goals.filter((g) => !g.ok).length;
  $("verdict").innerHTML = `<div class="verdict ${a.success ? "ok" : "bad"}">${a.success ? "✓" : "✕"} Physics: ${a.success ? "all goals met" : `${failed} goal${failed > 1 ? "s" : ""} failed`}<span style="margin-left:auto;font-weight:500;font-size:12px;opacity:.8">attempt ${a.attempt} · ${a.sim_time?.toFixed?.(1) ?? "—"} s sim</span></div>`;
  $("diag").innerHTML = a.plan.diagnosis
    ? `<div class="diag"><div class="h">${MB.spark("diagnose")}<span>from the sim log of attempt ${a.attempt - 1}</span></div>${esc(a.plan.diagnosis)}</div>` : "";
  $("goals").innerHTML = a.goals.map((g) =>
    `<li><span class="ic ${g.ok ? "ok" : "bad"}">${g.ok ? "✓" : "✕"}</span>
     <span><code>${esc(goalText(g))}</code>${g.implicit ? `<span class="tag-imp">implicit</span>` : `<span class="imp">${PRED[g.pred] || ""}</span>`}
     <div class="why">${esc(g.why)}</div></span></li>`).join("");
  const hitSet = new Set(unexpectedContacts(a));
  const ev = a.events.filter((e) => e.kind !== "contact" || !e.between.includes("table") || e.between.includes("robot_arm"));
  $("events").innerHTML = (a.execution_error ? `<li class="hit"><span class="t">—</span><span>${esc(a.execution_error.error)}</span></li>` : "") +
    ev.map((e) => `<li class="${hitSet.has(e) ? "hit" : ""}" data-t="${e.t}" title="Jump to this moment" tabindex="0"><span class="t">${e.t.toFixed(2)}s</span><span>${eventText(e)}</span></li>`).join("");
  eventRows = [...$("events").querySelectorAll("li[data-t]")].map((el) => ({ el, t: +el.dataset.t }));
  eventRows.forEach((r) => { r.el.onclick = () => jumpTo(r.t); r.el.onkeydown = (e) => { if (e.key === "Enter") jumpTo(r.t); }; });
}

function renderStress(objs) {
  const st = S.report.stress_test;
  if (!st) { $("scatter").innerHTML = `<div class="empty"><b>Not run</b>The final plan did not succeed, so there is nothing to stress-test.</div>`; $("fails").innerHTML = ""; $("stressSummary").textContent = ""; return; }
  const runs = st.runs, fails = runs.filter((r) => !r.success), oks = runs.filter((r) => r.success);
  const ids = objs.map((o) => o.id);
  const mean = (rs, id) => rs.reduce((s, r) => s + r.perception_error_cm[id], 0) / Math.max(rs.length, 1);
  const best = fails.length ? ids.slice().sort((a, b) => (mean(fails, b) - mean(oks, b)) - (mean(fails, a) - mean(oks, a)))[0] : ids[0];
  const sel = $("xobj");
  if (sel.dataset.key !== report_key()) {
    sel.innerHTML = ids.map((id) => `<option ${id === best ? "selected" : ""}>${esc(id)}</option>`).join("");
    sel.dataset.key = report_key();
    sel.onchange = () => { drawScatter(runs, sel.value); drawFails(fails, sel.value); };
  }
  $("stressSummary").innerHTML = `<b>${st.successes} of ${st.n}</b> randomized runs succeed. ` +
    (fails.length ? `Failures track the perception error of <code>${esc(best)}</code> (failed runs average ${mean(fails, best).toFixed(1)} cm, successful ones ${mean(oks, best).toFixed(1)} cm).` : "No failures.");
  drawScatter(runs, sel.value);
  drawFails(fails, sel.value);
}

function drawFails(fails, id) {
  $("fails").innerHTML = fails.length ? `<tr><th>seed</th><th>${esc(id)} error</th><th>friction</th><th>why it failed</th></tr>` +
    fails.map((r) => `<tr><td class="num">${r.seed}</td><td class="num">${r.perception_error_cm[id].toFixed(1)} cm</td>
      <td class="num">×${r.friction_scale.toFixed(2)}</td><td>${r.failed.map(esc).join("<br>")}</td></tr>`).join("")
    : `<tr><td class="empty">All runs succeeded</td></tr>`;
}

function drawScatter(runs, id) {
  const W = 560, H = 300, m = { l: 44, r: 12, t: 10, b: 36 };
  const xmax = Math.max(4, Math.ceil(Math.max(...runs.map((r) => r.perception_error_cm[id]))));
  const X = (v) => m.l + (v / xmax) * (W - m.l - m.r), Y = (v) => m.t + (1 - (v - 0.6) / 0.8) * (H - m.t - m.b);
  let g = "";
  for (let v = 0; v <= xmax; v++) g += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - m.b + 16}" text-anchor="middle">${v}</text>`;
  for (const v of [0.6, 0.8, 1.0, 1.2, 1.4]) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">×${v.toFixed(1)}</text>`;
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  g += `<text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">${esc(id)} position error (cm)</text>`;
  g += `<text transform="translate(12 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">friction scale</text>`;
  const failErr = runs.filter((r) => !r.success).map((r) => r.perception_error_cm[id]);
  if (failErr.length) {
    const mn = Math.min(...failErr);
    g += `<line x1="${X(mn)}" x2="${X(mn)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--bad)" stroke-dasharray="4 4"/>
          <text x="${X(mn) - 6}" y="${m.t + 12}" text-anchor="end">smallest failing error ${mn.toFixed(1)} cm</text>`;
  }
  for (const r of runs) {
    const x = X(r.perception_error_cm[id]), y = Y(r.friction_scale);
    const t = esc(`<b>seed ${r.seed} · ${r.success ? "✓ success" : "✕ failure"}</b>${id} error ${r.perception_error_cm[id].toFixed(1)} cm<br>friction ×${r.friction_scale.toFixed(2)} · mass ×${r.mass_scale.toFixed(2)}` +
      (r.failed.length ? "<br>" + r.failed.map(esc).join("<br>") : ""));
    g += `<g data-tip="${t}"><circle cx="${x}" cy="${y}" r="10" fill="transparent"/>` + (r.success
      ? `<circle cx="${x}" cy="${y}" r="4.5" fill="var(--good)" stroke="var(--surface)" stroke-width="2"/>`
      : `<path d="M${x - 5} ${y - 5}L${x + 5} ${y + 5}M${x + 5} ${y - 5}L${x - 5} ${y + 5}" stroke="var(--bad)" stroke-width="2.5" stroke-linecap="round"/>`) + `</g>`;
  }
  $("scatter").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Stress test scatter plot">${g}</svg>`;
  bindTips($("scatter"));
}

function renderLook() {
  const ap0 = S.report.active_perception;
  $("lookCard").hidden = !ap0;
  if (!ap0) return;
  const num = (v) => typeof v === "number" ? v : Math.max(...Object.values(v || { a: 0 }).map(Number));
  const ap = { ...ap0, sigma_cm: num(ap0.sigma_cm), refined_sigma_cm: num(ap0.refined_sigma_cm) };
  $("sig0").textContent = ap.sigma_cm.toFixed(1); $("sig1").textContent = ap.refined_sigma_cm.toFixed(1);
  const n = ap.n, b0 = S.report.stress_test.successes;
  if (!ap.steps.length) {
    $("lookSummary").textContent = b0 === n ? "Every stress run already succeeds — no need to look again." : "Looking again at any single object does not help: the bottleneck is not perception.";
    $("lookBars").innerHTML = ""; $("lookChain").innerHTML = ""; return;
  }
  const first = ap.steps[0], vals = first.value_by_object;
  const ranked = Object.keys(vals).sort((a, b) => vals[b] - vals[a]);
  const useless = ranked.filter((k) => vals[k] <= first.before);
  const last = ap.steps[ap.steps.length - 1];
  $("lookSummary").innerHTML = `Take one more look at <code>${ap.steps.map((s) => esc(s.look_at)).join("</code>, <code>")}</code>: success goes from <b>${first.before}/${n}</b> to <b>${last.after}/${n}</b>.` +
    (useless.length ? ` Looking again at ${useless.map((k) => `<code>${esc(k)}</code>`).join(", ")} would not help.` : "");
  const W = 560, rowH = 30, m = { l: 96, r: 70, t: 6 }, H = m.t + ranked.length * rowH + 6;
  const maxGain = Math.max(1, ...ranked.map((k) => vals[k] - first.before));
  const X = (v) => m.l + (v / maxGain) * (W - m.l - m.r);
  let g = `<line x1="${X(0)}" x2="${X(0)}" y1="${m.t - 2}" y2="${H - 4}" stroke="var(--axis)"/>`;
  ranked.forEach((k, i) => {
    const gain = vals[k] - first.before, y = m.t + i * rowH, chosen = k === first.look_at;
    const tipTxt = esc(`<b>Look again at ${k}</b>${first.before}/${n} → ${vals[k]}/${n}`);
    g += `<g data-tip="${tipTxt}"><rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>
      <text x="${m.l - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" style="${chosen ? "fill:var(--text);font-weight:600" : ""}">${esc(k)}</text>`;
    if (gain > 0) {
      const w = X(gain) - X(0);
      g += `<path d="M${X(0)} ${y + 7}h${w - 4}a4 4 0 0 1 4 4v${rowH - 22}a4 4 0 0 1 -4 4h${-(w - 4)}z" fill="${chosen ? "var(--accent)" : "var(--muted)"}"/>`;
    }
    g += `<text x="${X(Math.max(gain, 0)) + 8}" y="${y + rowH / 2 + 4}">${gain > 0 ? "+" + gain : "0"} · ${vals[k]}/${n}</text></g>`;
  });
  $("lookBars").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Value of looking again at each object">${g}</svg>`;
  bindTips($("lookBars"));
  let chain = `<div class="step"><div class="lbl">One photo · every object σ=${ap.sigma_cm.toFixed(1)} cm</div><div class="big">${first.before}/${n}</div></div>`;
  ap.steps.forEach((s, i) => {
    chain += `<div class="arrow">↓ look again at <code>${esc(s.look_at)}</code></div>
      <div class="step ${i === ap.steps.length - 1 ? "final" : ""}"><div class="lbl"><code>${esc(s.look_at)}</code> σ=${ap.refined_sigma_cm.toFixed(1)} cm</div><div class="big">${s.after}/${n}</div></div>`;
  });
  $("lookChain").innerHTML = chain;
}

// ------------------------------------------------------------------ go
if (params.has("demo")) {
  document.getElementById("app").hidden = true;
  [hero, twin, replay].forEach((r) => r && r.dispose());
  (await import("./demo.js")).runDemo(api);
} else {
  main().catch((e) => { console.error(e); toast(`Something went wrong: ${e.message}`); });
}
