// Demo mode (?demo=1): a self-running, screen-recordable sequence of the "money shot".
// attempt 1 fails → Muse Spark reads the sim log → attempt 2 succeeds → stress test universes fan out
// → active perception picks what to look at again. Keys: R restart · C cursor · Space pause.
import { Replay } from "./replay.js";
import { $, esc, sleep, GLOSS, hitList, goalText, LOGO, MB, REL } from "./common.js";

const P = new URLSearchParams(location.search);
const SPEED = +(P.get("speed") || 1);         // global pacing multiplier for rehearsing the take
const wait = (s) => sleep((s * 1000) / SPEED);
let paused = false;
async function hold(s) { const end = performance.now() + (s * 1000) / SPEED; while (performance.now() < end || paused) await sleep(50); }

export async function runDemo(api) {
  document.body.classList.add("demo");
  const root = $("demoRoot");
  root.hidden = false;
  root.innerHTML = `
  <div class="demo-stage">
    <div class="dv"><div class="stage" id="dStage"></div>
      <div class="demo-banner" id="dBanner"></div>
      <div class="demo-hud" id="dHud"></div>
      <div class="loader" id="dLoader"><div>Loading physics twin…<div class="ring"><span></span></div></div></div>
      <div class="demo-title" id="dTitle"><div><h1>Rehearsal</h1><p>Models propose, physics decides.</p>
        <div class="by">${MB.spark("plans &amp; diagnoses")}${MB.sam("segments")}${MB.phys("decides")}</div></div></div>
    </div>
    <aside class="dpanel">
      <div class="brand">${LOGO}Rehearsal<span class="tl">physics twin rehearsal</span></div>
      <div class="phase" id="dPhase">&nbsp;</div>
      <div class="dblock" id="bInstr"><div class="dlabel">Instruction</div><div class="instr" id="dInstr"></div></div>
      <div class="dblock" id="bGoals"><div class="dlabel">Checkable goals ${MB.spark("compile goals")}</div><ul class="dgoals" id="dGoals"></ul></div>
      <div class="dblock" id="bPlan"><div class="dlabel">Plan ${MB.spark("plan")}</div><div class="dlog" id="dPlan"></div></div>
      <div class="dblock" id="bLog" hidden><div class="dlabel">Sim log ${MB.phys("contacts")}</div><div class="dlog" id="dLog"></div></div>
      <div class="dblock" id="bDiag" hidden><div class="dlabel">Diagnosis ${MB.spark("reads the sim log")}</div><div class="ddiag" id="dDiag"></div></div>
      <div class="dblock" id="bStress" hidden><div class="dlabel" id="dStressLbl">Stress test · parallel universes ${MB.phys("× N")}</div>
        <div class="bigstat"><span class="n" id="dN">0</span><span class="of" id="dOf">/ 40</span><span class="lab" id="dLab">universes succeed</span></div>
        <div class="dlog" id="dStressNote" style="margin-top:14px"></div></div>
    </aside>
  </div>
  <div class="demo-ctl">R restart · Space pause · C cursor</div>`;
  document.addEventListener("keydown", (e) => {
    if (e.key === "r" || e.key === "R") location.reload();
    if (e.key === "c" || e.key === "C") document.body.classList.toggle("cursor");
    if (e.code === "Space") { e.preventDefault(); paused = !paused; if (r) paused ? r.pause() : (r.frames.length && r.t < r.duration && r.play()); }
  });

  let r;
  try {
    const gal = await api.gallery();
    const scene = P.get("scene") || (gal.find((g) => g.has_report) || {}).scene;
    const rep = scene && await api.report(scene);
    if (!rep) throw new Error("no precomputed report to demo — run a rehearsal first");
    const base = `/out/${encodeURIComponent(scene)}`;
    r = new Replay($("dStage"), {});
    r.controls.enabled = false;
    await r.loadScene(base, (f) => { $("dLoader").querySelector(".ring span").style.width = `${(f * 100).toFixed(0)}%`; });
    const getJson = (n) => fetch(`${base}/${n}`).then((x) => x.ok ? x.json() : null).catch(() => null);
    const A = rep.attempts;
    const failA = A.find((a) => !a.success), okA = A[A.length - 1];
    const frames = {};
    for (const a of A) frames[a.attempt] = await getJson(`attempt_${a.attempt}.traj.json`);
    const trails = rep.stress_test ? await getJson("stress_trails.json") : null;
    const after = rep.active_perception?.steps.length ? await getJson("stress_trails_after.json") : null;
    $("dLoader").hidden = true;
    await sequence(r, rep, failA, okA, frames, trails, after);
  } catch (e) {
    console.error(e);
    $("dLoader").hidden = false;
    $("dLoader").innerHTML = `<div style="font-size:16px">Demo unavailable: ${esc(e.message)}</div>`;
    $("dTitle").classList.add("out");
  }
}

function show(id, on = true) { const el = $(id); el.hidden = !on; requestAnimationFrame(() => requestAnimationFrame(() => el.classList.toggle("in", on))); }
function phase(t) { $("dPhase").textContent = t; }
let bannerTimer;
function banner(html, kind = "info", secs = 2.6) {
  const b = $("dBanner");
  b.className = `demo-banner ${kind}`; b.innerHTML = html;
  requestAnimationFrame(() => b.classList.add("in"));
  clearTimeout(bannerTimer);
  if (secs) bannerTimer = setTimeout(() => b.classList.remove("in"), (secs * 1000) / SPEED);
}
async function type(el, text, cps = 34) {
  el.innerHTML = `<span></span><span class="caret"></span>`;
  const span = el.firstChild;
  for (let i = 1; i <= text.length; i++) { span.textContent = text.slice(0, i); await hold(1 / cps); }
}
async function untilDone(r) { while (r.playing || paused) await sleep(60); }
function setGoal(i, state) {
  const st = $("dGoals").children[i]?.querySelector(".st");
  if (!st) return;
  st.className = `st ${state || ""}`; st.textContent = state === "ok" ? "✓" : state === "bad" ? "✕" : "";
}
function hud(html) { $("dHud").innerHTML = html; }

const WIDE = [[1.75, 1.05, 0.95], [0.5, 0, 0.08]];
const LOW = [[1.45, -0.95, 0.7], [0.5, 0, 0.1]];

async function sequence(r, rep, failA, okA, frames, trails, after) {
  const goals = rep.goals;
  // --- title over a slowly orbiting twin
  const first = failA || okA;
  r.setTrajectory(frames[first.attempt], first.events, []);
  r.camera.position.set(1.9, 1.4, 1.2);
  r.orbit(true, 0.5);
  await hold(3.6);
  $("dTitle").classList.add("out");
  r.orbit(false);
  r.flyTo(...WIDE, 2.2);

  // --- instruction → goals
  phase("01 · Instruction → checkable goals");
  show("bInstr");
  const gloss = GLOSS[rep.instruction];
  await type($("dInstr"), gloss || rep.instruction, 40);
  if (gloss) $("dInstr").insertAdjacentHTML("beforeend", `<span class="orig">${esc(rep.instruction)}</span>`);
  await hold(0.5);
  show("bGoals");
  $("dGoals").innerHTML = goals.map((g) => `<li><span class="st"></span><span>${esc(goalText(g))}${g.implicit ? ` <span style="color:#f5c25e;font-family:var(--font);font-size:12px">implicit</span>` : ""}</span></li>`).join("");
  for (const li of $("dGoals").children) { li.classList.add("in"); await hold(0.35); }
  await hold(0.6);

  // --- plan
  phase("02 · Muse Spark proposes a plan");
  const planHtml = (a, prev) => a.plan.actions.map((x, i) => {
    const p = prev && prev.plan.actions[i], chg = p && p.transit_height !== x.transit_height;
    return `<div>${i + 1}. ${esc(x.obj)} ${REL[x.relation] || x.relation} ${esc(x.dest)} · <span style="${chg ? "color:#7eaaff;font-weight:600" : ""}">${chg ? `${p.transit_height} → ` : ""}${x.transit_height} m</span></div>`;
  }).join("");
  $("dPlan").innerHTML = planHtml(first);
  show("bPlan");
  await hold(1.4);

  if (failA) {
    // --- attempt 1: physics says no
    phase(`03 · Physics runs attempt ${failA.attempt}`);
    hud(`<span class="glass"><span class="live-dot"></span>MuJoCo · attempt ${failA.attempt}</span>`);
    const hits = hitList(failA);
    let hitShown = false;
    const firstHit = hits[0];
    r.setTrajectory(frames[failA.attempt], failA.events, hits);
    r.onHit = async (h) => {
      if (hitShown) return;
      hitShown = true;
      r.speed = 0.22;
      const p = r.bodyPosition(h.obj) || [0.42, 0, 0.1];
      r.flyTo([p[0] + 0.55, p[1] + 0.42, p[2] + 0.32], [p[0], p[1], p[2] + 0.05], 1.4);
      const ev = failA.events.find((e) => e.t === h.t);
      banner(`✕ Collision — ${esc(ev?.while_holding || "arm")} hits ${esc(h.obj)}`, "bad", 3.4);
      $("dLog").innerHTML = `<div class="hit">t=${h.t.toFixed(2)}s  ${esc(ev.between.join(" ↔ "))} · holding ${esc(ev.while_holding)} · tool z ${ev.tcp_z} m</div>`;
      show("bLog");
      await wait(2.6);
      r.speed = 1.6;
      r.flyTo(...WIDE, 2);
    };
    r.speed = 1;
    r.play();
    await untilDone(r);
    r.onHit = () => {};
    failA.goals.forEach((g, i) => setGoal(i, g.ok ? "ok" : "bad"));
    const nf = failA.goals.filter((g) => !g.ok).length;
    banner(`Physics: ${nf} goal${nf > 1 ? "s" : ""} failed`, "bad", 2.6);
    const moved = failA.goals.filter((g) => !g.ok).map((g) => `<div>✕ ${esc(goalText(g))}</div>`).join("");
    $("dLog").insertAdjacentHTML("beforeend", moved);
    if (!firstHit) show("bLog");
    await hold(2.4);

    // --- diagnosis
    phase("04 · Muse Spark reads the sim log");
    hud(`<span class="glass">self-correction · round 1</span>`);
    show("bDiag");
    await type($("dDiag"), okA.plan.diagnosis || "—", 30);
    await hold(0.8);
    $("dPlan").innerHTML = planHtml(okA, failA);
    banner(`Re-plan: carry height ${failA.plan.actions[0].transit_height} → ${okA.plan.actions[0].transit_height} m`, "info", 2.4);
    await hold(2.4);
  }

  // --- the fixed plan
  phase(`05 · Physics runs attempt ${okA.attempt}`);
  hud(`<span class="glass"><span class="live-dot"></span>MuJoCo · attempt ${okA.attempt}</span>`);
  goals.forEach((_, i) => setGoal(i, null));
  show("bLog", false);
  r.setTrajectory(frames[okA.attempt], okA.events, hitList(okA));
  r.flyTo(...LOW, 2.5);
  r.speed = 1.5;
  r.play();
  setTimeout(() => r.flyTo(...WIDE, 4), 3500 / SPEED);
  await untilDone(r);
  okA.goals.forEach((g, i) => setGoal(i, g.ok ? "ok" : "bad"));
  banner(okA.success ? "✓ All goals met — verified on sim ground truth" : "Still failing", okA.success ? "ok" : "bad", 3);
  await hold(3);

  // --- stress test: parallel universes
  const st = rep.stress_test;
  if (st && trails) {
    phase("06 · Stress test · parallel universes");
    show("bDiag", false); show("bPlan", false);
    hud(`<span class="glass">${st.n} randomized runs · friction, mass, ±${st.sigma_cm ?? 1.2} cm perception error</span>`);
    r.showRobot(false);
    r.flyTo([0.95, -0.55, 1.05], [0.52, 0.02, 0.02], 2.4);
    await hold(1.2);
    $("dOf").textContent = `/ ${st.n}`;
    show("bStress");
    const fails = st.runs.filter((x) => !x.success);
    $("dStressNote").innerHTML = `${st.n} copies of the world, each slightly wrong.<br><span style="color:#4fd58f">green</span> = goals met · <span style="color:#ff7b80">red</span> = failed`;
    r.setTrails(trails, { animate: 5 });
    r.showTrails(true);
    await countTo($("dN"), st.successes, 2.6);
    await hold(1.4);
    if (fails.length) $("dStressNote").innerHTML = `${fails.length} universes fail: ${esc(fails[0].failed[0] || "")}`;
    await hold(2.6);

    const ap = rep.active_perception;
    if (ap && ap.steps.length && after) {
      phase("07 · Active perception · 再看一眼");
      const looks = ap.steps.map((s) => s.look_at);
      banner(`Look again at ${esc(looks.join(", "))} — worth +${ap.steps[ap.steps.length - 1].after - ap.steps[0].before} universes`, "info", 3.2);
      $("dStressNote").innerHTML = `Which single object is worth a second photo? Re-run with σ ${ap.sigma_cm} → ${ap.refined_sigma_cm} cm for each candidate, same seeds:<br>` +
        Object.entries(ap.steps[0].value_by_object).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k === looks[0] ? "▶" : "&nbsp;"} ${esc(k)} ${v}/${ap.n}`).join("<br>");
      await hold(2.6);
      r.setTrails(after, { animate: 4 });
      $("dLab").textContent = `after looking again at ${looks.join(", ")}`;
      await countTo($("dN"), ap.final.successes, 2, st.successes);
      await hold(3);
    }
  }

  // --- end card
  phase("Models propose, physics decides.");
  hud("");
  r.orbit(true, 0.4);
  $("dTitle").innerHTML = `<div><h1>Models propose,<br>physics decides.</h1><p>Rehearsal · a phone photo becomes a physics twin</p>
    <div class="by">${MB.spark("list · goals · plan · diagnose")}${MB.sam("segment")}${MB.phys("verify")}</div></div>`;
  await hold(0.6);
  $("dTitle").classList.remove("out");
}

async function countTo(el, n, secs, from = 0) {
  const t0 = performance.now();
  for (;;) {
    const u = Math.min(1, (performance.now() - t0) / ((secs * 1000) / SPEED));
    el.textContent = Math.round(from + (n - from) * (1 - (1 - u) ** 3));
    if (u >= 1) return;
    await sleep(30);
  }
}
