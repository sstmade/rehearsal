// Demo mode (?demo=1): cinematic auto-play of the money shot for the 3-minute video.
// fail → model diagnosis (typewriter) → fix → success → 40 universes fan out → look again.
// Uses only real report data; no debug chrome. ?loop=1 restarts at the end.
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function run(app) {
  const { replay, state, esc } = app;
  await app.ready();
  const r = state.report;
  if (!r || !replay) return;
  const ui = $("demoUi"); ui.hidden = false;
  const vig = document.createElement("div"); vig.className = "demo-vignette"; ui.prepend(vig);
  const scrim = document.createElement("div"); scrim.className = "demo-scrim"; ui.insertBefore(scrim, $("demoTitle"));
  const phase = (name) => { app.phase = name; };
  const statEl = document.createElement("div"); statEl.className = "demo-stat"; ui.appendChild(statEl);
  const lower = (chip, text, cls = "") => { $("demoChip").textContent = chip; $("demoChip").className = `demo-chip ${cls}`; $("demoText").innerHTML = text; $("demoLower").classList.toggle("on", !!(chip || text)); };
  const waitT = (t) => new Promise((res) => { const f = () => (replay.t >= t || !replay.playing ? res() : requestAnimationFrame(f)); f(); });
  const waitEnd = () => new Promise((res) => { const f = () => (!replay.playing ? res() : requestAnimationFrame(f)); f(); });
  const failIdx = r.attempts.findIndex((a) => !a.success), fixIdx = r.attempts.findIndex((a, i) => i > failIdx && a.success);
  const lastIdx = r.attempts.length - 1;

  // 0 · title over a slow orbit of the twin
  app.loadAttempt(lastIdx, { play: false });
  replay.flyTo("home", 0); replay.orbit(true, 0.5);
  phase("title"); scrim.classList.add("on"); $("demoTitle").classList.add("on");
  await sleep(4200);
  $("demoTitle").classList.remove("on"); scrim.classList.remove("on");
  await sleep(700);

  // 1 · twin
  phase("twin"); replay.flyTo("twin", 1.8);
  const low = {};
  for (const o of r.scene_spec.objects) if (o.perception && o.perception.confidence === "low") low[o.id] = "warn";
  for (const [id, ok] of Object.entries(r.reachability)) if (!ok) low[id] = "bad";
  replay.setFocus(low);
  lower("01 · Photo → twin", `${r.scene_spec.objects.length} objects proposed by <b>Muse Spark</b>, segmented by <b>SAM 3.1</b>, placed in MuJoCo`);
  await sleep(3600);

  // 2 · plan
  phase("plan"); replay.flyTo("top", 1.6); replay.showRobot(false);
  const f = {}; for (const x of r.attempts[0].plan.actions) { f[x.obj] = "accent"; if (!f[x.dest]) f[x.dest] = "good"; }
  replay.setFocus(f);
  const a0 = r.attempts[0].plan.actions[0];
  lower("02 · Instruction → plan", `“${esc(r.instruction)}” → ${r.goals.length} checkable goals · ${r.attempts[0].plan.actions.length} actions · carry <b>${esc(a0.obj)}</b> at ${a0.transit_height} m`);
  await sleep(3600);
  replay.setFocus({}); replay.showRobot(true);

  // 3 · attempt 1 fails
  if (failIdx >= 0) {
    const a = r.attempts[failIdx];
    const hits = app.unexpectedContacts(a);
    const hit = hits[0];
    app.loadAttempt(failIdx, { play: false });
    phase("attempt1"); replay.flyTo("cinema", 1.6);
    lower("03 · Rehearse · attempt 1", `physics runs the plan`);
    await sleep(1200);
    replay.speed = 1.3; replay.play();
    if (hit) {
      await waitT(hit.t - 1.0);
      const obj = hit.between.find((x) => x !== "robot_arm" && x !== hit.while_holding) || hit.between[0];
      const p = replay.bodyByName[obj].position.clone();
      replay.flyTo({ pos: [p.x + 0.62, p.y + 0.5, 0.46], target: [p.x, p.y, 0.14] }, 1.0);
      replay.speed = 0.5;
      await waitT(hit.t - 0.02);
      replay.speed = 0.18; phase("collision");
      lower(`collision · t = ${hit.t.toFixed(3)} s`, `<b>${esc(hit.between.join(" ↔ "))}</b> while holding ${esc(hit.while_holding)} at z = ${hit.tcp_z} m`, "bad");
      await sleep(2600);
      replay.speed = 1.6;
      await waitT(Math.min(replay.duration, hit.t + 3.0));
      replay.pause();
    } else { await waitEnd(); }
    await sleep(300);
    showGoals(a, false); phase("goals1");
    lower("attempt 1 · failed", `${a.goals.filter((g) => !g.ok).length} of ${a.goals.length} goals failed — the simulator says why`, "bad");
    await sleep(3400);
    hideGoals();
  }

  // 4 · diagnosis
  if (fixIdx >= 0 && r.attempts[fixIdx].plan.diagnosis) {
    const a = r.attempts[fixIdx], prev = r.attempts[failIdx];
    const changes = a.plan.actions.map((x, j) => prev.plan.actions[j] && prev.plan.actions[j].transit_height !== x.transit_height
      ? `<span><code>${esc(x.obj)}</code> transit_height <s>${prev.plan.actions[j].transit_height}</s> → <b>${x.transit_height} m</b></span>` : "").filter(Boolean);
    const card = $("demoCard"); card.hidden = false;
    card.innerHTML = `<div class="who">Muse Spark · reads the sim log</div><div class="q"><span class="cursor"></span></div>${app.GLOSS[a.plan.diagnosis] ? `<div class="q" style="font-size:.78em;color:#aab1c2;min-height:0;margin-top:8px" id="demoGloss"></div>` : ""}<div class="fix" hidden>${changes.join(" · ")}</div>`;
    lower("Self-correction", `the model is handed contacts with timestamps, IK errors and failed predicates`, "spark");
    replay.flyTo("cinema", 2.0);
    await sleep(100); card.classList.add("on"); phase("diag");
    const q = card.querySelector(".q"), text = a.plan.diagnosis;
    for (let i = 1; i <= text.length; i++) { q.innerHTML = `${esc(text.slice(0, i))}<span class="cursor"></span>`; await sleep(text.charCodeAt(i - 1) > 255 ? 42 : 18); }
    q.textContent = text;
    const g = $("demoGloss"); if (g) { g.textContent = app.GLOSS[text]; }
    await sleep(900);
    card.querySelector(".fix").hidden = false; phase("fix");
    await sleep(2800);
    card.classList.remove("on"); await sleep(600); card.hidden = true;
  }

  // 5 · attempt 2 succeeds
  if (fixIdx >= 0) {
    const a = r.attempts[fixIdx], a1 = a.plan.actions[0];
    app.loadAttempt(fixIdx, { play: false });
    phase("attempt2"); replay.flyTo("cinema", 1.2);
    lower("attempt 2", `same instruction · carry <b>${esc(a1.obj)}</b> at <b>${a1.transit_height} m</b>`, "ok");
    await sleep(900);
    replay.speed = 2.2; replay.play();
    await waitEnd();
    showGoals(a, true); phase("goals2");
    lower("attempt 2 · success", `all ${a.goals.length} goals verified by physics`, "ok");
    await sleep(3200);
    hideGoals();
  }

  // 6 · universes fan out
  if (state.trailSets.length) {
    replay.flyTo("universe", 2.0);
    replay.showRobot(false);
    const s0 = state.trailSets[0];
    lower("04 · Parallel universes", `${s0.runs.length} randomised replays · friction, mass and perception noise`);
    phase("universes"); await app.fanOut(0, 3200); phase("universes-stat");
    statEl.innerHTML = `<div class="n">${s0.successes}<span class="arr">/</span>${s0.runs.length}</div><div class="l">universes where the plan still works</div>`;
    statEl.classList.add("on");
    await sleep(2600);
    statEl.classList.remove("on");

    // 7 · look again
    if (state.trailSets.length > 1 && r.active_perception) {
      const s1 = state.trailSets[1], look = r.active_perception.steps.map((st) => st.look_at);
      replay.setFocus(Object.fromEntries(look.map((k) => [k, "accent"])));
      lower("05 · Look again · 再看一眼", `which object is worth a second photo? <b>${esc(look.join(", "))}</b> · same seeds, σ ${r.active_perception.sigma_cm} → ${r.active_perception.refined_sigma_cm} cm`);
      await sleep(1600);
      await app.fanOut(1, 2200);
      phase("look-stat"); statEl.innerHTML = `<div class="n">${s0.successes}<span class="arr">→</span><span class="ok">${s1.successes}</span><span class="arr">/</span>${s1.runs.length}</div><div class="l">after re-observing ${esc(look.join(", "))}</div>`;
      statEl.classList.add("on");
      await sleep(3000);
      statEl.classList.remove("on");
    }
    replay.setFocus({});
  }

  // 8 · outro
  lower("", "");
  replay.showRobot(true); replay.showTrails(false);
  app.loadAttempt(lastIdx, { play: true }); replay.speed = 1;
  replay.flyTo("home", 2.4); replay.orbit(true, 0.5);
  await sleep(600);
  phase("outro"); scrim.classList.add("on"); $("demoTitle").classList.add("on");
  await sleep(5000);
  if (new URLSearchParams(location.search).has("loop")) { $("demoTitle").classList.remove("on"); await sleep(800); run(app); }

  function showGoals(a, ok) {
    const el = $("demoGoals"); el.hidden = false;
    el.innerHTML = a.goals.map((g) => `<div class="g ${g.ok ? "ok" : "bad"}"><i>${g.ok ? "✓" : "✕"}</i><span>${esc(g.pred)}(${g.args.map(esc).join(", ")})</span><small>${esc(g.why && g.why !== "ok" ? g.why.split(",")[0] : "")}</small></div>`).join("");
    [...el.children].forEach((c, i) => setTimeout(() => c.classList.add("on"), 120 + i * 260));
  }
  function hideGoals() { const el = $("demoGoals"); [...el.children].forEach((c) => c.classList.remove("on")); setTimeout(() => { el.hidden = true; }, 400); }
}
