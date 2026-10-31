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
  const ag = app.agentInfo(r), perc = app.hasPerception(r.scene_spec);
  const who = ag.kind === "muse" ? `Muse Spark${ag.sub ? ` (${ag.sub})` : ""}` : ag.kind === "offline" ? "offline reference replay (no model call)" : ag.name;
  $("demoTitle").querySelector(".t3").textContent = `phone photo → MuJoCo twin → ${ag.kind === "muse" ? "Muse Spark plans" : "reference plan"} → physics judges → ${ag.kind === "muse" ? "model self-corrects" : "rule-based correction"} → ${r.stress_test ? r.stress_test.n : 40} universes`;
  const once = (cls, place) => { let el = ui.querySelector("." + cls); if (!el) { el = document.createElement("div"); el.className = cls; place(el); } el.className = cls; return el; };
  const vig = once("demo-vignette", (el) => ui.prepend(el));
  const scrim = once("demo-scrim", (el) => ui.insertBefore(el, $("demoTitle")));
  const phase = (name) => { app.phase = name; };
  const statEl = once("demo-stat", (el) => ui.appendChild(el));
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
  lower("01 · Photo → twin", perc ? `${r.scene_spec.objects.length} objects proposed by <b>Muse Spark</b>, segmented by <b>SAM 3.1</b>, placed in MuJoCo` : `${r.scene_spec.objects.length} objects from a <b>hand-authored scene</b> (no perception run), placed in MuJoCo`);
  await sleep(3600);

  // 2 · plan
  phase("plan"); replay.flyTo("top", 1.6); replay.showRobot(false);
  const f = {}; for (const x of r.attempts[0].plan.actions) { f[x.obj] = "accent"; if (!f[x.dest]) f[x.dest] = "good"; }
  replay.setFocus(f);
  const a0 = r.attempts[0].plan.actions[0];
  lower("02 · Instruction → plan", `“${esc(r.instruction)}” → ${r.goals.length} checkable goals · ${r.attempts[0].plan.actions.length} actions${a0 ? ` · carry <b>${esc(a0.obj)}</b> at ${a0.transit_height} m${a0.via && a0.via.length ? " via a detour" : ""}` : ""}`);
  await sleep(3600);
  replay.setFocus({}); replay.showRobot(true);

  // 3 · attempt 1 fails
  if (failIdx >= 0) {
    const a = r.attempts[failIdx];
    const hits = app.unexpectedContacts(a);
    const hit = hits.find((e) => e.kind === "protective_stop") || (a.execution_error ? null : hits[0]);
    app.loadAttempt(failIdx, { play: false });
    phase("attempt1"); replay.flyTo("cinema", 1.6);
    lower(`03 · Rehearse · attempt ${a.attempt}`, `physics runs the plan`);
    await sleep(1200);
    replay.speed = 1.3; replay.play();
    if (hit) {
      await waitT(hit.t - 1.0);
      const obj = app.otherOf(hit);
      const p = replay.bodyByName[obj] ? replay.bodyByName[obj].position.clone() : { x: 0.5, y: 0, z: 0 };
      replay.flyTo({ pos: [p.x + 0.62, p.y + 0.5, 0.46], target: [p.x, p.y, 0.14] }, 1.0);
      replay.speed = 0.5;
      await waitT(hit.t - 0.02);
      replay.speed = 0.18;
      if (hit.kind === "protective_stop") lower(`protective stop · t = ${hit.t.toFixed(3)} s`, `the robot's <b>${esc(hit.link_role || "arm")}</b> pressed on <b>${esc(hit.obj)}</b> with ${Math.round(hit.force_n || 0)} N${hit.phase ? ` during “${esc(hit.phase)}”` : ""}`, "bad");
      else lower(`collision · t = ${hit.t.toFixed(3)} s`, `<b>${esc(hit.between.join(" ↔ "))}</b>${hit.link_role ? ` (${esc(hit.link_role)})` : ""}${hit.while_holding ? ` while holding ${esc(hit.while_holding)}` : ""}${hit.tcp_z != null ? ` at z = ${hit.tcp_z} m` : ""}`, "bad");
      await sleep(2600);
      replay.speed = 1.6;
      await waitT(Math.min(replay.duration, hit.t + 3.0));
      replay.pause();
    } else if (a.execution_error) {
      replay.speed = 1.6; await waitEnd();
      lower("execution error", `<b>${esc(a.execution_error.error)}</b>`, "bad");
      await sleep(2600);
    } else { await waitEnd(); }
    await sleep(300);
    showGoals(a, false); phase("goals1");
    lower(`attempt ${a.attempt} · failed`, `${a.goals.filter((g) => !g.ok).length} of ${a.goals.length} goals failed${a.execution_error ? " · the plan was aborted" : ""} — the simulator says why`, "bad");
    await sleep(3400);
    hideGoals();
  }

  // 4 · diagnosis
  if (fixIdx >= 0 && (r.attempts[fixIdx].plan.diagnosis || r.attempts[fixIdx].plan.diagnosis_en)) {
    const a = r.attempts[fixIdx], prev = r.attempts[failIdx];
    const { primary, original } = app.diagText(a);
    const changes = app.planChanges(a, prev).map((c) => `<span>${c}</span>`);
    const card = $("demoCard"); card.hidden = false;
    card.innerHTML = `<div class="who">${esc(who)} · reads the sim log</div><div class="q"><span class="cursor"></span></div>${original ? `<div class="q" style="font-size:.78em;color:#aab1c2;min-height:0;margin-top:8px" id="demoGloss" lang="zh"></div>` : ""}<div class="fix" ${changes.length ? "" : "hidden"}>${changes.join(" · ") || ""}</div>`;
    lower(ag.kind === "muse" ? "Self-correction" : "Reference correction", ag.kind === "muse" ? `the model is handed contacts with timestamps, IK errors and failed predicates` : `rule-based diagnosis from the same sim log: contacts with timestamps, IK errors, failed predicates`, "spark");
    replay.flyTo("cinema", 2.0);
    await sleep(100); card.classList.add("on"); phase("diag");
    const q = card.querySelector(".q"), text = primary;
    for (let i = 1; i <= text.length; i++) { q.innerHTML = `${esc(text.slice(0, i))}<span class="cursor"></span>`; await sleep(text.charCodeAt(i - 1) > 255 ? 42 : 16); }
    q.textContent = text;
    const g = $("demoGloss"); if (g) { g.textContent = original; }
    await sleep(900);
    if (changes.length) card.querySelector(".fix").hidden = false; phase("fix");
    await sleep(2800);
    card.classList.remove("on"); await sleep(600); card.hidden = true;
  }

  // 5 · attempt 2 succeeds
  if (fixIdx >= 0) {
    const a = r.attempts[fixIdx], a1 = a.plan.actions[0];
    app.loadAttempt(fixIdx, { play: false });
    phase("attempt2"); replay.flyTo("cinema", 1.2);
    const what = a1 ? (a1.via && a1.via.length ? `detour for <b>${esc(a1.obj)}</b> at ${a1.transit_height} m` : `carry <b>${esc(a1.obj)}</b> at <b>${a1.transit_height} m</b>`) : "revised plan";
    lower(`attempt ${a.attempt}`, `same instruction · ${what}`, "ok");
    await sleep(900);
    replay.speed = 2.2; replay.play();
    await waitEnd();
    showGoals(a, true); phase("goals2");
    lower(`attempt ${a.attempt} · success`, `all ${a.goals.length} goals verified by physics`, "ok");
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
      lower("05 · Look again · 再看一眼", `which object is worth a second photo? <b>${esc(look.join(", "))}</b> · σ ${typeof r.active_perception.sigma_cm === "number" ? r.active_perception.sigma_cm : (r.active_perception.default_sigma_cm ?? 1.2)} → ${r.active_perception.refined_sigma_cm} cm`);
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
