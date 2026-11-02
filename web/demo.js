// Demo mode (?demo=1): cinematic auto-play of the money shot for the 3-minute video.
// With an LLM-only baseline (out_llm_v3): a language model alone fails in physics → "the model states the goal,
// search finds the motion" → Rehearsal's run passes → stress test. Without one: the report's own tries
// (fail → model diagnosis → fix) or simply the first try. Then 40 universes fan out → look again.
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
  const who = ag.kind === "muse" ? "Meta Muse Spark" : ag.kind === "offline" ? "a rule-based stand-in (no model call)" : ag.name;
  const { the, list, nameOf, tryName, cap } = app;
  const base = state.baseline && state.baseline.frames ? state.baseline : null;
  $("demoTitle").querySelector(".t3").textContent = base ? `a task in plain words → the model says what it understood → you confirm → a language model alone fails in physics → the model states the goal, a search finds the motion → tried ${r.stress_test ? r.stress_test.n : 40} times` : `one photo → a physics copy of the table → ${ag.kind === "muse" ? "Meta Muse Spark plans" : "a stand-in plan"} → physics tries it → ${ag.kind === "muse" ? "the model learns from the failure" : "a rule-based fix"} → tried ${r.stress_test ? r.stress_test.n : 40} times`;
  const once = (cls, place) => { let el = ui.querySelector("." + cls); if (!el) { el = document.createElement("div"); el.className = cls; place(el); } el.className = cls; return el; };
  const vig = once("demo-vignette", (el) => ui.prepend(el));
  const scrim = once("demo-scrim", (el) => ui.insertBefore(el, $("demoTitle")));
  const phase = (name) => { app.phase = name; };
  const statEl = once("demo-stat", (el) => ui.appendChild(el));
  const lower = (chip, text, cls = "") => { $("demoChip").textContent = chip; $("demoChip").className = `demo-chip ${cls}`; $("demoText").innerHTML = text; $("demoLower").classList.toggle("on", !!(chip || text)); };
  const waitT = (t) => new Promise((res) => { const f = () => (replay.t >= t || !replay.playing ? res() : requestAnimationFrame(f)); f(); });
  const waitEnd = () => new Promise((res) => { const f = () => (!replay.playing ? res() : requestAnimationFrame(f)); f(); });
  const failIdx = r.attempts.findIndex((a) => !a.success);
  const lastFail = r.attempts.map((a, i) => (!a.success ? i : -1)).filter((i) => i >= 0).pop() ?? -1;
  const fixIdx = r.attempts.findIndex((a, i) => i > lastFail && a.success);
  const lastIdx = r.attempts.length - 1;

  // 0 · title over a slow orbit of the twin
  app.loadAttempt(lastIdx, { play: false });
  replay.flyTo("home", 0); replay.orbit(true, 0.5);
  phase("title"); scrim.classList.add("on"); $("demoTitle").classList.add("on");
  await sleep(4200);
  $("demoTitle").classList.remove("on"); scrim.classList.remove("on");
  await sleep(700);

  // 1 · the task: typed in plain words, over the physics copy of the table
  phase("twin"); replay.flyTo("twin", 1.8);
  const low = {};
  for (const o of r.scene_spec.objects) if (o.perception && o.perception.confidence === "low") low[o.id] = "warn";
  for (const [id, ok] of Object.entries(r.reachability)) if (!ok) low[id] = "bad";
  replay.setFocus(low);
  const tableLine = perc ? `${r.scene_spec.objects.length} objects, listed by <b>Meta Muse Spark</b> and found in the photo by <b>Meta SAM 3.1</b>` : `${r.scene_spec.objects.length} objects from a <b>hand-built table</b>`;
  lower("1 · The task", `<q class="demo-typed"></q><small class="demo-sub">${tableLine}, in a physics simulation</small>`);
  const typed = $("demoText").querySelector(".demo-typed");
  for (let i = 1; i <= r.instruction.length; i++) { typed.textContent = r.instruction.slice(0, i); await sleep(r.instruction.charCodeAt(i - 1) > 255 ? 70 : 34); }
  await sleep(1300);

  // 2 · what the model understood: the report's own intent record (Muse Spark's compile step + Rehearsal's goal check)
  const it = r.intent, pv = (it && it.gate && it.gate.preview) || null;
  if (it && it.status === "goals" && pv) {
    phase("intent"); replay.flyTo("top", 1.6); replay.showRobot(false);
    const f = {}; for (const m of pv.moves || []) { f[m.obj] = "accent"; if (!f[m.dest]) f[m.dest] = "good"; }
    replay.setFocus(f);
    const plain = (t) => cap(String(t || "").replace(/\s*\([A-Za-z0-9_]+\)/g, "").replace(/^must not move or tip: /i, ""));
    const stays = (pv.stays || []).map((id) => the(id));
    const asm = it.assumptions || [], SHOW = 3;
    const card = $("demoCard"); card.hidden = false; card.className = "demo-card intent";
    card.innerHTML = `<div class="who"><span class="dot"></span>Here’s what I understood</div>
      <q class="task">${esc(r.instruction)}</q>
      <div class="cols"><div><h5>Will move</h5><ul>${(pv.moves || []).map((m) => `<li>${esc(plain(m.text))}</li>`).join("")}</ul></div>
      <div><h5>Must not move or tip</h5><p>${esc(stays.length ? cap(list(stays)) : "Everything else")}</p></div></div>
      <div class="asm"><h5>Assuming ${asm.length}</h5><ul>${asm.slice(0, SHOW).map((x) => `<li>${esc(x)}</li>`).join("")}${asm.length > SHOW ? `<li class="more">+ ${asm.length - SHOW} more</li>` : ""}</ul></div>
      <div class="cta"><span class="btn-c">Confirm</span><span class="src">Read by Meta Muse Spark · passed Rehearsal’s goal check</span></div>`;
    lower("2 · What the model understood", `goals physics can check, and the assumptions behind them`, "spark");
    await sleep(100); card.classList.add("on");
    const parts = [...card.querySelectorAll(".cols > div, .asm li, .cta")];
    for (const el of parts) { el.classList.add("pre"); }
    for (const el of parts) { await sleep(el.matches(".asm li") ? 650 : 900); el.classList.remove("pre"); }
    await sleep(2200);
    const btn = card.querySelector(".btn-c"); btn.classList.add("press"); phase("confirm");
    await sleep(380); btn.classList.remove("press"); btn.classList.add("done"); btn.textContent = "Confirmed ✓";
    lower("2 · Confirmed", `now physics gets a say`, "ok");
    await sleep(1500);
    card.classList.remove("on"); await sleep(600); card.hidden = true; card.className = "demo-card";
    replay.setFocus({}); replay.showRobot(true);
  } else {
    // a report from before the intent step: the plan overview instead
    phase("plan"); replay.flyTo("top", 1.6); replay.showRobot(false);
    const f = {}; for (const x of r.attempts[0].plan.actions) { f[x.obj] = "accent"; if (!f[x.dest]) f[x.dest] = "good"; }
    replay.setFocus(f);
    lower("2 · The task", `“${esc(r.instruction)}” → ${r.goals.length} things to check · ${r.attempts[0].plan.actions.length} move${r.attempts[0].plan.actions.length === 1 ? "" : "s"}`);
    await sleep(3600);
    replay.setFocus({}); replay.showRobot(true);
  }

  // 3/4 · every failed attempt, briefly: the failure beat, then the diagnosis that fixed it
  const hasDiag = (x) => x && !/^geometric search/i.test(x.plan.diagnosis || "") && (x.plan.diagnosis || x.plan.diagnosis_en);
  const submitted = [...r.attempts].reverse().find(hasDiag);
  const failed = base ? [-1] : r.attempts.map((a, i) => (!a.success ? i : -1)).filter((i) => i >= 0);
  const CAP = 330;   // ~5 lines on the demo card
  for (const k of failed) {
    const a = k < 0 ? base.att : r.attempts[k];
    const label = k < 0 ? "A language model alone" : tryName(k);
    const hits = app.unexpectedContacts(a);
    const hit = hits.find((e) => e.kind === "protective_stop") || (a.execution_error ? null : hits[0]);
    app.loadAttempt(k, { play: false });
    phase(k < 0 ? "baseline" : `attempt${a.attempt}`); replay.flyTo("cinema", 1.6);
    lower(`3 · ${label}`, k < 0 ? `Muse Spark plans every arm motion itself (an earlier, LLM-only run of this task)` : k === failed[0] ? `a simulated arm runs the plan` : `the corrected plan, back in physics`, k < 0 ? "bad" : "");
    await sleep(1200);
    replay.speed = 1.3; replay.play();
    if (hit) {
      await waitT(hit.t - 1.0);
      const obj = app.otherOf(hit);
      const p = replay.bodyByName[obj] ? replay.bodyByName[obj].position.clone() : { x: 0.5, y: 0, z: 0 };
      replay.flyTo({ pos: [p.x + 0.62, p.y + 0.5, 0.46], target: [p.x, p.y, 0.14] }, 1.0);
      replay.speed = 0.5;
      await waitT(hit.t - 0.02);
      replay.speed = 0.18; phase("collision");
      if (hit.kind === "protective_stop") lower(`the arm stops itself · ${hit.t.toFixed(1)} s`, `its <b>${esc(hit.link_role || "hand")}</b> pressed on <b>${esc(the(hit.obj))}</b> with ${Math.round(hit.force_n || 0)} N`, "bad");
      else lower(`collision · ${hit.t.toFixed(1)} s`, `${hit.while_holding ? `while carrying <b>${esc(the(hit.while_holding))}</b>, the arm` : `the arm's <b>${esc(hit.link_role || "wrist")}</b>`} hits <b>${esc(the(obj))}</b>`, "bad");
      await sleep(2600);
      replay.speed = 1.6;
      await waitT(Math.min(replay.duration, hit.t + 3.0));
      replay.pause();
    } else if (a.execution_error) {
      replay.speed = 1.8; await waitEnd(); phase("collision");
      lower(`${label} · stopped`, `<b>${esc(app.failurePlain(a))}</b>`, "bad");
      await sleep(2600);
    } else { replay.speed = 1.8; await waitEnd(); }
    await sleep(300);
    showGoals(a, false); phase("goals1");
    lower(`${label} · failed`, `${a.goals.filter((g) => !g.ok).length} of ${a.goals.length} goals missed${a.execution_error ? " · the plan was cut short" : ""} — and the simulator can say why`, "bad");
    await sleep(3000);
    hideGoals();

    // the diagnosis that answers this failure: the next attempt that carries one (agent loops put it on the submitted attempt)
    if (k < 0) {   // the thesis beat, instead of a diagnosis
      const card = $("demoCard"); card.hidden = false;
      card.innerHTML = `<div class="who">Rehearsal</div><div class="q"><span class="cursor"></span></div>`;
      lower("Rehearsal", "same table, same words", "spark");
      replay.flyTo("cinema", 2.0);
      await sleep(100); card.classList.add("on"); phase("thesis");
      const q = card.querySelector(".q"), text = "The model states the goal. A search finds the motion. Physics checks it.";
      for (let i = 1; i <= text.length; i++) { q.innerHTML = `${esc(text.slice(0, i))}<span class="cursor"></span>`; await sleep(26); }
      q.textContent = text;
      await sleep(2200);
      card.classList.remove("on"); await sleep(600); card.hidden = true;
      continue;
    }
    const next = r.attempts[k + 1];
    const diagA = r.attempts.slice(k + 1).find(hasDiag) || submitted;
    if (diagA && next) {
      let { primary, original } = app.diagText(diagA);
      if (primary.length > CAP) { primary = primary.slice(0, CAP).replace(/\s+\S*$/, "") + " …"; original = null; }
      const changes = app.planChangesPlain(next, a).map((c) => `<span>${esc(c)}</span>`);
      const card = $("demoCard"); card.hidden = false;
      card.innerHTML = `<div class="who">What ${esc(who)} learned from the log</div><div class="q"><span class="cursor"></span></div>${original ? `<div class="q" style="font-size:.78em;color:#55504a;min-height:0;margin-top:8px" id="demoGloss" lang="zh"></div>` : ""}<div class="fix" ${changes.length ? "" : "hidden"}><b>The fix:</b> ${changes.join(" ") || ""}</div>`;
      lower(ag.kind === "muse" ? "The model reads what physics saw" : "A rule-based fix from what physics saw", `what touched what, when, and how hard; what the arm could not reach; which goals were missed`, "spark");
      replay.flyTo("cinema", 2.0);
      await sleep(100); card.classList.add("on"); phase("diag");
      const q = card.querySelector(".q"), text = primary;
      for (let i = 1; i <= text.length; i++) { q.innerHTML = `${esc(text.slice(0, i))}<span class="cursor"></span>`; await sleep(text.charCodeAt(i - 1) > 255 ? 42 : 14); }
      q.textContent = text;
      const g = $("demoGloss"); if (g) { g.textContent = original; }
      await sleep(900);
      if (changes.length) card.querySelector(".fix").hidden = false; phase("fix");
      await sleep(2600);
      card.classList.remove("on"); await sleep(600); card.hidden = true;
    }
  }

  // 5 · attempt 2 succeeds
  if (fixIdx >= 0) {
    const a = r.attempts[fixIdx], a1 = a.plan.actions[0];
    app.loadAttempt(fixIdx, { play: false });
    phase("attempt2"); replay.flyTo("cinema", 1.2);
    const what = a1 ? app.actionPlain(a1).replace(/\.$/, "") : "the revised plan";
    lower(base ? "Rehearsal" : `${tryName(fixIdx)}`, base ? `the motion search found · ${esc(what)}` : `same task · ${esc(what)}`, "ok");
    await sleep(900);
    replay.speed = 2.2; replay.play();
    await waitEnd();
    showGoals(a, true); phase("goals2");
    lower(`${base ? "Rehearsal" : tryName(fixIdx)} · worked`, `all ${a.goals.length} goals checked by physics`, "ok");
    await sleep(3200);
    hideGoals();
  }

  // 6 · universes fan out
  if (state.trailSets.length) {
    replay.flyTo("universe", 2.0);
    replay.showRobot(false);
    const s0 = state.trailSets[0];
    lower("4 · Tried again and again", `${s0.runs.length} runs with small random changes to weight, friction and where things are`);
    phase("universes"); await app.fanOut(0, 3200); phase("universes-stat");
    statEl.innerHTML = `<div class="n">${s0.successes}<span class="arr">/</span>${s0.runs.length}</div><div class="l">runs where the plan still worked</div>`;
    statEl.classList.add("on");
    await sleep(2600);
    statEl.classList.remove("on");

    // 7 · look again
    if (state.trailSets.length > 1 && r.active_perception) {
      const s1 = state.trailSets[1], look = r.active_perception.steps.map((st) => st.look_at);
      replay.setFocus(Object.fromEntries(look.map((k) => [k, "accent"])));
      lower("5 · A closer look at one object", `which object is worth a second photo? <b>${esc(list(look.map(the)))}</b>`);
      await sleep(1600);
      await app.fanOut(1, 2200);
      phase("look-stat"); statEl.innerHTML = `<div class="n">${s0.successes}<span class="arr">→</span><span class="ok">${s1.successes}</span><span class="arr">/</span>${s1.runs.length}</div><div class="l">after a closer look at ${esc(list(look.map(the)))}</div>`;
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
    el.innerHTML = a.goals.map((g) => `<div class="g ${g.ok ? "ok" : "bad"}"><i>${g.ok ? "✓" : "✕"}</i><span>${esc(app.goalPlain(g))}</span></div>`).join("");
    [...el.children].forEach((c, i) => setTimeout(() => c.classList.add("on"), 120 + i * 260));
  }
  function hideGoals() { const el = $("demoGoals"); [...el.children].forEach((c) => c.classList.remove("on")); setTimeout(() => { el.hidden = true; }, 400); }
}
