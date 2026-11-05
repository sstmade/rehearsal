"""eval_v7 runner: re-rehearse the AI-generated tables with the extended search (scripts/eval_v7/ext.py), record
the effort of every search, and execute + record the found plan (or nothing) for the clips.

Physics decides, unchanged: rehearsal.executor.run_plan + rehearsal.predicates, goals + implicit undisturbed().
Writes ONLY under footage/tables_sim/eval_v7/ (runs/, _cache/, trace/).  tables_v6, scenes/ and hero/exec are read.

Run (rehearsal-mvp/):  .venv/bin/python scripts/eval_v7/run_v7.py [--jobs 5] [--budget 2400] [NN ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eval_v7"))
sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))

EV = ROOT.parent / "footage" / "tables_sim" / "eval_v7"
RUNS, CACHE, TRACE = EV / "runs", EV / "_cache", EV / "trace"
V6 = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec"
REC_EVERY = 2


def scene_file(nn):
    """A v7 twin fix (eval_v7/scenes/) wins over the shipped scene file; both are recorded in the run."""
    f = EV / "scenes" / f"t6_{nn}.json"
    return f if f.exists() else ROOT / "scenes" / f"t6_{nn}.json"


def intent_file(nn):
    f = EV / "intent" / f"{nn}.json"
    return f if f.exists() else ROOT / "out" / "tables_v6" / f"{nn}.json"


def install_all():
    """executor.transport -> near@ (run_user_desk demo extension) | in@/on@ (eval_v7 extension) | stock."""
    import ext
    import run_user_desk as U
    from rehearsal import executor
    ext.install()

    def tr(sim, obj, dest, relation, transit_height, via=None):
        if relation.startswith("near@"):
            return U.transport_ext(sim, obj, dest, relation, transit_height, via)
        return ext.transport_v7(sim, obj, dest, relation, transit_height, via)
    executor.transport = tr


def plan_one(nn, budget, sim_budget):
    import ext
    from rehearsal import executor
    from rehearsal.predicates import with_implicit_goals
    from rehearsal.search_planner import assignments_from_goals
    from rehearsal.sim import Sim
    install_all()
    for d in (RUNS, CACHE, TRACE):
        d.mkdir(parents=True, exist_ok=True)
    sf = scene_file(nn)
    sc = json.loads(sf.read_text())
    comp = json.loads(intent_file(nn).read_text())
    rec = {"scene": nn, "scene_file": str(sf.relative_to(ROOT.parent)), "instruction": comp.get("instruction"),
           "intent_status": (comp.get("intent") or {}).get("status")}
    if rec["intent_status"] != "goals" or not comp["intent"].get("goals"):
        rec.update(planned=False, executed=False, outcome="not_goals")
        return rec
    goals = with_implicit_goals(comp["intent"]["goals"], [o["id"] for o in sc["objects"]])
    rec["goals_compiled"] = comp["intent"]["goals"]
    trace = []
    t0 = time.time()
    has_near = any(g["pred"] == "near" for g in goals)
    if has_near and executor.run_plan(sc, {"actions": []}, goals)["success"]:
        # like the shipped search: if every goal already holds, the empty plan is the answer (no needless move)
        res = {"success": True, "plan": {"diagnosis": "every goal already holds", "actions": []},
               "reason": None, "stats": {"sims": 1, "found_at_sim": 1}}
        rec["planner"] = "empty plan (all goals already hold) + stock executor/predicates"
    elif has_near:
        import run_user_desk as U
        orig = executor.run_plan

        def traced(scene, plan, gl, *a, **k):
            r = orig(scene, plan, gl, *a, **k)
            acts = plan.get("actions") or []
            if acts:
                trace.append({"n": len(trace) + 1, "depth": len(acts), "full": gl is goals, "ok": bool(r["success"]),
                              "why": None if r["success"] else ext.classify(r),
                              "detail": None if r["success"] else ((r["execution_error"] or {}).get("error") or "; ".join(
                                  f"{g['pred']}({','.join(g['args'][:2])}): {g['why']}" for g in r["goals"] if not g["ok"])[:300]),
                              "last": {kk: acts[-1].get(kk) for kk in ("obj", "dest", "relation", "transit_height")},
                              "t": round(time.time() - t0, 1)})
            return r
        executor.run_plan = traced
        try:
            res = U.search_ext(sc, goals, time_budget_s=budget, sim_budget=sim_budget, log=lambda *_: None)
        finally:
            executor.run_plan = orig
        res["stats"]["found_at_sim"] = (next((t["n"] for t in reversed(trace) if t["ok"] and t["full"]), None)
                                        if res["success"] else None)
        rec["planner"] = ("near-placement DEMO EXTENSION (scripts/user_desk/run_user_desk.search_ext) + "
                          "stock executor/predicates")
    else:
        res = ext.search_v7(sc, goals, assignments_from_goals(goals), sim_budget=sim_budget, time_budget_s=budget,
                            trace=trace)
        rec["planner"] = "eval_v7 extended search (scripts/eval_v7/ext.py) + stock executor/predicates"
    rec["search"] = {"success": res["success"], "reason": res.get("reason"), "stats": res.get("stats"),
                     "wall_s": round(time.time() - t0, 1), "budget_s": budget, "sim_budget": sim_budget}
    (TRACE / f"{nn}.json").write_text(json.dumps(trace, default=float))
    plan = res["plan"]
    rec["planned"] = plan is not None
    if plan is None:
        rec.update(executed=False, success=False, outcome="no_safe_plan")
        return rec

    # execute the found plan once more, recorded (same physics; this run is what the clip shows)
    class RecSim(Sim):
        last = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.qrec = [self.d.qpos.copy()]
            self.nstep = 0
            RecSim.last = self

        def _step(self):
            try:
                super()._step()
            finally:
                self.nstep += 1
                if self.nstep % REC_EVERY == 0:
                    self.qrec.append(self.d.qpos.copy())
    orig = executor.Sim
    executor.Sim = RecSim
    t1 = time.time()
    try:
        r = executor.run_plan(sc, plan, goals)
    finally:
        executor.Sim = orig
    sim = RecSim.last
    rec["plan_exec"] = plan
    disp = []
    for a in plan["actions"]:       # display copy for captions: base relation, the extension params kept beside
        rel = a["relation"]
        if rel.startswith("near@"):
            b, ex = "next to", {"place_xy": [float(v) for v in rel[5:].split(",")[:2]]}
        else:
            b, dx, dy, dyaw, drop = ext.parse_rel(rel)
            ex = ({"offset": [dx, dy], "yaw_turn_rad": dyaw, "release_above_rim": drop,
                   "grip_offset": [float(v) for v in ext.parse_grip(rel)]} if "@" in rel else {})
        disp.append(dict(a, relation=b, **({"ext": ex} if ex else {})))
    rec["plan"] = {"diagnosis": plan.get("diagnosis", ""), "actions": disp}
    rec["sim_wall_s"] = round(time.time() - t1, 1)
    rec["sim_time_s"] = r["sim_time"]
    rec["executed"] = True
    rec["success"] = bool(r["success"])
    rec["execution_error"] = (r["execution_error"] or {}).get("error")
    rec["goals"] = [{"pred": g["pred"], "args": g["args"], "ok": g["ok"], "why": g["why"],
                     "implicit": bool(g.get("implicit"))} for g in r["goals"]]
    explicit = [g for g in rec["goals"] if not g["implicit"]]
    rec["goals_held"] = f"{sum(g['ok'] for g in explicit)}/{len(explicit)}"
    rec["undisturbed_ok"] = all(g["ok"] for g in rec["goals"] if g["implicit"])
    rec["disturbed"] = r["disturbed"]
    rec["events"] = [e for e in r["events"] if e["kind"] in ("grasp", "release", "protective_stop",
                                                            "ik_failure", "grasp_miss")]
    rec["dt"] = float(sim.m.opt.timestep) * REC_EVERY
    rec["nq_rigid"] = int(sim.m.nq)
    rec["rim_audit"] = ext.rim_audit(sc, r, goals) if r["success"] else ""
    rec["outcome"] = ("already_done" if not plan["actions"] and r["success"] else
                      "done" if r["success"] and not rec["rim_audit"] else "missed")
    (CACHE / f"{nn}_scene.json").write_text(json.dumps(sc))
    np.savez_compressed(CACHE / f"{nn}_traj.npz", q=np.asarray(sim.qrec, np.float32))
    (CACHE / f"{nn}_run.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def run_one(nn, budget, sim_budget):
    t0 = time.time()
    try:
        rec = plan_one(nn, budget, sim_budget)
    except Exception as e:
        rec = {"scene": nn, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
    rec["total_wall_s"] = round(time.time() - t0, 1)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{nn}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(nn, rec.get("outcome"), rec.get("goals_held"), (rec.get("search") or {}).get("reason"),
          rec.get("error", ""), rec["total_wall_s"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--budget", type=float, default=2400)
    ap.add_argument("--sims", type=int, default=1500)
    ap.add_argument("--one", action="store_true")
    a = ap.parse_args()
    if a.one:
        for nn in a.scenes:
            run_one(nn, a.budget, a.sims)
        return
    queue, procs = list(a.scenes), []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    (EV / "logs").mkdir(parents=True, exist_ok=True)
    while queue or procs:
        while queue and len(procs) < a.jobs:
            nn = queue.pop(0)
            cmd = [sys.executable, __file__, "--one", "--budget", str(a.budget), "--sims", str(a.sims), nn]
            procs.append(subprocess.Popen(cmd, env=env, stdout=open(EV / "logs" / f"{nn}.log", "w"),
                                          stderr=subprocess.STDOUT))
        time.sleep(3)
        procs = [p for p in procs if p.poll() is None]


if __name__ == "__main__":
    main()
