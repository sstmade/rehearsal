"""Re-run the 06 geometric search OFFLINE with observers, to log every candidate future it rehearsed.

Read-only w.r.t. shipped results: reads footage/tables_sim/hero/exec/_cache/06_scene.json (the exact scene
the shipped run used) and out/tables_v6/06.json (compiled goals). Writes only footage/vfx/_data/06_futures.*.
No model calls. Same search() call and budgets as scripts/exec_tables.py plan_and_run.

Logged:
  kin[]   every kinematic dry run: prefix length, action (obj, height, via), ik_ok, first hit / near, the TCP
          waypoints and joint configs it visited (IK per ~2 cm), so ghosts can be drawn exactly.
  sims[]  every physics rehearsal: actions, success, error, failed goals, disturbed objects, evidence and
          the full body trajectory (Sim.start_recording('full'), 50 Hz) + arm qpos samples.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rehearsal import search_planner as sp, executor
from rehearsal.predicates import with_implicit_goals

NN = sys.argv[1] if len(sys.argv) > 1 else "06"
CACHE = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec" / "_cache"
OUT = ROOT.parent / "footage" / "vfx" / "_data"
OUT.mkdir(parents=True, exist_ok=True)

sc = json.loads((CACHE / f"{NN}_scene.json").read_text())
comp = json.loads((ROOT / "out" / "tables_v6" / f"{NN}.json").read_text())
goals = with_implicit_goals(comp["intent"]["goals"], [o["id"] for o in sc["objects"]])
assigns = sp.assignments_from_goals(goals)

KIN, SIMS = [], []
_cur = {"wp": None, "q": None}

orig_init = sp.Kinematics.__init__
def init(self, scene):
    orig_init(self, scene)
    ik0 = self.sim.ik
    def ik(wp, q0, *a, **k):
        q = ik0(wp, q0, *a, **k)
        if _cur["wp"] is not None:
            _cur["wp"].append([round(float(x), 4) for x in wp]); _cur["q"].append([round(float(x), 4) for x in q[:7]])
        return q
    self.sim.ik = ik
sp.Kinematics.__init__ = init

orig_check = sp.Kinematics.check
def check(self, act, poses, start_tcp, start_q):
    _cur["wp"], _cur["q"] = [], []
    try:
        r = orig_check(self, act, poses, start_tcp, start_q)
    finally:
        wp, q = _cur["wp"], _cur["q"]; _cur["wp"] = _cur["q"] = None
    KIN.append({"i": len(KIN), "act": {k: act[k] for k in ("obj", "dest", "relation", "transit_height", "via")},
                "start_tcp": [round(float(x), 4) for x in start_tcp],
                "ik_ok": r["ik_ok"], "ik_phase": r.get("ik_phase"),
                "hits": r["hits"][:8], "n_hits": len(r["hits"]), "near": r["near"][:4], "n_near": len(r["near"]),
                "wp": wp, "q": q})
    return r
sp.Kinematics.check = check

orig_run = sp.run_plan
def run_plan(scene, plan, gl, **kw):
    t = time.time()
    r = orig_run(scene, plan, gl, record="full", **kw)
    SIMS.append({"i": len(SIMS), "actions": plan["actions"], "n_goals": len(gl), "success": bool(r["success"]),
                 "execution_error": r["execution_error"] and {k: r["execution_error"].get(k) for k in ("action_index", "error")},
                 "failed": [f"{g['pred']}({', '.join(g['args'])}): {g['why']}" for g in r["goals"] if not g["ok"]],
                 "disturbed": r["disturbed"], "evidence": r["evidence"], "events": r["events"],
                 "arm_contacts": r["arm_contacts"], "sim_time": r["sim_time"], "wall": round(time.time() - t, 2),
                 "traj": r["trajectory"]})
    print(f"sim {len(SIMS)}: n_act={len(plan['actions'])} success={r['success']} "
          f"err={(r['execution_error'] or {}).get('error')} disturbed={[d.get('obj') for d in r['disturbed']]}", flush=True)
    return r
sp.run_plan = run_plan

t0 = time.time()
res = sp.search(sc, goals, assigns, sim_budget=300, time_budget_s=900.0, log=lambda m: print(m, flush=True))
print("RESULT", res["success"], res["stats"])
shipped = json.loads((ROOT.parent / f"footage/tables_sim/hero/exec/{NN}.json").read_text())
out = {"scene": NN, "stats": res["stats"], "plan": res["plan"], "success": res["success"],
       "shipped_stats": shipped and shipped["search"]["stats"], "shipped_plan": shipped and shipped["plan"],
       "body_names": None, "kin": KIN, "sims": SIMS}
# body names for the recorded 'full' trajectory (bodies 1..nbody-1)
from rehearsal.scene import build_model
m = build_model(sc)
out["body_names"] = [m.body(b).name for b in range(1, m.nbody)]
(OUT / f"{NN}_futures.json").write_text(json.dumps(out, default=float))
print("wrote", OUT / f"{NN}_futures.json", "kin", len(KIN), "sims", len(SIMS), round(time.time() - t0, 1), "s")
