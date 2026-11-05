"""Extra *simulated candidate rollouts* for the mosaic tiles.

For every scene with a cached shipped run (footage/tables_sim/hero/exec/_cache/NN_run.json) re-run the MuJoCo
executor on a few candidate variants of the plan -- the same kinds of variation the search planner tries
(action order, transit height) and the stress tests apply (perception error, friction / mass scale) -- and
record qpos exactly like exec_tables.plan_and_run does.

READ-ONLY w.r.t. shipped results: reads _cache/*, writes only footage/mosaic/_work/rollouts/NN_cK.{npz,json}.

    rehearsal-mvp/.venv/bin/python scripts/mosaic/rollouts.py --jobs 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec" / "_cache"
OUT = ROOT.parent / "footage" / "mosaic" / "_work" / "rollouts"
REC_EVERY = 2
MAX_SIM_S = 40.0


def variants(nn, rec, sc, rng):
    acts = rec["plan"]["actions"]
    if not acts:
        from rehearsal.predicates import with_implicit_goals
        from rehearsal.search_planner import assignments_from_goals
        comp = json.loads((ROOT / "out" / "tables_v6" / f"{nn}.json").read_text())
        goals = with_implicit_goals(comp["intent"]["goals"], [o["id"] for o in sc["objects"]])
        acts = [dict(a, transit_height=0.3, via=[]) for a in assignments_from_goals(goals)]
    if not acts:
        return []
    n_obj = len(sc["objects"])
    out = []
    order = list(range(len(acts)))
    rev = order[::-1]
    shuf = list(order)
    rng.shuffle(shuf)

    def plan(idx, th=None, keep_via=False):
        return {"actions": [dict(acts[i], transit_height=th or acts[i].get("transit_height", 0.3),
                                 via=acts[i].get("via", []) if keep_via else []) for i in idx]}

    out.append(("order_reversed", plan(rev), {}))
    out.append(("low_transit_0.16", plan(order, 0.16, True), {}))
    out.append(("order_shuffled_high_0.42", plan(shuf, 0.42), {}))
    out.append(("perception_err_2cm", plan(order, None, True),
                {"perception_error": rng.normal(0, 0.02, (n_obj, 2))}))
    out.append(("friction_0.6_err_3cm", plan(order, 0.24, True),
                {"friction_scale": 0.6, "perception_error": rng.normal(0, 0.03, (n_obj, 2))}))
    out.append(("mass_1.5_order_rot", plan(order[1:] + order[:1], 0.34), {"mass_scale": 1.5}))
    return out


def run_one(job):
    nn, k, name, plan, kw = job
    dst = OUT / f"{nn}_c{k}.npz"
    if dst.exists():
        return nn, k, "cached"
    from rehearsal import executor
    from rehearsal.predicates import with_implicit_goals
    from rehearsal.sim import IKFailure, Sim

    sc = json.loads((CACHE / f"{nn}_scene.json").read_text())
    comp = json.loads((ROOT / "out" / "tables_v6" / f"{nn}.json").read_text())
    goals = with_implicit_goals(comp["intent"]["goals"], [o["id"] for o in sc["objects"]])

    class RecSim(Sim):
        last = None

        def __init__(self, *a, **k_):
            super().__init__(*a, **k_)
            self.qrec = [self.d.qpos.copy()]
            self.nstep = 0
            RecSim.last = self

        def _step(self):
            if self.nstep * self.m.opt.timestep > MAX_SIM_S:
                raise IKFailure("mosaic rollout time cap")
            try:
                super()._step()
            finally:
                self.nstep += 1
                if self.nstep % REC_EVERY == 0:
                    self.qrec.append(self.d.qpos.copy())

    orig = executor.Sim
    executor.Sim = RecSim
    t0 = time.time()
    meta = {"scene": nn, "candidate": k, "variant": name, "kind": "simulated candidate rollout (not shipped)",
            "plan": plan}
    try:
        r = executor.run_plan(sc, plan, goals, **kw)
        meta.update(success=bool(r["success"]), error=(r["execution_error"] or {}).get("error"))
    except Exception as e:  # a candidate that crashes is still a rehearsal; keep what was recorded
        meta.update(success=False, error=f"{type(e).__name__}: {e}")
    finally:
        executor.Sim = orig
    sim = RecSim.last
    if sim is None or len(sim.qrec) < 50:
        meta["skipped"] = True
        (OUT / f"{nn}_c{k}.json").write_text(json.dumps(meta, default=float))
        return nn, k, "too short"
    meta["dt"] = float(sim.m.opt.timestep) * REC_EVERY
    meta["nq_rigid"] = int(sim.m.nq)
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["frames"] = len(sim.qrec)
    np.savez_compressed(dst, q=np.asarray(sim.qrec, np.float32))
    (OUT / f"{nn}_c{k}.json").write_text(json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    return nn, k, f"ok {meta['wall_s']}s ok={meta['success']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in sorted(CACHE.glob("*_run.json")):
        rec = json.loads(f.read_text())
        nn = rec["scene"]
        if not rec.get("executed") or not (CACHE / f"{nn}_scene.json").exists():
            continue
        sc = json.loads((CACHE / f"{nn}_scene.json").read_text())
        rng = np.random.default_rng(int(nn) * 7 + 1)
        try:
            for k, (name, plan, kw) in enumerate(variants(nn, rec, sc, rng), 1):
                jobs.append((nn, k, name, plan, kw))
        except Exception:
            traceback.print_exc()
    print(len(jobs), "candidate jobs", flush=True)
    with Pool(a.jobs, maxtasksperchild=4) as p:
        for res in p.imap_unordered(run_one, jobs):
            print(*res, flush=True)


if __name__ == "__main__":
    main()
