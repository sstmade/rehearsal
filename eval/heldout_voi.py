"""Held-out check of the VoI result: same final plan, seeds 1000..1000+N (not used for selection).

Original by the research analyst (docs/claims.md §4), copied into eval/ for reproducibility.
Adapted: report path is an argument. Superseded by eval/active_perception_baseline.py (Experiment C).
    .venv/bin/python eval/heldout_voi.py out/desk_01/report.json [N=200] [out.json]
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from rehearsal import rehearse as R

rep = json.load(open(sys.argv[1]))
scene, goals = rep["scene_spec"], rep["goals"]
plan = rep["attempts"][-1]["plan"]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ids = [o["id"] for o in scene["objects"]]

def run(sigmas, pool):
    jobs = [(scene, plan, goals, 1000 + s, sigmas, False) for s in range(N)]
    return [r["success"] for r in pool.map(R._variant, jobs)]

def main():
    t = time.time()
    out = {}
    with ProcessPoolExecutor() as pool:
        base = run([R.SIGMA] * len(ids), pool)
        out["baseline"] = base
        for i, oid in enumerate(ids):
            s = [R.SIGMA] * len(ids); s[i] = R.SIGMA_REFINED
            out[oid] = run(s, pool)
        for sig in [0.0, 0.004, 0.008, 0.016, 0.02, 0.03]:
            out[f"all_sigma_{sig}"] = run([sig] * len(ids), pool)
    b = np.array(base)
    for k, v in out.items():
        v = np.array(v)
        disc_up = int((~b & v).sum()); disc_down = int((b & ~v).sum())
        print(f"{k:18s} {v.sum():4d}/{N}  vs baseline: +{disc_up} -{disc_down}")
    print("wall", round(time.time() - t, 1))
    json.dump({k: list(map(bool, v)) for k, v in out.items()}, open(sys.argv[3] if len(sys.argv) > 3 else "/dev/null", "w"))


if __name__ == "__main__":
    main()
