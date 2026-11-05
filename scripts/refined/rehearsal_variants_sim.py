"""IDEA "many times, in simulation": the candidates the 02 v2 near-placement search really simulated
(_runs/logs/02_laptop_v2_a_plan.log), re-simulated one by one with the real executor (deterministic replay of each
candidate, real physics, real verdicts).  Writes _runs/cache/rv_NN_{run.json,traj.npz,scene.json} + an index.
"""
import json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))
import run_user_desk as U  # noqa: E402
REF = ROOT.parent / "footage/tables_sim/hero/refined"
U.CACHE, U.INTENT, U.SCENE = REF / "_runs/cache", REF / "_runs/intent", ROOT / "scenes/t6_02.json"
o = lambda xy, h: {"obj": "orange_1", "dest": "milk_carton_1", "relation": f"near@{xy[0]},{xy[1]}", "transit_height": h, "via": []}
s = lambda xy, h: {"obj": "spoon_1", "dest": "milk_carton_1", "relation": f"near@{xy[0]},{xy[1]}", "transit_height": h, "via": []}
OK_O = (0.5402, 0.096)
C = [[o((0.5102, -0.034), .15)], [o((0.5402, 0.096), .15)], [o((0.5402, 0.006), .15)], [o((0.5202, -0.074), .15)],
     [o((0.5602, 0.056), .15)], [o((0.5502, -0.034), .15)], [o((0.5802, 0.096), .15)], [o((0.5602, -0.074), .15)],
     [o((0.5102, -0.034), .22)], [o((0.5102, -0.034), .30)], [o(OK_O, .22)],
     [o(OK_O, .22), s((0.5637, -0.0106), .15)], [o(OK_O, .22), s((0.6237, 0.1094), .15)]]
idx = []
for n, cand in enumerate(C, 1):
    key = f"rv_{n:02d}"
    shutil.copy(U.INTENT / "02_laptop_v2_a.json", U.INTENT / f"{key}.json")
    r = U.plan_and_run(key, force=True, replay=cand)
    idx.append({"n": n, "key": key, "candidate": cand, "success": r["success"], "execution_error": r.get("execution_error"),
                "goals_failed": [f"{g['pred']}({','.join(g['args'])}): {g['why']}" for g in r["goals"] if not g["ok"]]})
(REF / "_runs/cache/rv_index.json").write_text(json.dumps(idx, indent=1))
for e in idx:
    print(e["n"], e["success"], (e["execution_error"] or "")[:80], e["goals_failed"][:2])
