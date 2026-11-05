"""rehearsal_variants/index.json: one row per rendered tile, from the real re-simulated verdicts."""
import json, re
from pathlib import Path
REF = Path(__file__).resolve().parents[3] / "footage/tables_sim/hero/refined"
I = json.loads((REF / "_runs/cache/rv_index.json").read_text())
V = REF / "rehearsal_variants"
logs = (V / "render_A.log").read_text() + (V / "render_B.log").read_text()
chip = dict(re.findall(r"tile (\d\d) (?:OK|FAIL) (.+)", logs))
rows = []
for e in I:
    n = f"{e['n']:02d}"
    if not (V / f"{n}.mp4").exists():
        continue
    a = e["candidate"]
    rows.append({"file": f"{n}.mp4", "attempt_id": e["key"], "outcome": "PASS" if e["success"] else "FAIL",
                 "chip": chip.get(n), "candidate": [f"{x['obj']} next to {x['dest']} at {x['relation'][5:]}, carry height "
                                                   f"{x['transit_height']} m" for x in a],
                 "execution_error": e["execution_error"], "checks_failed": e["goals_failed"]})
(V / "index.json").write_text(json.dumps({
    "source": "02_laptop_v2 near-placement search (refined/_runs/logs/02_laptop_v2_a_plan.log): the candidates it really "
              "simulated, re-simulated one by one with the real executor (deterministic), t6_02, same camera",
    "instruction": "The front of the table. Just put the orange and the spoon beside the milk carton.",
    "note": "Each candidate is checked against the FULL goal (orange AND spoon near the carton + nothing else disturbed); "
            "single-move candidates therefore also fail near(spoon,...) - the chip names the physical failure. "
            "Tile 13 is the plan that was sent (5/5 goals). Primitive Cycles, 1280x720, ~3.5 s each.",
    "tiles": rows}, indent=1))
print(len(rows), [r["outcome"] for r in rows])
