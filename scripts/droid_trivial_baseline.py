"""Trivial baseline for the DROID replay check: no twin, no physics, no images.

Predict "success" iff, during some gripper-close segment, the gripper ended up holding something and the end
effector was then lifted. Two variants:

  finger   holding = the *measured* finger position plateaus below HOLD_MAX (closed on nothing reaches ~0.92-0.97)
  command  holding = the gripper was *commanded* closed (no finger feedback at all)

Lifted = EE z rose at least LIFT_MIN above its height at the close command, while the gripper was closed.
Reads label.json only to score. $0, ~1 s.

    .venv/bin/python scripts/droid_trivial_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DROID = ROOT / "datasets" / "droid"
EPISODES = ["duck_success", "duck_missed_grasp", "duck_fail_nudge", "bowl_fail", "cup_success", "cup_fail"]
HOLD_MAX = 0.90
LIFT_MIN = 0.05
SETTLE_STEPS = 8      # ~0.5 s at 15 Hz for the fingers to finish closing
# The replay's predictions (docs/real_robot.md, eval/results/droid_replay/summary.json).
REPLAY_PRED = {"duck_success": True, "duck_missed_grasp": False, "duck_fail_nudge": False, "bowl_fail": False,
               "cup_success": False, "cup_fail": False}


def close_segments(cmd: list[float]) -> list[tuple[int, int]]:
    segs, i = [], 0
    while i < len(cmd):
        if cmd[i] > 0.5:
            j = i
            while j < len(cmd) and cmd[j] > 0.5:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def features(traj: dict) -> list[dict]:
    g, z = traj["gripper_position"], [p[2] for p in traj["cartesian_position"]]
    out = []
    for a, b in close_segments(traj["gripper_command"]):
        settled = g[min(b - 1, a + SETTLE_STEPS):b]
        out.append({"steps": [a, b], "finger_plateau": round(max(settled), 3),
                    "lift_m": round(max(z[a:b]) - z[a], 3)})
    return out


def predict(segs: list[dict], variant: str) -> bool:
    return any(s["lift_m"] >= LIFT_MIN and (variant == "command" or s["finger_plateau"] < HOLD_MAX) for s in segs)


def main() -> dict:
    rows = []
    for e in EPISODES:
        traj = json.loads((DROID / e / "trajectory.json").read_text())
        segs = features(traj)
        rows.append({"episode": e, "segments": segs, "finger": predict(segs, "finger"),
                     "command": predict(segs, "command"), "replay": REPLAY_PRED[e]})
    for r in rows:   # label read only after every prediction is made
        r["real"] = bool(json.loads((DROID / r["episode"] / "label.json").read_text())["success"])
    res = {"hold_max": HOLD_MAX, "lift_min_m": LIFT_MIN, "episodes": rows,
           "agree": {k: sum(r[k] == r["real"] for r in rows) for k in ("finger", "command", "replay")},
           "n": len(rows)}
    # Threshold sensitivity of the finger variant: the range of thresholds over which its score is unchanged.
    plateaus = sorted({s["finger_plateau"] for r in rows for s in r["segments"]})
    res["finger_plateaus"] = plateaus
    for r in rows:
        print(f"{r['episode']:18s} real={r['real']!s:5s} finger={r['finger']!s:5s} command={r['command']!s:5s} "
              f"replay={r['replay']!s:5s} " + " ".join(f"[g{s['finger_plateau']} lift {s['lift_m']}]" for s in r["segments"]))
    print("agreement with the DROID label:", res["agree"], "of", res["n"])
    out = ROOT / "eval" / "results" / "droid_replay" / "trivial_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
