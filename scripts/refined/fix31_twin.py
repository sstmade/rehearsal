"""t6_31 twin, corrected against its photo (writes footage/tables_sim/hero/refined/31_dinner/twin/t6_31_fixed.json;
scenes/t6_31.json is NOT modified).

What the shipped twin got wrong (checked by projecting every object into the photo camera):
  * the stack of three plates (photo: top-left) became plate_2 + plate_3 overlapping side by side, and plate_1 a
    loose plate on empty table 20 cm away;
  * the bread (photo: on the cutting board) sits next to the board; the board itself is ~19 cm off;
  * the butter (photo: in its dish) and the spoon (photo: in the jam jar) are separate loose objects;
  * the three loose plates are ~10 cm off.
Fix: positions from the photo camera (ray through the object's centre pixel, intersected at its height); the stack,
board+bread and dish+butter become single bodies (compound / taller cylinder: the scene format has no
object-on-object start poses); the spoon is drawn in the jar (visual only).  Overlapping plate rims in the photo
(plates resting on each other's edges) are pushed apart just enough to rest flat on the table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rehearsal.calib import Camera  # noqa: E402

OUT = ROOT.parent / "footage/tables_sim/hero/refined/31_dinner/twin"
sc = json.loads((ROOT / "scenes/t6_31.json").read_text())
cam = Camera(**{k: sc["photo"]["camera"][k] for k in ("K", "R", "t")})
O = {o["id"]: o for o in sc["objects"]}
px = lambda u, v, z: [round(float(a), 4) for a in cam.intersect_z(u, v, z)]
HALF_PI = 1.5708

new = []
# plates the robot will move: photo centres (rim ellipse centres), pushed apart to rest flat
P = {"plate_4": px(482, 662, 0.011), "plate_5": px(620, 560, 0.010), "plate_6": px(665, 735, 0.010)}
R = {k: O[k]["shape"]["radius"] for k in P}
xy = {k: np.array(v, float) for k, v in P.items()}
for _ in range(400):
    for a in P:
        for b in P:
            if a < b:
                d = xy[b] - xy[a]
                dist = np.linalg.norm(d)
                need = R[a] + R[b] + 0.006
                if dist < need:
                    m = d / dist * (need - dist) / 2
                    xy[a] -= m
                    xy[b] += m
    # the cutting board (fixed, axis-aligned in world: long side along y) is an obstacle too
    bc = np.array(px(800, 450, 0.01))
    for a in P:
        q = np.clip(xy[a], bc - [0.115 + 0.006, 0.185 + 0.006], bc + [0.115 + 0.006, 0.185 + 0.006])
        d = xy[a] - q
        dist = np.linalg.norm(d)
        if dist < R[a]:
            xy[a] += d / max(dist, 1e-9) * (R[a] - dist)
for k in P:
    o = dict(O[k], pos=[round(float(v), 4) for v in xy[k]])
    o["refined_fix"] = {"photo_xy": P[k], "pushed_cm": round(float(np.linalg.norm(xy[k] - P[k])) * 100, 1)}
    new.append(o)
# the stack of three plates: one body, three plates tall
p1 = O["plate_1"]
stack = dict(p1, id="plate_stack_1", label="stack of three plates", category="plate stack",
             shape={"type": "cylinder", "radius": 0.128, "height": 0.066}, pos=px(437, 425, 0.03), yaw=0.0,
             mass_kg={"nominal": 1.8, "range": [1.4, 2.2]}, rgba=O["plate_2"]["rgba"])
stack["refined_fix"] = "was plate_1 + plate_2 + plate_3 (spread out); one stack as in the photo"
new.append(stack)
# cutting board with the loaf and two slices on it
cb, br = O["cutting_board_1"], O["bread_1"]
bx, by, bh = 0.37, 0.23, 0.021
H = bh + 0.095
c = np.array(px(800, 450, 0.01))
loaf = np.array(px(828, 405, 0.05)) - c
sl = np.array(px(745, 425, 0.03)) - c
yaw = HALF_PI              # the board's long side runs left-right in the photo = world y
Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
loaf, sl = Rz @ loaf, Rz @ sl  # into the board frame
parts = [{"type": "box", "size": [bx, by, bh], "pos": [0, 0, -H / 2 + bh / 2], "rgba": cb["rgba"], "name": "board"},
         {"type": "cylinder", "radius": 0.072, "height": 0.09, "pos": [float(loaf[0]), float(loaf[1]), -H / 2 + bh + 0.045],
          "rgba": br["rgba"], "name": "loaf"},
         {"type": "box", "size": [0.075, 0.12, 0.035], "pos": [float(sl[0]), float(sl[1]), -H / 2 + bh + 0.0175],
          "rgba": O["bread_2"]["rgba"], "name": "slices"}]
board = dict(cb, id="cutting_board_1", label="cutting board with bread", pos=[round(float(v), 4) for v in c],
             yaw=yaw, shape={"type": "compound", "parts": parts, "bbox": [bx, by, H]},
             mass_kg={"nominal": 1.6, "range": [1.2, 2.0]})
board["refined_fix"] = "bread_1/2/3 put on the board (photo); board re-placed from the photo camera"
new.append(board)
jar = dict(O["jar_1"], pos=px(1015, 445, 0.06), label="jam jar with a spoon")
jar["refined_fix"] = "spoon_1 drawn in the jar (as in the photo); not a separate body"
new.append(jar)
base, bu = O["butter_dish_base_1"], O["butter_1"]
H2 = 0.027 + 0.04
dish = dict(base, label="butter dish with butter", pos=px(1125, 548, 0.02), yaw=HALF_PI,
            shape={"type": "compound", "bbox": [0.211, 0.112, H2], "parts": [
                {"type": "box", "size": [0.211, 0.112, 0.027], "pos": [0, 0, -H2 / 2 + 0.0135], "rgba": base["rgba"],
                 "name": "dish"},
                {"type": "box", "size": [0.12, 0.06, 0.04], "pos": [0, 0, -H2 / 2 + 0.027 + 0.02 - 0.012],
                 "rgba": bu["rgba"], "name": "butter"}]},
            mass_kg={"nominal": 0.6, "range": [0.45, 0.75]})
dish["refined_fix"] = "butter_1 merged into its dish (photo)"
new.append(dish)
lid = px(1210, 510, 0.03)
new.append(dict(O["butter_dish_lid_1"], pos=[round(lid[0] + 0.05, 4), lid[1]], yaw=HALF_PI,
                refined_fix="5 cm back from the photo ray estimate so it does not intersect the dish"))
new.append(dict(O["napkin_1"], pos=px(1110, 760, 0.01)))
new.append(dict(O["keys_1"], pos=px(1225, 690, 0.007)))
fixed = dict(sc, objects=new)
fixed["source"] += " | refined/31: twin corrected against the photo (scripts/refined/fix31_twin.py)."
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "t6_31_fixed.json").write_text(json.dumps(fixed, indent=1))
for o in new:
    print(o["id"], o["pos"], o.get("refined_fix", ""))
