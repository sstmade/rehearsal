"""The robot is always visibly supported: a stand from the floor whenever the table does not carry its base."""
import json
from pathlib import Path

import mujoco

from rehearsal.scene import FLOOR_Z, build_model, support_boxes, table_supports_base

SCENES = Path(__file__).resolve().parent.parent / "scenes"


def test_stand_only_when_base_is_off_the_table():
    for p in SCENES.glob("*.json"):
        t = json.loads(p.read_text())["table"]
        stands = [b for b in support_boxes(t) if b["name"].startswith("robot_stand")]
        assert bool(stands) == (not table_supports_base(t)), p.name
        for b in stands:   # reaches the floor and the base (z=0), never pokes into the table
            assert abs(b["center"][2] - b["size"][2] / 2 - FLOOR_Z) < 1e-9
            (cx, cy), (sx, sy) = t["center"], t["size"]
            assert b["center"][0] + b["size"][0] / 2 <= cx - sx / 2 + 1e-9
        main = [b for b in stands if b["name"] == "robot_stand"]
        if main:
            assert abs(main[0]["center"][2] + main[0]["size"][2] / 2) < 1e-9


def test_supports_add_no_bodies_and_legs_do_not_collide():
    sc = json.loads((SCENES / "aigen_desk_walnut.json").read_text())
    m = build_model(sc)
    world_boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    assert len(world_boxes) == 6  # stand, foot, four legs
    assert sum(1 for g in world_boxes if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0) == 4
