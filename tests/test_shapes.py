"""Lying cylinders ("axis") and compound shapes: scene helpers, MuJoCo build, settle, export, validation."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from rehearsal.export3d import export_scene3d
from rehearsal.pipeline import validate_scene
from rehearsal.scene import build_model, footprint_polygon, footprint_radius, object_height, shape_bbox
from rehearsal.sim import Sim

ROOT = Path(__file__).resolve().parents[1]

DRILL = {"type": "compound", "parts": [
    {"type": "box", "size": [0.18, 0.05, 0.07], "pos": [0.0, 0.0, 0.05], "rgba": [0.9, 0.4, 0.1, 1]},
    {"type": "cylinder", "radius": 0.02, "height": 0.04, "axis": "x", "pos": [0.11, 0.0, 0.05]},
    {"type": "box", "size": [0.05, 0.04, 0.09], "pos": [-0.04, 0.0, -0.03]},
    {"type": "box", "size": [0.08, 0.06, 0.03], "pos": [-0.04, 0.0, -0.085], "rgba": [0.1, 0.1, 0.1, 1]}]}


def _scene():
    sc = json.loads((ROOT / "scenes" / "desk_01.json").read_text())
    sc["objects"] = sc["objects"][:1]
    base = copy.deepcopy(sc["objects"][0])
    marker = {**copy.deepcopy(base), "id": "marker_1", "label": "marker", "pos": [0.55, 0.3], "yaw": 0.6,
              "shape": {"type": "cylinder", "radius": 0.007, "height": 0.12, "axis": "x"},
              "mass_kg": {"nominal": 0.02, "range": [0.01, 0.03]}}
    drill = {**copy.deepcopy(base), "id": "drill_1", "label": "drill", "pos": [0.5, -0.2], "yaw": -0.4,
             "shape": DRILL, "mass_kg": {"nominal": 1.0, "range": [0.8, 1.2]}}
    sc["objects"] += [marker, drill]
    return sc


def test_bbox_helpers():
    assert shape_bbox({"type": "cylinder", "radius": 0.01, "height": 0.1, "axis": "x"}) == [0.1, 0.02, 0.02]
    assert shape_bbox({"type": "cylinder", "radius": 0.01, "height": 0.1, "axis": "y"}) == [0.02, 0.1, 0.02]
    assert shape_bbox({"type": "cylinder", "radius": 0.01, "height": 0.1}) == [0.02, 0.02, 0.1]
    bb = shape_bbox(DRILL)
    assert np.allclose(bb, [0.22, 0.06, 0.185], atol=1e-9)
    assert shape_bbox({**DRILL, "bbox": [0.2, 0.06, 0.2]}) == [0.2, 0.06, 0.2]
    o = {"shape": DRILL, "pos": [0, 0], "yaw": 0.0}
    assert abs(object_height(o) - 0.185) < 1e-9 and abs(footprint_radius(o) - 0.03) < 1e-9
    assert footprint_polygon(o).shape == (4, 2)
    assert footprint_polygon({"shape": {"type": "cylinder", "radius": 0.03, "height": 0.1}, "pos": [1, 1]}).shape == (16, 2)


def test_build_settle_export(tmp_path):
    sc = _scene()
    m = build_model(sc)
    sim = Sim(m, sc)
    for oid, h in (("marker_1", 0.014), ("drill_1", 0.185)):
        st = sim.obj_state(oid)
        assert abs(st["pos"][2] - h / 2) < 0.01, (oid, st["pos"])
        assert st["up_cos"] > 0.99, oid  # lying marker and standing drill both rest in their own orientation
        assert np.linalg.norm(st["pos"][:2] - np.array(sim.objs[oid]["pos"])) < 0.01
    # every compound part is a collidable geom on the drill body, with its own colour where given
    b = m.body("drill_1").id
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    assert len(gs) == 4 and m.geom_contype[gs[0]] != 0
    assert np.allclose(m.geom_rgba[gs[0]][:3], [0.9, 0.4, 0.1])
    # the lying cylinder's axis is +x in the body frame
    g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("marker_1").id][0]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
    assert abs(R.reshape(3, 3)[0, 2]) > 0.99
    export_scene3d(m, tmp_path)
    desc = json.loads((tmp_path / "scene3d.json").read_text())
    assert sum(1 for g in desc["geoms"] if g["body"] == b) == 4


def test_validation():
    sc = _scene()
    assert validate_scene(sc) == []
    bad = copy.deepcopy(sc)
    bad["objects"][2]["shape"]["parts"][0]["type"] = "bowl"
    assert any("parts[0]" in e for e in validate_scene(bad))
    bad = copy.deepcopy(sc)
    bad["objects"][1]["shape"]["axis"] = "w"
    assert any("axis" in e for e in validate_scene(bad))
    bad = copy.deepcopy(sc)
    bad["objects"][2]["shape"]["parts"] = []
    assert validate_scene(bad)
