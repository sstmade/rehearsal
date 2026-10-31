"""Scene spec (JSON, the output of perception) -> MuJoCo model with a Franka Panda.

Frame: table top is z=0, robot base at the origin facing +x. Units are meters / kg.
Every object body's origin sits at the center of its bounding box.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

G = mujoco.mjtGeom


def load_scene(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def object_height(obj: dict) -> float:
    s = obj["shape"]
    return s["size"][2] if s["type"] == "box" else s["height"]


def footprint_radius(obj: dict) -> float:
    """Radius of the object's horizontal footprint (used by `on`/`inside`)."""
    s = obj["shape"]
    if s["type"] == "box":
        return 0.5 * min(s["size"][0], s["size"][1])
    return s["radius"]


def _add_object(wb, obj: dict, friction_scale: float, mass_scale: float):
    s = obj["shape"]
    mass = obj["mass_kg"]["nominal"] * mass_scale
    fr = [obj.get("friction", {}).get("nominal", 0.8) * friction_scale, 0.005, 0.0001]
    rgba = obj.get("rgba", [0.7, 0.7, 0.7, 1])
    h = object_height(obj)
    body = wb.add_body(name=obj["id"], pos=[obj["pos"][0], obj["pos"][1], h / 2 + 0.001],
                       quat=_yaw_quat(obj.get("yaw", 0.0)))
    body.add_freejoint()
    if s["type"] == "box":
        body.add_geom(type=G.mjGEOM_BOX, size=[x / 2 for x in s["size"]], mass=mass,
                      rgba=rgba, friction=fr)
    elif s["type"] == "cylinder":
        body.add_geom(type=G.mjGEOM_CYLINDER, size=[s["radius"], h / 2, 0], mass=mass,
                      rgba=rgba, friction=fr)
    elif s["type"] == "bowl":
        # Container approximated as a thin base plus a ring of wall boxes (convex pieces).
        r, wall, n = s["radius"], 0.006, 16
        body.add_geom(type=G.mjGEOM_CYLINDER, size=[r, 0.004, 0], pos=[0, 0, -h / 2 + 0.004],
                      mass=mass * 0.4, rgba=rgba, friction=fr)
        seg = 2 * np.pi * r / n * 0.55
        for k in range(n):
            a = 2 * np.pi * k / n
            body.add_geom(type=G.mjGEOM_BOX, size=[wall, seg, h / 2],
                          pos=[r * np.cos(a), r * np.sin(a), 0], quat=_yaw_quat(a),
                          mass=mass * 0.6 / n, rgba=rgba, friction=fr)
    else:
        raise ValueError(f"unknown shape {s['type']}")


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def build_model(scene: dict, friction_scale: float = 1.0, mass_scale: float = 1.0,
                pos_noise: np.ndarray | None = None) -> mujoco.MjModel:
    scene = copy.deepcopy(scene)
    if pos_noise is not None:
        for obj, dxy in zip(scene["objects"], pos_noise):
            obj["pos"][0] += float(dxy[0])
            obj["pos"][1] += float(dxy[1])

    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):  # the menagerie keyframe only covers the arm's qpos
        spec.delete(key)
    spec.option.timestep = 0.002
    wb = spec.worldbody
    wb.add_light(pos=[0.4, 0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    wb.add_geom(type=G.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75], rgba=[0.25, 0.25, 0.28, 1])
    tx, ty = scene["table"]["size"]
    table = wb.add_body(name="table", pos=[scene["table"]["center"][0], scene["table"]["center"][1], -0.02])
    table.add_geom(type=G.mjGEOM_BOX, size=[tx / 2, ty / 2, 0.02], rgba=[0.62, 0.5, 0.38, 1],
                   friction=[0.8, 0.005, 0.0001])
    spec.add_exclude(bodyname1="link0", bodyname2="table")

    for obj in scene["objects"]:
        _add_object(wb, obj, friction_scale, mass_scale)
        # One suction weld per object, switched on at grasp time.
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"suction_{obj['id']}",
                          name1="hand", name2=obj["id"], objtype=mujoco.mjtObj.mjOBJ_BODY,
                          active=False)
    return spec.compile()
