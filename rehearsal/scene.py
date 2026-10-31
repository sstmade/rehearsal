"""Scene spec (JSON, the output of perception) -> MuJoCo model with a Franka Panda.

Frame: table top is z=0, robot base at the origin facing +x. Units are meters / kg.
Every object body's origin sits at the center of its bounding box.
Shapes: box | cylinder (optionally lying, "axis": "x"|"y") | bowl | compound (list of box/cylinder parts in the
object frame, each with pos/quat/rgba; see shape_bbox).
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


AXIS_QUAT = {"z": [1.0, 0.0, 0.0, 0.0],
             "x": [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0],    # cylinder axis along +x
             "y": [float(np.cos(np.pi / 4)), -float(np.sin(np.pi / 4)), 0.0, 0.0]}   # cylinder axis along +y


def shape_bbox(s: dict) -> list[float]:
    """Axis-aligned extent [sx, sy, sz] of a shape in the object frame (before yaw).
    box: size | cylinder/bowl: 2r x 2r x h, or lying on its side with "axis": "x"|"y" (h along that axis, 2r tall)
    | compound: explicit "bbox" or the union of its parts."""
    t = s["type"]
    if t == "box":
        return [float(v) for v in s["size"]]
    if t == "compound":
        if s.get("bbox"):
            return [float(v) for v in s["bbox"]]
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for p in s["parts"]:
            half = np.asarray(shape_bbox(p)) / 2
            c = np.asarray(p.get("pos", [0, 0, 0]), float)
            lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
        return (hi - lo).tolist()
    r, h = float(s["radius"]), float(s["height"])
    axis = s.get("axis", "z")
    return [h, 2 * r, 2 * r] if axis == "x" else [2 * r, h, 2 * r] if axis == "y" else [2 * r, 2 * r, h]


def object_height(obj: dict) -> float:
    return shape_bbox(obj["shape"])[2]


def contact_extent(s: dict) -> tuple[list[float], list[float]]:
    """([cx, cy], [sx, sy]) of the part of the shape that touches the table, in the object frame: the whole
    footprint for primitives; for a compound only the parts whose bottom reaches the bounding box's bottom
    (a drill stands on its battery, not on its full length)."""
    bb = shape_bbox(s)
    if s["type"] != "compound":
        return [0.0, 0.0], bb[:2]
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for p in s["parts"]:
        pb = shape_bbox(p)
        c = np.asarray(p.get("pos", [0, 0, 0]), float)
        if c[2] - pb[2] / 2 <= -bb[2] / 2 + 0.005:
            lo, hi = np.minimum(lo, c[:2] - np.asarray(pb[:2]) / 2), np.maximum(hi, c[:2] + np.asarray(pb[:2]) / 2)
    if not np.all(np.isfinite(lo)):
        return [0.0, 0.0], bb[:2]
    return ((lo + hi) / 2).tolist(), (hi - lo).tolist()


def footprint_radius(obj: dict) -> float:
    """Radius of the object's horizontal footprint (used by `on`/`inside`)."""
    _, ext = contact_extent(obj["shape"])
    return 0.5 * min(ext[0], ext[1])


def footprint_polygon(obj: dict, n: int = 16) -> np.ndarray:
    """Horizontal outline of the object in the table frame (rotated rectangle for boxes, lying cylinders and
    compounds; a circle for upright cylinders/bowls)."""
    s, pos, yaw = obj["shape"], obj["pos"], obj.get("yaw", 0.0)
    if s["type"] in ("cylinder", "bowl") and s.get("axis", "z") == "z":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)] * s["radius"] + pos
    ctr, (L, W) = contact_extent(s)
    c, si = np.cos(yaw), np.sin(yaw)
    loc = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]]) + ctr
    return loc @ np.array([[c, si], [-si, c]]) + pos


def shape_bands(s: dict) -> list[tuple[dict, list[float]]]:
    """Split a banded box / cylinder into (primitive, local pos) pieces, bottom (or -axis end) to top.
    bands: [{"frac": share of the length, "rgba": [...]}, ...]; fractions are normalised."""
    fr = np.array([max(1e-6, float(b.get("frac", 1.0))) for b in s["bands"]])
    fr = fr / fr.sum()
    axis = s.get("axis", "z") if s["type"] == "cylinder" else "z"
    k = {"x": 0, "y": 1, "z": 2}[axis]
    length = shape_bbox(s)[k]
    out, start = [], -length / 2
    for b, f in zip(s["bands"], fr):
        seg = length * f
        pos = [0.0, 0.0, 0.0]
        pos[k] = start + seg / 2
        if s["type"] == "box":
            size = list(s["size"]); size[2] = seg
            part = {"type": "box", "size": size}
        else:
            part = {"type": "cylinder", "radius": s["radius"], "height": seg, "axis": axis}
        if b.get("rgba"):
            part["rgba"] = b["rgba"]
        part["_frac"] = float(f)
        out.append((part, pos))
        start += seg
    return out


def _part_volume(p: dict) -> float:
    if p["type"] == "box":
        return float(np.prod(p["size"]))
    return float(np.pi * p["radius"] ** 2 * p["height"])


def _add_primitive(body, s: dict, mass: float, rgba, fr, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    """One collidable geom (box / cylinder, any axis) at a local pose inside the body."""
    if s["type"] == "box":
        body.add_geom(type=G.mjGEOM_BOX, size=[x / 2 for x in s["size"]], mass=mass, rgba=rgba, friction=fr,
                      pos=list(pos), quat=list(quat))
    elif s["type"] == "cylinder":
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.asarray(quat, float), np.asarray(AXIS_QUAT[s.get("axis", "z")], float))
        if s.get("axis", "z") != "z":  # lying on its side: rolling resistance so it does not roll off by itself
            fr = [fr[0], fr[1], max(fr[2], 0.002)]
        body.add_geom(type=G.mjGEOM_CYLINDER, size=[s["radius"], s["height"] / 2, 0], mass=mass,
                      rgba=rgba, friction=fr, pos=list(pos), quat=q.tolist())
    else:
        raise ValueError(f"unsupported compound part {s['type']}")


def _add_object(wb, obj: dict, friction_scale: float, mass_scale: float):
    s = obj["shape"]
    mass = obj["mass_kg"]["nominal"] * mass_scale
    fr = [obj.get("friction", {}).get("nominal", 0.8) * friction_scale, 0.005, 0.0001]
    rgba = obj.get("rgba", [0.7, 0.7, 0.7, 1])
    h = object_height(obj)
    body = wb.add_body(name=obj["id"], pos=[obj["pos"][0], obj["pos"][1], h / 2 + 0.001],
                       quat=_yaw_quat(obj.get("yaw", 0.0)))
    body.add_freejoint()
    if s["type"] in ("box", "cylinder") and s.get("bands"):
        # Colour bands from the photo: the same solid, split along its height (or a lying cylinder's axis)
        # into stacked geoms with their own rgba.  Visual only; collision is identical to the single primitive.
        for part, pos in shape_bands(s):
            _add_primitive(body, part, mass * part.pop("_frac"), part.get("rgba", rgba), fr, pos)
    elif s["type"] in ("box", "cylinder"):
        _add_primitive(body, s, mass, rgba, fr)
    elif s["type"] == "compound":
        # Several collidable primitives in the object frame (origin = centre of the bounding box); each part may
        # carry its own rgba.  Mass is split by volume.
        vols = [_part_volume(p) for p in s["parts"]]
        tot = sum(vols) or 1.0
        for p, v in zip(s["parts"], vols):
            _add_primitive(body, p, mass * v / tot, p.get("rgba", rgba), fr, p.get("pos", (0, 0, 0)),
                           p.get("quat", (1, 0, 0, 0)))
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
