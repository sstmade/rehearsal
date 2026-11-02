"""Scene spec (JSON, the output of perception) -> MuJoCo model with a Franka Panda.

Frame: table top is z=0, robot base at the origin facing +x. Units are meters / kg.
Optional `scene["robot"]` = {"base_x", "base_y", "base_z", "base_yaw", "gripper"} moves the Panda's link0 (default: all 0,
on the table plane at the origin) and can swap the tool for an approximate Robotiq 2F-85 ("gripper": "robotiq_2f85",
used to replay real DROID episodes, see rehearsal/replay.py).  Scenes without "robot" build exactly as before.
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


# ---------------------------------------------------------------- supports (what holds the robot and the table up)
FLOOR_Z = -0.75                      # floor plane height in the table frame
TABLE_THICKNESS = 0.04
# Footprint of the Panda's link0 (base) around the origin, from the menagerie mesh, plus a small margin.
ROBOT_BASE_FOOTPRINT = (-0.17, 0.09, -0.11, 0.11)   # xmin, xmax, ymin, ymax
STAND_RGBA = [0.80, 0.82, 0.84, 1.0]  # neutral light grey, next to the arm's off-white
LEG_RGBA = [0.42, 0.34, 0.26, 1.0]


def robot_pose(scene: dict) -> tuple[float, float, float, float]:
    """(base_x, base_y, base_z, base_yaw) of link0 in the table frame; all 0 unless scene["robot"] says otherwise."""
    r = scene.get("robot") or {}
    return tuple(float(r.get(k, 0.0)) for k in ("base_x", "base_y", "base_z", "base_yaw"))


def _base_footprint(robot=None) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = ROBOT_BASE_FOOTPRINT
    if robot is None:
        return x0, x1, y0, y1
    bx, by, _, yaw = robot
    if abs(yaw) > 1e-9:   # axis-aligned box around the rotated footprint
        c, s = np.cos(yaw), np.sin(yaw)
        P = np.array([[x, y] for x in (x0, x1) for y in (y0, y1)]) @ np.array([[c, s], [-s, c]])
        x0, y0, x1, y1 = P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
    return x0 + bx, x1 + bx, y0 + by, y1 + by


def table_supports_base(table: dict, robot=None) -> bool:
    """True when the table top lies under the whole robot base footprint (the base stands on the table).
    robot: optional robot_pose(scene); a base raised above the table (base_z > 0) always needs its own stand."""
    (cx, cy), (sx, sy) = table["center"], table["size"]
    if robot is not None and robot[2] > 1e-6:
        return False
    x0, x1, y0, y1 = _base_footprint(robot)
    return cx - sx / 2 <= x0 and cx + sx / 2 >= x1 and cy - sy / 2 <= y0 and cy + sy / 2 >= y1


def support_boxes(table: dict, robot=None) -> list[dict]:
    """Static structure under the robot and the table, in the table frame: [{"name", "center", "size", "rgba",
    "collide"}].  When the table does not carry the robot base (small photo tables: the base is placed off the near
    edge), a stand from the floor to the base: a column under link0 with a wider foot, clipped so it never pokes into
    the table.  Always: four visual-only table legs (no collision, so they cannot change any physics outcome).
    Mirrored by web/replay.js supportBoxes() for the editable twin view."""
    out = []
    (cx, cy), (sx, sy) = table["center"], table["size"]
    tz = -TABLE_THICKNESS
    if not table_supports_base(table, robot):
        x0, x1, y0, y1 = _base_footprint(robot)
        tx0, ty0, ty1 = cx - sx / 2, cy - sy / 2, cy + sy / 2
        if ty0 < y1 and ty1 > y0 and tx0 > x0:     # table beside the base: stop the stand at the table edge
            x1 = min(x1, tx0 - 0.005)
        h = (0.0 if robot is None else robot[2]) - FLOOR_Z
        out.append({"name": "robot_stand", "center": [(x0 + x1) / 2, (y0 + y1) / 2, FLOOR_Z + h / 2],
                    "size": [x1 - x0, y1 - y0, h], "rgba": STAND_RGBA, "collide": True})
        fx1 = min(x1 + 0.05, tx0 - 0.005) if tx0 > x0 else x1 + 0.05
        out.append({"name": "robot_stand_foot", "center": [(x0 - 0.05 + fx1) / 2, (y0 + y1) / 2, FLOOR_Z + 0.01],
                    "size": [fx1 - x0 + 0.05, y1 - y0 + 0.10, 0.02], "rgba": STAND_RGBA, "collide": True})
    leg, inset = 0.045, 0.05
    for kx in (-1, 1):
        for ky in (-1, 1):
            lx = cx + kx * (sx / 2 - inset - leg / 2)
            ly = cy + ky * (sy / 2 - inset - leg / 2)
            out.append({"name": "table_leg", "center": [lx, ly, (FLOOR_Z + tz) / 2],
                        "size": [leg, leg, tz - FLOOR_Z], "rgba": LEG_RGBA, "collide": False})
    return out


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


# Approximate Robotiq 2F-85 on the DROID Franka: the DROID EE point is the flange (= the Panda hand origin, checked by
# forward kinematics against the recorded cartesian_position to < 0.1 mm); the 2F-85 fingertips end ~0.18 m below it
# (0.17-0.19 m from projecting candidate tip points into the grasp frames) and its fingers close along the flange x axis
# (two separated fingers visible from both side cameras when the EE yaw is 0).  We keep the Panda finger meshes and
# move the hand 7 cm further out on a coupling cylinder, turn it so the fingers slide along flange x, and let each
# finger open to 42.5 mm (85 mm stroke).
ROBOTIQ_EXTENSION = 0.07
ROBOTIQ_HALF_STROKE = 0.0425
PANDA_FINGER_GAIN = 0.01568627451 / 100          # actuator8: tendon length (per-finger opening, m) per ctrl unit


def _robotiq_2f85(spec):
    hand = spec.body("hand")
    hand.pos = [0.0, 0.0, 0.107 + ROBOTIQ_EXTENSION]
    hand.quat = [float(np.cos(-np.pi / 4)), 0.0, 0.0, float(np.sin(-np.pi / 4))]
    spec.body("link7").add_geom(type=G.mjGEOM_CYLINDER, size=[0.037, ROBOTIQ_EXTENSION / 2, 0],
                                pos=[0, 0, 0.107 + ROBOTIQ_EXTENSION / 2], rgba=[0.12, 0.12, 0.13, 1], mass=0.3)
    for j in ("finger_joint1", "finger_joint2"):
        spec.joint(j).range = [0.0, ROBOTIQ_HALF_STROKE]
    spec.actuator("actuator8").ctrlrange = [0.0, ROBOTIQ_HALF_STROKE / PANDA_FINGER_GAIN]


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
    robot = robot_pose(scene) if scene.get("robot") else None
    if robot is not None:
        bx, by, bz, yaw = robot
        spec.body("link0").pos = [bx, by, bz]
        spec.body("link0").quat = _yaw_quat(yaw)
        if (scene.get("robot") or {}).get("gripper") == "robotiq_2f85":
            _robotiq_2f85(spec)
    wb = spec.worldbody
    wb.add_light(pos=[0.4, 0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    wb.add_geom(type=G.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75], rgba=[0.25, 0.25, 0.28, 1])
    tx, ty = scene["table"]["size"]
    table = wb.add_body(name="table", pos=[scene["table"]["center"][0], scene["table"]["center"][1], -0.02])
    table.add_geom(type=G.mjGEOM_BOX, size=[tx / 2, ty / 2, 0.02], rgba=[0.62, 0.5, 0.38, 1],
                   friction=[0.8, 0.005, 0.0001])
    spec.add_exclude(bodyname1="link0", bodyname2="table")
    # Stand under the robot base (only when the table does not carry it) and table legs, all static geoms of the
    # world body: no new bodies, so recorded trajectories (per-body poses) stay valid.
    for sb in support_boxes(scene["table"], robot):
        kw = {} if sb["collide"] else {"contype": 0, "conaffinity": 0}
        wb.add_geom(type=G.mjGEOM_BOX, size=[v / 2 for v in sb["size"]], pos=sb["center"], rgba=sb["rgba"],
                    friction=[0.8, 0.005, 0.0001], **kw)

    for obj in scene["objects"]:
        _add_object(wb, obj, friction_scale, mass_scale)
        # One suction weld per object, switched on at grasp time.
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"suction_{obj['id']}",
                          name1="hand", name2=obj["id"], objtype=mujoco.mjtObj.mjOBJ_BODY,
                          active=False)
    return spec.compile()
