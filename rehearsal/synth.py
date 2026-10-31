"""Synthetic ground truth for perception: render a scene with a phone-like camera and an A4 sheet.

The scene is built by rehearsal.scene.build_model (same geometry the simulator uses). The robot is hidden
(geom group filtering), an A4 sheet is drawn as an extra visual geom at a known pose, and the exact camera
(K, R, t) is read back from MuJoCo so the A4 corners and object centers project to pixels exactly.

    python -m rehearsal.synth render            # writes photos/synth/*.png + *.gt.json (free, no API)
    python -m rehearsal.synth eval              # runs perceive() on every render (Meta API, data="public")
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .scene import build_model, load_scene, object_height

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "photos" / "synth"
W, H = 1280, 960
FOVY = 55.0  # degrees; a phone main camera in 4:3 portrait-free landscape is ~50-55 deg vertical
A4 = (0.297, 0.210)

# Phone-like poses: (name, lookat xyz, distance m, azimuth deg, elevation deg).  Azimuth 0 looks along +x
# (standing behind/beside the robot); 90 looks along +y (standing at the robot's right side); 180 looks
# along -x (standing across the table, facing the robot).
POSES = [
    ("rear50", [0.62, 0.0, 0.0], 1.05, 0, -50),     # behind the robot, 0.80 m above table, 50 deg down
    ("side40", [0.62, 0.0, 0.0], 1.15, 90, -40),    # robot's right side, 0.74 m, 40 deg down
    ("front55", [0.62, 0.0, 0.0], 1.00, 180, -55),  # across the table, 0.82 m, 55 deg down
    ("oblq45", [0.62, 0.0, 0.0], 1.10, 45, -45),    # rear-right corner, 0.78 m, 45 deg down
]


def paper_corners_world(x0: float, y0: float, size=A4) -> np.ndarray:
    """Corner convention (robot frame, robot at origin facing +x, robot's left = +y):
    near-left, near-right, far-right, far-left.  (x0, y0) is the near-left corner; the long side runs along +x."""
    L, Wd = size
    return np.array([[x0, y0], [x0, y0 - Wd], [x0 + L, y0 - Wd], [x0 + L, y0]], float)


def random_layout(seed: int, paper_xy, n: int = 6) -> dict:
    """Random non-overlapping objects on the desk_01 table, avoiding the paper. Ground truth for eval."""
    rng = np.random.default_rng(seed)
    kinds = [
        ("mug", "ceramic mug", {"type": "cylinder", "radius": 0.04, "height": 0.10}, [0.9, 0.9, 0.95, 1]),
        ("bottle", "water bottle", {"type": "cylinder", "radius": 0.035, "height": 0.24}, [0.25, 0.6, 0.9, 1]),
        ("bowl", "bowl", {"type": "bowl", "radius": 0.075, "height": 0.06}, [0.9, 0.5, 0.25, 1]),
        ("book", "book", {"type": "box", "size": [0.21, 0.15, 0.03]}, [0.7, 0.12, 0.12, 1]),
        ("box", "cardboard box", {"type": "box", "size": [0.12, 0.10, 0.08]}, [0.72, 0.55, 0.35, 1]),
        ("can", "soda can", {"type": "cylinder", "radius": 0.033, "height": 0.12}, [0.85, 0.1, 0.1, 1]),
        ("cup", "paper cup", {"type": "cylinder", "radius": 0.04, "height": 0.09}, [0.95, 0.95, 0.9, 1]),
        ("block", "wooden block", {"type": "box", "size": [0.06, 0.06, 0.06]}, [0.85, 0.7, 0.3, 1]),
    ]
    pc = paper_corners_world(*paper_xy)
    px0, py0 = pc.min(0) - 0.03
    px1, py1 = pc.max(0) + 0.03
    objs, placed, count = [], [], {}
    for k in rng.permutation(len(kinds))[:n]:
        cat, label, shape, rgba = kinds[k]
        r = shape.get("radius") or 0.5 * float(np.hypot(*shape["size"][:2]))
        for _ in range(500):
            x, y = rng.uniform(0.30 + r, 1.0 - r), rng.uniform(-0.45 + r, 0.45 - r)
            if px0 - r < x < px1 + r and py0 - r < y < py1 + r:
                continue
            if all(np.hypot(x - a, y - b) > r + rb + 0.03 for a, b, rb in placed):
                break
        else:
            continue
        placed.append((x, y, r))
        count[cat] = count.get(cat, 0) + 1
        objs.append({"id": f"{cat}_{count[cat]}", "label": label, "category": cat, "shape": shape,
                     "pos": [round(float(x), 4), round(float(y), 4)],
                     "yaw": round(float(rng.uniform(-0.6, 0.6)), 3) if shape["type"] == "box" else 0.0,
                     "mass_kg": {"nominal": 0.3, "range": [0.2, 0.4]},
                     "friction": {"nominal": 0.6, "range": [0.4, 0.8]}, "fragile": False, "rgba": rgba})
    return {"name": f"rand_{seed}", "source": "synthetic random layout (rehearsal.synth)",
            "table": {"center": [0.45, 0.0], "size": [1.3, 1.2]}, "objects": objs}


# --------------------------------------------------------------------------------------------- render
def _camera_matrices(glcam, w: int, h: int):
    """K, R (world->camera, OpenCV convention: x right, y down, z forward), t, from MuJoCo's GL camera."""
    fwd = np.array(glcam.forward, float)
    up = np.array(glcam.up, float)
    right = np.cross(fwd, up)
    R = np.stack([right, -up, fwd])
    pos = np.array(glcam.pos, float)  # (the caller passes the average of MuJoCo's two stereo eyes)
    t = -R @ pos
    f = h * glcam.frustum_near / (glcam.frustum_top - glcam.frustum_bottom)
    cy = h / 2 + f * (glcam.frustum_top + glcam.frustum_bottom) / 2 / glcam.frustum_near
    cx = w / 2 - f * glcam.frustum_center / glcam.frustum_near
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return K, R, t


def project(K, R, t, pts3: np.ndarray) -> np.ndarray:
    pc = (R @ np.asarray(pts3, float).T).T + t
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def _add_box(scn, pos, half, rgba, yaw=0.0):
    if scn.ngeom >= scn.maxgeom:
        return
    c, s = np.cos(yaw), np.sin(yaw)
    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float).flatten()
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, float),
                        np.asarray(pos, float), mat, np.asarray(rgba, np.float32))
    scn.ngeom += 1


def render(scene: dict, pose, paper_xy, w: int = W, h: int = H, seed: int = 0):
    """Returns (rgb uint8 HxWx3, ground-truth dict)."""
    name, lookat, dist, az, el = pose
    m = build_model(scene)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 640), max(h, 480)
    m.vis.global_.fovy = FOVY
    m.vis.quality.shadowsize = 8192
    m.vis.quality.offsamples = 8
    m.vis.headlight.ambient[:] = [0.25, 0.25, 0.25]
    m.vis.headlight.diffuse[:] = [0.35, 0.35, 0.35]
    m.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
    m.vis.map.znear = 0.01
    # Make the scene's one light a soft, slightly off-axis "window" light with shadows.
    for i in range(m.nlight):
        m.light_pos[i] = [0.2, -0.6, 2.2]
        d = np.array([0.35, 0.45, -2.2]); m.light_dir[i] = d / np.linalg.norm(d)
        m.light_castshadow[i] = 1
        m.light_diffuse[i] = [0.75, 0.73, 0.70]
        m.light_specular[i] = [0.15, 0.15, 0.15]
    for g in range(m.ngeom):  # soften plastic look
        m.geom_rgba[g][:3] = np.clip(m.geom_rgba[g][:3], 0, 1)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # world/table/objects only; Panda visual meshes are group 2

    with mujoco.Renderer(m, h, w) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        scn = r.scene
        pc = paper_corners_world(*paper_xy)
        ctr = pc.mean(0)
        _add_box(scn, [ctr[0], ctr[1], 0.0006], [A4[0] / 2, A4[1] / 2, 0.0005], [0.97, 0.97, 0.95, 1])
        img = r.render().copy()
        K, R, t = _camera_matrices(mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1]), w, h)

    # mild sensor realism: noise + slight vignetting (deterministic per image)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - 0.18 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    img = np.clip(img * vig[..., None] + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)

    corners_px = project(K, R, t, np.c_[pc, np.zeros(4)])
    gt = {"pose": {"name": name, "lookat": lookat, "distance": dist, "azimuth": az, "elevation": el,
                   "fovy": FOVY, "cam_height_m": float(-(R.T @ t)[2])},
          "camera": {"K": K.tolist(), "R": R.tolist(), "t": t.tolist()},
          "paper_xy": list(paper_xy), "paper_corners_px": corners_px.tolist(),
          "objects": [{"id": o["id"], "pos": o["pos"], "category": o.get("category"),
                       "center_px": project(K, R, t, [[*o["pos"], object_height(o) / 2]])[0].tolist()}
                      for o in scene["objects"]],
          "scene": scene, "width": w, "height": h}
    return img, gt


SCENES = [  # (scene, paper near-left corner xy)
    ("desk_01", (0.78, 0.0)),
    ("rand_1", (0.15, 0.105)),
    ("rand_2", (0.55, -0.15)),
]


def get_scene(name: str, paper_xy) -> dict:
    if name.startswith("rand_"):
        return random_layout(int(name.split("_")[1]), paper_xy)
    return load_scene(ROOT / "scenes" / f"{name}.json")


def render_all(out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (sname, pxy) in enumerate(SCENES):
        sc = get_scene(sname, pxy)
        for j, pose in enumerate(POSES):
            img, gt = render(sc, pose, pxy, seed=100 * k + j)
            p = out / f"{sname}__{pose[0]}.jpg"
            Image.fromarray(img).save(p, quality=92)
            p.with_suffix(".gt.json").write_text(json.dumps(gt, indent=1))
            paths.append(p)
            print("rendered", p.name, f"cam height {gt['pose']['cam_height_m']:.2f} m")
    return paths


# ----------------------------------------------------------------------------------------------- eval
def _match(est: list[dict], gt: dict, max_d: float = 0.15):
    """Match estimated objects to ground truth by nearest GT *pixel center* to the SAM box center."""
    pairs = []
    for i, e in enumerate(est):
        b = e["perception"]["box_px"]
        bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        for j, g in enumerate(gt["objects"]):
            inside = b[0] <= g["center_px"][0] <= b[2] and b[1] <= g["center_px"][1] <= b[3]
            pairs.append((np.linalg.norm(bc - g["center_px"]) - (1e4 if inside else 0), i, j))
    used_e, used_g, out = set(), set(), []
    for c, i, j in sorted(pairs):
        if i in used_e or j in used_g:
            continue
        used_e.add(i); used_g.add(j)
        out.append((est[i], gt["objects"][j]))
    return out, [g for j, g in enumerate(gt["objects"]) if j not in used_g], [e for i, e in enumerate(est) if i not in used_e]


def evaluate(paths: list[Path], methods=("bottom_center", "bottom_ray", "ray_mid", "box_fit_fixed", "box_fit"), use_paper_gt=True,
             log=print) -> dict:
    from .perception import perceive_raw, estimate_scene
    rows = []
    for p in paths:
        gt = json.loads(p.with_suffix(".gt.json").read_text())
        corners = np.array(gt["paper_corners_px"])
        raw = perceive_raw(p, data="public")
        for meth in methods:
            sc = estimate_scene(raw, corners, tuple(gt["paper_xy"]), method=meth, clamp=False)
            matched, missed, extra = _match(sc["objects"], gt)
            for e, g in matched:
                err = float(np.hypot(e["pos"][0] - g["pos"][0], e["pos"][1] - g["pos"][1]))
                rows.append({"image": p.name, "pose": gt["pose"]["name"], "method": meth, "gt_id": g["id"],
                             "est_id": e["id"], "source": e["perception"]["source"], "err_cm": err * 100})
            if meth == methods[0]:
                log(f"{p.name}: {len(raw['objects'])} listed, {len(matched)} matched, missed "
                    f"{[g['id'] for g in missed]}, extra {[e['id'] for e in extra]}")
    return {"rows": rows}


def summarize(rows: list[dict], key=lambda r: r["method"]) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r["err_cm"])
    return {k: {"n": len(v), "median_cm": round(float(np.median(v)), 2),
                "p90_cm": round(float(np.percentile(v, 90)), 2), "max_cm": round(float(np.max(v)), 2)}
            for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "eval"])
    ap.add_argument("--only", default="", help="substring filter on image names")
    a = ap.parse_args()
    if a.cmd == "render":
        render_all()
        return
    paths = sorted(p for p in OUT.glob("*.jpg") if a.only in p.name and "overlay" not in p.name)
    res = evaluate(paths)
    rows = res["rows"]
    (OUT / "eval.json").write_text(json.dumps({
        "rows": rows,
        "by_method": summarize(rows),
        "by_method_pose": summarize(rows, lambda r: f"{r['method']}|{r['pose']}"),
        "by_method_source": summarize(rows, lambda r: f"{r['method']}|{r['source']}"),
    }, indent=1))
    for k, v in summarize(rows).items():
        print(k, v)
    for k, v in summarize(rows, lambda r: f"{r['method']}|{r['pose']}").items():
        print(k, v)
    for k, v in summarize(rows, lambda r: f"{r['method']}|{r['source']}").items():
        print(k, v)


if __name__ == "__main__":
    main()
