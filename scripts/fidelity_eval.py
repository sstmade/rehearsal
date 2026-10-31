"""How much does the physics twin look like the photo it was built from?  (the yardstick)

    .venv/bin/python scripts/fidelity_eval.py scenes/real_ycbv59.json [--out photos/real/real_ycbv59] [--tag before]

Renders the twin (table + objects, no robot) with MuJoCo from the photo's own camera (K, R, t from
scene["photo"]["camera"], exact intrinsics incl. the principal point), then per object:

  * silhouette IoU  - rendered visible silhouette (segmentation render) vs the SAM 3.1 mask of that object
  * dE_render       - CIE76 distance in Lab between the mean colour of the rendered object pixels and the mean
                      colour of the photo pixels inside its SAM mask (what you see vs what was photographed)
  * dE_rgba         - the same for the object's flat rgba vs the photo's median mask colour (material only)
  * pos_err_cm      - xy error vs the dataset ground truth, copied from scene.perception_notes (measure only)

and writes <out>.twin_render.png, <out>.twin_compare.png (SAM overlay | render) and <out>.twin_overlay.png
(50 % photo + 50 % render) from the SAME viewpoint, plus <out>.fidelity.json with the numbers.
SAM masks come from perception.perceive_raw, which is served from the on-disk Meta cache (free for a rerun).

The ground truth is never used to build the twin, only to report the position error here.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rehearsal.calib import Camera  # noqa: E402
from rehearsal.perception import draw_overlay, perceive_raw  # noqa: E402
from rehearsal.scene import G, _add_object  # noqa: E402


# ----------------------------------------------------------------------------------------------- colour
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] (..., 3) -> CIE Lab (D65)."""
    c = np.asarray(rgb, float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def delta_e(lab1, lab2) -> float:
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


# ----------------------------------------------------------------------------------------------- render
def objects_spec(scene: dict) -> mujoco.MjSpec:
    """Table + objects only (no robot), same geometry/colours as scene.build_model."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    wb = spec.worldbody
    wb.add_light(pos=[0.4, 0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    tx, ty = scene["table"]["size"]
    table = wb.add_body(name="table", pos=[scene["table"]["center"][0], scene["table"]["center"][1], -0.02])
    table.add_geom(type=G.mjGEOM_BOX, size=[tx / 2, ty / 2, 0.02], rgba=[0.62, 0.5, 0.38, 1])
    for obj in scene["objects"]:
        _add_object(wb, obj, 1.0, 1.0)
    return spec


def add_photo_camera(spec: mujoco.MjSpec, cam: Camera, w: int, h: int, name="photo"):
    """MuJoCo camera that reproduces an OpenCV pinhole (K, R, t): looks down -z with y up, so the OpenCV
    axes (x right, y down, z forward) become (x, -y, -z)."""
    Rw = cam.R.T                                   # columns = camera axes in world coords (OpenCV)
    M = np.c_[Rw[:, 0], -Rw[:, 1], -Rw[:, 2]]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, M.ravel())
    c = spec.worldbody.add_camera(name=name, pos=cam.center.tolist(), quat=q.tolist())
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    c.resolution = [w, h]
    c.focal_pixel = [fx, fy]
    c.sensor_size = [w / fx, h / fy]               # sensorsize in "length" units with focal_length = 1
    c.principal_pixel = [cx - w / 2, cy - h / 2]   # MuJoCo: offset from the image centre
    return c


def render_twin(scene: dict, cam: Camera, w: int, h: int, settle: int = 300):
    """-> (rgb uint8 (h, w, 3), object-id raster (h, w) of int body ids, {body id: object id})."""
    spec = objects_spec(scene)
    add_photo_camera(spec, cam, w, h)
    spec.visual.global_.offwidth = max(w, spec.visual.global_.offwidth)
    spec.visual.global_.offheight = max(h, spec.visual.global_.offheight)
    m = spec.compile()
    d = mujoco.MjData(m)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera="photo")
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="photo")
    seg = r.render()
    r.close()
    gid = seg[..., 0].astype(int)
    body_of_geom = m.geom_bodyid
    body_raster = np.where(gid >= 0, body_of_geom[np.clip(gid, 0, m.ngeom - 1)], -1)
    names = {m.body(b).id: m.body(b).name for b in range(m.nbody)}
    return rgb, body_raster, names


# ----------------------------------------------------------------------------------------------- masks
def full_masks(scene: dict, raw: dict) -> dict:
    """{object id: bool (h, w)} SAM mask pasted into the full image, matched to scene objects by box_px."""
    h, w = raw["height"], raw["width"]
    by_box = {tuple(int(v) for v in o["box"]): o for o in raw["objects"] if o.get("box")}
    out = {}
    for o in scene["objects"]:
        box = tuple(int(v) for v in (o.get("perception") or {}).get("box_px", ()))
        rec = by_box.get(box)
        if rec is None or rec.get("mask") is None:
            continue
        full = np.zeros((h, w), bool)
        x1, y1 = box[0], box[1]
        mk = rec["mask"].astype(bool)
        y2, x2 = min(h, y1 + mk.shape[0]), min(w, x1 + mk.shape[1])
        full[y1:y2, x1:x2] = mk[:y2 - y1, :x2 - x1]
        out[o["id"]] = full
    return out


# ----------------------------------------------------------------------------------------------- main
def evaluate(scene_path: Path, out_stem: Path | None = None, tag: str = "", write_images: bool = True):
    scene = json.loads(scene_path.read_text())
    ph = scene["photo"]
    cam = Camera(**{k: ph["camera"][k] for k in ("K", "R", "t")})
    w, h = ph["width"], ph["height"]
    photo_path = ROOT / ph["file"]
    photo = np.asarray(Image.open(photo_path).convert("RGB"))
    raw = perceive_raw(photo_path, data=scene.get("privacy", "private"))
    masks = full_masks(scene, raw)
    rgb, body_raster, names = render_twin(scene, cam, w, h)
    body_id = {n: b for b, n in names.items()}
    gt_rows = {r["est"]: r for r in (scene.get("perception_notes", {}).get("ground_truth_eval", {}).get("rows", []))}

    rows = []
    photo_lab = srgb_to_lab(photo / 255.0)
    render_lab = srgb_to_lab(rgb / 255.0)
    for o in scene["objects"]:
        oid = o["id"]
        M = masks.get(oid)
        R = body_raster == body_id.get(oid, -99)
        row = {"id": oid, "label": o["label"], "shape": o["shape"]["type"]}
        if M is None:
            row.update(iou=None, dE_render=None, dE_rgba=None)
        else:
            inter, union = np.count_nonzero(R & M), np.count_nonzero(R | M)
            row["iou"] = round(inter / union, 3) if union else 0.0
            row["mask_px"] = int(M.sum())
            row["render_px"] = int(R.sum())
            p_mean = photo_lab[M].mean(0)
            row["dE_render"] = round(delta_e(render_lab[R].mean(0), p_mean), 1) if R.any() else None
            row["dE_rgba"] = round(delta_e(srgb_to_lab(np.asarray(o.get("rgba", [0.7, 0.7, 0.7, 1])[:3])),
                                           np.median(photo_lab[M], axis=0)), 1)
        g = gt_rows.get(oid)
        row["pos_err_cm"] = g["err_cm"] if g else None
        rows.append(row)

    ious = [r["iou"] for r in rows if r.get("iou") is not None]
    dEs = [r["dE_render"] for r in rows if r.get("dE_render") is not None]
    errs = [r["pos_err_cm"] for r in rows if r.get("pos_err_cm") is not None]
    summary = {"scene": scene["name"], "tag": tag, "n": len(rows),
               "mean_iou": round(float(np.mean(ious)), 3) if ious else None,
               "median_iou": round(float(np.median(ious)), 3) if ious else None,
               "mean_dE_render": round(float(np.mean(dEs)), 1) if dEs else None,
               "median_pos_err_cm": round(float(np.median(errs)), 2) if errs else None,
               "max_pos_err_cm": round(float(np.max(errs)), 2) if errs else None, "rows": rows}

    if write_images:
        out_stem = out_stem or (ROOT / "photos" / "real" / scene["name"])
        suffix = f".{tag}" if tag else ""
        render_im = Image.fromarray(rgb)
        render_im.save(f"{out_stem}{suffix}.twin_render.png")
        ov = draw_overlay(photo_path, scene)
        side = Image.new("RGB", (2 * w + 10, h), (255, 255, 255))
        side.paste(ov, (0, 0))
        side.paste(render_im, (w + 10, 0))
        side.save(f"{out_stem}{suffix}.twin_compare.png")
        blend = Image.blend(Image.fromarray(photo), render_im, 0.5)
        d = ImageDraw.Draw(blend)
        for oid, M in masks.items():  # SAM outline in green, rendered silhouette edge in white
            ys, xs = np.nonzero(M)
            if len(xs):
                d.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(40, 220, 90), width=1)
        blend.save(f"{out_stem}{suffix}.twin_overlay.png")
        Path(f"{out_stem}{suffix}.fidelity.json").write_text(json.dumps(summary, indent=1))
    return summary


def print_table(s: dict):
    print(f"{'object':16} {'shape':9} {'IoU':>6} {'dE_render':>9} {'dE_rgba':>8} {'pos_err':>8}")
    for r in s["rows"]:
        f = lambda v, fmt: ("   -" if v is None else format(v, fmt))
        print(f"{r['id']:16} {r['shape']:9} {f(r.get('iou'), '6.3f')} {f(r.get('dE_render'), '9.1f')} "
              f"{f(r.get('dE_rgba'), '8.1f')} {f(r.get('pos_err_cm'), '8.2f')}")
    print(f"mean IoU {s['mean_iou']}  median IoU {s['median_iou']}  mean dE_render {s['mean_dE_render']}  "
          f"median pos err {s['median_pos_err_cm']} cm (max {s['max_pos_err_cm']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output stem (default photos/real/<scene name>)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. 'before'")
    a = ap.parse_args()
    print_table(evaluate(a.scene, a.out, a.tag))
