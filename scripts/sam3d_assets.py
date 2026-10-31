"""Per-object look-alike meshes for a photo scene with Meta SAM 3D Objects (visual only; physics keeps primitives).

For every object of <scene.json> that has a SAM 3.1 mask (perception.mask_px):
  1. rasterise the mask polygon over the scene photo (same as experiments/sam3d/make_masks.py);
  2. run SAM 3D Objects (community MLX port `mlx-spatial`, 8-bit weights; splat only, ~4-5 min, ~11 GB peak);
  3. turn the Gaussian splat into a vertex-coloured mesh (experiments/sam3d/splat2mesh.py, <1 s), decimate it;
  4. orient it: of the 24 axis-aligned rotations of SAM 3D's canonical frame, keep the one whose reprojection into
     the photo (twin pose, scene camera) best matches the SAM mask and the photo colours;
  5. write assets/sam3d/<scene>/<obj>.glb (mesh in the twin object frame: x/y/z = the primitive's axes,
     bbox centred at the origin, longest side = 1) + <obj>.json (timing, memory, extents, orientation scores).
rehearsal/export3d.py then scales/places each GLB on its body (`visual_mesh` in scene3d.json).

Resumable: objects whose GLB exists are skipped; a finished splat (work/outputs/<obj>/gaussians.ply) is reused.
Runs reconstructions one at a time (each peaks at ~10-11 GB on a 36 GB Mac).

This needs its own venv, NOT the project .venv:
    uv venv --python 3.13 sam3d_env
    uv pip install --python sam3d_env/bin/python "mlx-spatial @ git+https://github.com/appautomaton/mlx-spatial.git@main" \
        huggingface_hub plyfile trimesh scikit-image scipy fast-simplification pillow
    sam3d_env/bin/hf download appautomaton/sam-3d-objects-mlx-8bit --local-dir weights/sam-3d-objects-mlx-8bit
usage (from the repo root):
    SAM3D_WEIGHTS=weights/sam-3d-objects-mlx-8bit sam3d_env/bin/python scripts/sam3d_assets.py scenes/real_ycbv59.json \
        [--work .scratch/sam3d/<scene>] [--only id,id] [--faces 18000] [--realign]
"""
from __future__ import annotations

import argparse, itertools, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "sam3d"))
from splat2mesh import splat_to_mesh  # noqa: E402


# ---------------------------------------------------------------- shapes (mirrors rehearsal/scene.py, no mujoco)
def shape_bbox(s: dict) -> list[float]:
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


def uniform_scale(bbox, ext) -> float:
    """Uniform scale for a mesh of extent `ext` to stand in for a primitive of extent `bbox`: geometric mean of the
    ratios on the primitive's two largest axes (keeps the mesh's own aspect; the thin axis is the least reliable
    from one photo). Same formula as rehearsal/export3d.py."""
    k = np.argsort(bbox)[-2:]
    return float(np.exp(np.mean(np.log(np.asarray(bbox)[k] / np.maximum(np.asarray(ext)[k], 1e-6)))))


def rotations24():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


# ---------------------------------------------------------------- photo reprojection score
def render(mesh_w: np.ndarray, faces, fcol, cam, box, pad=12):
    """Painter's-algorithm render of a world-space mesh into the photo crop `box` (x0,y0,x1,y1)."""
    K, Rc, t = (np.asarray(cam[k], float) for k in ("K", "R", "t"))
    pc = mesh_w @ Rc.T + t
    uv = pc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0 = box[0] - pad, box[1] - pad
    W, H = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    img = Image.new("RGB", (W, H), (0, 0, 0))
    sil = Image.new("L", (W, H), 0)
    di, ds = ImageDraw.Draw(img), ImageDraw.Draw(sil)
    depth = pc[faces, 2].mean(1)
    tri = uv[faces] - [x0, y0]
    for f in np.argsort(-depth):
        pts = [tuple(p) for p in tri[f]]
        di.polygon(pts, fill=tuple(int(c) for c in fcol[f]))
        ds.polygon(pts, fill=255)
    return np.asarray(img, float) / 255, np.asarray(sil) > 0, (x0, y0, W, H)


def score_pose(mesh, R, obj, cam, photo, mask, mesh_faces_col):
    v = (mesh.vertices - mesh.bounds.mean(0)) @ R.T
    ext = v.max(0) - v.min(0)
    bbox = shape_bbox(obj["shape"])
    s = uniform_scale(bbox, ext)
    v = v * s
    v[:, 2] += -bbox[2] / 2 - v[:, 2].min()          # sit on the primitive's bottom
    yaw = float(obj.get("yaw", 0.0))
    c, si = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1]])
    vw = v @ Rz.T + [obj["pos"][0], obj["pos"][1], bbox[2] / 2 + 0.001]
    box = obj["perception"]["box_px"]
    img, sil, (x0, y0, W, H) = render(vw, mesh.faces, mesh_faces_col, cam, box)
    ph = np.zeros((H, W, 3)); mk = np.zeros((H, W), bool)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + W, photo.shape[1]), min(y0 + H, photo.shape[0])
    ph[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = photo[ys0:ys1, xs0:xs1]
    mk[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = mask[ys0:ys1, xs0:xs1]
    iou = float((sil & mk).sum() / max((sil | mk).sum(), 1))
    both = sil & mk
    col = float(np.abs(img[both] - ph[both]).mean()) if both.sum() > 20 else 0.5
    up_prior = 0.02 if R[2, 2] == 1 else 0.0           # SAM 3D's canonical frame is usually z-up
    return iou - 0.5 * col + up_prior, {"iou": round(iou, 3), "colour_err": round(col, 3), "scale": round(s, 4)}


# ---------------------------------------------------------------- main
def peak_from_time(txt: str):
    m = re.search(r"(\d+)\s+peak memory footprint", txt)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--work", default=None, help="scratch dir for masks/splats (default .scratch/sam3d/<scene>)")
    ap.add_argument("--weights", default=os.environ.get("SAM3D_WEIGHTS", "weights/sam-3d-objects-mlx-8bit"))
    ap.add_argument("--only", default="", help="comma-separated object ids")
    ap.add_argument("--faces", type=int, default=18000, help="decimation target per object")
    ap.add_argument("--realign", action="store_true", help="redo mesh+orientation even if the GLB exists")
    ap.add_argument("--memory-profile", default="balanced")
    a = ap.parse_args()

    scene_path = Path(a.scene).resolve()
    scene = json.loads(scene_path.read_text())
    name = scene.get("name") or scene_path.stem
    work = Path(a.work or ROOT / ".scratch" / "sam3d" / name).resolve()
    (work / "in").mkdir(parents=True, exist_ok=True)
    out = ROOT / "assets" / "sam3d" / name
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(a.weights).resolve()
    cli = Path(sys.executable).parent / "mlx-spatial-sam3d"

    img = Image.open(ROOT / scene["photo"]["file"]).convert("RGB")
    img.save(work / "in" / "image.png")
    photo = np.asarray(img, float) / 255
    cam = scene["photo"]["camera"]
    only = set(filter(None, a.only.split(",")))

    for o in scene["objects"]:
        oid, poly = o["id"], (o.get("perception") or {}).get("mask_px")
        if not poly or (only and oid not in only):
            continue
        glb, side = out / f"{oid}.glb", out / f"{oid}.json"
        if glb.exists() and not a.realign:
            print(f"[skip] {oid}: {glb.relative_to(ROOT)} exists", flush=True)
            continue
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=255)
        m.save(work / "in" / f"mask_{oid}.png")
        mask = np.asarray(m) > 0
        meta = json.loads(side.read_text()) if side.exists() else {}
        if meta.get("status") == "failed":
            meta = {}
        meta.update({"object": oid, "scene": name, "model": "SAM 3D Objects (Meta, SAM License)",
                     "runtime": "mlx-spatial (MIT) + appautomaton/sam-3d-objects-mlx-8bit"})

        ply = work / "outputs" / oid / "gaussians.ply"
        if not ply.exists():
            ply.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["/usr/bin/time", "-l", str(cli), "reconstruct", str(weights), "in/image.png",
                   "--mask", f"in/mask_{oid}.png", "--output", f"outputs/{oid}/gaussians.ply",
                   "--trace-output", f"outputs/{oid}/trace.json", "--memory-profile", a.memory_profile, "--moge-root", str(weights / "moge")]
            print(f"[sam3d] {oid}: reconstructing ...", flush=True)
            t0 = time.time()
            p = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            (work / "outputs" / oid / "log.txt").write_text(p.stdout + "\n" + p.stderr)
            meta["reconstruct_s"] = round(time.time() - t0, 1)
            meta["peak_memory_gb"] = round((peak_from_time(p.stderr) or 0) / 1e9, 2)
            if p.returncode != 0 or not ply.exists():
                meta["status"] = "failed"
                meta["error"] = (p.stderr or p.stdout)[-800:]
                side.write_text(json.dumps(meta, indent=1))
                print(f"[fail] {oid}: exit {p.returncode}", flush=True)
                continue
        tr = work / "outputs" / oid / "trace.json"
        if tr.exists():
            g = json.loads(tr.read_text()).get("metadata", {}).get("gaussian_decoder", {})
            meta["gaussians"] = g.get("xyz_stats", {}).get("count")
            meta["quality"] = {"geometry": g.get("geometry_quality", {}).get("status"),
                               "opacity": g.get("opacity_quality", {}).get("status")}

        mesh, st = splat_to_mesh(ply, 128, 0.0)
        meta["splat2mesh"] = st
        if len(mesh.faces) > a.faces:
            mesh = mesh.simplify_quadric_decimation(face_count=a.faces)
            # simplification drops vertex colours: re-sample them from the dense mesh
            dense, _ = splat_to_mesh(ply, 128, 0.0)
            from scipy.spatial import cKDTree
            _, nn = cKDTree(dense.vertices).query(mesh.vertices, k=1)
            mesh.visual.vertex_colors = dense.visual.vertex_colors[nn]
        vc = mesh.visual.vertex_colors[:, :3].astype(float)
        fcol = vc[mesh.faces].mean(1)

        best = None
        scores = []
        for R in rotations24():
            sc, info = score_pose(mesh, R, o, cam, photo, mask, fcol)
            scores.append((sc, R, info))
        scores.sort(key=lambda x: -x[0])
        sc, R, info = scores[0]
        v = (mesh.vertices - mesh.bounds.mean(0)) @ R.T
        ext = v.max(0) - v.min(0)
        v = v / ext.max()
        aligned = trimesh.Trimesh(v, mesh.faces, process=False)
        aligned.visual.vertex_colors = mesh.visual.vertex_colors
        aligned.export(glb)
        meta.update({"status": "ok", "faces": len(mesh.faces), "glb_bytes": glb.stat().st_size,
                     "extent": (ext / ext.max()).round(4).tolist(), "frame": "twin object frame, bbox centred, longest side 1",
                     "rotation_from_sam3d": R.astype(int).tolist(), "orientation": info,
                     "orientation_runner_up": scores[1][2] | {"rotation": scores[1][1].astype(int).tolist()}})
        side.write_text(json.dumps(meta, indent=1))
        print(f"[ok] {oid}: {meta.get('reconstruct_s', 'reused')} s, {len(mesh.faces)} faces, "
              f"{glb.stat().st_size / 1e3:.0f} kB, {info}", flush=True)


if __name__ == "__main__":
    main()
