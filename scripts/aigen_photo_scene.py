"""AI-generated test photo -> scene, end to end, WITHOUT an A4 sheet.

    .venv/bin/python scripts/aigen_photo_scene.py photos/ai_gen/desk_lamp_mug.png --name aigen_desk_lamp_mug \
        --ref rect:smartphone_1:0.1466x0.0706x0.0078 --table-px "u,v u,v u,v u,v" --check laptop_1,book_1

The photos in photos/ai_gen/ were made by an image model (see photos/ai_gen/SOURCES.md). They are NOT real
captures, so `data="public"` (Contributor mode) is allowed, and every scene says so in its title and source.

There is no A4 sheet, so the metric scale comes from a different reference of known size lying flat on the table:

  rect:<object id>:<L>x<W>[x<thickness>]   a flat rectangle (phone, closed tablet, notebook). Its 4 corners come
        from the SAM 3.1 mask of that object (convex hull -> quad -> straight-edge refinement, calib.py helpers),
        are ordered with calib.order_corners (long side chosen by focal-length consistency) and give the
        pixel -> plane homography exactly like the A4 path, with a different size.  The rectangle's top face is
        `thickness` above the table, so the camera is shifted down by that amount afterwards.
  circle:<object id>:<diameter>:<f_px>   a flat circle (cork coaster).  One circle does not fix the focal length,
        so f is assumed; the tilt comes from the ellipse's minor/major ratio, the distance from its major axis
        (f * D / major).  Camera roll is taken from --horizon-px (a line that is horizontal on the table plane,
        e.g. the counter/backsplash joint); 0 if not given.

Robot frame: the robot stands at the near (bottom-of-photo) edge of the table, facing away from the camera.
--table-px gives 4 pixel points on the table edges: near-left, near-right, far-right, far-left (corners may be
extrapolated beyond the image).  Robot +y runs along the far edge (right -> left), the table's near edge sits at
x = --near-x and the table is centred on y = 0.  Positions are never hand-edited: estimate_scene(camera=...) with the
guarded fit, then the exact-footprint de-overlap from scripts/real_photo_scene.py (not perception.tidy).

The uncertainty of the scale reference is propagated by Monte Carlo: corners jittered by --corner-px (1 sigma) and
the reference size by --size-rel (1 sigma), the whole geometric fit re-run (no API), and the per-object position
spread is reported in perception_notes.scale_reference.mc and added in quadrature to each object's sigma_m.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rehearsal.calib import (Camera, _hull, _quad_by_edge_removal, _quad_from_hull, _refine_quad,  # noqa: E402
                             apply_h, camera_from_homography, fit_homography, order_corners, paper_world)
from rehearsal.perception import draw_overlay, estimate_scene, perceive_raw  # noqa: E402
from real_photo_scene import separate  # noqa: E402

LABEL = "AI-generated test photo (image model), not a real capture"


# ------------------------------------------------------------------------------------------ reference shapes
def mask_points(o):
    m = o["mask"]
    x1, y1 = o["box"][:2]
    ys, xs = np.nonzero(m)
    mp = np.pad(m > 0, 1)
    inner = mp[1:-1, 1:-1] & mp[:-2, 1:-1] & mp[2:, 1:-1] & mp[1:-1, :-2] & mp[1:-1, 2:]
    bys, bxs = np.nonzero((m > 0) & ~inner)
    return np.c_[xs + x1, ys + y1].astype(float), np.c_[bxs + x1, bys + y1].astype(float)


def rect_corners(o):
    pts, bnd = mask_points(o)
    h = _hull(pts)
    q = _quad_by_edge_removal(h)
    if q is None:
        q = _quad_from_hull(h)
    return _refine_quad(q, bnd)


def fit_ellipse(pts):
    """Algebraic conic fit to boundary points -> (centre, major, minor, angle of the major axis)."""
    x, y = pts[:, 0], pts[:, 1]
    mx, my, s = x.mean(), y.mean(), max(x.std(), y.std())
    X, Y = (x - mx) / s, (y - my) / s
    D = np.c_[X * X, X * Y, Y * Y, X, Y, np.ones_like(X)]
    a, b, c, d, e, f = np.linalg.svd(D)[2][-1]
    M = np.array([[a, b / 2], [b / 2, c]])
    ctr = np.linalg.solve(2 * M, [-d, -e])
    k = -(f + 0.5 * (d * ctr[0] + e * ctr[1]))
    ev, evec = np.linalg.eigh(M / k)
    axes = 1 / np.sqrt(ev)  # ev ascending -> axes descending
    major, minor = 2 * axes[0] * s, 2 * axes[1] * s
    ang = float(np.arctan2(evec[1, 0], evec[0, 0]))
    return ctr * s + [mx, my], major, minor, ang


def camera_from_circle(ctr, major, minor, diam, f, w, h, roll=0.0):
    """Camera (table frame: circle centre at the origin, z up, +x away from the camera) from one imaged circle."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    ray = np.linalg.solve(K, [ctr[0], ctr[1], 1.0])
    ray /= np.linalg.norm(ray)
    phi = np.arcsin(np.clip(minor / major, 0, 1))      # angle between the line of sight and the table plane
    dist = f * diam / major                               # distance camera -> circle centre
    # camera pitch so that this ray makes angle phi with the plane (roll applied around the optical axis)
    ray_el = np.arctan2(ray[1], np.hypot(ray[0], ray[2]))  # how far below the optical axis (v down = +)
    pitch = phi - ray_el                                  # optical axis tilt below the horizon
    cr, sr = np.cos(roll), np.sin(roll)
    Rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    # world -> camera for a camera looking along +x, tilted down by pitch: cam z = (cp, 0, -sp), cam x = (0, -1, 0)
    Rw = np.array([[0, -1, 0], [-sp, 0, -cp], [cp, 0, -sp]], float)
    R = Rroll @ Rw
    C0 = np.array([0.0, 0.0, 0.0])
    cam = Camera(K, R, -R @ C0)
    # place the camera so the circle centre (world origin) lies on `ray` at distance `dist`
    d_world = R.T @ ray
    C = -dist * d_world
    cam = Camera(K, R, -R @ C)
    return cam


def shift_down(cam, hgt):
    """The reference's top face was the z=0 plane; move the world so the table is z=0 (top face at +hgt)."""
    return Camera(cam.K, cam.R, cam.t - hgt * cam.R[:, 2])


def to_robot(cam, table_px, near_x):
    """Rigid transform of the world frame: robot +y along the far table edge (right -> left), +x away from the
    camera, table near edge at x = near_x, table centred on y = 0.  Returns (camera, table dict, table corners)."""
    G = cam.K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
    Hp = np.linalg.inv(G)
    tc = apply_h(Hp, table_px)                     # NL, NR, FR, FL in the current world frame
    ey = tc[3] - tc[2]
    ey /= np.linalg.norm(ey)
    ex = np.array([ey[1], -ey[0]])                 # right-handed (x, y, z up)
    if (tc[3] - tc[0]) @ ex < 0:                   # +x must point from the near edge to the far edge
        raise ValueError("table corners not in NL, NR, FR, FL order")
    Rz = np.eye(3)
    Rz[:2, :2] = np.c_[ex, ey].T                   # new = Rz @ old
    loc = tc @ Rz[:2, :2].T
    x0 = loc[[0, 1], 0].mean()
    yc = (loc[:, 1].max() + loc[:, 1].min()) / 2
    off = np.array([near_x - x0, -yc, 0.0])
    R = cam.R @ Rz.T                               # X_old = Rz.T (X_new - off)
    t = cam.t - R @ off
    cam2 = Camera(cam.K, R, t)
    loc = loc + off[:2]
    sx = float(np.mean([np.linalg.norm(loc[3] - loc[0]), np.linalg.norm(loc[2] - loc[1])]))
    sy = float(np.mean([np.linalg.norm(loc[1] - loc[0]), np.linalg.norm(loc[2] - loc[3])]))
    table = {"center": [round(float(loc[:, 0].mean()), 4), 0.0], "size": [round(sx, 4), round(sy, 4)]}
    return cam2, table, loc


def plane_h(cam):
    G = cam.K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
    H = np.linalg.inv(G)
    return H / H[2, 2]


def edges_cm(H, q):
    p = apply_h(H, q)
    return [round(float(np.linalg.norm(p[(i + 1) % 4] - p[i])) * 100, 1) for i in range(4)]


def cam_tilt(f, pitch, roll, w, h, height=1.0):
    """Camera looking along world +x, tilted down by pitch, rolled by roll, `height` above the z=0 plane."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array([[0, -1, 0], [-sp, 0, -cp], [cp, 0, -sp]], float)
    C = np.array([0.0, 0.0, height])
    return Camera(K, R, -R @ C)


def _angles_cos(Hp, q):
    p = apply_h(Hp, q)
    out = []
    for i in range(4):
        a, b = p[i - 1] - p[i], p[(i + 1) % 4] - p[i]
        out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return out


def fit_rectangles(quads, ref_q, ref_size, w, h):
    """Metric rectification from several rectangles of unknown size (right angles) plus the reference's aspect
    ratio: solve f, pitch, roll (square pixels, centred principal point).  Returns (f, pitch, roll, residual)."""
    from rehearsal.perception import _nelder_mead
    asp = np.log(ref_size[0] / ref_size[1])

    def cost(x):
        f, pitch, roll = np.exp(x[0]) * max(w, h), x[1], x[2]
        if not (0.05 < pitch < 1.55):
            return 1e3
        Hp = plane_h(cam_tilt(f, pitch, roll, w, h))
        c = sum(v * v for q in quads + [ref_q] for v in _angles_cos(Hp, q))
        e = edges_cm(Hp, ref_q)
        la, lb = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        c += 4 * (np.log(max(la, lb) / min(la, lb)) - asp) ** 2
        return c
    best = None
    for f0 in (0.8, 1.1, 1.5):
        for p0 in (0.5, 0.8, 1.1):
            x, v = _nelder_mead(cost, np.array([np.log(f0), p0, 0.0]), np.array([0.2, 0.15, 0.05]), iters=400)
            if best is None or v < best[1]:
                best = (x, v)
    x, v = best
    return float(np.exp(x[0]) * max(w, h)), float(x[1]), float(x[2]), float(v)


def build_camera(ref, raw, w, h, f_override=None, jitter=None, rng=None, table_px=None):
    """ref -> camera in the reference's own frame (table z = 0).  jitter = (corner_px, size_rel) for Monte Carlo."""
    o = next(x for x in raw["objects"] if x["id"] == ref["id"])
    if ref["kind"] == "rectfit":
        L, W = ref["size"]
        q = rect_corners(o)
        quads = [rect_corners(next(x for x in raw["objects"] if x["id"] == k)) for k in ref["others"]]
        tq = np.asarray(table_px, float)
        if jitter:
            q = q + rng.normal(0, jitter[0], q.shape)
            quads = [qq + rng.normal(0, jitter[0], qq.shape) for qq in quads]
            tq = tq + rng.normal(0, ref.get("table_px_sigma", 4.0), tq.shape)
            s = 1 + rng.normal(0, jitter[1])
            L, W = L * s, W * s
        f, pitch, roll, res = fit_rectangles(quads + ([tq] if ref.get("table_rect", True) else []), q, (L, W), w, h)
        Hp = plane_h(cam_tilt(f, pitch, roll, w, h))
        e = edges_cm(Hp, q)
        la, lb = max((e[0] + e[2]) / 2, (e[1] + e[3]) / 2) / 100, min((e[0] + e[2]) / 2, (e[1] + e[3]) / 2) / 100
        # height of the camera above the reference's top face; a long thin reference (a ruler) is scaled by its
        # length only (its width is a few pixels and its overall width varies between rulers)
        k = L / la if L / W > 4 else (L / la + W / lb) / 2
        cam = cam_tilt(f, pitch, roll, w, h, height=k + ref.get("thick", 0.0))
        return cam, {"f_px": round(f, 1), "f_source": "rectangles(right angles)+ref aspect",
                     "pitch_deg": round(float(np.degrees(pitch)), 2), "roll_deg": round(float(np.degrees(roll)), 2),
                     "rect_fit_residual": round(res, 5), "rectangles": [ref["id"]] + ref["others"] + (["table"] if ref.get("table_rect", True) else []),
                     "cam_height_m": round(float(cam.center[2]), 3), "ref_corners_px": np.round(q, 1).tolist()}
    if ref["kind"] == "rect":
        q = rect_corners(o)
        L, W = ref["size"]
        if jitter:
            q = q + rng.normal(0, jitter[0], q.shape)
            s = 1 + rng.normal(0, jitter[1])
            L, W = L * s, W * s
        q = order_corners(q, w, h, "bottom", size=(L, W))
        H = fit_homography(q, paper_world((0, 0), (L, W)))
        cam, info = camera_from_homography(H, w, h, f=f_override)
        cam = shift_down(cam, ref.get("thick", 0.0))
        info["ref_corners_px"] = np.round(q, 1).tolist()
        return cam, info
    _, bnd = mask_points(o)
    ctr, major, minor, ang = fit_ellipse(bnd)
    d, f = ref["diam"], ref["f"]
    if jitter:
        ctr = ctr + rng.normal(0, jitter[0], 2)
        major, minor = major + rng.normal(0, jitter[0]), minor + rng.normal(0, jitter[0])
        d = d * (1 + rng.normal(0, jitter[1]))
        f = f * np.exp(rng.normal(0, ref.get("f_rel", 0.0)))
    # the mask covers the top face AND the visible rim (thickness t): the outline's minor axis is longer than the
    # top face's by about (major / d) * t * cos(phi).  Fixed-point iteration (2-3 steps converge).
    minor_top = minor
    for _ in range(5):
        phi = np.arcsin(np.clip(minor_top / major, 0, 1))
        minor_top = minor - major / d * ref.get("thick", 0.0) * np.cos(phi)
    cam = camera_from_circle(ctr, major, minor_top, d, f, w, h, roll=ref.get("roll", 0.0))
    cam = shift_down(cam, ref.get("thick", 0.0))
    return cam, {"f_px": round(f, 1), "f_source": "assumed", "ellipse_px": {
        "centre": np.round(ctr, 1).tolist(), "major": round(float(major), 1), "minor": round(float(minor), 1),
        "minor_top_face": round(float(minor_top), 1), "view_elevation_deg": round(float(np.degrees(phi)), 1),
        "angle_rad": round(ang, 3)}, "cam_height_m": round(float(cam.center[2]), 3)}


def parse_ref(s):
    kind, oid, rest = s.split(":", 2)
    if kind == "rectfit":
        size, others = (rest.split(":") + [""])[:2]
        v = [float(t) for t in size.split("x")]
        return {"kind": "rectfit", "id": oid, "size": (v[0], v[1]), "thick": v[2] if len(v) > 2 else 0.0,
                "others": [t for t in others.split(",") if t]}
    if kind == "rect":
        v = [float(t) for t in rest.split("x")]
        return {"kind": "rect", "id": oid, "size": (v[0], v[1]), "thick": v[2] if len(v) > 2 else 0.0}
    v = rest.split(":")
    return {"kind": "circle", "id": oid, "diam": float(v[0]), "f": float(v[1]),
            "thick": float(v[2]) if len(v) > 2 else 0.0}


def pts(s):
    return np.array([[float(t) for t in c.split(",")] for c in s.split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-note", default="", help="why this reference and its size (free text, stored)")
    ap.add_argument("--table-px", required=True, help='"u,v x4": near-left, near-right, far-right, far-left')
    ap.add_argument("--horizon-px", default=None, help='circle refs: "u,v u,v" a table-plane line that is level')
    ap.add_argument("--f-rel", type=float, default=0.0, help="circle refs: 1-sigma log-uncertainty of assumed f")
    ap.add_argument("--no-table-rect", action="store_true", help="rectfit: the table top is not a rectangle")
    ap.add_argument("--near-x", type=float, default=0.15)
    ap.add_argument("--table-size", default=None, help="override table size sx,sy (e.g. counter cut to reach)")
    ap.add_argument("--check", default="", help="comma-separated ids of other flat rectangles to measure")
    ap.add_argument("--corner-px", type=float, default=1.5)
    ap.add_argument("--size-rel", type=float, default=0.01)
    ap.add_argument("--mc", type=int, default=30)
    ap.add_argument("--title", default="")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--upscaled-from", default="", help="note if the photo was upscaled before perception")
    a = ap.parse_args()
    t0 = time.time()
    photo = a.photo if a.photo.is_absolute() else ROOT / a.photo
    rel = str(photo.relative_to(ROOT))
    w, h = Image.open(photo).size
    raw = perceive_raw(photo, data="public")
    ref = parse_ref(a.ref)
    ref["f_rel"] = a.f_rel
    ref["table_rect"] = not a.no_table_rect
    if a.horizon_px and ref["kind"] == "circle":
        p = pts(a.horizon_px)
        ref["roll"] = float(np.arctan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
    table_px = pts(a.table_px)

    def fit(jitter=None, rng=None, fidelity=True):
        cam0, info = build_camera(ref, raw, w, h, jitter=jitter, rng=rng, table_px=table_px)
        cam, table, tloc = to_robot(cam0, table_px, a.near_x)
        if a.table_size:
            sx, sy = (float(v) for v in a.table_size.split(","))
            table = {"center": [round(a.near_x + sx / 2, 4), 0.0], "size": [sx, sy]}
        sc = estimate_scene(raw, None, method="guarded", camera=cam, table=table, privacy="public",
                            name=a.name, photo_file=rel, clamp=False, fidelity=fidelity)
        return sc, cam, info, table, tloc

    scene, cam, info, table, tloc = fit()
    raw_pos = {o["id"]: list(o["pos"]) for o in scene["objects"]}
    separate(scene["objects"])

    # Monte Carlo over the scale reference (geometry only, no API)
    rng = np.random.default_rng(0)
    samples = {o["id"]: [] for o in scene["objects"]}
    tsz = []
    for _ in range(a.mc):
        try:
            sc, _, _, tb, _ = fit((a.corner_px, a.size_rel), rng, fidelity=False)
        except Exception:
            continue
        tsz.append(tb["size"])
        for o in sc["objects"]:
            samples[o["id"]].append(o["pos"])
    mc = {}
    for o in scene["objects"]:
        P = np.asarray(samples[o["id"]])
        # robust spread (1.4826 * MAD per axis): an occasional flip of the guarded fit (mask vs 2D-box fallback)
        # in one Monte Carlo draw should not set the stress-test sigma on its own
        sd = (float(np.sqrt(np.sum((1.4826 * np.median(np.abs(P - np.median(P, 0)), 0)) ** 2)))
              if len(P) > 2 else None)
        mc[o["id"]] = None if sd is None else round(sd, 4)
        if sd is not None:
            o["perception"]["sigma_m_detector"] = o["perception"]["sigma_m"]
            o["perception"]["sigma_m_scale_ref"] = round(sd / np.sqrt(2), 4)  # per-axis
            o["perception"]["sigma_m"] = round(float(np.hypot(o["perception"]["sigma_m"], sd / np.sqrt(2))), 4)

    H = plane_h(cam)
    checks = {}
    for oid in [s for s in a.check.split(",") if s]:
        o = next((x for x in raw["objects"] if x["id"] == oid), None)
        if o is None or o.get("mask") is None:
            continue
        checks[oid] = {"edges_cm_on_table_plane": edges_cm(H, rect_corners(o)), "muse_dims_m": o["dims_m"]}
    for o in scene["objects"]:
        if o["category"] in ("coaster", "trivet") and o["shape"]["type"] == "cylinder":
            checks[o["id"]] = {"diameter_cm": round(o["shape"]["radius"] * 200, 1)}
    tl = np.asarray(tloc)
    ang = [float(np.degrees(np.arccos(np.clip(np.dot(tl[i - 1] - tl[i], tl[(i + 1) % 4] - tl[i])
                                               / np.linalg.norm(tl[i - 1] - tl[i])
                                               / np.linalg.norm(tl[(i + 1) % 4] - tl[i]), -1, 1))))
           for i in range(4)]
    scene["title"] = a.title or f"{a.name} ({LABEL})"
    scene["ai_generated"] = True
    scene["source"] = (f"{LABEL}: {photo.name}, see photos/ai_gen/SOURCES.md. Muse Spark listing + SAM 3.1 masks "
                       f"(Contributor mode, synthetic image) -> guarded fit with a camera from a scale reference "
                       f"({a.ref}; no A4 sheet in the photo) -> exact-footprint de-overlap "
                       f"(scripts/aigen_photo_scene.py). No hand-edited positions."
                       + (f" Upscaled: {a.upscaled_from}." if a.upscaled_from else ""))
    if a.instruction:
        scene["demo_instruction"] = a.instruction
    scene["photo"]["camera_info"].update(info)
    scene["photo"]["camera_info"]["ai_generated"] = LABEL
    scene["perception_notes"]["scale_reference"] = {
        "ref": a.ref, "note": a.ref_note, "table_px": table_px.tolist(),
        "table_corners_robot_m": np.round(tl, 4).tolist(), "table_corner_angles_deg": [round(v, 1) for v in ang],
        "table_size_m": table["size"],
        "table_size_mc_sd_m": np.round(np.std(np.asarray(tsz), 0), 4).tolist() if tsz else None,
        "checks": checks, "mc": {"n": a.mc, "corner_px_sigma": a.corner_px, "size_rel_sigma": a.size_rel,
                                 "f_rel_sigma": a.f_rel, "pos_sd_m": mc, "pos_sd_m_is": "robust (1.4826 MAD), xy combined",
                                 "pos_sd_m_plain": {k: (round(float(np.sqrt(np.sum(np.asarray(v).var(0)))), 4)
                                                        if len(v) > 2 else None) for k, v in samples.items()}},
        "deoverlap_moves_cm": {k: round(float(np.linalg.norm(np.subtract(v, next(
            o["pos"] for o in scene["objects"] if o["id"] == k)))) * 100, 2) for k, v in raw_pos.items()}}
    if a.upscaled_from:
        scene["perception_notes"]["upscaled"] = a.upscaled_from
    (ROOT / "scenes" / f"{a.name}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    draw_overlay(photo, scene).save(photo.parent / f"{a.name}.overlay.png")
    print(json.dumps(scene["perception_notes"]["scale_reference"], indent=1))
    print("camera:", scene["photo"]["camera_info"])
    for o in scene["objects"]:
        print(f'{o["id"]:20s} {o["shape"]["type"]:9s} pos {o["pos"]} yaw {o["yaw"]:.2f} '
              f'sigma {o["perception"]["sigma_m"]} conf {o["perception"]["confidence"]} '
              f'dims {o["shape"]}')
    print("not segmented:", scene["perception_notes"]["not_segmented"])
    print(f"wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
