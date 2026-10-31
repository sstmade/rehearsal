"""Table-plane calibration from a sheet of paper (A4 by default) lying flat on the table.

Corner convention (used everywhere: CLI, HTTP `paper_corners`, scene JSON):
    pixel corners are given in the order  near-left, near-right, far-right, far-left
    "near/far" = closer to / farther from the robot base; "left/right" = the robot's left/right when it
    looks along +x.  The paper's LONG side runs along the robot's +x axis (pointing away from the robot).
    `paper_xy` is the robot-frame (x, y) of the near-left corner, so the sheet covers
    x in [x0, x0 + 0.297], y in [y0 - 0.210, y0].

Everything here is numpy + PIL only (no OpenCV).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

A4 = (0.297, 0.210)


# ------------------------------------------------------------------------------------------ geometry
def paper_world(paper_xy, size=A4) -> np.ndarray:
    x0, y0 = paper_xy
    L, W = size
    return np.array([[x0, y0], [x0, y0 - W], [x0 + L, y0 - W], [x0 + L, y0]], float)


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT with Hartley normalisation; maps src (N,2) -> dst (N,2)."""
    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    Ts, Td = norm(src), norm(dst)
    s = (Ts @ np.c_[src, np.ones(len(src))].T).T
    d = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    A = []
    for (u, v, _), (x, y, _) in zip(s, d):
        A += [[u, v, 1, 0, 0, 0, -x * u, -x * v, -x], [0, 0, 0, u, v, 1, -y * u, -y * v, -y]]
    Hn = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(np.asarray(pts).reshape(-1, 2)))]
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def focal_from_homography(H_table_to_px: np.ndarray, w: int, h: int):
    """Focal length (px) from the two rotation constraints, principal point at the image center.
    Returns (f_orth, f_norm) — the estimates from r1.r2=0 and |r1|=|r2|; NaN if degenerate."""
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    M = T @ H_table_to_px
    (a, b), (c, d), (e, g) = M[0, :2], M[1, :2], M[2, :2]
    f2_orth = -(a * b + c * d) / (e * g) if abs(e * g) > 1e-12 else np.nan
    den = g * g - e * e
    f2_norm = (a * a + c * c - b * b - d * d) / den if abs(den) > 1e-12 else np.nan
    f = lambda v: float(np.sqrt(v)) if np.isfinite(v) and v > 0 else float("nan")
    return f(f2_orth), f(f2_norm)


class Camera:
    """Pinhole camera, OpenCV convention (x right, y down, z forward); world = robot frame (table z=0)."""

    def __init__(self, K, R, t):
        self.K, self.R, self.t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, pts3) -> np.ndarray:
        pc = (self.R @ np.asarray(pts3, float).reshape(-1, 3).T).T + self.t
        uv = (self.K @ pc.T).T
        return uv[:, :2] / uv[:, 2:3]

    def ray(self, u: float, v: float):
        d = self.R.T @ np.linalg.solve(self.K, [u, v, 1.0])
        return self.center, d / np.linalg.norm(d)

    def intersect_z(self, u: float, v: float, z: float) -> np.ndarray:
        o, d = self.ray(u, v)
        s = (z - o[2]) / d[2]
        return (o + s * d)[:2]

    def to_json(self) -> dict:
        return {"K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist()}


def camera_from_homography(H_px_to_table: np.ndarray, w: int, h: int, f: float | None = None):
    """Recover a full camera from the table-plane homography (square pixels, centered principal point).
    If f is None it is estimated from the homography; if that is degenerate or implausible, a typical phone
    field of view is assumed (f = 0.75 * max(w, h), ~67 deg horizontal).  Returns (Camera, info dict)."""
    G = np.linalg.inv(H_px_to_table)  # table (x, y, 1) -> pixel
    f_o, f_n = focal_from_homography(G, w, h)
    src = "given"
    if f is None:
        cands = [v for v in (f_o, f_n) if np.isfinite(v) and 0.4 * max(w, h) < v < 2.5 * max(w, h)]
        if cands:
            f, src = float(np.mean(cands)), "homography"
        else:
            f, src = 0.75 * max(w, h), "default_fov"
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    B = np.linalg.solve(K, G)
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if B[2, 2] * lam < 0:  # the table must be in front of the camera (positive depth)
        lam = -lam
    r1, r2, t = B[:, 0] * lam, B[:, 1] * lam, B[:, 2] * lam
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, r3])  # nearest rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    cam = Camera(K, R, t)
    info = {"f_px": round(f, 1), "f_source": src, "f_orth": round(f_o, 1) if np.isfinite(f_o) else None,
            "f_norm": round(f_n, 1) if np.isfinite(f_n) else None,
            "cam_height_m": round(float(cam.center[2]), 3)}
    return cam, info


# ----------------------------------------------------------------------------- paper corner detection
def _label(mask: np.ndarray):
    """4-connected component labelling (small images only). Returns (labels, sizes)."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    sizes = [0]
    flat, lf = mask.ravel(), lab.ravel()
    cur = 0
    for start in np.flatnonzero(flat):
        if lf[start]:
            continue
        cur += 1
        stack, n = [start], 0
        lf[start] = cur
        while stack:
            p = stack.pop()
            n += 1
            y, x = divmod(p, w)
            for q, ok in ((p - 1, x > 0), (p + 1, x < w - 1), (p - w, y > 0), (p + w, y < h - 1)):
                if ok and flat[q] and not lf[q]:
                    lf[q] = cur
                    stack.append(q)
        sizes.append(n)
    return lab, np.array(sizes)


def _hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, counter-clockwise in (x, y)."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], float)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quad_from_hull(hull: np.ndarray) -> np.ndarray:
    """4 hull vertices that (approximately) maximise the enclosed area."""
    n = len(hull)
    d = np.linalg.norm(hull[:, None] - hull[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    a, b = hull[i], hull[j]
    nrm = np.array([-(b - a)[1], (b - a)[0]])
    s = (hull - a) @ nrm
    k, l = int(np.argmax(s)), int(np.argmin(s))
    idx = sorted({i, j, k, l})
    if len(idx) < 4:
        return hull[np.linspace(0, n - 1, 4).astype(int)]
    best = hull[idx]
    for _ in range(3):  # coordinate ascent: move each corner to the hull vertex that maximises area
        for c in range(4):
            areas = []
            for v in range(n):
                q = best.copy(); q[c] = hull[v]
                areas.append(abs(_area(q)))
            best[c] = hull[int(np.argmax(areas))]
    return best


def _refine_quad(quad: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Fit a line to the boundary points near each side, intersect neighbouring lines."""
    lines = []
    for s in range(4):
        a, b = quad[s], quad[(s + 1) % 4]
        L = np.linalg.norm(b - a)
        dirv = (b - a) / L
        nrm = np.array([-dirv[1], dirv[0]])
        rel = boundary - a
        along, off = rel @ dirv, rel @ nrm
        sel = (along > 0.15 * L) & (along < 0.85 * L) & (np.abs(off) < max(3.0, 0.03 * L))
        P = boundary[sel] if sel.sum() >= 5 else np.array([a, b])
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        dv = vt[0]
        lines.append((c, dv))
    out = []
    for s in range(4):
        (c1, d1), (c2, d2) = lines[s - 1], lines[s]
        A = np.c_[d1, -d2]
        if abs(np.linalg.det(A)) < 1e-9:
            out.append(quad[s]); continue
        k = np.linalg.solve(A, c2 - c1)
        out.append(c1 + k[0] * d1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.1 * np.linalg.norm(quad[0] - quad[2]):
        return quad  # refinement went wild
    return out


def detect_paper(img: Image.Image | str, work_w: int = 480) -> dict | None:
    """Find the white sheet of paper. Returns {"corners_px": 4x2 (unordered, ccw in image), "score"} or None."""
    im = Image.open(img) if not isinstance(img, Image.Image) else img
    im = im.convert("RGB")
    W, H = im.size
    a = np.asarray(im, np.float32)
    mn, mx = a.min(2), a.max(2)
    white = mn - 1.5 * (mx - mn)  # bright AND unsaturated
    s = work_w / W
    small = np.asarray(Image.fromarray(np.clip(white, 0, 255).astype(np.uint8)).resize(
        (work_w, max(1, int(round(H * s)))), Image.BILINEAR), np.float32)
    best = None
    for thr in (np.percentile(small, 99) * 0.8, 170.0, 200.0, 140.0):
        mask = small > thr
        lab, sizes = _label(mask)
        for c in np.argsort(sizes)[::-1][:6]:
            if c == 0 or sizes[c] < 0.002 * mask.size:
                continue
            ys, xs = np.nonzero(lab == c)
            hull = _hull(np.c_[xs, ys].astype(float))
            if len(hull) < 4:
                continue
            ha = abs(_area(hull))
            quad = _quad_from_hull(hull)
            fill = sizes[c] / max(ha, 1)           # solid (not a ring / scattered)
            quadness = abs(_area(quad)) / max(ha, 1)  # ~1 for a quadrilateral, ~0.64 for an ellipse
            if fill < 0.85 or quadness < 0.9:
                continue
            # paper should not touch the image border
            if xs.min() == 0 or ys.min() == 0 or xs.max() == lab.shape[1] - 1 or ys.max() == lab.shape[0] - 1:
                continue
            score = sizes[c] * quadness ** 4
            if best is None or score > best[0]:
                best = (score, quad / s, thr, c, lab == c)
        if best is not None:
            break
    if best is None:
        return None
    score, quad, thr, _, compmask = best
    # full-resolution refinement: boundary of the thresholded component inside its (dilated) bbox
    x0, y0 = np.floor(quad.min(0)).astype(int) - 6
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 6
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    roi = white[y0:y1, x0:x1]
    big = np.asarray(Image.fromarray(compmask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST))[y0:y1, x0:x1] > 0
    grow = big.copy()
    for _ in range(4):  # cheap dilation so the full-res edge is inside the region
        grow[1:] |= grow[:-1].copy(); grow[:-1] |= grow[1:].copy()
        grow[:, 1:] |= grow[:, :-1].copy(); grow[:, :-1] |= grow[:, 1:].copy()
    inside = np.median(roi[big]) if big.any() else 220
    outside = np.median(roi[~grow]) if (~grow).any() else 0
    fm = (roi > (inside + outside) / 2) & grow
    ys, xs = np.nonzero(fm)
    if len(xs) < 20:
        return {"corners_px": quad.tolist(), "score": float(score), "refined": False}
    # row-wise extreme pixels = boundary (sub-pixel: +0.5 centre, edges at pixel borders)
    bnd = []
    for y in np.unique(ys):
        row = xs[ys == y]
        bnd += [(row.min() + x0, y + y0 + 0.5), (row.max() + 1 + x0, y + y0 + 0.5)]
    for x in np.unique(xs):
        col = ys[xs == x]
        bnd += [(x + x0 + 0.5, col.min() + y0), (x + x0 + 0.5, col.max() + 1 + y0)]
    bnd = np.array(bnd, float)
    hull = _hull(bnd)
    q = _quad_from_hull(hull)
    q = _refine_quad(q, bnd)
    return {"corners_px": q.tolist(), "score": float(score), "refined": True}


def order_corners(quad_px, w: int, h: int, robot_side: str = "bottom", size=A4) -> np.ndarray:
    """Order 4 detected corners as near-left, near-right, far-right, far-left (see module docstring).

    Which pair of opposite edges is the LONG side is decided by fitting both hypotheses and keeping the one
    whose two focal-length constraints agree best (the wrong aspect ratio gives an inconsistent camera).
    Which short edge is NEAR the robot cannot be known from the paper alone: `robot_side` says where the robot
    base is in the image ("bottom" = photo taken from behind the robot, the default; "top", "left", "right").
    Left/right then follow from the orientation of the table plane (camera above the table).
    """
    q = np.asarray(quad_px, float)
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # sorted by angle in image coords (v down): clockwise as seen on screen
    target = {"bottom": np.array([0, 1.0]), "top": np.array([0, -1.0]),
              "left": np.array([-1.0, 0]), "right": np.array([1.0, 0])}[robot_side]
    best = None
    for start in range(4):
        for rev in (False, True):
            cand = np.roll(q, -start, axis=0)
            if rev:
                cand = cand[::-1]
                cand = np.roll(cand, 1, axis=0)
            Hm = fit_homography(cand, paper_world((0, 0), size))
            G = np.linalg.inv(Hm)
            f_o, f_n = focal_from_homography(G, w, h)
            # Seen from above, the robot-frame order NL,NR,FR,FL (counter-clockwise in x,y) appears with a
            # NEGATIVE shoelace area in image coordinates (u right, v down) for any camera above the table.
            if _area(cand) >= 0:
                continue
            fs = [v for v in (f_o, f_n) if np.isfinite(v)]
            consist = abs(f_o - f_n) / max(f_o, f_n) if len(fs) == 2 else 1.0
            near_mid = (cand[0] + cand[1]) / 2
            nearness = float((near_mid - c) @ target) / (np.linalg.norm(near_mid - c) + 1e-9)
            key = (round(consist, 2) > 0.25, -nearness, consist)
            if best is None or key < best[0]:
                best = (key, cand)
    return best[1] if best is not None else q
