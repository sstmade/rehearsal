"""Table height from two exterior cameras by an object-space plane sweep.  $0, local, no Meta API.

    <python with cv2+scipy> -I datasets/droid/_tools/table_sweep.py <karlp_annot dir> <frames dir> <out json> <eid>...

For every episode, frame 0 of ext1 and ext2 (left lenses; decoded beforehand into <frames dir>/<eid>_<ext>_0.png,
never published: the other view can show people) is resampled onto a horizontal grid in the robot base frame
(x 0.30..0.95, y -0.35..0.45, 4 mm) at each candidate height z (-0.16..+0.04 m, 2 mm).  Where the hypothesised plane
is the real table, both cameras see the same wood grain at the same grid cell, so the local zero-mean normalised
cross-correlation (ZNCC, 13x13 cells = 5 cm) peaks.  Each 6 cm block votes for its best z (peak ZNCC >= 0.5 and
clearly above the block's mean); a RANSAC plane through the votes (1 cm inlier band) gives the table plane.
Camera poses: KarlP/droid cam2base_extrinsic_superset.json (per episode), intrinsics.json.
"""
import json
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

XS, YS, STEP = (0.30, 0.95), (-0.35, 0.45), 0.004
ZS = np.arange(-0.16, 0.0401, 0.002)
WIN, BLOCK = 13, 15


def cam(sup, intr, eid, serial):
    fx, cx, fy, cy = intr[eid][serial]["cameraMatrix"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    e = sup[eid][serial]
    Rc2b = Rot.from_euler("xyz", e[3:6]).as_matrix()
    return K, Rc2b.T, -Rc2b.T @ np.asarray(e[:3])          # X_cam = R X_base + t


def ortho(img, K, R, t, X, Y, z):
    P = np.stack([X, Y, np.full_like(X, z)], -1) @ R.T + t
    ok = P[..., 2] > 0.05
    uv = (P @ K.T)[..., :2] / np.maximum(P[..., 2:3], 1e-6)
    o = cv2.remap(img, uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                  borderValue=np.nan)
    o[~ok] = np.nan
    return o


def zncc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(m, a, 0), np.where(m, b, 0)
    k = (WIN, WIN)
    n = cv2.boxFilter(m.astype(np.float32), -1, k, normalize=False)
    sa, sb = cv2.boxFilter(a, -1, k, normalize=False), cv2.boxFilter(b, -1, k, normalize=False)
    saa, sbb = cv2.boxFilter(a * a, -1, k, normalize=False), cv2.boxFilter(b * b, -1, k, normalize=False)
    sab = cv2.boxFilter(a * b, -1, k, normalize=False)
    n1 = np.maximum(n, 1)
    cov = sab - sa * sb / n1
    va, vb = saa - sa * sa / n1, sbb - sb * sb / n1
    r = cov / np.sqrt(np.maximum(va * vb, 1e-9))
    r[(n < 0.8 * WIN * WIN) | (va < 1e-3 * n1) | (vb < 1e-3 * n1)] = np.nan
    return r


def _prep(g):
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    return g - cv2.GaussianBlur(g, (0, 0), 6.0)              # keep the grain, drop shading


def views_two_ext(sup, intr, frames, eid):
    """ext1 + ext2 left lenses (wide baseline; depends on BOTH exterior calibrations)."""
    serials = json.load(open(f"{frames}/{eid}.serials.json"))    # {"ext1": serial, "ext2": serial}
    return ([_prep(cv2.imread(f"{frames}/{eid}_{k}_0.png", cv2.IMREAD_GRAYSCALE)) for k in serials],
            [cam(sup, intr, eid, s) for s in serials.values()], "two exterior cameras (ext1+ext2 left lenses)")


def views_stereo(sup, intr, frames, eid, key, serial, baseline=0.12):
    """Left + right lens of one ZED 2 (rectified side-by-side frame, 12 cm baseline): only this camera's
    exterior calibration matters, the stereo rig itself is factory calibrated."""
    g = cv2.imread(f"{frames}/stereo_{eid}_{key}_0.png", cv2.IMREAD_GRAYSCALE)
    w = g.shape[1] // 2
    K, R, t = cam(sup, intr, eid, serial)
    return ([_prep(g[:, :w]), _prep(g[:, w:])], [(K, R, t), (K, R, t - np.array([baseline, 0, 0]))],
            f"ZED stereo pair of {key} ({serial})")


def sweep(views, eid):
    X, Y = np.meshgrid(np.arange(*XS, STEP), np.arange(*YS, STEP))
    X, Y = X.astype(np.float32), Y.astype(np.float32)
    ims, cams, label = views
    vol = np.stack([zncc(ortho(ims[0], *cams[0], X, Y, z), ortho(ims[1], *cams[1], X, Y, z)) for z in ZS])
    votes = []
    H, W = X.shape
    for i in range(0, H - BLOCK, BLOCK):
        for j in range(0, W - BLOCK, BLOCK):
            c = np.nanmean(vol[:, i:i + BLOCK, j:j + BLOCK].reshape(len(ZS), -1), axis=1)
            if not np.isfinite(c).any():
                continue
            k = int(np.nanargmax(c))
            if c[k] < 0.5 or c[k] - np.nanmean(c) < 0.2 or k in (0, len(ZS) - 1):
                continue
            # sub-sample peak (parabola through the three samples)
            y0, y1, y2 = c[k - 1], c[k], c[k + 1]
            den = y0 - 2 * y1 + y2
            dz = 0.5 * (y0 - y2) / den * (ZS[1] - ZS[0]) if np.isfinite(den) and abs(den) > 1e-9 else 0.0
            votes.append([float(X[i + BLOCK // 2, j + BLOCK // 2]), float(Y[i + BLOCK // 2, j + BLOCK // 2]),
                          float(ZS[k] + dz), float(y1)])
    V = np.array(votes)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(2000):
        p = V[rng.choice(len(V), 3, replace=False), :3]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.97:
            continue
        inl = np.abs((V[:, :3] - p[0]) @ n) < 0.01
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = V[best, :3]
    A = np.c_[Q[:, 0] - 0.6, Q[:, 1], np.ones(len(Q))]
    coef, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)        # z = a (x - 0.6) + b y + c
    res = Q[:, 2] - A @ coef
    n = np.array([-coef[0], -coef[1], 1.0]); n /= np.linalg.norm(n)
    # bootstrap: spread of the fitted height at (0.6, 0)
    bs = [np.linalg.lstsq(A[ii], Q[ii, 2], rcond=None)[0][2] for ii in
          (rng.choice(len(Q), len(Q)) for _ in range(300))]
    return {"eid": eid, "method": "plane sweep, frame 0, ZNCC on wood grain: " + label,
            "n_votes": int(len(V)), "n_inliers": int(best.sum()),
            "z_table_at_0.6_0": round(float(coef[2]), 4), "slope_dz_dx": round(float(coef[0]), 4),
            "slope_dz_dy": round(float(coef[1]), 4), "tilt_deg": round(float(np.degrees(np.arccos(n[2]))), 2),
            "normal_base": np.round(n, 4).tolist(), "rms_residual_m": round(float(np.sqrt(np.mean(res ** 2))), 4),
            "bootstrap_sd_m": round(float(np.std(bs)), 4),
            "x_range": np.round(np.percentile(Q[:, 0], [2, 98]), 3).tolist(),
            "y_range": np.round(np.percentile(Q[:, 1], [2, 98]), 3).tolist(),
            "median_peak_zncc": round(float(np.median(V[best, 3])), 3)}


def main():
    ann, frames, out = sys.argv[1:4]
    sup = json.load(open(f"{ann}/cam2base_extrinsic_superset.json"))
    intr = json.load(open(f"{ann}/intrinsics.json"))
    res = {}
    for arg in sys.argv[4:]:     # <eid> (two exterior cameras) or <eid>:<ext1|ext2>:<serial> (that camera's stereo)
        eid, *st = arg.split(":")
        v = views_stereo(sup, intr, frames, eid, *st) if st else views_two_ext(sup, intr, frames, eid)
        res[arg] = sweep(v, eid)
        print(json.dumps(res[arg]))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
