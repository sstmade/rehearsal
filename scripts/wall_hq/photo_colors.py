"""Per-table photo colours for the HQ wall: each object's dominant colour (pixels where its collision shapes project
into the photo), the table-top colour (table plane minus object footprints) and the room mood (photo outside the table).
  .venv/bin/python scripts/wall_hq/photo_colors.py TWIN_DIR OUT_DIR t6_NN ...   -> OUT_DIR/t6_NN.colors.json
"""
import json, sys, colorsys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
twin, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)


def pts_of(g, n=9):
    t, s = g["type"], np.array(g["size"])
    u = np.linspace(-0.7, 0.7, n)
    if t == 6:
        a, b = np.meshgrid(u, u)
        P = [np.stack([a.ravel() * s[0], b.ravel() * s[1], np.full(a.size, s[2])], 1)]          # top face
        for ax in (0, 1):
            for sg in (-1, 1):
                q = np.zeros((a.size, 3)); q[:, ax] = sg * s[ax]; o = 1 - ax
                q[:, o] = a.ravel() * s[o]; q[:, 2] = b.ravel() * s[2]; P.append(q)
        P = np.concatenate(P)
    elif t in (5, 3):
        r, h = s[0], s[1]
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rr = np.linspace(0, 0.7, 5)
        A, Rr = np.meshgrid(th, rr)
        top = np.stack([Rr.ravel() * r * np.cos(A.ravel()), Rr.ravel() * r * np.sin(A.ravel()), np.full(A.size, h)], 1)
        A2, Z = np.meshgrid(th, u)
        side = np.stack([r * np.cos(A2.ravel()), r * np.sin(A2.ravel()), Z.ravel() * h], 1)
        P = np.concatenate([top, side])
    elif t in (2, 4):
        sz = np.array([s[0]] * 3) if t == 2 else s[:3]
        v = np.random.default_rng(0).normal(size=(200, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        P = v * sz * 0.8
    elif t == 7:
        V = np.array(g["verts"]); c = V.mean(0); P = c + 0.7 * (V - c)
    else:
        return np.zeros((0, 3))
    R = np.array(g["mat"]).reshape(3, 3)
    return P @ R.T + np.array(g["pos"])


def dominant(px, avoid=None):
    px = px.reshape(-1, 3).astype(float)
    if avoid is not None:
        keep = np.linalg.norm(px - avoid, axis=1) > 45
        if keep.sum() >= max(6, 0.15 * len(px)):
            px = px[keep]
    if len(px) < 4:
        return None
    k = min(3, len(px))
    rng = np.random.default_rng(0)
    C = px[rng.choice(len(px), k, replace=False)]
    for _ in range(12):
        lab = np.argmin(((px[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([px[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
    cnt = np.bincount(lab, minlength=k)
    return C[np.argmax(cnt)] / 255.0


def mute(rgb, sat=0.82, vmax=0.9):
    h, s_, v = colorsys.rgb_to_hsv(*np.clip(rgb, 0, 1))
    return list(colorsys.hsv_to_rgb(h, s_ * sat, min(v, vmax)))


for name in sys.argv[3:]:
    J = json.loads((twin / f"{name}.json").read_text())
    im = np.asarray(Image.open(ROOT / J["photo"]).convert("RGB"))
    Hh, Ww = im.shape[:2]
    K, Rc, t = (np.array(J["camera"][k], float) for k in ("K", "R", "t"))
    sx, sy = Ww / J["W"], Hh / J["H"]

    def proj(X):
        c = X @ Rc.T + t
        ok = c[:, 2] > 1e-3
        uv = (c @ K.T)[:, :2] / np.maximum(c[:, 2:3], 1e-6)
        uv = uv * [sx, sy]
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < Ww - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh - 1)
        return uv[ok].astype(int)

    objs = {}
    for g in J["geoms"]:
        if g["kind"] == "obj":
            objs.setdefault(g["body"], []).append(pts_of(g))
    res = {"objects": {}}
    foot = [np.concatenate(L)[:, :2] for L in objs.values()]
    # table top: grid on the top plane, minus object footprints
    cx, cy = J["table"]["center"]; L_, W_ = J["table"]["size"]
    a, b_ = np.meshgrid(np.linspace(-0.45, 0.45, 40) * L_ + cx, np.linspace(-0.45, 0.45, 40) * W_ + cy)
    TP = np.stack([a.ravel(), b_.ravel(), np.zeros(a.size)], 1)
    if foot:
        F = np.concatenate(foot)
        dmin = np.min(np.linalg.norm(TP[:, None, :2] - F[None], axis=2), 1)
        TP = TP[dmin > 0.03]
    uv = proj(TP)
    tp = im[uv[:, 1], uv[:, 0]].reshape(-1, 3) / 255.0
    res["table"] = list(np.median(tp, 0)) if len(tp) else J.get("table_rgba", [0.5, 0.5, 0.5])[:3]
    res["table_spread"] = float(np.mean(np.std(tp, 0))) if len(tp) else 0.0   # wood grain / pattern -> rougher

    for b, L in objs.items():
        P = np.concatenate(L)
        uv = proj(P)
        d = dominant(im[uv[:, 1], uv[:, 0]], np.array(res['table']) * 255) if len(uv) else None
        if d is not None:
            res["objects"][b] = mute(d)
    # mood: top 20% of the photo (wall / room behind)
    res["mood"] = list(np.median(im[: Hh // 5].reshape(-1, 3), 0) / 255.0)
    (out / f"{name}.colors.json").write_text(json.dumps(res, indent=1))
    print(name, len(res["objects"]), "objs table", np.round(res["table"], 2), "mood", np.round(res["mood"], 2), flush=True)
