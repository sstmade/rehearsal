"""Target animations for the mosaic (any video/image works as a target; these are the prepared defaults).

    targets.py arm        -> footage/mosaic/_work/target_arm.npy       (scene 06 shipped run, arm close 3/4 view)
    targets.py wordmark   -> footage/mosaic/_work/target_wordmark.npy  (REHEARSAL, light sweep)
    targets.py house      -> footage/mosaic/_work/target_house.npy     (row of homes, windows lighting up)
All (N, 288, 512, 3) uint8 at 30 fps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiles  # noqa: E402

WORK = tiles.WORK
W, H, N = 512, 288, 600


def arm():
    import mujoco

    import exec_tables as ET
    nn = "06"
    rec, ros = tiles.rollouts_for(nn)
    m, d, sc = tiles.build(nn, rec, "dark")
    q = np.load(ros[0][1])["q"].astype(float)
    nq1 = rec["nq_rigid"]
    m.vis.quality.shadowsize = 4096
    mujoco.mj_forward(m, d)
    names = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")
    arm_b = [b for b in range(m.nbody) if m.body(b).name in names]
    P = []
    for k in range(0, len(q), 40):
        d.qpos[:nq1] = q[k]
        mujoco.mj_kinematics(m, d)
        P.extend(d.xpos[b] for b in arm_b)
    P = np.asarray(P)
    hero = -125.0
    cam = ET.fit_cam(P, hero + 35, 22, margin=0.92, safe=(0.2, 0.8, 0.08, 0.92))
    sh = tiles.Tiny(m, W, H)
    out = np.lib.format.open_memmap(WORK / "target_arm.npy", "w+", np.uint8, (N, H, W, 3))
    idx = np.linspace(0, len(q) - 1, N)
    for i in range(N):
        k = int(round(idx[i]))
        d.qpos[:nq1] = q[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        c = ET.Cam(cam.lookat, cam.dist, cam.az + 14 * (i / N - 0.5), cam.el, 0, cam.fy)
        out[i] = sh.render(d, c)
    out.flush()


def _font(size):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wordmark():
    from PIL import Image, ImageDraw
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    f = _font(78)
    txt = "REHEARSAL"
    bb = dr.textbbox((0, 0), txt, font=f)
    dr.text(((W - (bb[2] - bb[0])) / 2 - bb[0], (H - (bb[3] - bb[1])) / 2 - bb[1]), txt, font=f, fill=255)
    base = np.asarray(img, np.float32) / 255
    xx = np.arange(W)[None, :]
    out = np.lib.format.open_memmap(WORK / "target_wordmark.npy", "w+", np.uint8, (N, H, W, 3))
    for i in range(N):
        cx = -150 + (W + 300) * ((i / 150) % 1.6) / 1.6
        sweep = np.exp(-((xx - cx) / 60.0) ** 2)
        lum = base * (0.72 + 0.28 * sweep)
        col = np.stack([lum * 235, lum * 242, lum * 255], -1) + 6
        out[i] = np.clip(col, 0, 255).astype(np.uint8)
    out.flush()


def house():
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(3)
    houses = []
    x = 18
    while x < W - 60:
        w = int(rng.integers(52, 84))
        h = int(rng.integers(50, 92))
        houses.append((x, w, h))
        x += w + int(rng.integers(8, 18))
    gy = 232
    win = []
    sil = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(sil)
    for x, w, h in houses:
        dr.rectangle([x, gy - h, x + w, gy], fill=70)
        dr.polygon([(x - 5, gy - h), (x + w / 2, gy - h - 0.55 * w), (x + w + 5, gy - h)], fill=70)
        for r in range(max(1, h // 30)):
            for c in range(max(1, w // 26)):
                wx = x + 8 + c * 24
                wy = gy - h + 10 + r * 28
                if wx + 12 < x + w - 4 and wy + 14 < gy - 6:
                    win.append((wx, wy, 12, 14))
    dr.line([(0, gy + 1), (W, gy + 1)], fill=120, width=2)
    sil = np.asarray(sil, np.float32) / 255
    on_at = rng.permutation(len(win)) * (N * 0.75 / len(win))
    out = np.lib.format.open_memmap(WORK / "target_house.npy", "w+", np.uint8, (N, H, W, 3))
    for i in range(N):
        a = np.stack([sil * 160, sil * 175, sil * 205], -1)
        for (wx, wy, ww, wh), t0 in zip(win, on_at):
            k = np.clip((i - t0) / 12, 0, 1)
            a[wy:wy + wh, wx:wx + ww] = (1 - k) * a[wy:wy + wh, wx:wx + ww] + k * np.array([255, 205, 120])
        out[i] = np.clip(a + 5, 0, 255).astype(np.uint8)
    out.flush()


if __name__ == "__main__":
    for name in sys.argv[1:]:
        {"arm": arm, "wordmark": wordmark, "house": house}[name]()
        print(name, "done")
