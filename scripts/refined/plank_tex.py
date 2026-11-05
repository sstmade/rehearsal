"""Rebuild a plank table texture from a clean band of the photo's own (rectified) wood (visual only).
The shipped t6_31 table texture is ~50 % inpainting blotches; the band at rows 665-712 (front of the table) is
clean photo wood.  Planks (~0.19 m) are filled with that band (stretched x1.6 vertically, mirrored, rolled in x per
plank, +-6 % tone), separated by a dark seam.  Usage: plank_tex.py IN.png OUT.png"""
import sys

import numpy as np
from PIL import Image

src = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32)
H, W = src.shape[:2]
band = src[665:712]
# flatten the photo's lighting along the band (column-wise low-pass), keep the wood detail
from PIL import ImageFilter  # noqa: E402
lo = np.asarray(Image.fromarray(band.astype(np.uint8)).filter(ImageFilter.GaussianBlur(60)), np.float32)
band = band / np.maximum(lo, 1) * lo.reshape(-1, 3).mean(0)
band = np.asarray(Image.fromarray(np.clip(band, 0, 255).astype(np.uint8)).resize((W, int(band.shape[0] * 1.6)), Image.BICUBIC), np.float32)
plank = int(round(0.19 / 0.967 * H))
rng = np.random.default_rng(31)
out = np.zeros_like(src)
y = 0
k = 0
while y < H:
    rows = []
    while sum(r.shape[0] for r in rows) < plank:
        rows.append(np.roll(band, int(rng.integers(0, W)), axis=1))
    p = np.concatenate(rows)[:plank]
    sh = int(rng.integers(0, W))
    p = np.roll(p, sh, axis=1) * rng.uniform(0.88, 1.04)
    p[:3] *= 0.35                                   # seam between planks
    p[:, sh - 2:sh + 1] *= 0.5 if 2 < sh < W else 1  # butt joint
    n = min(plank, H - y)
    out[y:y + n] = p[:n]
    y += plank
    k += 1
img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
if "--transpose" in sys.argv:      # the table's UVs run u <-> world x, v <-> world y (checked in the render)
    img = img.transpose(Image.Transpose.TRANSPOSE).resize((W, H), Image.LANCZOS)
img.save(sys.argv[2])
