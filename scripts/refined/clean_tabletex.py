"""Remove the inpainting blotches from a photo-rectified table texture (visual only).

The table texture is the photo's table top, rectified; where objects stood it was inpainted, which leaves smooth
fuzzy blotches.  Those are found as low-detail regions (local Laplacian energy) and refilled ALONG THE GRAIN
(image rows: planks run along x in the texture) by repeating the nearest clean run of the same row, so the
photo's own wood (tone, knots, cracks) fills them.  Usage: clean_tabletex.py IN.png OUT.png [--axis x|y]
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

src, dst = sys.argv[1], sys.argv[2]
axis = sys.argv[sys.argv.index("--axis") + 1] if "--axis" in sys.argv else "x"
im = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
if axis == "y":
    im = im.transpose(1, 0, 2)
# the inpainted regions = where the photo's objects stood: their SAM masks (photo px) -> table plane (the
# scene's homography) -> texture px (texture = the rectified top: u from +y to -y, v from +x (far) to -x)
import json
from PIL import ImageDraw
sc = json.loads(open(sys.argv[sys.argv.index("--scene") + 1]).read())
Hm = np.array(sc["photo"]["homography_px_to_table"])
cx, cy = sc["table"]["center"]
sx, sy = sc["table"]["size"]
Hh, Ww = im.shape[:2]
mk = Image.new("L", (Ww, Hh), 0)
dr = ImageDraw.Draw(mk)
for o in sc["objects"]:
    P = np.array(o.get("perception", {}).get("mask_px") or [], float)
    if len(P) < 3:
        continue
    q = (Hm @ np.c_[P, np.ones(len(P))].T).T
    q = q[:, :2] / q[:, 2:]
    u = (cy + sy / 2 - q[:, 1]) / sy * Ww
    v = (cx + sx / 2 - q[:, 0]) / sx * Hh
    dr.polygon(list(zip(u, v)), fill=255)
mk = mk.filter(ImageFilter.MaxFilter(15))
mask = np.asarray(mk) > 127
mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(41))) > 127
# rows touched by an inpainted region are replaced by whole clean rows of the photo's own wood (grain runs along
# the rows, so whole-row swaps read as other planks), rolled in x so repeats do not line up
H, W = mask.shape
frac = mask.mean(1)
clean_rows = np.where(frac < 0.02)[0]
clean_rows = clean_rows[(clean_rows > H * 0.06) & (clean_rows < H * 0.97)]      # not the photo's table edges
out = im.copy()
rng = np.random.default_rng(31)
bad = np.where(frac >= 0.02)[0]
# contiguous bands of bad rows get one contiguous clean band each (keeps plank structure)
k = 0
while k < len(bad):
    k1 = k
    while k1 + 1 < len(bad) and bad[k1 + 1] == bad[k1] + 1:
        k1 += 1
    y0, y1 = bad[k], bad[k1] + 1
    n = y1 - y0
    start = rng.integers(0, max(1, len(clean_rows) - n))
    src = clean_rows[start:start + n]
    if len(src) < n:
        src = np.resize(clean_rows, n)
    shift = int(rng.integers(0, W))
    out[y0:y1] = np.roll(im[src], shift, axis=1)
    k = k1 + 1
if axis == "y":
    out = out.transpose(1, 0, 2)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst)
Image.fromarray((mask * 255).astype(np.uint8)).save(dst.replace(".png", "_mask.png"))
print("masked", round(float(mask.mean()) * 100, 1), "%")
