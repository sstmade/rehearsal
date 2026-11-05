"""Hero tabletop for t6_06: a clean brushed-stainless top (procedural: anisotropic brushing + scratches), coloured
from the photo's own clean steel. Replaces the inpainted photo texture, whose object holes read as grey blobs.
Visual only. Run: .venv/bin/python scripts/steel_tex.py  ->  photos/ai_gen/tables_v6/t6_06.table_tex_hero.png"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "photos/ai_gen/tables_v6/t6_06.table_tex.png"
OUT = ROOT / "photos/ai_gen/tables_v6/t6_06.table_tex_hero.png"
CLEAN = (1100, 430, 1340, 740)        # clean steel in the photo texture (colour reference)


def smooth_noise(rng, h, w, sigma):
    """Smooth noise with feature size ~sigma px: low-res random field, bicubic upscaled (no blur edge artefacts)."""
    gh, gw = max(2, int(h / sigma) + 3), max(2, int(w / sigma) + 3)
    g = Image.fromarray(rng.standard_normal((gh, gw)).astype(np.float32), mode="F")
    a = np.asarray(g.resize((int(gw * sigma), int(gh * sigma)), Image.BICUBIC))[:h, :w]
    return (a - a.mean()) / (a.std() + 1e-6)


def brushed(rng, h, w, angle_deg, length):
    """Noise streaked along angle (deg, image coords)."""
    D = int(np.hypot(h, w)) + 4
    n = Image.fromarray((rng.random((D, D)) * 255).astype(np.uint8))
    n = n.rotate(angle_deg, resample=Image.BILINEAR)
    a = np.asarray(n).astype(np.float32)
    k = np.ones(length) / length
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
    n = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).rotate(-angle_deg, resample=Image.BILINEAR)
    a = np.asarray(n).astype(np.float32)
    y0, x0 = (D - h) // 2, (D - w) // 2
    a = a[y0:y0 + h, x0:x0 + w]
    return (a - a.mean()) / (a.std() + 1e-6)


def main(seed=6, scale=2):
    rng = np.random.default_rng(seed)
    ref = Image.open(SRC).convert("RGB")
    W, H = ref.size[0] * scale, ref.size[1] * scale
    base = np.median(np.asarray(ref.crop(CLEAN)).reshape(-1, 3), 0).astype(np.float32)
    base = base * 0.3 + np.array([128, 131, 135], np.float32) * 0.7      # cooler/cleaner than the photo's warm cast
    L = 1.0 + 0.025 * smooth_noise(rng, H, W, 220 * scale) + 0.008 * smooth_noise(rng, H, W, 30 * scale)
    L += 0.022 * brushed(rng, H, W, 58, 60 * scale)                    # fine brushing, same direction as photo
    img = base[None, None, :] * L[..., None]
    # scratches: thin bright arcs, mostly along the brushing direction, drawn at 2x and filtered down
    S = 2
    lay = Image.new("L", (W * S, H * S), 0)
    dr = ImageDraw.Draw(lay)
    for _ in range(int(1400 * (W * H) / (2740 * 1536))):
        x, y = rng.random() * W * S, rng.random() * H * S
        a = np.radians(58 + rng.normal(0, 7))
        ln = rng.lognormal(np.log(60 * scale * S), 0.7)
        curv = rng.normal(0, 0.0006 / scale)
        pts, aa = [], a
        for _ in range(16):
            pts.append((x, y))
            x += np.cos(aa) * ln / 16
            y += np.sin(aa) * ln / 16
            aa += curv * ln / 16
        dr.line(pts, fill=int(rng.uniform(40, 150)), width=1)
    lay = lay.filter(ImageFilter.GaussianBlur(0.6)).resize((W, H), Image.LANCZOS)
    sc = np.asarray(lay).astype(np.float32) / 255.0
    img = img + sc[..., None] * 70.0
    # a few soft smudges (fingerprints / water marks), slightly darker
    sm = np.clip(smooth_noise(rng, H, W, 25 * scale) - 1.6, 0, None)
    img *= (1 - 0.0 * sm)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(OUT)
    print("wrote", OUT, img.shape)


if __name__ == "__main__":
    main()
