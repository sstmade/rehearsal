"""Procedural textures for the set dressing (scripts/hero_dress.py): clean table tops by material class, room walls
and floors.  Every generator is coloured from a few clean photo pixels passed in by the caller, so a top keeps the
photo's hue but loses the inpainting blobs where objects were.  Visual only.

Table tops are written as top views in the same layout as the scenes' table_texture (row 0 = far edge, column 0 =
left as seen from the robot), rows = table x extent, columns = table y extent.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PX_PER_M = 1500


def _norm(a):
    return (a - a.mean()) / (a.std() + 1e-6)


def aniso_noise(rng, h, w, sy, sx):
    """Smooth noise with feature size sy px along rows (vertical) and sx px along columns (horizontal)."""
    gh, gw = max(2, int(h / sy) + 3), max(2, int(w / sx) + 3)
    g = Image.fromarray(rng.standard_normal((gh, gw)).astype(np.float32), mode="F")
    a = np.asarray(g.resize((int(gw * sx), int(gh * sy)), Image.BICUBIC))[:h, :w]
    return _norm(a)


def noise(rng, h, w, s):
    return aniso_noise(rng, h, w, s, s)


def fbm(rng, h, w, s, octaves=4):
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for _ in range(octaves):
        out += amp * noise(rng, h, w, max(2.0, s))
        s, amp = s / 2, amp * 0.5
    return _norm(out)


def _save(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def top_size(sx, sy, ppm=PX_PER_M):
    return int(round(sx * ppm)), int(round(sy * ppm))          # (rows, cols)


# ------------------------------------------------------------------------------------------------ table tops
def marble(path, sx, sy, base, vein, seed=0, angle=-28, density=1.0):
    """White Carrara-style marble: broad soft grey drifts, thin sharper veins along them, faint secondary veins and
    a cloudy ground."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy)
    yy, xx = np.mgrid[:h, :w].astype(np.float32) / PX_PER_M
    a = np.radians(angle)
    t = xx * np.cos(a) + yy * np.sin(a)
    turb = fbm(rng, h, w, 700, 6)
    m1 = np.abs(np.sin((t * 2.6 * density + 0.42 * turb) * np.pi))
    soft = (1 - m1) ** 5
    thin = (1 - m1) ** 40
    turb2 = fbm(rng, h, w, 400, 6)
    m2 = np.abs(np.sin((t * 7 * density + 0.6 * turb2 + 0.3) * np.pi))
    v2 = (1 - m2) ** 50
    amp = np.clip(0.55 + 0.45 * noise(rng, h, w, 260), 0, 1.3)        # veins fade in and out
    cloud = fbm(rng, h, w, 350, 4)
    k = np.clip(0.30 * soft * amp + 0.55 * thin * amp + 0.28 * v2 * np.clip(noise(rng, h, w, 200), 0, 1)
                + 0.05 * np.clip(cloud, 0, None), 0, 1)
    base, vein = np.asarray(base, np.float32), np.asarray(vein, np.float32)
    img = base * (1 + 0.010 * cloud)[..., None]
    img = img * (1 - k[..., None]) + vein * k[..., None]
    img += rng.normal(0, 1.0, (h, w, 1))
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    im.save(path)


def laminate(path, sx, sy, base, seed=0):
    """Plain plastic laminate / painted top: very low-contrast mottling, a few faint marks."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 900)
    img = np.asarray(base, np.float32) * (1 + 0.010 * noise(rng, h, w, 250) + 0.004 * noise(rng, h, w, 25))[..., None]
    img += rng.normal(0, 0.8, (h, w, 1))
    _save(img, path)


def wood(path, sx, sy, base, dark, seed=0, style="slab", strip_m=0.022, period_m=0.012, contrast=1.0):
    """Wood top with grain running left-right (along the table's y axis, the photo's horizontal).
    style 'slab': flowing flat-sawn grain (walnut); 'strips': bamboo / butcher-block strips across the depth."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy)
    base, dark = np.asarray(base, np.float32), np.asarray(dark, np.float32)
    yy = np.mgrid[:h, :w][0].astype(np.float32) / PX_PER_M
    fib = aniso_noise(rng, h, w, 1.5, 120)                    # fine fibres, long along the grain
    if style == "strips":
        n = int(np.ceil(sx / strip_m)) + 1
        idx = np.clip((yy / strip_m).astype(int), 0, n - 1)
        tone = rng.normal(0, 0.035, n)[idx]                    # each strip its own shade
        warp = 0.4 * aniso_noise(rng, h, w, 40, 900)
        t = yy / (period_m * 0.6) + warp + rng.uniform(0, 10, n)[idx]
        lines = (0.5 + 0.5 * np.sin(2 * np.pi * t)) ** 6
        k = np.clip(0.40 + tone * 3 + 0.16 * lines * contrast + 0.06 * fib, 0, 1)
        # thin glue lines between strips
        edge = np.abs((yy / strip_m) - np.round(yy / strip_m)) * strip_m * PX_PER_M < 0.8
        k = np.where(edge, np.clip(k + 0.18, 0, 1), k)
    else:
        warp = 2.2 * aniso_noise(rng, h, w, 260, 1400) + 0.5 * aniso_noise(rng, h, w, 60, 500)
        t = yy / period_m + warp * 3
        lines = (0.5 + 0.5 * np.sin(2 * np.pi * t)) ** 3
        figure = aniso_noise(rng, h, w, 300, 900)
        k = np.clip(0.45 + 0.30 * lines * contrast + 0.10 * figure + 0.08 * fib, 0, 1)
    img = base * (1 - k[..., None]) + dark * k[..., None]
    img *= (1 + 0.03 * noise(rng, h, w, 500))[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    im.save(path)


# ------------------------------------------------------------------------------------------------ room
def plaster(path, base, px=512, seed=4):
    rng = np.random.default_rng(seed)
    a = noise(rng, px, px, 60)
    img = np.asarray(base, np.float32) * (1 + 0.022 * a)[..., None] + rng.normal(0, 1.8, (px, px, 1))
    _save(img, path)


def tiles(path, base, grout, n=4, px=1024, seed=3):
    """n x n ceramic tiles with grout and a little per-tile variation (06's floor, any colour)."""
    rng = np.random.default_rng(seed)
    t = px // n
    img = np.zeros((px, px, 3), np.float32)
    base = np.asarray(base, np.float32)
    for i in range(n):
        for j in range(n):
            c = base * rng.uniform(0.96, 1.03)
            yy, xx = np.mgrid[:t, :t]
            shade = 1 + 0.03 * np.sin(xx / t * np.pi * rng.uniform(.5, 1.5) + rng.uniform(0, 6))[..., None]
            img[i * t:(i + 1) * t, j * t:(j + 1) * t] = c * shade
    img += rng.normal(0, 2.0, img.shape[:2])[..., None]
    g = 3
    for k in range(n + 1):
        a = min(k * t, px - 1)
        img[max(0, a - g // 2):a + g // 2 + 1, :] = grout
        img[:, max(0, a - g // 2):a + g // 2 + 1] = grout
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)).save(path)


def planks(path, base, dark, n=6, px=1024, seed=5):
    """Light oak floor boards (n boards across the texture), staggered end joints."""
    rng = np.random.default_rng(seed)
    base, dark = np.asarray(base, np.float32), np.asarray(dark, np.float32)
    b = px // n
    img = np.zeros((px, px, 3), np.float32)
    for i in range(n):
        warp = 1.5 * aniso_noise(rng, b, px, 30, 400)
        yy = np.mgrid[:b, :px][0].astype(np.float32)
        lines = (0.5 + 0.5 * np.sin(2 * np.pi * (yy / 9 + warp))) ** 3
        fib = aniso_noise(rng, b, px, 1.5, 80)
        k = np.clip(0.30 + 0.25 * lines + 0.06 * fib + rng.normal(0, 0.06), 0, 1)
        img[i * b:(i + 1) * b] = base * (1 - k[..., None]) + dark * k[..., None]
        j = int(rng.uniform(0.2, 0.9) * px)
        img[i * b:(i + 1) * b, j:j + 2] *= 0.88
        img[i * b:i * b + 2] *= 0.9
    _save(img, path)


# ------------------------------------------------------------------------------------------------ soft goods
def floral(path, base=(236, 232, 222), flower=(70, 104, 170), leaf=(120, 140, 100), w=1024, h=384, seed=7,
           density=1.0, scale=1.0):
    """Cotton print: small blue five-petal flowers and leaves on cream, tiling horizontally."""
    rng = np.random.default_rng(seed)
    S = 2
    im = Image.new("RGB", (w * S, h * S), tuple(int(v) for v in base))
    dr = ImageDraw.Draw(im)
    n = int(140 * density * w * h / (1024 * 384))
    for _ in range(n):
        x, y = rng.random() * w * S, rng.random() * h * S
        r = rng.uniform(7, 12) * S * scale
        c = tuple(int(v * rng.uniform(0.85, 1.1)) for v in flower)
        a0 = rng.uniform(0, 2 * np.pi)
        for xs in (x - w * S, x, x + w * S):
            for k in range(5):
                a = a0 + k * 2 * np.pi / 5
                px, py = xs + np.cos(a) * r * 0.55, y + np.sin(a) * r * 0.55
                dr.ellipse([px - r * 0.42, py - r * 0.42, px + r * 0.42, py + r * 0.42], fill=c)
            dr.ellipse([xs - r * 0.2, y - r * 0.2, xs + r * 0.2, y + r * 0.2], fill=(240, 226, 170))
            for k in range(2):
                a = rng.uniform(0, 2 * np.pi)
                lx, ly = xs + np.cos(a) * r * 1.3, y + np.sin(a) * r * 1.3
                dr.ellipse([lx - r * 0.35, ly - r * 0.18, lx + r * 0.35, ly + r * 0.18], fill=tuple(leaf))
    im = im.resize((w, h), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    a *= (1 + 0.025 * aniso_noise(rng, h, w, 2, 2))[..., None]        # weave
    _save(a, path)


def yarn(path, rgb, w=512, h=256, seed=11):
    """Yarn ball: bundles of strands wound at a few angles, with dark gaps between strands."""
    rng = np.random.default_rng(seed)
    rgb = np.asarray(rgb, np.float32)
    out = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for ang in rng.uniform(-60, 60, 5):
        a = np.radians(ang)
        t = (xx * np.sin(a) + yy * np.cos(a)) / 3.2 + 1.2 * noise(rng, h, w, 40)
        strands = (0.5 + 0.5 * np.sin(2 * np.pi * t)) ** 0.6
        band = np.clip(noise(rng, h, w, 45) + 0.2, 0, 1)
        out = np.maximum(out, strands * band)
    k = 0.55 + 0.45 * out
    _save(rgb * k[..., None] + rng.normal(0, 3, (h, w, 1)), path)


# ------------------------------------------------------------------------------------------------ more table tops
# (the photo-texture tops whose inpainting left blobs: 08 18 19 20 25 26 29 30 31 36 37 40 41 46 48)
def brushed(path, sx, sy, base, seed=0, slat_m=0.0):
    """Brushed stainless / aluminium: fine streaks along the table's y axis, soft large-scale sheen; slat_m > 0
    adds the grooves of a slatted camping top."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    streak = aniso_noise(rng, h, w, 0.8, 300)
    sheen = noise(rng, h, w, 500)
    img = np.asarray(base, np.float32) * (1 + 0.035 * streak + 0.03 * sheen)[..., None]
    if slat_m > 0:
        xx = np.mgrid[:h, :w][0].astype(np.float32) / 1000
        f = (xx / slat_m) % 1.0
        img *= np.where(f < 0.05, 0.72, np.where(f < 0.09, 1.06, 1.0))[..., None]
    img += rng.normal(0, 1.2, (h, w, 1))
    _save(img, path)


def stone(path, sx, sy, base, seed=0, contrast=1.0):
    """Honed slate / dark stone: cloudy low-contrast mottling and faint pale specks."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    cloud = fbm(rng, h, w, 260, 5)
    img = np.asarray(base, np.float32) * (1 + 0.06 * contrast * cloud + 0.02 * noise(rng, h, w, 6))[..., None]
    specks = rng.random((h, w)) < 0.0015
    img[specks] *= 1.35
    _save(img, path)


def terrazzo(path, sx, sy, base, chips=((90, 88, 86), (232, 230, 226), (150, 138, 124), (60, 58, 56)), seed=0,
             density=1.0):
    """Terrazzo / speckled concrete: a pale cement ground with small angular chips of a few stone colours."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    S = 2
    ground = np.asarray(base, np.float32) * (1 + 0.025 * fbm(rng, h, w, 200, 4))[..., None]
    im = Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8)).resize((w * S, h * S), Image.BILINEAR)
    dr = ImageDraw.Draw(im)
    n = int(9000 * density * sx * sy)
    for _ in range(n):
        x, y = rng.random() * w * S, rng.random() * h * S
        r = rng.choice([1.5, 2.5, 3.5, 5.5], p=[0.45, 0.3, 0.17, 0.08]) * S
        c = np.asarray(chips[rng.integers(len(chips))], float) * rng.uniform(0.9, 1.08)
        a0 = rng.uniform(0, 2 * np.pi)
        k = rng.integers(4, 7)
        pts = [(x + r * rng.uniform(0.6, 1.0) * np.cos(a0 + 2 * np.pi * j / k),
                y + r * rng.uniform(0.6, 1.0) * np.sin(a0 + 2 * np.pi * j / k)) for j in range(k)]
        dr.polygon(pts, fill=tuple(int(v) for v in np.clip(c, 0, 255)))
    im = im.resize((w, h), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) + rng.normal(0, 1.0, (h, w, 1))
    _save(a, path)


def tiletop(path, sx, sy, base, grout, tile_m=0.15, seed=0):
    """Tiled table top: square glazed tiles tile_m wide with grout lines, sized to the table."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    yy, xx = np.mgrid[:h, :w].astype(np.float32) / 1000
    i, j = (yy / tile_m).astype(int), (xx / tile_m).astype(int)
    tone = rng.uniform(0.97, 1.03, (i.max() + 2, j.max() + 2))[i, j]
    img = np.asarray(base, np.float32) * (tone * (1 + 0.012 * noise(rng, h, w, 120)))[..., None]
    gy = np.abs(yy / tile_m - np.round(yy / tile_m)) * tile_m
    gx = np.abs(xx / tile_m - np.round(xx / tile_m)) * tile_m
    g = np.minimum(gx, gy) < 0.0025
    img[g] = grout
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    im.save(path)


def rust(path, sx, sy, base, rust_rgb, dark, seed=0):
    """Weathered painted metal: patches of orange rust and dark pitting over a worn brown-grey ground."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    a = fbm(rng, h, w, 220, 5)
    b = fbm(rng, h, w, 60, 4)
    k1 = np.clip((a - 0.1) * 1.2, 0, 1)[..., None]
    k2 = 0.6 * np.clip((b - 1.3) * 0.6, 0, 1)[..., None]
    img = np.asarray(base, np.float32) * (1 - k1) + np.asarray(rust_rgb, np.float32) * k1
    img = img * (1 - k2) + np.asarray(dark, np.float32) * k2
    img *= (1 + 0.04 * noise(rng, h, w, 4))[..., None]
    _save(img, path)


def felt(path, sx, sy, base, seed=0):
    """Card-table felt / baize: even colour, a fine fibrous nap."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    img = np.asarray(base, np.float32) * (1 + 0.035 * noise(rng, h, w, 1.5) + 0.02 * noise(rng, h, w, 150))[..., None]
    _save(img, path)


def patchwork(path, sx, sy, base, ink, seed=0, cell_m=0.12):
    """Printed cotton tablecloth: a faded patchwork of squares with small motifs, borders and stripes."""
    rng = np.random.default_rng(seed)
    h, w = top_size(sx, sy, 1000)
    S = 2
    im = Image.new("RGB", (w * S, h * S), tuple(int(v) for v in base))
    dr = ImageDraw.Draw(im)
    c = int(cell_m * 1000 * S)
    base, ink = np.asarray(base, float), np.asarray(ink, float)
    for y in range(0, h * S, c):
        for x in range(0, w * S, c):
            k = rng.uniform(0.15, 0.55)
            col = tuple(int(v) for v in base * (1 - k) + ink * k)
            dr.rectangle([x + 6, y + 6, x + c - 6, y + c - 6], outline=col, width=5)
            pat = rng.integers(3)
            col2 = tuple(int(v) for v in base * (1 - k * 0.7) + ink * k * 0.7)
            if pat == 0:                           # dot grid
                for yy in range(y + 24, y + c - 16, 28):
                    for xx in range(x + 24, x + c - 16, 28):
                        dr.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=col2)
            elif pat == 1:                         # stripes
                for yy in range(y + 20, y + c - 14, 18):
                    dr.line([x + 18, yy, x + c - 18, yy], fill=col2, width=4)
            else:                                  # a medallion
                m = c // 2
                for r in (m * 0.55, m * 0.35, m * 0.15):
                    dr.ellipse([x + m - r, y + m - r, x + m + r, y + m + r], outline=col2, width=5)
    im = im.resize((w, h), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    a *= (1 + 0.03 * aniso_noise(rng, h, w, 2, 2) + 0.03 * noise(rng, h, w, 300))[..., None]   # weave, fading
    _save(a, path)


def grid_floor(path, size_m, px=4096, minor_m=0.1, major_m=0.5, line=(90, 150, 210), bg=(4, 5, 7), fade_m=1.45,
               pool=(13, 15, 19)):
    """Dark-studio floor: near-black with a faint light pool in the middle and a thin cool grid (major + minor
    lines) that fades out with distance.  The texture covers size_m x size_m, centred."""
    yy, xx = (np.mgrid[:px, :px].astype(np.float32) + 0.5) / px * size_m - size_m / 2
    r = np.hypot(xx, yy)
    fade = np.exp(-(r / fade_m) ** 2)
    mpp = size_m / px

    def lines(step, width_m):
        d = np.minimum(np.abs(xx / step - np.round(xx / step)), np.abs(yy / step - np.round(yy / step))) * step
        return np.clip(1 - d / max(width_m, mpp * 0.7), 0, 1)
    k = np.maximum(0.55 * lines(major_m, 0.0024), 0.16 * lines(minor_m, 0.0012)) * fade
    bgc = np.asarray(bg, np.float32) + (np.asarray(pool, np.float32) - bg) * np.exp(-(r / (fade_m * 0.8)) ** 2)[..., None]
    img = bgc * (1 - k[..., None]) + np.asarray(line, np.float32) * k[..., None]
    _save(img, path)
