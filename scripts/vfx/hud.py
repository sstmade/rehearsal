"""Shared HUD drawing for the VFX sequences: same language as scripts/opener/compose.py
(thin SF Mono, dark translucent chips with a 2 px accent bar, corner brackets, status line top-left)."""
import math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, SS = 1920, 1080, 2
FONT_PATH = "/System/Library/Fonts/SFNSMono.ttf"
ACC = (64, 156, 255)
RED = (255, 72, 64)
GREEN = (110, 235, 160)
WHITE = (255, 255, 255)
GREY = (150, 160, 175)
FF = None


def font(px, wt=500):
    f = ImageFont.truetype(FONT_PATH, px * SS)
    try:
        f.set_variation_by_axes([wt])
    except Exception:
        pass
    return f


F = {k: font(v) for k, v in (("lab", 14), ("small", 11), ("hud", 12), ("huds", 10), ("big", 22), ("mid", 17))}
F["bigb"] = font(22, 650)
F["title"] = font(30, 400)


def rgba(c, a):
    return (int(c[0]), int(c[1]), int(c[2]), int(255 * max(0.0, min(1.0, a))))


class Layer:
    """supersampled RGBA overlay; .composite(img_float_rgb) -> float rgb"""

    def __init__(self):
        self.im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def arr(self):
        a = np.asarray(self.im).astype(np.float32) / 255.0
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        return a

    def over(self, img):
        a = self.arr()
        al = a[..., 3:4]
        return img * (1 - al) + a[..., :3] * al

    # -------------------------------------------------- primitives (coords in output px)
    def text(self, xy, s, f="lab", col=WHITE, a=1.0, anchor="lm"):
        if a <= 0.01 or not s:
            return
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=F[f], fill=rgba(col, a), anchor=anchor)

    def tlen(self, s, f="lab"):
        return F[f].getlength(s) / SS

    def chip(self, x, y, parts, a=1.0, acc=ACC, typed=1.0, pad=9, h=24, anchor="l"):
        """parts: [(text, font, colour)], drawn left to right; returns (x0, y0, x1, y1)"""
        if a <= 0.01:
            return None
        gap = 7
        ws = [self.tlen(t, f) for t, f, _ in parts]
        w = pad * 2 + sum(ws) + gap * (len(parts) - 1)
        if anchor == "r":
            x = x - w
        elif anchor == "c":
            x = x - w / 2
        x0, y0, x1, y1 = x * SS, (y - h / 2) * SS, (x + w) * SS, (y + h / 2) * SS
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=3 * SS, fill=(9, 11, 15, int(170 * a)),
                                 outline=(255, 255, 255, int(52 * a)), width=SS)
        self.d.rectangle([x0, y0 + 5 * SS, x0 + 2 * SS, y1 - 5 * SS], fill=rgba(acc, a))
        cx = x + pad
        total = sum(len(t) for t, _, _ in parts)
        budget = int(round(total * typed))
        for (t, f, c), wi in zip(parts, ws):
            n = max(0, min(len(t), budget)); budget -= len(t)
            self.text((cx, y), t[:n], f, c, a)
            cx += wi + gap
        return (x, y - h / 2, x + w, y + h / 2)

    def brackets(self, bb, col=WHITE, a=1.0, w=2, L=None, grow=0.0):
        if a <= 0.01:
            return
        x0, y0, x1, y1 = bb
        x0 -= grow; y0 -= grow; x1 += grow; y1 += grow
        L = L or min(16, 0.3 * min(x1 - x0, y1 - y0) + 5)
        c = rgba(col, a)
        for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            self.d.line([(px * SS, (py + dy * L) * SS), (px * SS, py * SS), ((px + dx * L) * SS, py * SS)],
                        fill=c, width=w * SS, joint="curve")

    def status(self, i, title, sub=None, dot=ACC, a=1.0, x=48, y=44):
        if a <= 0.01:
            return
        blink = 0.65 + 0.35 * math.cos(i * 0.35)
        self.d.ellipse([x * SS, (y - 3) * SS, (x + 6) * SS, (y + 3) * SS], fill=rgba(dot, a * blink))
        self.text((x + 14, y), title, "hud", WHITE, a * 0.9)
        if sub:
            self.text((x + 14, y + 18), sub, "huds", (170, 180, 195), a * 0.9)

    def poly(self, pts, col, a, w=2):
        if a <= 0.01 or len(pts) < 2:
            return
        self.d.line([(p[0] * SS, p[1] * SS) for p in pts], fill=rgba(col, a), width=int(w * SS), joint="curve")


def glow(img, thresh=0.75, strength=0.55, sizes=(9, 31, 81)):
    """additive bloom of the bright parts (img float 0..1)"""
    b = np.clip(img - thresh, 0, None) / (1 - thresh)
    out = np.zeros_like(img)
    small = cv2.resize(b, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    for s in sizes:
        k = max(3, int(s / 4) | 1)
        out += cv2.resize(cv2.GaussianBlur(small, (0, 0), k / 2.5), (W, H), interpolation=cv2.INTER_LINEAR)
    return img + out * strength / len(sizes)


_VIG = None


def vignette(img, k=0.35):
    global _VIG
    if _VIG is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((x - W / 2) / (W / 2), (y - H / 2) / (H / 2)) / 1.414
        _VIG = (1 - k * np.clip(r, 0, 1) ** 2.2)[..., None]
    return img * _VIG


def grain(img, i, amt=0.012):
    rng = np.random.default_rng(i)
    n = rng.normal(0, amt, (H // 2, W // 2, 1)).astype(np.float32)
    return img + cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]


def finish(img, i, bloom=True, strength=0.55, thresh=0.75):
    if bloom:
        img = glow(img, thresh=thresh, strength=strength)
    img = vignette(img)
    img = grain(img, i)
    return np.clip(img, 0, 1)


class Encoder:
    def __init__(self, path, ff, fps=30):
        self.p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def smooth(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(max(t, 0.0), 1.0); return 1 - (1 - t) ** 3


def load(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im[..., ::-1].astype(np.float32) / 255.0
