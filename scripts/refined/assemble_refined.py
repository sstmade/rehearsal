"""Cycles frames -> NN_task.mp4 (clean) + NN_task_labeled.mp4 (opener HUD style), photo_vs_render.jpg.

  .venv/bin/python scripts/refined/assemble_refined.py OUT_DIR KEY EXP_DIR FRAMES_DIR INTENT_JSON [--name NAME]

Timeline: the AI-generated photo (1.3 s) -> smoothstep crossfade (1.2 s) into the Cycles twin seen from the photo's
camera (shot frame 0) -> the shot frames.  Captions only on the labelled version (thin monospace chips, the opener's
top-left REHEARSAL · SIMULATION block, the verdict chip).  Every caption is taken from the run / intent records.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS, SS = 1920, 1080, 30, 2
FONT = "/System/Library/Fonts/SFNSMono.ttf"
ACC = (64, 156, 255)
GREEN = (120, 230, 170)
RED = (255, 98, 84)


def font(px):
    f = ImageFont.truetype(FONT, px * SS)
    try:
        f.set_variation_by_axes([500])
    except Exception:
        pass
    return f


F_HUD, F_HUDS, F_CHIP, F_SMALL, F_BIG = font(11), font(10), font(15), font(12), font(20)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def chip(d, x, y, text, a, accent=ACC, f=None, sub=None, maxw=None):
    """Thin monospace chip: dark translucent fill, hairline border, accent tick at the left. Returns its height."""
    f = f or F_CHIP
    lines = wrap(text, f, maxw) if maxw else [text]
    lh = f.size * 1.35
    w = max(f.getlength(t) for t in lines) + 26 * SS
    if sub:
        w = max(w, F_SMALL.getlength(sub) + 26 * SS)
    h = lh * len(lines) + 14 * SS + (F_SMALL.size * 1.5 if sub else 0)
    X, Y = x * SS, y * SS
    d.rounded_rectangle([X, Y, X + w, Y + h], radius=3 * SS, fill=(9, 11, 15, int(170 * a)),
                        outline=(255, 255, 255, int(52 * a)), width=SS)
    d.rectangle([X, Y + 5 * SS, X + 2 * SS, Y + h - 5 * SS], fill=(*accent, int(255 * a)))
    yy = Y + 7 * SS
    if sub:
        d.text((X + 14 * SS, yy), sub, font=F_SMALL, fill=(150, 160, 175, int(255 * a)))
        yy += F_SMALL.size * 1.5
    for t in lines:
        d.text((X + 14 * SS, yy), t, font=f, fill=(255, 255, 255, int(255 * a)))
        yy += lh
    return h / SS


def wrap(text, f, maxw):
    out, cur = [], ""
    for w_ in text.split():
        t = (cur + " " + w_).strip()
        if f.getlength(t) > maxw * SS and cur:
            out.append(cur)
            cur = w_
        else:
            cur = t
    return out + ([cur] if cur else [])


def hud(d, i, n_obj, a=1.0):
    x, y = 48 * SS, 44 * SS
    blink = 0.65 + 0.35 * math.cos(i * 0.35)
    d.ellipse([x, y - 3 * SS, x + 6 * SS, y + 3 * SS], fill=(*GREEN, int(255 * a * blink)))
    d.text((x + 14 * SS, y), "REHEARSAL  ·  SIMULATION", font=F_HUD, fill=(255, 255, 255, int(220 * a)), anchor="lm")
    d.text((x + 14 * SS, y + 18 * SS), f"OBJECTS {n_obj:02d}  ·  PHYSICS: MUJOCO", font=F_HUDS,
           fill=(170, 180, 195, int(220 * a)), anchor="lm")


def overlay(img, layer):
    L = np.asarray(layer.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
    out = img.astype(np.float32) / 255
    out = out * (1 - L[..., 3:4]) + L[..., :3] * L[..., 3:4]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main():
    out_dir, key, exp, frames = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    intent = json.loads(Path(sys.argv[5]).read_text())
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else key
    shots = json.loads((exp / f"{key}_shots.json").read_text())["frames"]
    meta = json.loads((exp / f"{key}_traj_meta.json").read_text())
    run = json.loads(Path(meta.get("run_json", "")).read_text()) if meta.get("run_json") else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    photo = Image.open(meta["photo"]).convert("RGB")
    k = W / photo.width
    photo = np.asarray(photo.resize((W, int(round(photo.height * k))), Image.LANCZOS))
    y0 = (photo.shape[0] - H) // 2
    photo = photo[y0:y0 + H]

    def fr(i):
        im = Image.open(frames / f"f_{i:04d}.png").convert("RGB")
        return np.asarray(im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS))

    twin0 = fr(0)
    # photo vs render
    blend = (photo.astype(np.float32) * 0.5 + twin0.astype(np.float32) * 0.5).astype(np.uint8)
    sheet = Image.new("RGB", (W * 3 // 2, H // 2 + 40), (12, 12, 14))
    for j, (im, lab) in enumerate(((photo, "AI-generated photo"), (twin0, "Cycles twin, photo camera"),
                                   (blend, "50 / 50 blend"))):
        sheet.paste(Image.fromarray(im).resize((W // 2, H // 2), Image.LANCZOS), (j * W // 2, 40))
        ImageDraw.Draw(sheet).text((j * W // 2 + 12, 10), lab, font=ImageFont.truetype(FONT, 20), fill=(230, 230, 235))
    sheet.save(out_dir / ("photo_vs_primitive.jpg" if name.endswith("_primitive") else "photo_vs_render.jpg"), quality=90)

    ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()

    def enc(p):
        return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                 f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf",
                                 "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(p)], stdin=subprocess.PIPE)
    pc, pl = enc(out_dir / f"{name}.mp4"), enc(out_dir / f"{name}_labeled.mp4")
    n_photo, n_xf = int(1.3 * FPS), int(1.2 * FPS)
    instr = intent["instruction"]
    q = None
    if intent.get("answer_of"):
        q0 = json.loads((Path(sys.argv[5]).parent / f"{intent['answer_of']}.json").read_text())
        q = (q0["intent"].get("question") or q0["intent"].get("reason"), intent.get("answer"))
    success = bool(meta.get("success") and meta.get("planned"))
    catch = meta.get("catch")
    acts = meta["actions"]
    n_obj = int(meta.get("n_objects", 0))
    ext = bool(run.get("demo_extension"))
    total = n_photo + n_xf + len(shots)
    first_task = next(i for i, s in enumerate(shots) if s["shot"] not in ("photo", "move"))
    for t in range(total):
        if t < n_photo:
            img = photo
        elif t < n_photo + n_xf:
            x = ease((t - n_photo) / n_xf)
            img = (photo.astype(np.float32) * (1 - x) + twin0.astype(np.float32) * x).astype(np.uint8)
        else:
            img = fr(t - n_photo - n_xf)
        pc.stdin.write(np.ascontiguousarray(img).tobytes())
        # ---- labelled
        L = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(L)
        i = t - n_photo - n_xf
        if t < n_photo + n_xf // 2:
            chip(d, 48, 44, "AI-GENERATED PHOTO", 1.0, (170, 180, 195), F_HUDS)
        else:
            hud(d, t, n_obj, 1.0)
            sh = shots[min(max(i, 0), len(shots) - 1)]
            y = 92
            a = ease((i + 20) / 20)
            y += chip(d, 48, y, f"“{instr}”", a, ACC, F_CHIP, "INSTRUCTION", maxw=560) + 8
            if q:
                y += chip(d, 48, y, q[0], a, (245, 200, 120), F_SMALL, "REHEARSAL ASKED", maxw=560) + 8
                y += chip(d, 48, y, q[1], a, ACC, F_SMALL, "ANSWER", maxw=560) + 8
            if sh["shot"] == "ff":
                chip(d, W - 140, 44, f"×{sh['speed']:.0f}  FAST", 1.0, (170, 180, 195), F_HUDS)
            if ext and i >= first_task:
                chip(d, W - 420, H - 60, "near-placement: demo extension", 0.9, (170, 180, 195), F_HUDS)
            end_i = len(shots) - int(2.5 * FPS) if success else next(
                (j for j, s in enumerate(shots) if s["shot"] == "freeze"), len(shots))
            if i >= end_i:
                av = ease((i - end_i) / 12)
                if success:
                    chip(d, 48, H - 150, "PLAN VERIFIED  ·  SENT", av, GREEN, F_BIG,
                         f"{len(acts)} MOVES CHECKED IN PHYSICS  ·  OTHER OBJECTS UNDISTURBED")
                else:
                    reason = (catch or {}).get("reason", "")
                    chip(d, 48, H - 190, "REHEARSAL DOESN'T SEND IT", av, RED, F_BIG,
                         "NO SAFE PLAN FOUND  ·  NOTHING SENT TO THE ROBOT")
                    if reason:
                        chip(d, 48, H - 108, "A direct attempt would " + reason, av, RED, F_SMALL, maxw=900)
        pl.stdin.write(np.ascontiguousarray(overlay(img, L)).tobytes())
    for p in (pc, pl):
        p.stdin.close()
        p.wait()
    print("wrote", out_dir / f"{name}.mp4", total, "frames")


if __name__ == "__main__":
    main()
