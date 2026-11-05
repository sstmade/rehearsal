"""Cycles frames -> 06_cycles.mp4 (+ 06_cycles_labeled.mp4), frame-for-frame swappable with exec/06.mp4.

Timeline of the MuJoCo clip (747 frames @30 fps, see exec_tables.render_clip):
  frames 0..38   the AI-generated photo                          -> the same photo here
  frames 39..74  photo -> twin crossfade (smoothstep over 1.2 s)  -> same crossfade onto the Cycles twin
  frames 75..98  twin through the photo camera, nothing moves     -> one Cycles render (frame 98) reused
  frames 99..746 camera swing, arm fade-in, task, hold            -> one Cycles render per frame
Captions for the labeled version: exec_tables.caption with the exact per-frame text recorded by export_traj.py.

.venv/bin/python scripts/blender06/assemble.py [--frames DIR] [--out NAME] [--first 98]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import exec_tables as X  # noqa: E402

OUT = ROOT.parent / "footage/tables_sim/hero/exec_blender"
EXP = OUT / "_export"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(OUT / "frames"))
    ap.add_argument("--out", default="06_cycles")
    ap.add_argument("--first", type=int, default=98)
    ap.add_argument("--no-labeled", action="store_true")
    a = ap.parse_args()
    D = np.load(EXP / "06_traj.npz")
    texts = [json.loads(s) for s in D["text"]]
    NF = len(texts)
    meta = json.loads((EXP / "06_traj_check.json").read_text())
    W, H, FPS = X.W, X.H, X.FPS
    photo = Image.open(meta["photo"]).convert("RGB")
    k = W / photo.width
    photo = np.asarray(photo.resize((W, int(round(photo.height * k))), Image.LANCZOS))
    y0 = (photo.shape[0] - H) // 2
    photo = photo[y0:y0 + H].astype(np.float32)
    fr = Path(a.frames)

    def frame(i):
        p = fr / f"f_{max(i, a.first):04d}.png"
        im = Image.open(p).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        return np.asarray(im)

    twin = frame(a.first).astype(np.float32)
    ff = X.ffmpeg_exe()

    def enc(path):
        return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                 f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                                 "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                stdin=subprocess.PIPE)
    pc = enc(OUT / f"{a.out}.mp4")
    pl = None if a.no_labeled else enc(OUT / f"{a.out}_labeled.mp4")
    T_PHOTO, T_XF, T_SW0 = 1.3, 1.2, 3.3
    for i in range(NF):
        t = i / FPS
        if t < T_SW0:
            x = X._ease((t - T_PHOTO) / T_XF)
            img = photo * (1 - x) + twin * x if 0 < x < 1 else (photo if x <= 0 else twin)
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = frame(i)
        pc.stdin.write(np.ascontiguousarray(img).tobytes())
        if pl is not None:
            text, sub, alpha, note = texts[i]
            col = (150, 230, 160) if (text or "").startswith("Rehearsed in physics") else (255, 255, 255)
            lab = X.caption(img, text, sub, alpha, col, note=note)
            pl.stdin.write(np.ascontiguousarray(lab).tobytes())
    for p in (pc, pl):
        if p is not None:
            p.stdin.close()
            p.wait()
    print("wrote", OUT / f"{a.out}.mp4", NF, "frames")


if __name__ == "__main__":
    main()
