"""Drop-in assembly for an exec_dark clip re-rendered in the refined (Cycles) style, frame for frame.
  .venv/bin/python scripts/refined/assemble_dropin.py ORIG.mp4 FRAMES_DIR SHOTS.json MASK.png "u,v" OUT.mp4
Photo frames: the original clip's own photo (frame 0), smoothstep crossfade into the render at 1.3-2.5 s exactly as
exec_tables.render_clip. Caught failure: from the first frozen frame on, the freeze render with the same red glow and
contact ring (exec_tables.glow), same strength curve. No captions (the clean variant).
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exec_tables as X  # noqa: E402
orig, fdir, shots, maskp, uv, outp = sys.argv[1:7]
S = json.loads(Path(shots).read_text())
N, ff0 = len(S["frames"]), S["info"].get("first_frozen")
W, H, FPS = 1920, 1080, 30
FF = X.ffmpeg_exe()
photo = np.frombuffer(subprocess.run([FF, "-loglevel", "error", "-i", orig, "-vframes", "1", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-"], capture_output=True).stdout, np.uint8).reshape(H, W, 3)
def frame(i):
    j = min(i, ff0) if ff0 is not None else i
    im = Image.open(Path(fdir) / f"f_{j:04d}.png").convert("RGB")
    return np.asarray(im.resize((W, H), Image.LANCZOS) if im.size != (W, H) else im)
masks = ring = None
if ff0 is not None:
    m = np.asarray(Image.open(maskp).convert("L").resize((W, H), Image.NEAREST)) > 127
    masks = X.glow_masks(m)
    u, v = (float(a) for a in uv.split(","))
    ring = (u * W / 1600.0, v * H / 900.0)
enc = subprocess.Popen([FF, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                        "-preset", "medium", outp], stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    img = frame(i)
    if t < 1.3 + 1.2:
        a = X._ease((t - 1.3) / 1.2)
        img = (photo * (1 - a) + img * a).astype(np.uint8)
    if ff0 is not None and i >= ff0:
        tf = (i - ff0) / FPS
        img = X.glow(img, masks, X._ease(tf / 0.5) * (0.85 + 0.15 * np.cos(tf * 5.0)), ring)
    enc.stdin.write(np.ascontiguousarray(img).tobytes())
enc.stdin.close(); enc.wait()
print("wrote", outp, N, "frames")
