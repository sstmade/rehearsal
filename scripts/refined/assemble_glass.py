"""IDEA clip: frames -> idea_glass_fall.mp4 (1920x1080).  The last HOLD frames (glass on the floor) get the exec_tables
red glow on the tumbler + ring, and a red 'REJECTED' chip with the real failed check.
  .venv/bin/python scripts/refined/assemble_glass.py FRAMES N HOLD MASK.png u,v OUT.mp4 "chip"
"""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exec_tables as X  # noqa: E402
fd, nf, hold, maskp, uv, outp, chip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
W, H = 1920, 1080
m = np.asarray(Image.open(maskp).convert("L").resize((W, H), Image.NEAREST)) > 127
masks = X.glow_masks(m)
u, v = (float(a) for a in uv.split(","))
ring = (u * W / 1600, v * H / 900)
font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 34)
RED = (255, 70, 60)
enc = subprocess.Popen([X.ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", outp], stdin=subprocess.PIPE)
last = None
for i in range(nf):
    p = Path(fd) / f"f_{i:04d}.png"
    if p.exists():
        last = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS))
    img = last
    k = i - (nf - hold)
    if k >= 0:
        tf = k / 30
        img = X.glow(img, masks, X._ease(tf / 0.5) * (0.85 + 0.15 * np.cos(tf * 5.0)), ring)
        a = min(1.0, k / 10)
        im = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        A = int(255 * a)
        tw = d.textlength(chip, font=font)
        d.rounded_rectangle([48, 52, 48 + tw + 48, 120], 12, fill=(10, 10, 12, int(210 * a)), outline=RED + (A,), width=3)
        d.text((72, 64), chip, font=font, fill=RED + (A,))
        img = np.asarray(Image.alpha_composite(im, ov).convert("RGB"))
    enc.stdin.write(np.ascontiguousarray(img).tobytes())
enc.stdin.close(); enc.wait()
print("wrote", outp)
