"""One rehearsal variant tile: frames -> NN.mp4 (1280x720) with its real verdict: a thin border and a chip that turn
green (PASS) or red (REJECTED · first failed check) when the attempt ends (last HOLD frames).
  .venv/bin/python scripts/refined/assemble_variant.py FRAMES_DIR N_FRAMES HOLD OUT.mp4 OK|FAIL "chip text"
"""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
fd, nf, hold, outp, verdict, text = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
W, H = 1280, 720
col = (120, 230, 170) if verdict == "OK" else (255, 70, 60)
font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 22)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exec_tables as X  # noqa: E402
enc = subprocess.Popen([X.ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", outp], stdin=subprocess.PIPE)
last = None
for i in range(nf):
    p = Path(fd) / f"f_{i:04d}.png"
    im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS) if p.exists() else last
    last = im
    k = i - (nf - hold)
    if k >= 0:
        a = min(1.0, k / 8)
        im = im.copy()
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        A = int(255 * a)
        for t in range(6):
            d.rectangle([t, t, W - 1 - t, H - 1 - t], outline=col + (A,))
        tw = d.textlength(text, font=font)
        d.rounded_rectangle([24, H - 70, 24 + tw + 32, H - 26], 8, fill=(10, 10, 12, int(200 * a)), outline=col + (A,), width=2)
        d.text((40, H - 61), text, font=font, fill=col + (A,))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    enc.stdin.write(np.asarray(im).tobytes())
enc.stdin.close(); enc.wait()
print("wrote", outp)
