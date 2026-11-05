"""Render a refined .blend on the Metal GPU, resumable; identical frames (same physics frame, camera, fade -- the
locked-off holds and the frozen catch) are rendered once and copied.

blender -b X.blend -P scripts/refined/render_refined.py -- --shots KEY_shots.json --out DIR
        [--frames 0,114,150 | --start A --end B] [--samples N] [--thresh T] [--res 1920x1080] [--engine EEVEE]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(n, d=None):
    return argv[argv.index(n) + 1] if n in argv else d


OUT = Path(arg("--out"))
OUT.mkdir(parents=True, exist_ok=True)
SH = json.loads(Path(arg("--shots")).read_text())["frames"]
scn = bpy.context.scene
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"
prefs.get_devices()
for dv in prefs.devices:
    dv.use = dv.type != "CPU"
scn.cycles.device = "GPU"
if arg("--samples"):
    scn.cycles.samples = int(arg("--samples"))
if arg("--thresh"):
    scn.cycles.adaptive_threshold = float(arg("--thresh"))
if arg("--res"):
    w, h = arg("--res").split("x")
    scn.render.resolution_x, scn.render.resolution_y = int(w), int(h)
scn.render.use_persistent_data = True
scn.render.use_overwrite = True
if arg("--frames"):
    frames = [int(x) for x in arg("--frames").split(",")]
else:
    frames = list(range(int(arg("--start", 0)), int(arg("--end", len(SH) - 1)) + 1))


def sig(i):
    f = SH[i]
    return (f["src"], tuple(round(v, 5) for v in f["cam"]), tuple(round(v, 5) for v in f["look"]),
            round(f["lens"], 4), round(f["fade"], 4))


done = {}
for i in range(len(SH)):                       # previously rendered identical frames (resume)
    p = OUT / f"f_{i:04d}.png"
    if p.exists() and p.stat().st_size > 0:
        done.setdefault(sig(i), p)
for i in frames:
    p = OUT / f"f_{i:04d}.png"
    if p.exists() and p.stat().st_size > 0 and "--overwrite" not in argv:
        continue
    s = sig(i)
    if s in done and "--no-dedupe" not in argv and i > 0 and sig(i - 1) == s:
        shutil.copyfile(done[s], p)
        print(f"FRAME {i} copy", flush=True)
        continue
    t0 = time.time()
    scn.frame_set(i)
    scn.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    done[s] = p
    print(f"FRAME {i} {time.time() - t0:.1f}s", flush=True)
