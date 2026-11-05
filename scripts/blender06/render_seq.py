"""Render a frame range of 06_cycles.blend on the Metal GPU (resumable: existing frames are skipped).

blender -b footage/tables_sim/hero/exec_blender/06_cycles.blend -P scripts/blender06/render_seq.py -- \
        [--start 98] [--end 746] [--step 1] [--out DIR] [--samples N] [--thresh T]
GPU device choice lives in the user preferences, not in the .blend: this script enables Metal before rendering
(a plain `blender -b file.blend -a` would silently fall back to the CPU).
"""
import sys
import time
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(n, d=None):
    return argv[argv.index(n) + 1] if n in argv else d


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(arg("--out", str(ROOT / "footage/tables_sim/hero/exec_blender/frames")))
OUT.mkdir(parents=True, exist_ok=True)
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
scn.render.use_persistent_data = True
a, b, st = int(arg("--start", 98)), int(arg("--end", scn.frame_end)), int(arg("--step", 1))
for f in range(a, b + 1, st):
    p = OUT / f"f_{f:04d}.png"
    if p.exists() and p.stat().st_size > 0 and "--overwrite" not in argv:
        continue
    t0 = time.time()
    scn.frame_set(f)
    scn.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    print(f"FRAME {f} {time.time() - t0:.1f}s", flush=True)
