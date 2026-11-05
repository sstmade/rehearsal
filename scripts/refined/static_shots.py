"""Static-camera shot list over a window of export frames (+ a hold).  Used for the IDEA clips.
  .venv/bin/python scripts/refined/static_shots.py EXP KEY --src A B [--hold N] [--cam x,y,z --look x,y,z --lens mm]
  [--focus x,y,z] [--sample N  (resample the window to N output frames)]
"""
import json, sys
from pathlib import Path
import numpy as np
a = sys.argv
exp, key = Path(a[1]), a[2]
g = lambda n, d=None: a[a.index(n) + 1] if n in a else d
v3 = lambda s: [float(x) for x in s.split(",")]
s0, s1 = int(a[a.index("--src") + 1]), int(a[a.index("--src") + 2])
hold = int(g("--hold", 0))
cam, look, lens = v3(g("--cam")), v3(g("--look")), float(g("--lens", 35))
focus = v3(g("--focus", g("--look")))
n = int(g("--sample", 0))
srcs = list(range(s0, s1 + 1)) if not n else [int(round(x)) for x in np.linspace(s0, s1, n)]
srcs += [s1] * hold
out = [{"src": s, "cam": cam, "look": look, "lens": lens, "focus": focus, "fade": 1.0, "shot": "wide", "speed": 1.0,
        "off": [0.0, 0.0]} for s in srcs]
(exp / f"{key}_shots.json").write_text(json.dumps({"info": {"key": key, "frames": len(out), "static": True,
                                                            "src": [s0, s1], "hold": hold}, "frames": out}))
print(key, len(out), "frames")
