"""Shot list that reproduces an exec_tables clip frame for frame (drop-in): every export frame i -> output frame i,
with the MuJoCo clip's own camera (position, forward, fy, principal-point offset) and the arm fade of render_clip.
  .venv/bin/python scripts/refined/dropin_shots.py EXP KEY
"""
import json, sys
from pathlib import Path
import numpy as np
exp, key = Path(sys.argv[1]), sys.argv[2]
D = np.load(exp / f"{key}_traj.npz")
meta = json.loads((exp / f"{key}_traj_meta.json").read_text())
W, H, FPS = int(D["W"]), int(D["H"]), int(D["fps"])
T_PHOTO, T_XF, T_HOLD, T_SWING = 1.3, 1.2, 0.8, 4.0
t_swing0 = T_PHOTO + T_XF + T_HOLD
ease = lambda x: float(np.clip(x, 0, 1)) ** 2 * (3 - 2 * float(np.clip(x, 0, 1)))
fp = np.asarray((meta.get("catch") or {}).get("point") or [0.6, 0.0, 0.05], float)
frozen = D["frozen"].astype(bool)
out = []
for i, c in enumerate(D["cam"]):
    t = i / FPS
    C, f = c[0:3], c[3:6]
    if t < t_swing0:
        fade, shot = 0.0, "photo"
    elif t < t_swing0 + T_SWING:
        fade, shot = ease(((t - t_swing0) / T_SWING - 0.5) / 0.45), "move"
    else:
        fade, shot = 1.0, ("freeze" if frozen[i] else "wide")
    look = C + f
    out.append({"src": i, "cam": C.tolist(), "look": look.tolist(), "lens": float(c[9] * 36.0 / W),
                "focus": fp.tolist(), "fade": fade, "shot": shot, "speed": 1.0, "off": [float(c[10]), float(c[11])]})
info = {"key": key, "frames": len(out), "seconds": round(len(out) / FPS, 2), "dropin": True,
        "first_frozen": int(np.argmax(frozen)) if frozen.any() else None}
(exp / f"{key}_shots.json").write_text(json.dumps({"info": info, "frames": out}))
print(json.dumps(info))
