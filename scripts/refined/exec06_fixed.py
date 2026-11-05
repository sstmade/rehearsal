"""exec_dark/06 visual-only fix: the static spoon's drawn handle runs into the rendered (SAM 3D) spice-box rim.

Re-renders hero/exec_dark/06 from the SAME cached trajectory (no re-simulation, physics untouched) with one change
after the model is dressed: the spoon's visual-only geoms (contype = conaffinity = 0) are slid in the table plane, away
from the spice box, just far enough that the drawn spoon clears the drawn box.  Output: exec_dark/06_fixed.mp4,
06_fixed_labeled.mp4, 06_fixed_still.png (the originals are kept).
  .venv/bin/python scripts/refined/exec06_fixed.py [--diag] [--shift M]
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DRESS_STYLE", "dark")
import exec_tables as ET  # noqa: E402
import mujoco  # noqa: E402

DIAG = "--diag" in sys.argv
SHIFT = float(sys.argv[sys.argv.index("--shift") + 1]) if "--shift" in sys.argv else None
BOX_R = float(sys.argv[sys.argv.index("--box-r") + 1]) if "--box-r" in sys.argv else None   # fit the drawn box to r
INFO = {}


def world_xy(m, d, g):
    mid = m.geom_dataid[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    else:
        s = m.geom_size[g]
        V = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
    return d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T


orig_hide = ET.hide_visual


def hide_visual(m, sc):
    orig_hide(m, sc)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    bs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "spoon_1")
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "spice_box_1")
    c = d.xpos[bb][:2]
    box_vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bb and m.geom_rgba[g][3] > 0
               and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    sp_vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bs and m.geom_rgba[g][3] > 0
              and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    sp_col = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bs and (m.geom_contype[g] or m.geom_conaffinity[g])]
    bx = np.vstack([world_xy(m, d, g) for g in box_vis]) if box_vis else np.zeros((1, 3))
    r_box = float(np.max(np.hypot(*(bx[:, :2] - c).T)))
    zt = float(bx[:, 2].max())
    sp = np.vstack([world_xy(m, d, g) for g in sp_vis])
    r_sp = float(np.min(np.hypot(*(sp[:, :2] - c).T)))
    u = d.xpos[bs][:2] - c; u /= np.linalg.norm(u)
    need = max(0.0, r_box - r_sp + 0.006)
    sh = need if SHIFT is None else SHIFT
    INFO.update(box_visual_geoms=len(box_vis), spoon_visual_geoms=len(sp_vis), spoon_collision_geoms=len(sp_col),
                box_visual_r=round(r_box, 4), box_top_z=round(zt, 4), physics_box_r=0.0968,
                spoon_visual_min_r=round(r_sp, 4), spoon_z=[round(float(sp[:, 2].min()), 4), round(float(sp[:, 2].max()), 4)],
                shift_m=round(sh, 4), dir_xy=u.round(3).tolist())
    print("06 visual fix:", INFO, flush=True)
    if BOX_R is not None and r_box > BOX_R:           # shrink the drawn spice box (xy, about its body axis) to BOX_R
        k = BOX_R / r_box
        for g in box_vis:
            mid = m.geom_dataid[g]
            sl = slice(m.mesh_vertadr[mid], m.mesh_vertadr[mid] + m.mesh_vertnum[mid])
            Rg = d.geom_xmat[g].reshape(3, 3)
            Wv = d.geom_xpos[g] + m.mesh_vert[sl] @ Rg.T
            Wv[:, 0] = c[0] + (Wv[:, 0] - c[0]) * k
            Wv[:, 1] = c[1] + (Wv[:, 1] - c[1]) * k
            m.mesh_vert[sl] = (Wv - d.geom_xpos[g]) @ Rg
        INFO.update(box_render_r_after=round(BOX_R, 4), box_xy_scale=round(k, 4))
        mujoco.mj_forward(m, d)
        bx2 = np.vstack([world_xy(m, d, g) for g in box_vis])
        INFO.update(box_render_r_check=round(float(np.max(np.hypot(*(bx2[:, :2] - c).T))), 4))
        print("06 box fit:", INFO, flush=True)
    if DIAG:
        raise SystemExit(0)
    Rb = d.xmat[bs].reshape(3, 3)
    dl = Rb.T @ np.array([u[0] * sh, u[1] * sh, 0.0])           # world shift -> spoon body frame
    for g in sp_vis:
        m.geom_pos[g] += dl


ET.hide_visual = hide_visual
WORK = ET.OUT / "_06_fixed_work"
WORK.mkdir(parents=True, exist_ok=True)
rec = json.loads((ET.OUT / "06.json").read_text())
ET.OUT = WORK
r = ET.render_clip("06", rec)
for a, b in (("06.mp4", "06_fixed.mp4"), ("06_labeled.mp4", "06_fixed_labeled.mp4"), ("06_still.png", "06_fixed_still.png")):
    shutil.move(WORK / a, WORK.parent / b)
(WORK.parent / "06_fixed.json").write_text(json.dumps({"visual_fix": INFO, "render": r, "source": "06.json (same trajectory)"},
                                                       indent=1, default=float))
print("wrote", WORK.parent / "06_fixed.mp4")
