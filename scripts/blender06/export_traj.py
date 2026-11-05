"""Export the t6_06 hero execution clip (scripts/exec_tables.py --hq 06) as per-frame poses for a Blender re-render.

Non-invasive: exec_tables.py is NOT modified. This script imports it, points its OUT dir at a scratch folder (so
footage/tables_sim/hero/exec/06.* is never overwritten), uses the cached plan + trajectory (never --force), and
wraps two hooks:
  * Shooter.render(d, cam, arm)  -> snapshot of the MuJoCo state (bodies, geoms, flex vertices) and the camera
  * caption(img, ...)            -> called exactly once per output frame (success clip): closes the frame record
So the exported frames are the clip's own frames, at the clip's own times, from the clip's own (cloth-stepped)
MjData. The outcome JSON written to the scratch dir is compared against the shipped 06.json.

Run:  EXPORT_TRAJ=<out.npz> .venv/bin/python scripts/blender06/export_traj.py
Writes <out.npz> (per-frame arrays), <out>_model.npz (meshes, geoms, materials, flex topology), textures in
<out>_tex/, and <out>_check.json (outcome comparison).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["EXEC_HQ"] = "1"                       # identical to `exec_tables.py --hq 06`
os.environ.pop("EXEC_PLAN_ONLY", None)
os.environ.pop("DRESS_STYLE", None)               # the original (photo) dressing: same geometry and cloth

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import exec_tables as X  # noqa: E402

NN = "06"
OUTP = Path(os.environ.get("EXPORT_TRAJ", ROOT.parent / "footage/tables_sim/hero/exec_blender/_export/06_traj.npz"))
OUTP.parent.mkdir(parents=True, exist_ok=True)
SCRATCH = OUTP.parent / "_mj_clip"                # exec_tables writes its (identical) mp4s/json here, not to exec/
SCRATCH.mkdir(parents=True, exist_ok=True)
X.OUT = SCRATCH                                   # CACHE stays the shipped cache (read-only: hash matches, no force)

S = {"m": None, "pending": None, "frames": [], "init": None, "refs": {}}


def snap(m, d, cam, arm):
    C, f, up = cam.frame()
    return dict(xpos=d.xpos.copy(), xquat=d.xquat.copy(), gpos=d.geom_xpos.copy(), gmat=d.geom_xmat.copy(),
                flex=d.flexvert_xpos.copy() if m.nflexvert else np.zeros((0, 3)),
                cam=np.r_[C, f, up, cam.fy, cam.off].astype(np.float64), arm={bool(arm)})


class RecShooter(X.Shooter):
    def __init__(self, m, sc, ss=1):
        super().__init__(m, sc, ss)
        S["m"], S["sc"] = m, sc

    def render(self, d, cam, arm=True):
        img = super().render(d, cam, arm)
        s = snap(self.m, d, cam, arm)
        if S["init"] is None:                      # twin0: the photo-camera twin (arm hidden) before the loop
            S["init"] = s
        elif S["pending"] is not None and np.allclose(S["pending"]["cam"], s["cam"]):
            S["pending"]["arm"] |= s["arm"]        # same frame rendered twice (arm fade-in): union of arm flags
            S["pending"]["img_" + str(arm)] = img
        else:
            S["pending"] = s
        S["last_img"] = img
        return img


_orig_caption = X.caption


def rec_caption(img, text, sub=None, alpha=1.0, color=(255, 255, 255), note=None):
    i = len(S["frames"])
    if S["pending"] is not None:
        S["frames"].append(S["pending"])
        S["pending"] = None
    else:                                          # photo phase: no render this frame (photo / twin0 crossfade)
        s = dict(S["init"])
        s["arm"] = {False}
        s["photo"] = True
        S["frames"].append(s)
    if i % 60 == 0 or i in (98, 99, 220, 400, 600):
        S["refs"][i] = np.asarray(img).copy()     # reference MuJoCo frames, for alignment checks
    S["frames"][-1]["text"] = (text, sub, float(alpha), note)
    return _orig_caption(img, text, sub, alpha, color, note)


X.Shooter = RecShooter
X.caption = rec_caption


def export_model(m, tdir: Path):
    tdir.mkdir(exist_ok=True)
    G = mujoco.mjtGeom
    out = {}
    names = lambda obj, n: [mujoco.mj_id2name(m, obj, i) or "" for i in range(n)]
    out["body_name"] = np.array(names(mujoco.mjtObj.mjOBJ_BODY, m.nbody))
    out["geom_name"] = np.array(names(mujoco.mjtObj.mjOBJ_GEOM, m.ngeom))
    out["mat_name"] = np.array(names(mujoco.mjtObj.mjOBJ_MATERIAL, m.nmat))
    out["mesh_name"] = np.array(names(mujoco.mjtObj.mjOBJ_MESH, m.nmesh))
    for k in ("geom_type", "geom_size", "geom_dataid", "geom_matid", "geom_rgba", "geom_group", "geom_bodyid",
              "geom_pos", "geom_quat", "mat_rgba", "mat_texrepeat", "mat_texuniform", "mat_specular",
              "mat_shininess", "mat_emission", "body_parentid", "mesh_vertadr", "mesh_vertnum", "mesh_faceadr",
              "mesh_facenum", "mesh_vert", "mesh_face", "mesh_normal", "mesh_normaladr", "mesh_facenormal",
              "mesh_texcoordadr", "mesh_texcoordnum", "mesh_texcoord", "mesh_facetexcoord"):
        out[k] = np.asarray(getattr(m, k)).copy()
    out["mat_texid"] = np.asarray(m.mat_texid)[:, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)].copy()
    # textures used as RGB maps -> PNG (row order as MuJoCo stores it)
    tex_files = []
    for t in range(m.ntex):
        a, h, w, c = m.tex_adr[t], m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t) or f"tex{t}"
        if m.tex_type[t] != int(mujoco.mjtTexture.mjTEXTURE_2D) or h * w > 4096 * 4096:
            tex_files.append("")
            continue
        px = np.asarray(m.tex_data[a:a + h * w * c]).reshape(h, w, c)
        f = tdir / f"{t:02d}_{nm}.png"
        Image.fromarray(px[..., :3] if c >= 3 else px[..., 0]).save(f)
        tex_files.append(f.name)
    out["tex_file"] = np.array(tex_files)
    # flex (cloth) topology
    out["nflex"] = np.array(m.nflex)
    if m.nflex:
        for k in ("flex_vertadr", "flex_vertnum", "flex_elemadr", "flex_elemnum", "flex_elem", "flex_dim",
                  "flex_matid", "flex_texcoordadr", "flex_texcoord", "flex_elemdataadr", "flex_radius",
                  "flex_rgba"):
            if hasattr(m, k):
                out[k] = np.asarray(getattr(m, k)).copy()
    return out


def main():
    rec = X.plan_and_run(NN, force=False)
    assert rec.get("executed") and rec.get("success"), "expected the cached successful run"
    rend = X.render_clip(NN, rec)
    rec["render"] = rend
    m = S["m"]
    F = S["frames"]
    print("frames:", len(F), "clip frames:", rend["frames"], flush=True)
    assert len(F) == rend["frames"], (len(F), rend["frames"])
    arm = np.array([[False in f["arm"], True in f["arm"]] for f in F])
    np.savez_compressed(
        OUTP,
        fps=X.FPS, W=X.W, H=X.H,
        xpos=np.stack([f["xpos"] for f in F]).astype(np.float32),
        xquat=np.stack([f["xquat"] for f in F]).astype(np.float32),
        gpos=np.stack([f["gpos"] for f in F]).astype(np.float32),
        gmat=np.stack([f["gmat"] for f in F]).astype(np.float32),
        flex=np.stack([f["flex"] for f in F]).astype(np.float32),
        cam=np.stack([f["cam"] for f in F]),
        arm_off=arm[:, 0], arm_on=arm[:, 1],
        photo=np.array([bool(f.get("photo")) for f in F]),
        text=np.array([json.dumps(f["text"]) for f in F]),
    )
    mdl = export_model(m, Path(str(OUTP).replace(".npz", "_tex")))
    np.savez_compressed(str(OUTP).replace(".npz", "_model.npz"), **mdl)
    rdir = OUTP.parent / "mj_ref"
    rdir.mkdir(exist_ok=True)
    for i, img in S["refs"].items():
        Image.fromarray(img).save(rdir / f"mj_{i:04d}.png")
    # timeline metadata (for the README / compositing)
    meta = {"frames": len(F), "fps": X.FPS, "hero_cam": rec.get("hero_cam"), "render": rend,
            "photo": str(ROOT / S["sc"]["photo"]["file"]), "nbody": int(m.nbody), "ngeom": int(m.ngeom),
            "nflexvert": int(m.nflexvert)}
    # outcome check: the shipped 06.json vs what this run produced (render timing fields excluded)
    shipped = json.loads((ROOT.parent / "footage/tables_sim/hero/exec/06.json").read_text())
    skip = {"render", "total_wall_s"}
    a = {k: v for k, v in shipped.items() if k not in skip}
    b = json.loads(json.dumps({k: v for k, v in rec.items() if k not in skip}, default=float))
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    meta["outcome_identical"] = not diff
    meta["outcome_diff_keys"] = diff
    meta["shipped_render"] = shipped.get("render")
    Path(str(OUTP).replace(".npz", "_check.json")).write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
