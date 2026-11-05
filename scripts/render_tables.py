"""Render the batch_tables.py scenes (scenes/t6_NN.json) with the Franka arm and the photo-textured tabletop.

Writes out/tables_v6/render<suffix>/NN_view.png (3/4 view) and NN_photo.png (through the photo's calibrated camera).
Visual only: the texture is a non-colliding plane on the tabletop, the physics model is the shipped one.
DRESS_STYLE (scripts/dress_style.py): dark (default) -> render_dark/, primitive -> render_primitive/,
real -> render/ (the realistic room, as before).
Run: .venv/bin/python scripts/render_tables.py [01 02 ...]
"""
import inspect
import os
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rehearsal import scene as S  # noqa: E402
from rehearsal.replay import EpisodeCamera  # noqa: E402
from rehearsal.sim import HOME_Q  # noqa: E402

import dress_style  # noqa: E402

OUT = ROOT / "out" / "tables_v6" / ("render" + dress_style.suffix())
G = mujoco.mjtGeom

# build_model without the final compile, so the texture can be added to the spec
_src = inspect.getsource(S.build_model).replace("return spec.compile()", "return spec").replace(
    "def build_model(", "def build_spec(")
exec(compile(_src, S.__file__, "exec"), S.__dict__)


def model_for(sc):
    return finish(base_spec(sc))


def base_spec(sc):
    spec = S.build_spec(sc)
    vis = sc.get("visual") or {}
    (cx, cy), (sx, sy) = sc["table"]["center"], sc["table"]["size"]
    tg = spec.body("table").first_geom()
    if vis.get("table_rgba"):
        tg.rgba = vis["table_rgba"]
    if vis.get("table_texture"):
        # stored as a top view (row 0 = far edge, column 0 = left); MuJoCo's plane maps image columns to +x and
        # image rows to -y, so transpose and flip
        top = np.asarray(Image.open(ROOT / vis["table_texture"]).convert("RGB"))
        mj = OUT.parent / "_mj_tex" / f"{sc['name']}.png"
        mj.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(top.transpose(1, 0, 2)[:, ::-1])).save(mj)
        tex = spec.add_texture(name="tabletex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(mj))
        mat = spec.add_material(name="tablemat")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tabletex"
        mat.texrepeat = [1, 1]
        mat.texuniform = False
        mat.specular = 0.0
        mat.shininess = 0.0
        mat.reflectance = 0.0
        if vis.get("table_top") in ("round", "oval"):
            # round / oval top: an elliptical textured slab (visual only); the collision box stays, hidden
            tg.rgba = [0, 0, 0, 0]
            n, th = 96, 0.03
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            ex, ey = sx / 2 * np.cos(ang), sy / 2 * np.sin(ang)
            V = [[0, 0, 0.0006]] + [[x, y, 0.0006] for x, y in zip(ex, ey)] + [[x, y, -th] for x, y in zip(ex, ey)]
            T = [[0.5, 0.5]] + [[0.5 + x / sx, 0.5 + y / sy] for x, y in zip(ex, ey)] * 2
            F = []
            for k in range(n):
                a, b = 1 + k, 1 + (k + 1) % n
                F += [0, a, b, a, a + n, b, b, a + n, b + n]
            mesh = spec.add_mesh(name="tabletop_ellipse")
            mesh.uservert = np.asarray(V, float).ravel().tolist()
            mesh.usertexcoord = np.asarray(T, float).ravel().tolist()
            mesh.userface = F
            spec.worldbody.add_geom(type=G.mjGEOM_MESH, meshname="tabletop_ellipse", pos=[cx, cy, 0],
                                    material="tablemat", contype=0, conaffinity=0, group=1, density=0)
        else:
            spec.worldbody.add_geom(type=G.mjGEOM_PLANE, size=[sx / 2, sy / 2, 0.01], pos=[cx, cy, 0.0006],
                                    material="tablemat", contype=0, conaffinity=0, group=1)
        del tex
    # round / oval tops: pull the corner legs in under the ellipse
    if vis.get("table_top") in ("round", "oval"):
        g = spec.worldbody.first_geom()
        while g is not None:
            if g.type == G.mjGEOM_BOX and g.pos[2] < -0.1 and abs(g.pos[0] - cx) < sx / 2 + 0.01 \
                    and abs(g.pos[1] - cy) < sy / 2 + 0.01 and g.size[0] < 0.06:
                g.pos = [cx + (g.pos[0] - cx) * 0.55, cy + (g.pos[1] - cy) * 0.55, g.pos[2]]
            g = spec.worldbody.next_geom(g)
    # a floor big enough that no camera sees its edge (the black wedge behind the table)
    for g in [spec.worldbody.first_geom()]:
        while g is not None:
            if g.type == G.mjGEOM_PLANE and g.pos[2] < -0.5:
                g.size = [40, 40, 0.1]
            g = spec.worldbody.next_geom(g)
    # the model's single top spotlight leaves a hot spot on large tables: make it a soft directional light
    for lt in spec.worldbody.lights:
        lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        lt.diffuse = [0.27, 0.27, 0.27]
        lt.specular = [0.05, 0.05, 0.05]
    # softer, warmer room light than the default single top light
    spec.worldbody.add_light(pos=[cx - 1.0, 1.2, 1.8], dir=[0.5, -0.6, -0.7], diffuse=[0.22, 0.22, 0.21],
                             castshadow=False)
    spec.worldbody.add_light(pos=[cx + 1.5, -1.0, 1.5], dir=[-0.6, 0.4, -0.6], diffuse=[0.12, 0.12, 0.13],
                             castshadow=False)
    return spec


def finish(spec):
    m = spec.compile()
    m.vis.headlight.ambient = [0.22, 0.22, 0.22]
    m.vis.headlight.diffuse = [0.18, 0.18, 0.18]
    m.vis.headlight.specular = [0.05, 0.05, 0.05]
    m.vis.quality.shadowsize = 4096
    m.vis.quality.offsamples = 8
    # the 40 m floor inflates the scene extent; keep the shadow map on the table area (no jagged dark band far off)
    m.stat.extent = 2.5
    m.vis.map.shadowclip = 1.5
    m.vis.map.shadowscale = 0.6
    return m


def settle(m, sc):
    d = mujoco.MjData(m)
    d.qpos[:7] = HOME_Q
    d.ctrl[:7] = HOME_Q
    mujoco.mj_forward(m, d)
    for _ in range(250):
        mujoco.mj_step(m, d)
    return d


def render_view(m, d, sc, w=1200, h=900):
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, m.vis.global_.offwidth), max(h, m.vis.global_.offheight)
    (cx, cy), (sx, sy) = sc["table"]["center"], sc["table"]["size"]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [cx * 0.8, cy, 0.3]         # between the table and the arm: both fill the frame
    cam.distance = 0.78 + 0.92 * max(sx, sy)
    cam.azimuth = 152          # from across the table and above: the table fills the frame, the arm stands behind it
    cam.elevation = -46
    opt = mujoco.MjvOption()
    opt.geomgroup[4] = 1               # set dressing: the room wall behind the robot (this view looks from across)
    with mujoco.Renderer(m, h, w) as r:
        r.update_scene(d, cam, scene_option=opt)
        return dress_style.post(r.render().copy())


def render_photo(m, d, sc):
    ph = sc["photo"]
    c = ph["camera"]
    w, h = ph["width"], ph["height"]
    ec = EpisodeCamera(m, c["K"], c["R"], c["t"], w, h)
    ec.opt.geomgroup[5] = 1            # set dressing: the room wall behind the table, as in the photo
    ec.opt.geomgroup[2] = 0            # hide the arm (menagerie visual group): it stands where the photographer stood
    # and its pedestal (world boxes left of the table edge), which would show under the table in this view
    x0 = sc["table"]["center"][0] - sc["table"]["size"][0] / 2
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == 0 and m.geom_type[i] == G.mjGEOM_BOX and m.geom_pos[i][0] + m.geom_size[i][0] < x0 + 0.05:
            m.geom_rgba[i][3] = 0
    try:
        return dress_style.post(ec.render(d))
    finally:
        ec.renderer.close()


def run(nn):
    f = ROOT / "scenes" / f"t6_{nn}.json"
    sc = json.loads((f if f.exists() else OUT.parent / f"t6_{nn}.json").read_text())
    from photo_skin import dress
    bs = base_spec
    if (ROOT / "assets" / "sam3d" / sc["name"]).is_dir() and not os.environ.get("NO_SAM3D"):
        # Meta SAM 3D Objects meshes for the hero tables (visual only; the primitives still collide)
        from sam3d_mj import add_sam3d_visuals

        def bs(s):
            sp = base_spec(s)
            add_sam3d_visuals(sp, s)
            return sp
    # set dressing (visual only): room, table shape/legs, clean procedural top, templates; HERO_DRESS=0 turns it off
    import hero_dress
    import photo_skin
    bs = hero_dress.wrap(sc, bs, auto=True)
    bs = dress_style.wrap(sc, bs)          # dark studio / primitive (DRESS_STYLE); identity for real
    saved = (photo_skin.CLOTH_SOFT, photo_skin.CLOTH_RADIUS, photo_skin.CLOTH_SHRINK, photo_skin.is_cloth)
    if hero_dress.cloth_soft(sc):
        photo_skin.CLOTH_SOFT, photo_skin.CLOTH_RADIUS = True, 0.0049
    photo_skin.CLOTH_SHRINK = hero_dress.cloth_shrink(sc)
    rigid = set(hero_dress.not_cloth(sc))
    photo_skin.is_cloth = lambda o, _f=saved[3]: o["id"] not in rigid and _f(o)
    try:
        m, d = dress(sc, bs, finish, settle)
    finally:
        photo_skin.CLOTH_SOFT, photo_skin.CLOTH_RADIUS, photo_skin.CLOTH_SHRINK, photo_skin.is_cloth = saved
    hero_dress.model_fix(m, sc, auto=True)
    dress_style.model_fix(m, sc)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_view(m, d, sc)).save(OUT / f"{nn}_view.png")
    Image.fromarray(render_photo(m, d, sc)).save(OUT / f"{nn}_photo.png")
    print(nn, "ok", flush=True)


if __name__ == "__main__":
    for nn in sys.argv[1:] or sorted({p.stem[3:] for p in [*(ROOT / "scenes").glob("t6_*.json"), *OUT.parent.glob("t6_*.json")]}):
        try:
            run(nn)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(nn, "FAILED", e, flush=True)
