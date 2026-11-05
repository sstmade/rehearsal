"""Render style switch for the table renders (scripts/render_tables.py, scripts/exec_tables.py,
footage/tables_sim/make_cards.py).  Visual only: physics, plans and outcomes are identical in every style.

    DRESS_STYLE=dark       (default) the dressed objects and table top (SAM 3D meshes, mask-gated photo decals,
                           procedural top, soft towel) in a dark "simulation studio": near-black void, a thin cool
                           grid on a glossy black floor that fades out, key + cool rim lights, a faint glowing line
                           along the table edge, a soft vignette.
    DRESS_STYLE=primitive  the same studio, camera and lights, but every visual-only geom is hidden: only the physics
                           collision primitives are drawn, each object in one flat matte colour (pixel-aligned with
                           dark, so an editor can wipe between the two).
    DRESS_STYLE=real       the realistic room (walls, skirting, tiled / wooden floor, window light) as before.

Every geom added here has contype = conaffinity = 0 and density 0; no body or joint is added.  The lights and
materials of the realistic room are switched off, not removed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("dark", "primitive", "real")
TEXDIR = ROOT / "out" / "tables_v6" / "_mj_tex" / "studio"

GRID_SIZE = 7.0                     # m, the grid floor square (centred between the robot and the table)
EDGE_RGBA = [0.55, 0.82, 1.0, 1.0]  # the thin glowing line along the table top's edge

# flat matte object colours for the primitive style (muted, distinct, readable on black)
PALETTE = [(0.36, 0.67, 0.93), (0.96, 0.62, 0.30), (0.42, 0.80, 0.62), (0.90, 0.42, 0.48), (0.66, 0.56, 0.90),
           (0.95, 0.82, 0.38), (0.36, 0.78, 0.82), (0.86, 0.55, 0.78), (0.62, 0.76, 0.40), (0.98, 0.70, 0.62),
           (0.52, 0.62, 0.98), (0.80, 0.68, 0.52)]
PRIM_TABLE = (0.30, 0.32, 0.36)
PRIM_LEG = (0.22, 0.23, 0.26)


def style() -> str:
    s = os.environ.get("DRESS_STYLE", "dark").strip().lower()
    if s not in STYLES:
        raise ValueError(f"DRESS_STYLE={s!r}: expected one of {STYLES}")
    return s


def room() -> bool:
    return style() == "real"


def studio() -> bool:
    return style() in ("dark", "primitive")


def suffix() -> str:
    """'' for real (the original output folders), '_dark' / '_primitive' otherwise."""
    return "" if style() == "real" else "_" + style()


# ---------------------------------------------------------------------------------------------- spec stage
def _grid_png():
    TEXDIR.mkdir(parents=True, exist_ok=True)
    p = TEXDIR / "grid_floor_v4.png"
    if not p.exists():
        import dress_tex
        dress_tex.grid_floor(p, GRID_SIZE)
    return p


def wrap(sc, bs):
    """base_spec wrapper (outermost): the dark studio around whatever bs builds.  Identity for DRESS_STYLE=real."""
    if not studio():
        return bs
    import mujoco
    G = mujoco.mjtGeom

    def bs2(scene):
        spec = bs(scene)
        (cx, cy), (sx, sy) = scene["table"]["center"], scene["table"]["size"]
        wb = spec.worldbody
        # sky: near black (the gradient skybox of exec_tables, if any, is re-coloured)
        sky = None
        for t in spec.textures:
            if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                sky = t
        if sky is None:
            sky = spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                                   builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=512, height=3072)
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1, sky.rgb2 = [0.030, 0.034, 0.042], [0.004, 0.004, 0.006]
        # floor: glossy black (reflects the arm and the table a little) + the fading grid on a finite square
        spec.add_material(name="studio_black", rgba=[0.012, 0.013, 0.016, 1], specular=0.0, shininess=0.0,
                          reflectance=0.0)
        spec.add_texture(name="studio_grid_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(_grid_png()))
        gm = spec.add_material(name="studio_grid", texrepeat=[1, 1], texuniform=False, specular=0.25,
                               shininess=0.6, reflectance=0.12, emission=1.0)
        gm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "studio_grid_tex"
        fz = -0.75
        g = wb.first_geom()
        while g is not None:
            if g.type == G.mjGEOM_PLANE and g.pos[2] < -0.5:
                fz = float(g.pos[2])
                g.material, g.rgba = "studio_black", [0.5, 0.5, 0.5, 1]     # default rgba: the material's colour
            g = wb.next_geom(g)
        gx = cx * 0.5
        wb.add_geom(name="studio_grid", type=G.mjGEOM_PLANE, size=[GRID_SIZE / 2, GRID_SIZE / 2, 0.1],
                    pos=[gx, cy, fz + 0.0008], material="studio_grid", contype=0, conaffinity=0, group=1,
                    density=0)
        # a thin cool line of light along the top's edge
        import hero_dress
        ol, _th = hero_dress.top_outline(scene)
        if style() == "primitive":         # the collision box's own edge
            ol = hero_dress._rounded_rect(sx, sy, 0.002, 0.002)
        c0 = ol.mean(0)
        ring = c0 + (ol - c0) * [(sx + 0.004) / sx, (sy + 0.004) / sy]
        spec.add_material(name="studio_edge", rgba=EDGE_RGBA, emission=1.0, specular=0.0, shininess=0.0)
        for k in range(len(ring)):
            a_, b_ = ring[k], ring[(k + 1) % len(ring)]
            if np.linalg.norm(b_ - a_) < 1e-4:
                continue
            wb.add_geom(name=f"studio_edge_{k}", type=G.mjGEOM_CAPSULE, size=[0.0011, 0, 0], material="studio_edge",
                        fromto=[cx + a_[0], cy + a_[1], 0.0002, cx + b_[0], cy + b_[1], 0.0002], contype=0,
                        conaffinity=0, group=1, density=0)
        # lights: the room's lights off; a key with shadows, a soft fill and two cool rims from behind
        for lt in spec.lights:
            lt.active = False
        D, SP = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, mujoco.mjtLightType.mjLIGHT_SPOT
        # key: a soft spot from high above, aimed between the arm and the table (light falls off toward the void)
        tgt = np.array([cx * 0.6, cy, 0.0])
        src = tgt + [0.35, -0.55, 2.6]
        kd = (tgt - src) / np.linalg.norm(tgt - src)
        wb.add_light(name="studio_key", type=SP, pos=src.tolist(), dir=kd.tolist(), cutoff=50, exponent=4,
                     diffuse=[0.80, 0.80, 0.82], specular=[0.30, 0.30, 0.32], castshadow=True)
        wb.add_light(name="studio_fill", type=D, pos=[cx + 1.5, cy + 1.0, 1.5], dir=[-0.55, -0.45, -0.70],
                     diffuse=[0.07, 0.08, 0.10], specular=[0, 0, 0], castshadow=False)
        # cool rims, low and from behind (they catch silhouettes, not the top)
        wb.add_light(name="studio_rim_a", type=D, pos=[cx - 1.5, cy + 1.5, 1.0], dir=[0.70, -0.66, -0.26],
                     diffuse=[0.24, 0.34, 0.50], specular=[0.45, 0.55, 0.70], castshadow=False)
        wb.add_light(name="studio_rim_b", type=D, pos=[cx + 1.5, cy + 1.8, 1.0], dir=[-0.50, -0.83, -0.24],
                     diffuse=[0.16, 0.24, 0.38], specular=[0.35, 0.45, 0.60], castshadow=False)
        if style() == "primitive":
            for i, c in enumerate(PALETTE):
                spec.add_material(name=f"prim_{i}", rgba=[*c, 1], specular=0.12, shininess=0.25, reflectance=0)
            spec.add_material(name="prim_table", rgba=[*PRIM_TABLE, 1], specular=0.1, shininess=0.2)
            spec.add_material(name="prim_leg", rgba=[*PRIM_LEG, 1], specular=0.1, shininess=0.2)
            # four plain legs so the table box doesn't float (the dressed legs are hidden in this style)
            rnd = (scene.get("visual") or {}).get("table_top") in ("round", "oval")
            ix, iy = (sx * 0.25, sy * 0.25) if rnd else (0.06, 0.06)
            for k, (kx, ky) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                wb.add_geom(name=f"prim_leg_{k}", type=G.mjGEOM_BOX, size=[0.02, 0.02, (-0.04 - fz) / 2],
                            pos=[cx + kx * (sx / 2 - ix), cy + ky * (sy / 2 - iy), (fz - 0.04) / 2],
                            material="prim_leg", contype=0, conaffinity=0, group=1, density=0)
        return spec
    return bs2


# ---------------------------------------------------------------------------------------------- model stage
def model_fix(m, sc):
    """After compile (after hero_dress.model_fix): studio headlight and shadow settings; primitive look."""
    if not studio():
        return
    import mujoco
    m.vis.quality.shadowsize = max(int(m.vis.quality.shadowsize), 8192)
    m.vis.quality.numslices = 96            # round things render round (default 28 facets)
    m.vis.quality.numstacks = 24
    m.vis.headlight.ambient = [0.09, 0.09, 0.10]
    m.vis.headlight.diffuse = [0.13, 0.13, 0.14]
    m.vis.headlight.specular = [0.04, 0.04, 0.05]
    # SAM 3D meshes are made slightly self-lit for the room (sam3d_mj emission 0.3): less so on black
    for k in range(m.nmat):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or ""
        if nm.startswith("sam3d_") and m.mat_emission[k] > 0.15:
            m.mat_emission[k] = 0.15
    if style() == "primitive":
        primitive_fix(m, sc)


def primitive_fix(m, sc):
    """Hide every visual-only geom of the objects and the table; draw each collision primitive in a flat colour.
    The arm, its stand, the floor and the studio geoms are unchanged."""
    import mujoco
    mid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n)
    objs = {}
    for i, o in enumerate(sc["objects"]):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        if b >= 0:
            objs[b] = mid(f"prim_{i % len(PALETTE)}")
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    x0 = sc["table"]["center"][0] - sc["table"]["size"][0] / 2
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        coll = bool(m.geom_contype[g] or m.geom_conaffinity[g])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if b in objs or b == table:
            if coll:
                k = objs[b] if b in objs else mid("prim_table")
                m.geom_matid[g] = k
                m.geom_rgba[g] = m.mat_rgba[k]                 # (a non-default geom rgba overrides the material)
            else:
                m.geom_rgba[g][3] = 0.0
        elif b == 0:
            if name.startswith(("studio_", "prim_leg")) or \
                    m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and m.geom_pos[g][2] < -0.5:
                continue
            is_stand = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_pos[g][0] + m.geom_size[g][0] < x0 + 0.05
            if coll or is_stand:
                continue                                   # the robot's stand: as in dark
            m.geom_rgba[g][3] = 0.0                        # set dressing: legs, apron, rims, slab, plane texture
    if m.nflex:
        m.flex_rgba[:, 3] = 0.0                            # the soft towel: its rigid core pad shows instead
        m.flex_group[:] = 5


# ---------------------------------------------------------------------------------------------- image stage
_VIG = {}


def post(img):
    """Soft vignette (studio styles only), same for every style so dark and primitive stay matched."""
    if not studio():
        return img
    h, w = img.shape[:2]
    v = _VIG.get((h, w))
    if v is None:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
        v = 1.0 - 0.42 * np.clip((r - 0.45) / 0.55, 0, 1) ** 1.6
        _VIG[(h, w)] = v = v[..., None]
    return np.clip(img.astype(np.float32) * v, 0, 255).astype(np.uint8)
