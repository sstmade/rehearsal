"""Set dressing for the hero table (t6_06, the Indian-kitchen photo): visual only, nothing here collides or has mass.

- the top: a stainless-steel slab with the photo's rounded front corners and a brushed-steel texture
  (scripts/steel_tex.py) instead of the inpainted photo texture
- round steel legs, a lower shelf with pots (the photo shows one), a cream back wall, a beige tiled floor
- the knife and the spoon: a thin blade + black handle and a steel spoon instead of photo decals on a box
- shiny steel for the SAM 3D bowl and spice box, warm window light

Physics, planning and the recorded run are unchanged: every added geom has contype=conaffinity=0 and density 0,
and no body or joint is added.  Used by scripts/exec_tables.py (render only).  HERO_DRESS=0 turns it off.
"""
import copy
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXDIR = ROOT / "out" / "tables_v6" / "_mj_tex"

STEEL = [0.70, 0.71, 0.72, 1.0]
FLOOR_Z = -0.75
SCENES = {"t6_06", "t6_09", "t6_27", "t6_45", "t6_44", "t6_50"}     # hand-tuned (exec hero clips)


def enabled(sc, auto=False):
    """Hand-tuned scenes always; with auto=True (the 50 poster/card renders) every scene gets the automatic pass."""
    if os.environ.get("HERO_DRESS", "1") == "0":
        return False
    return sc.get("name") in SCENES or (auto and os.environ.get("AUTO_DRESS", "1") != "0")


def _floor_tiles(path, n=4, px=1024, seed=3):
    """n x n beige ceramic tiles with grout and a little per-tile variation."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    t = px // n
    img = np.zeros((px, px, 3), np.float32)
    base = np.array([178, 168, 152], np.float32)
    for i in range(n):
        for j in range(n):
            c = base * rng.uniform(0.96, 1.03)
            yy, xx = np.mgrid[:t, :t]
            shade = 1 + 0.03 * np.sin(xx / t * np.pi * rng.uniform(.5, 1.5) + rng.uniform(0, 6))[..., None]
            img[i * t:(i + 1) * t, j * t:(j + 1) * t] = c * shade
    img += rng.normal(0, 2.0, img.shape[:2])[..., None]
    g = 3
    for k in range(n + 1):
        a = min(k * t, px - 1)
        img[max(0, a - g // 2):a + g // 2 + 1, :] = [128, 120, 108]
        img[:, max(0, a - g // 2):a + g // 2 + 1] = [128, 120, 108]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    im.save(path)


def _wall_tex(path, px=512, seed=4):
    """Matte cream plaster: very soft mottling."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    g = Image.fromarray(rng.standard_normal((9, 9)).astype(np.float32), mode="F").resize((px, px), Image.BICUBIC)
    a = np.asarray(g)
    a = (a - a.mean()) / (a.std() + 1e-6)
    base = np.array([238, 229, 210], np.float32)
    img = base * (1 + 0.025 * a)[..., None] + rng.normal(0, 2.0, (px, px, 3))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def _rounded_rect(sx, sy, r_near, r_far, n=10):
    """Outline (counter-clockwise) of an sx x sy rectangle centred at 0; corners at -x (near the robot, toward the
    photo camera) rounded with r_near, at +x with r_far."""
    pts = []
    corners = [((sx / 2 - r_far, -sy / 2 + r_far), r_far, -90),   # +x,-y
               ((sx / 2 - r_far, sy / 2 - r_far), r_far, 0),      # +x,+y
               ((-sx / 2 + r_near, sy / 2 - r_near), r_near, 90),  # -x,+y
               ((-sx / 2 + r_near, -sy / 2 + r_near), r_near, 180)]  # -x,-y
    for (cx, cy), r, a0 in corners:
        for a in np.linspace(np.radians(a0), np.radians(a0 + 90), n):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.asarray(pts)


def _prism(spec, name, outline, z0, z1, uv=None):
    """Closed prism mesh: outline (N,2) polygon (convex or star-shaped around its centroid), from z0 up to z1."""
    n = len(outline)
    c = outline.mean(0)
    V = [[c[0], c[1], z1]] + [[x, y, z1] for x, y in outline] + [[x, y, z0] for x, y in outline] + \
        [[c[0], c[1], z0]]
    F = []
    for k in range(n):
        a, b = 1 + k, 1 + (k + 1) % n
        F += [0, a, b, a, a + n, b, b, a + n, b + n, 2 * n + 1, b + n, a + n]
    mesh = spec.add_mesh(name=name)
    mesh.uservert = np.asarray(V, float).ravel().tolist()
    if uv is not None:
        T = [uv(c)] + [uv(p) for p in outline] * 2 + [uv(c)]
        mesh.usertexcoord = np.asarray(T, float).ravel().tolist()
    mesh.userface = F
    return mesh


def _room():
    """The realistic room (walls, skirting, window light) only in DRESS_STYLE=real."""
    import dress_style
    return dress_style.room()


def _vis(**kw):
    return dict(contype=0, conaffinity=0, density=0, group=1, **kw)


def _wrap06(sc, bs, poster=False):
    """t6_06 (the first hero table): base_spec wrapper adding its set dressing."""
    import mujoco
    G = mujoco.mjtGeom
    hero_tex = ROOT / "photos" / "ai_gen" / "tables_v6" / f"{sc['name']}.table_tex_hero.png"
    TEXDIR.mkdir(parents=True, exist_ok=True)
    floor_png, wall_png = TEXDIR / "hero_floor.png", TEXDIR / "hero_wall.png"
    if not floor_png.exists():
        _floor_tiles(floor_png)
    if not wall_png.exists():
        _wall_tex(wall_png)

    def bs2(scene):
        scene = copy.deepcopy(scene)
        if hero_tex.exists():
            scene.setdefault("visual", {})["table_texture"] = str(hero_tex.relative_to(ROOT))
        spec = bs(scene)
        (cx, cy), (sx, sy) = scene["table"]["center"], scene["table"]["size"]
        wb = spec.worldbody

        # materials
        steel = spec.add_material(name="hero_steel", rgba=STEEL, specular=0.6, shininess=0.55, reflectance=0.0)
        del steel
        spec.add_texture(name="hero_floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(floor_png))
        fm = spec.add_material(name="hero_floor_mat", texrepeat=[1 / 1.2, 1 / 1.2], texuniform=True,
                               specular=0.15, shininess=0.3, reflectance=0.02)
        fm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hero_floor_tex"
        spec.add_texture(name="hero_wall_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(wall_png))
        wm = spec.add_material(name="hero_wall_mat", texrepeat=[1, 1], texuniform=False, specular=0.05,
                               shininess=0.05, emission=0.35 if poster else 0.0)
        wm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hero_wall_tex"
        spec.add_material(name="hero_pot", rgba=[0.86, 0.86, 0.87, 1], specular=0.8, shininess=0.7)
        spec.add_material(name="hero_black", rgba=[0.05, 0.05, 0.055, 1], specular=0.3, shininess=0.4)
        spec.add_material(name="hero_blade", rgba=[0.80, 0.81, 0.83, 1], specular=0.9, shininess=0.85)

        # tabletop: hide the flat plane and the collision box's look, add a rounded steel slab
        g = wb.first_geom()
        floor = None
        while g is not None:
            if g.type == G.mjGEOM_PLANE and g.pos[2] < -0.5:
                floor = g
            elif g.type == G.mjGEOM_PLANE and getattr(g, "material", "") == "tablemat":
                g.material, g.rgba = "", [0, 0, 0, 0]
            elif g.type == G.mjGEOM_BOX and g.pos[2] < -0.1 and g.size[0] < 0.06 and g.size[1] < 0.06 \
                    and abs(g.pos[0] - cx) < sx / 2 and abs(g.pos[1] - cy) < sy / 2:
                g.type, g.size = G.mjGEOM_CYLINDER, [0.019, g.size[2], 0]          # round steel tube legs
                g.material, g.rgba = "hero_steel", [1, 1, 1, 1]
            g = wb.next_geom(g)
        if floor is not None:
            floor.material, floor.rgba = "hero_floor_mat", [1, 1, 1, 1]
        tb = spec.body("table")
        if tb is not None:
            tg = tb.first_geom()
            while tg is not None:
                tg.rgba = [0, 0, 0, 0]
                tg = tb.next_geom(tg)
        outline = _rounded_rect(sx, sy, 0.075, 0.015)
        _prism(spec, "hero_top", outline, -0.035, 0.0006, uv=lambda p: [0.5 + p[0] / sx, 0.5 + p[1] / sy])
        wb.add_geom(type=G.mjGEOM_MESH, meshname="hero_top", pos=[cx, cy, 0], material="tablemat", **_vis())
        # rolled steel lip under the edge (a slightly larger, thin, darker band)
        _prism(spec, "hero_lip", _rounded_rect(sx + 0.008, sy + 0.008, 0.079, 0.019), -0.045, -0.033)
        wb.add_geom(type=G.mjGEOM_MESH, meshname="hero_lip", pos=[cx, cy, 0], material="hero_steel",
                    **_vis())

        # lower shelf with pots and a basket (as in the photo)
        zs = FLOOR_Z + 0.17
        wb.add_geom(type=G.mjGEOM_BOX, size=[sx / 2 - 0.035, sy / 2 - 0.035, 0.008], pos=[cx, cy, zs],
                    material="hero_steel", **_vis())
        for (dx, dy, r, h) in [(-0.02, -0.30, 0.10, 0.07), (0.02, 0.0, 0.11, 0.08), (0.0, 0.30, 0.09, 0.055)]:
            wb.add_geom(type=G.mjGEOM_CYLINDER, size=[r, h / 2, 0], pos=[cx + dx, cy + dy, zs + 0.008 + h / 2],
                        material="hero_pot", **_vis())
        # the stand the robot sits on is wood-free here: nothing to change

        # room: a cream back wall behind the far edge, and a side wall on the window (+y) side
        # (DRESS_STYLE=dark|primitive: no room, scripts/dress_style.py puts the table in a dark studio)
        wk = dict(_vis())
        room = _room()
        if poster:            # poster view looks from across the table: this wall only in the photo view (group 5)
            wk["group"] = 5
        if room:
            wb.add_geom(type=G.mjGEOM_BOX, size=[0.01, 2.2, 1.05], pos=[cx + sx / 2 + 0.03, cy, FLOOR_Z + 1.05],
                        material="hero_wall_mat", **wk)
            # skirting
            wb.add_geom(type=G.mjGEOM_BOX, size=[0.012, 2.2, 0.04], pos=[cx + sx / 2 + 0.02, cy, FLOOR_Z + 0.04],
                        rgba=[0.62, 0.57, 0.50, 1], **wk)
        if poster and room:            # and a wall behind the robot for the poster view (group 4)
            wk["group"] = 4
            wb.add_geom(type=G.mjGEOM_BOX, size=[0.01, 2.6, 1.1], pos=[-0.86, cy, FLOOR_Z + 1.1],
                        material="hero_wall_mat", **wk)
            wb.add_geom(type=G.mjGEOM_BOX, size=[0.012, 2.6, 0.04], pos=[-0.848, cy, FLOOR_Z + 0.04],
                        rgba=[0.62, 0.57, 0.50, 1], **wk)

        # knife: blade + handle instead of a 3 cm tall box
        for o in scene["objects"]:
            b = spec.body(o["id"])
            if b is None:
                continue
            cat = (o.get("category", "") + " " + o.get("label", "")).lower()
            sh = o["shape"]
            if "knife" in cat and sh.get("type") == "box":
                L, Wd, Hh = sh["size"]
                z = -Hh / 2
                hl = 0.45 * L                                  # handle length (handle at -x, blade at +x)
                b.add_geom(name="hd_knife_handle", type=G.mjGEOM_CAPSULE, size=[0.010, hl / 2 - 0.01, 0],
                           pos=[-L / 2 + hl / 2, 0, z + 0.010], quat=[0.7071068, 0, 0.7071068, 0],
                           material="hero_black", **_vis())
                bl = L - hl
                blade = np.array([[0, -0.011], [bl * 0.72, -0.012], [bl, 0.004], [bl * 0.8, 0.010], [0, 0.011]])
                _prism(spec, "hd_knife_blade_mesh", blade, -0.0008, 0.0008)
                b.add_geom(name="hd_knife_blade", type=G.mjGEOM_MESH, meshname="hd_knife_blade_mesh",
                           pos=[-L / 2 + hl, 0, z + 0.006], material="hero_blade", **_vis())
            elif "spoon" in cat and sh.get("type") == "box":
                L, Wd, Hh = sh["size"]
                b.add_geom(name="hd_spoon_handle", type=G.mjGEOM_CAPSULE, size=[0.0035, L * 0.33, 0],
                           pos=[L * 0.12, 0, 0], quat=[0.7071068, 0, 0.7071068, 0], material="hero_blade",
                           **_vis())
                b.add_geom(name="hd_spoon_bowl", type=G.mjGEOM_ELLIPSOID, size=[0.022, 0.014, 0.005],
                           pos=[-L / 2 + 0.022, 0, -0.002], material="hero_blade", **_vis())
        _bowl_shells(spec, scene)
        _plate_rims(spec, scene)
        _mug_handles(spec, scene)

        # warm window light from the +y / far side (the photo's window is upper left), with shadows
        if room:
            wb.add_light(pos=[cx - 0.4, cy + 1.6, 1.9], dir=[0.25, -0.7, -0.65], diffuse=[0.42, 0.37, 0.30],
                         specular=[0.25, 0.22, 0.18], castshadow=True, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        return spec
    return bs2


def _model_fix06(m, sc):
    """t6_06 after compile: steel looks like steel, knife/spoon decals off, warmer room light."""
    import mujoco
    for oid in ("mixing_bowl_1", "spice_box_1"):
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"sam3d_{oid}_mat")
        if k >= 0:
            m.mat_specular[k], m.mat_shininess[k], m.mat_reflectance[k] = 0.7, 0.75, 0.0
    # the SAM 3D bowl's texture carries the photo's warm light: pull it to neutral steel grey
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "sam3d_mixing_bowl_1_tex")
    if t >= 0:
        a, h, w, c = m.tex_adr[t], m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
        px = m.tex_data[a:a + h * w * c].reshape(-1, c)
        L = px[:, :3].astype(float) @ [0.299, 0.587, 0.114]
        px[:, :3] = np.clip(np.stack([L * 0.98, L * 1.0, L * 1.03], 1) * 1.05, 0, 255).astype(np.uint8)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "tablemat")
    if k >= 0:
        m.mat_specular[k], m.mat_shininess[k] = 0.35, 0.45
    for o in sc["objects"]:
        cat = (o.get("category", "") + " " + o.get("label", "")).lower()
        if not (("knife" in cat or "spoon" in cat) and o["shape"].get("type") == "box"):
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") \
                    .startswith("hd_"):
                m.geom_rgba[g][3] = 0.0
    m.vis.headlight.ambient = [0.30, 0.29, 0.27]
    m.vis.headlight.diffuse = [0.20, 0.195, 0.185]
    glassify(m, sc)


# =================================================================================================================
# Config-driven dressing: the other hand-tuned hero tables (HERO) and the automatic pass for the rest (auto_config).
# Same rules as t6_06: visual only (contype=conaffinity=0, density 0, no bodies or joints added).
# =================================================================================================================
import re  # noqa: E402

DRESSDIR = TEXDIR / "dress"


def _c(rgb):
    return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0]


HONEY = (178, 120, 66)
WALNUT = (92, 58, 38)
RATTAN = (204, 168, 118)
WINDOW = dict(pos=[-0.4, 1.6, 1.9], dir=[0.25, -0.7, -0.65], diffuse=[0.42, 0.37, 0.30], specular=[0.25, 0.22, 0.18])

HERO = {
    # white Carrara marble, thick rounded slab on a honey-oak frame with a slatted lower shelf; cream wall, tiles
    "t6_09": dict(top=dict(kind="marble", base=(214, 213, 211), vein=(112, 117, 130), seed=9, angle=-30, density=1.3,
                           shape=("rrect", 0.20, 0.20), thick=0.04, spec=(0.45, 0.7)),
                  legs=dict(style="square", w=0.055, rgb=HONEY, inset=(0.10, 0.12)),
                  apron=dict(h=0.07, rgb=HONEY), shelf=dict(kind="slats", z=0.20, rgb=HONEY),
                  wall=(234, 226, 214), skirt=(240, 237, 230), floor=("tiles", (200, 188, 172), 3),
                  cloth_soft=True, az=(-110,),
                  sam_mat={"bowl_1": "ceramic", "mug_1": "ceramic", "egg_1": "egg", "egg_2": "egg", "egg_3": "egg"},
                  tex_fix={"bowl_1": (0.35, 1.04)}),
    # white laminate folding table: thin top, grey edge band, grey steel legs; white wall, beige tiles
    "t6_27": dict(top=dict(kind="laminate", base=(236, 236, 233), seed=27, shape=("rrect", 0.035, 0.035),
                           thick=0.022, spec=(0.25, 0.35), edge=dict(rgb=(150, 150, 152), h=0.024)),
                  legs=dict(style="square", w=0.026, rgb=(120, 122, 126), inset=(0.04, 0.05), metal=True),
                  apron=dict(h=0.035, rgb=(130, 132, 136), metal=True),
                  wall=(238, 236, 231), skirt=(228, 226, 220), floor=("tiles", (206, 196, 182), 3),
                  az=(-110,), sam_mat={"bucket_1": "plastic", "spray_bottle_1": "clear", "sponge_1": "matte"},
                  tex_fix={"spray_bottle_1": (0.2, 1.15)},
                  flatten=("rubber_gloves_1", "cleaning_cloth_1", "keys_1", "coaster_1")),
    # bamboo strip top with rounded front corners on a black steel frame; cream wall, beige tiles
    "t6_45": dict(top=dict(kind="wood", style="strips", base=(216, 172, 122), dark=(178, 126, 76), seed=45,
                           strip_m=0.021, shape=("rrect", 0.07, 0.035), thick=0.024, spec=(0.30, 0.45)),
                  legs=dict(style="square", w=0.028, rgb=(22, 22, 24), inset=(0.045, 0.05), metal=True),
                  apron=dict(h=0.04, rgb=(24, 24, 26), metal=True),
                  wall=(236, 232, 224), skirt=(230, 226, 218), floor=("tiles", (204, 194, 178), 3), az=(-105,),
                  tmpl_rgb={"screwdriver_1": (230, 180, 30)}),
    # walnut kidney coffee-table top on splayed round legs; white wall and skirting, light oak floor
    "t6_44": dict(top=dict(kind="wood", style="slab", base=(124, 80, 54), dark=(64, 36, 22), seed=44,
                           period_m=0.016, shape=("kidney", 0.05), thick=0.035, spec=(0.40, 0.6)),
                  legs=dict(style="tapered", w=0.026, rgb=WALNUT, inset=(0.14, 0.16), splay=0.05),
                  wall=(240, 236, 229), skirt=(244, 242, 238), floor=("planks", (214, 178, 136), (170, 128, 88)),
                  az=(-105,), cloth_soft=True, not_cloth=("knitting_needle_1", "knitting_needle_2"),
                  flatten=("coaster_1", "keychain_1"), cloth_shrink={"knit_fabric_1": 0.6}),
    # light plywood tray top with a rattan rim, rattan legs and stretchers; cream wall, pale tiles
    "t6_50": dict(top=dict(kind="wood", style="slab", base=(204, 176, 140), dark=(170, 138, 102), seed=50,
                           period_m=0.02, contrast=0.5, shape=("rrect", 0.06, 0.06), thick=0.02, spec=(0.2, 0.3),
                           rim=dict(r=0.016, rgb=RATTAN)),
                  legs=dict(style="rattan", w=0.021, rgb=RATTAN, inset=(0.045, 0.05)),
                  wall=(236, 233, 226), skirt=(222, 212, 196), floor=("tiles", (210, 202, 190), 3), az=(-110,),
                  flatten=("sunglasses_1", "cable_1", "key_1", "coaster_1", "map_1")),
}

# category -> look (regex on "category label"), for SAM 3D materials and decal-less templates
MAT_LOOK = {"metal": (0.75, 0.75), "ceramic": (0.35, 0.55), "clear": (0.6, 0.85), "plastic": (0.3, 0.45),
            "egg": (0.08, 0.15), "matte": (0.0, 0.02)}
CAT_LOOK = [(r"steel|metal|knife|spoon|fork|pot\b|pan\b|kettle|scissors|tin\b|\bcan\b|wrench|pliers|key", "metal"),
            (r"glass|jar|vase|wine|tumbler", "clear"),
            (r"mug|cup|plate|bowl|ceramic|porcelain|teapot|saucer", "ceramic"),
            (r"bucket|bottle|plastic|container|tupperware|lid|box", "plastic")]


def look_of(o):
    t = (o.get("category", "") + " " + o.get("label", "")).lower()
    for rx, look in CAT_LOOK:
        if re.search(rx, t):
            return look
    return None


def config(sc, auto=True):
    name = sc.get("name")
    if name in HERO:
        return HERO[name]
    return auto_config(sc) if auto else None


def cloth_soft(sc):
    """Soft towel (photo_skin.CLOTH_SOFT) only where configured: nothing may rest on the cloth."""
    if sc.get("name") == "t6_06":
        return enabled(sc)
    c = config(sc) if enabled(sc) else None
    return bool(c and c.get("cloth_soft"))


def not_cloth(sc):
    """Objects the cloth regex catches but that are rigid things (knitting needles)."""
    if not enabled(sc):
        return ()
    c = config(sc, auto=False) or {}
    return tuple(c.get("not_cloth", ()))


def cloth_shrink(sc):
    c = config(sc, auto=False) if enabled(sc) else None
    return dict((c or {}).get("cloth_shrink", {}))


def hero_azs(sc):
    c = HERO.get(sc.get("name"))
    return c.get("az") if c and enabled(sc) else None


# ------------------------------------------------------------------------------------------------ outlines
def _kidney(sx, sy, depth, n=120):
    """Bean outline: a superellipse with a soft concave dip on the near (-x, photo-camera) edge."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    e = 2 / 3.2
    x = sx / 2 * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** e
    y = sy / 2 * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** e
    dip = depth * np.exp(-(y / (0.28 * sy)) ** 2) * (x < 0) * np.clip(-x / (sx / 2), 0, 1)
    return np.c_[x + dip, y]


def outline_for(shape, sx, sy):
    if shape[0] == "ellipse":
        a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        return np.c_[sx / 2 * np.cos(a), sy / 2 * np.sin(a)]
    if shape[0] == "kidney":
        return _kidney(sx, sy, shape[1])
    r1, r2 = min(shape[1], sx / 2 - 1e-3, sy / 2 - 1e-3), min(shape[2], sx / 2 - 1e-3, sy / 2 - 1e-3)
    return _rounded_rect(sx, sy, r1, r2)


def poking_out(sc, outline, margin=0.004):
    """Objects whose footprint leaves the dressed outline (they would look like they float)."""
    from matplotlib.path import Path as MPath
    from rehearsal.scene import footprint_polygon
    (cx, cy) = sc["table"]["center"]
    P = MPath(outline)
    bad = []
    for o in sc["objects"]:
        fp = np.asarray(footprint_polygon(o)) - [cx, cy]
        c = fp.mean(0)
        fp = c + (fp - c) * 0.85               # the footprint corners of round-ish things are generous
        if not P.contains_points(fp, radius=-margin).all() and not P.contains_points(fp, radius=margin).all():
            bad.append(o["id"])
    return bad


def safe_outline(sc, shape):
    """Shrink corner radii / the kidney dip until no object pokes out."""
    (sx, sy) = sc["table"]["size"]
    shape = list(shape)
    for _ in range(12):
        ol = outline_for(shape, sx, sy)
        if not poking_out(sc, ol):
            return ol, shape
        shape = [shape[0]] + [v * 0.75 for v in shape[1:]]
    if shape[0] == "ellipse":
        return outline_for(shape, sx, sy), shape
    return outline_for([shape[0]] + [0.0] * (len(shape) - 1), sx, sy) if shape[0] == "rrect" else \
        _rounded_rect(sx, sy, 0.005, 0.005), shape


# ------------------------------------------------------------------------------------------------ textures
def _top_texture(sc, cfg):
    import dress_tex as T
    t = cfg["top"]
    if t["kind"] == "photo":
        return None
    (sx, sy) = sc["table"]["size"]
    DRESSDIR.mkdir(parents=True, exist_ok=True)
    import hashlib
    key = hashlib.md5((repr(sorted((k, str(v)) for k, v in t.items() if k != "shape")) + repr((sx, sy)))
                      .encode()).hexdigest()[:8]
    path = DRESSDIR / f"{sc['name']}_top_{key}.png"
    if path.exists():
        return path
    if t["kind"] == "marble":
        T.marble(path, sx, sy, t["base"], t["vein"], seed=t.get("seed", 0), angle=t.get("angle", -28),
                 density=t.get("density", 1.0))
    elif t["kind"] == "laminate":
        T.laminate(path, sx, sy, t["base"], seed=t.get("seed", 0))
    elif t["kind"] == "wood":
        T.wood(path, sx, sy, t["base"], t["dark"], seed=t.get("seed", 0), style=t.get("style", "slab"),
               strip_m=t.get("strip_m", 0.022), period_m=t.get("period_m", 0.012), contrast=t.get("contrast", 1.0))
    elif t["kind"] == "steel":
        T.laminate(path, sx, sy, t["base"], seed=t.get("seed", 0))
    elif t["kind"] == "brushed":
        T.brushed(path, sx, sy, t["base"], seed=t.get("seed", 0), slat_m=t.get("slat_m", 0.0))
    elif t["kind"] == "stone":
        T.stone(path, sx, sy, t["base"], seed=t.get("seed", 0), contrast=t.get("contrast", 1.0))
    elif t["kind"] == "terrazzo":
        T.terrazzo(path, sx, sy, t["base"], seed=t.get("seed", 0))
    elif t["kind"] == "tiletop":
        T.tiletop(path, sx, sy, t["base"], t["grout"], tile_m=t.get("tile_m", 0.15), seed=t.get("seed", 0))
    elif t["kind"] == "rust":
        T.rust(path, sx, sy, t["base"], t["rust"], t["dark"], seed=t.get("seed", 0))
    elif t["kind"] == "felt":
        T.felt(path, sx, sy, t["base"], seed=t.get("seed", 0))
    elif t["kind"] == "patchwork":
        T.patchwork(path, sx, sy, t["base"], t["ink"], seed=t.get("seed", 0))
    return path


def _room_textures(cfg, name):
    import dress_tex as T
    DRESSDIR.mkdir(parents=True, exist_ok=True)
    wall = DRESSDIR / ("wall_%d_%d_%d.png" % tuple(cfg["wall"]))
    if not wall.exists():
        T.plaster(wall, cfg["wall"])
    fl = cfg["floor"]
    if fl[0] == "tiles":
        floor = DRESSDIR / ("tiles_%d_%d_%d_%d.png" % (*fl[1], fl[2]))
        if not floor.exists():
            grout = tuple(int(v * 0.72) for v in fl[1])
            T.tiles(floor, fl[1], grout, n=fl[2])
    else:
        floor = DRESSDIR / ("planks_%d_%d_%d.png" % tuple(fl[1]))
        if not floor.exists():
            T.planks(floor, fl[1], fl[2])
    return wall, floor


# ------------------------------------------------------------------------------------------------ templates
THIN = r"\bknife|\bspoon|\bfork\b|\bpen\b|\bpencil|\bmarker|screwdriver|scissors|chopstick|paint ?brush|" \
       r"tooth ?brush|make-?up brush|\bcrayon"


def _templates(spec, scene, mats):
    """Thin tools drawn as a primitive box with photo decals -> a simple shaped stand-in (visual only).
    The long axis is the box's longer side; the handle sits at -long unless the object has "hd_flip"."""
    import mujoco
    G = mujoco.mjtGeom
    done = []
    for o in scene["objects"]:
        sh = o["shape"]
        t = (o.get("category", "") + " " + o.get("label", "")).lower()
        m_ = re.search(THIN, t)
        if not m_ or sh.get("type") != "box":
            continue
        if sh["size"][2] > 0.5 * max(sh["size"][:2]) or \
                (ROOT / "assets" / "sam3d" / scene["name"] / f"{o['id']}.glb").exists():
            continue                                   # standing in a jar / has its SAM 3D mesh: leave it
        b = spec.body(o["id"])
        if b is None:
            continue
        L, Wd, Hh = sh["size"]
        along_y = Wd > L
        if along_y:
            L, Wd = Wd, L
        s = -1.0 if o["id"] in scene.get("_hd_flip", ()) else 1.0
        z0 = -Hh / 2
        kind = m_.group(0).strip()
        if "brush" in kind or kind == "crayon":
            kind = "pen"
        if o["id"] not in mats and kind in ("screwdriver", "pen", "pencil", "marker", "chopstick"):
            rgb = tuple(int(c_ * 255) for c_ in o.get("rgba", [0.9, 0.7, 0.1])[:3])
            spec.add_material(name=f"hd_t_{o['id']}", rgba=_c(rgb), specular=0.35, shininess=0.5)
            mats[o["id"]] = f"hd_t_{o['id']}"
        rot = [0.7071068, 0, 0, 0.7071068] if along_y else [1, 0, 0, 0]

        def P(x, y=0.0, z=0.0):
            x *= s
            return [-y, x, z] if along_y else [x, y, z]

        def axq():   # capsule/cylinder axis along the long side
            return [0.5, 0.5, 0.5, 0.5] if along_y else [0.7071068, 0, 0.7071068, 0]
        gs = []
        if kind == "knife":
            hl = 0.45 * L
            r = min(0.010, Hh / 2 + 0.002)
            gs.append(dict(type=G.mjGEOM_CAPSULE, size=[r, hl / 2 - r, 0], pos=P(-L / 2 + hl / 2, 0, z0 + r),
                           quat=axq(), material=mats["black"]))
            bl = L - hl
            blade = np.array([[0, -0.011], [bl * 0.72, -0.012], [bl, 0.004], [bl * 0.8, 0.010], [0, 0.011]])
            blade[:, 0] *= s
            nm = f"hd_{o['id']}_blade"
            _prism(spec, nm, blade if not along_y else blade[:, ::-1] * [-1, 1], -0.0008, 0.0008)
            gs.append(dict(type=G.mjGEOM_MESH, meshname=nm, pos=P(-L / 2 + hl, 0, z0 + 0.006),
                           material=mats["blade"]))
        elif kind in ("spoon", "fork"):
            gs.append(dict(type=G.mjGEOM_CAPSULE, size=[0.0035, L * 0.33, 0], pos=P(-L * 0.12, 0, z0 + 0.004),
                           quat=axq(), material=mats["blade"]))
            if kind == "spoon":
                gs.append(dict(type=G.mjGEOM_ELLIPSOID, size=[0.022, 0.014, 0.005], pos=P(L / 2 - 0.022, 0, z0 + 0.005),
                               quat=rot, material=mats["blade"]))
            else:
                for k in (-1.5, -0.5, 0.5, 1.5):
                    gs.append(dict(type=G.mjGEOM_BOX, size=[0.02, 0.0014, 0.0012],
                                   pos=P(L / 2 - 0.02, k * 0.0045, z0 + 0.004), quat=rot, material=mats["blade"]))
        elif kind == "screwdriver":
            hl = 0.42 * L
            r = min(0.016, max(0.008, Hh / 2))
            col = mats.get(o["id"], mats["handle"])
            gs.append(dict(type=G.mjGEOM_CAPSULE, size=[r, hl / 2 - r, 0], pos=P(-L / 2 + hl / 2, 0, z0 + r),
                           quat=axq(), material=col))
            gs.append(dict(type=G.mjGEOM_CYLINDER, size=[r * 1.04, hl * 0.18, 0], pos=P(-L / 2 + hl * 0.62, 0, z0 + r),
                           quat=axq(), material=mats["black"]))
            sl = L - hl
            gs.append(dict(type=G.mjGEOM_CYLINDER, size=[0.0032, sl / 2, 0], pos=P(-L / 2 + hl + sl / 2, 0, z0 + r),
                           quat=axq(), material=mats["blade"]))
        elif kind in ("pen", "pencil", "marker", "chopstick", "brush"):
            r = max(0.004, min(Wd, Hh) / 2)
            gs.append(dict(type=G.mjGEOM_CAPSULE, size=[r, L / 2 - r, 0], pos=P(0, 0, z0 + r), quat=axq(),
                           material=mats.get(o["id"], mats["handle"])))
        elif kind == "scissors":
            bl = 0.55 * L
            for sgn in (-1, 1):
                blade = np.array([[0, -0.006], [bl, -0.001], [bl, 0.001], [0, 0.006]]) + [0, sgn * 0.003]
                blade[:, 0] *= s
                nm = f"hd_{o['id']}_b{sgn}"
                _prism(spec, nm, blade if not along_y else blade[:, ::-1] * [-1, 1], -0.0008, 0.0008)
                gs.append(dict(type=G.mjGEOM_MESH, meshname=nm, pos=P(L / 2 - bl, 0, z0 + 0.004 + 0.0018 * sgn),
                               material=mats["blade"]))
                gs.append(dict(type=G.mjGEOM_ELLIPSOID, size=[0.018, 0.011, 0.004],
                               pos=P(-L / 2 + 0.02, sgn * 0.013, z0 + 0.004), quat=rot, material=mats["black"]))
        for i, g in enumerate(gs):
            b.add_geom(name=f"hd_{o['id']}_{i}", **g, **_vis())
        if gs:
            done.append(o["id"])
    return done


def _revolve(spec, name, prof, nt=48, nz=12, two_sided=True, ripple=None):
    """Surface of revolution: prof(v) -> (r, z) for v in [0, 1]; ripple(theta, v) -> radius factor. UV = (theta, v)."""
    V, T, F = [], [], []
    for j in range(nz + 1):
        v = j / nz
        r, z = prof(v)
        for i in range(nt + 1):
            th = 2 * np.pi * i / nt
            rr = r * (ripple(th, v) if ripple else 1.0)
            V.append([rr * np.cos(th), rr * np.sin(th), z])
            T.append([i / nt, 1 - v])
    W = nt + 1
    for j in range(nz):
        for i in range(nt):
            a, b, c, d = j * W + i, j * W + i + 1, (j + 1) * W + i, (j + 1) * W + i + 1
            F += [a, b, d, a, d, c]
            if two_sided:
                F += [a, d, b, a, c, d]
    mesh = spec.add_mesh(name=name)
    mesh.uservert = np.asarray(V, float).ravel().tolist()
    mesh.usertexcoord = np.asarray(T, float).ravel().tolist()
    mesh.userface = F
    import mujoco
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return mesh


def _shell(spec, name, outline, z0, z1, nz=10):
    """Side wall through the closed outlines outline(v) (N,2) at heights z0..z1, two-sided, UV = (k/N, v)."""
    V, T, F = [], [], []
    n = None
    for j in range(nz + 1):
        v = j / nz
        ol = outline(v)
        n = len(ol)
        for i in range(n + 1):
            x, y = ol[i % n]
            V.append([x, y, z0 + (z1 - z0) * v])
            T.append([i / n * 3, 1 - v])
    W = n + 1
    for j in range(nz):
        for i in range(n):
            a, b, c, d = j * W + i, j * W + i + 1, (j + 1) * W + i, (j + 1) * W + i + 1
            F += [a, b, d, a, d, c, a, d, b, a, c, d]
    mesh = spec.add_mesh(name=name)
    mesh.uservert = np.asarray(V, float).ravel().tolist()
    mesh.usertexcoord = np.asarray(T, float).ravel().tolist()
    mesh.userface = F
    import mujoco
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return mesh


def _texmat(spec, name, png, spec_=0.05, shin=0.05):
    import mujoco
    spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
    mt = spec.add_material(name=name, specular=spec_, shininess=shin)
    mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name + "_tex"
    return name


def _soft_templates(spec, scene, cfg):
    """Fabric bags (bowl primitives) -> a soft printed cotton bag with a cream lining; yarn balls (cylinders) -> a
    wound ball.  Visual only; the bowl / cylinder primitive keeps colliding and is hidden in model_fix."""
    import mujoco
    import dress_tex as T
    G = mujoco.mjtGeom
    DRESSDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for o in scene["objects"]:
        sh, t = o["shape"], (o.get("category", "") + " " + o.get("label", "")).lower()
        b = spec.body(o["id"])
        if b is None:
            continue
        if sh["type"] == "bowl" and re.search(r"drawstring|cloth bag|fabric bag|tote|sack|project bag", t):
            r, h = sh["radius"], sh["height"]
            png = DRESSDIR / "floral3.png"
            if not png.exists():
                T.floral(png, base=(228, 222, 207), w=2048, h=320, density=2.4, scale=0.75)
            lin = DRESSDIR / "lining2.png"
            if not lin.exists():
                T.laminate(lin, 0.2, 0.2, (206, 192, 168))
            fm = _texmat(spec, f"hd_{o['id']}_print", png)
            lm = _texmat(spec, f"hd_{o['id']}_lining", lin)
            ph = rng.uniform(0, 6)
            rip = (lambda th, v, ph=ph: 1 + 0.035 * np.sin(9 * th + ph) * (0.3 + v) + 0.015 * np.sin(4 * th))
            _revolve(spec, f"hd_{o['id']}_out", lambda v: ((r + 0.008) * (1 + 0.07 * np.sin(np.pi * v * 0.9)),
                                                           -h / 2 + h * 0.97 * v), ripple=rip)
            _revolve(spec, f"hd_{o['id']}_in", lambda v: (r - 0.008, -h / 2 + 0.009 + (h * 0.9) * v), ripple=rip)
            _revolve(spec, f"hd_{o['id']}_cuff", lambda v: ((r + 0.004) * (1 + 0.08 * np.sin(np.pi * v)),
                                                            h / 2 - 0.03 + 0.04 * v), ripple=rip)
            b.add_geom(name=f"hd_{o['id']}_out", type=G.mjGEOM_MESH, meshname=f"hd_{o['id']}_out", material=fm, **_vis())
            b.add_geom(name=f"hd_{o['id']}_in", type=G.mjGEOM_MESH, meshname=f"hd_{o['id']}_in", material=lm, **_vis())
            b.add_geom(name=f"hd_{o['id']}_cuff", type=G.mjGEOM_MESH, meshname=f"hd_{o['id']}_cuff", material=lm,
                       **_vis())
            b.add_geom(name=f"hd_{o['id']}_floor", type=G.mjGEOM_CYLINDER, size=[r - 0.006, 0.002, 0],
                       pos=[0, 0, -h / 2 + 0.009], material=lm, **_vis())
        elif sh["type"] == "compound" and sh.get("template") == "bottle" and len(sh["parts"]) >= 3 and \
                all(p_["type"] == "cylinder" and p_.get("axis", "z") == "z" for p_ in sh["parts"]) and \
                "cap" not in o.get("category", ""):
            # smooth bottle: body -> rounded shoulder -> neck, and a cap
            P = sorted(sh["parts"], key=lambda p_: p_["pos"][2])
            bd, nk, cp = P[0], P[1], P[-1]
            rb, z0, z1 = bd["radius"], bd["pos"][2] - bd["height"] / 2, bd["pos"][2] + bd["height"] / 2
            rn, zn1 = nk["radius"] * 0.85, nk["pos"][2] + nk["height"] / 2
            sh_h = min(0.25 * bd["height"], 1.2 * rb)

            def prof(v, rb=rb, z0=z0, z1=z1, rn=rn, zn1=zn1, sh_h=sh_h):
                zb = z1 - sh_h
                if v < 0.05:                        # rounded foot
                    return rb * (0.92 + 0.08 * v / 0.05), z0 + 0.003 * v / 0.05
                if v < 0.6:
                    return rb, z0 + 0.003 + (zb - z0 - 0.003) * (v - 0.05) / 0.55
                if v < 0.88:
                    u = (v - 0.6) / 0.28
                    return rn + (rb - rn) * np.cos(u * np.pi / 2), zb + (z1 - zb) * np.sin(u * np.pi / 2)
                return rn, z1 + (zn1 - z1) * (v - 0.88) / 0.12
            col = list(bd.get("rgba", o.get("rgba", [0.85, 0.85, 0.85, 1])))[:3]
            clear = bool(re.search(r"clear|transparent|glass", t))
            rgba = [min(1, c_ * 1.08) for c_ in col] + [0.55 if clear else 1.0]
            _revolve(spec, f"hd_{o['id']}_bottle", prof, nt=40, nz=40, two_sided=False)
            b.add_geom(name=f"hd_{o['id']}_bottle", type=G.mjGEOM_MESH, meshname=f"hd_{o['id']}_bottle", rgba=rgba,
                       **_vis())
            if clear:                               # contents, seen through the plastic
                b.add_geom(name=f"hd_{o['id']}_liquid", type=G.mjGEOM_CYLINDER, size=[rb * 0.9, (z1 - z0) * 0.3, 0],
                           pos=[0, 0, z0 + (z1 - z0) * 0.32], rgba=[0.93, 0.86, 0.62, 1], **_vis())
            cc = list(cp.get("rgba", rgba))[:3]
            b.add_geom(name=f"hd_{o['id']}_cap", type=G.mjGEOM_CYLINDER, size=[cp["radius"] * 1.05, cp["height"] / 2, 0],
                       pos=list(cp["pos"]), rgba=cc + [1], **_vis())
        elif sh["type"] == "compound" and sh.get("template") == "open_box" and \
                re.search(r"pouch|toiletry|cosmetic|wash bag|pencil case", t):
            # soft zip pouch: a rounded fabric shell, open at the top, with a grey lining
            lo = np.min([np.asarray(p_["pos"]) - np.asarray(p_["size"]) / 2 for p_ in sh["parts"]], 0)
            hi = np.max([np.asarray(p_["pos"]) + np.asarray(p_["size"]) / 2 for p_ in sh["parts"]], 0)
            L, Wd = hi[0] - lo[0], hi[1] - lo[1]
            col = tuple(int(min(255, c_ * 255 * 0.95)) for c_ in o.get("rgba", [0.6, 0.57, 0.54])[:3])
            fab = DRESSDIR / ("fabric_%d_%d_%d.png" % col)
            if not fab.exists():
                T.laminate(fab, 0.3, 0.3, col)
            fm = _texmat(spec, f"hd_{o['id']}_fab", fab, 0.08, 0.1)
            lin = DRESSDIR / "lining_grey.png"
            if not lin.exists():
                T.laminate(lin, 0.2, 0.2, (150, 146, 142))
            lm = _texmat(spec, f"hd_{o['id']}_lin", lin)
            for nm, inset, mt in (("out", 0.0, fm), ("in", 0.008, lm)):
                def outline(v, inset=inset):
                    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                    bulge = 1 + 0.06 * np.sin(np.pi * min(v, 0.95))
                    e = 2 / 5.0
                    x = (L / 2 - inset) * bulge * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** e
                    y = (Wd / 2 - inset) * bulge * (0.85 + 0.15 * v) * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** e
                    return np.c_[x, y]
                _shell(spec, f"hd_{o['id']}_{nm}", outline, lo[2] + 0.002 + inset, hi[2] - 0.004)
                b.add_geom(name=f"hd_{o['id']}_{nm}", type=G.mjGEOM_MESH, meshname=f"hd_{o['id']}_{nm}", material=mt,
                           **_vis())
            b.add_geom(name=f"hd_{o['id']}_floor", type=G.mjGEOM_BOX, size=[L / 2 - 0.012, Wd / 2 - 0.012, 0.002],
                       pos=[0, 0, lo[2] + 0.01], material=lm, **_vis())
        elif sh["type"] == "cylinder" and sh.get("axis", "z") in ("x", "y") and sh["radius"] < 0.012 and \
                re.search(r"needle|chopstick|\bpencil|\bpen\b|dowel|skewer", t):
            r, L = sh["radius"], sh["height"]
            ax = [1, 0, 0] if sh["axis"] == "x" else [0, 1, 0]
            rgb = tuple(int(min(255, c * 255)) for c in o.get("rgba", [0.9, 0.8, 0.6])[:3])
            e = np.asarray(ax, float) * (L / 2 - r * 0.6)
            b.add_geom(name=f"hd_{o['id']}_rod", type=G.mjGEOM_CAPSULE, size=[r * 0.62, 0, 0], fromto=[*(-e), *e],
                       rgba=_c(rgb), **_vis())
            b.add_geom(name=f"hd_{o['id']}_knob", type=G.mjGEOM_SPHERE, size=[r * 1.15, 0, 0], pos=list(-e),
                       rgba=_c(tuple(int(v * 0.8) for v in rgb)), **_vis())
        elif sh["type"] == "cylinder" and sh.get("axis", "z") == "z" and re.search(r"yarn|wool ball|ball of", t):
            r, h = sh["radius"], sh["height"]
            rgb = tuple(int(min(255, c * 255 * 1.05)) for c in o.get("rgba", [0.8, 0.7, 0.6])[:3])
            png = DRESSDIR / ("yarn_%d_%d_%d.png" % rgb)
            if not png.exists():
                T.yarn(png, rgb)
            ym = _texmat(spec, f"hd_{o['id']}_yarn", png)
            rz = h / 2
            _revolve(spec, f"hd_{o['id']}_ball", lambda v: (max(1e-4, r * np.sin(np.pi * v)), -rz * np.cos(np.pi * v)),
                     nt=32, nz=20, two_sided=False,
                     ripple=lambda th, v: 1 + 0.02 * np.sin(7 * th + 5 * v))
            b.add_geom(name=f"hd_{o['id']}_ball", type=G.mjGEOM_MESH, meshname=f"hd_{o['id']}_ball", material=ym,
                       **_vis())


BOWL_FOOT = 0.62     # photo_skin.BOWL_FOOD_* must agree: the contents disc sits inside this curve
BOWL_SKIP = r"drawstring|cloth bag|fabric bag|tote|sack|project bag"


def _bowl_shells(spec, scene, nt=96):
    """Bowl primitives (a base disc + 16 wall boxes, which read as an octagon-ish ring) -> a smooth turned shell
    of the same size and colour, nt segments around (visual only; the wall boxes keep colliding, drawn
    invisible here).  Skipped where a SAM 3D mesh or a fabric-bag template already draws the object."""
    import mujoco
    G = mujoco.mjtGeom
    for o in scene["objects"]:
        sh = o["shape"]
        if sh["type"] != "bowl" or sh.get("axis", "z") != "z":
            continue
        if re.search(BOWL_SKIP, (o.get("category", "") + " " + o.get("label", "")).lower()) or \
                (ROOT / "assets" / "sam3d" / scene["name"] / f"{o['id']}.glb").exists():
            continue
        b = spec.body(o["id"])
        if b is None:
            continue
        r, h, w = sh["radius"], sh["height"], 0.006
        rgba = None
        g = b.first_geom()
        while g is not None:
            if g.type == G.mjGEOM_BOX:
                rgba = list(g.rgba)
                g.rgba = [g.rgba[0], g.rgba[1], g.rgba[2], 0.0]     # the ring of boxes: still collides, not drawn
            g = b.next_geom(g)
        if rgba is None:
            continue
        z0, z1, zf = -h / 2, h / 2, -h / 2 + 0.008
        ro, ri = r + w, r - w
        F0 = BOWL_FOOT                         # radius at the foot as a fraction of the rim radius (a curved bowl)

        def prof(v, ro=ro, ri=ri, z0=z0, z1=z1, zf=zf):
            # foot ring -> curved outer wall -> rounded rim -> curved inner wall -> the floor, closed at the centre
            if v < 0.04:
                return ro * F0 * (v / 0.04), z0
            if v < 0.44:
                u = (v - 0.04) / 0.40
                return ro * (F0 + (1 - F0) * np.sin(u * np.pi / 2)), z0 + (z1 - 0.003 - z0) * u
            if v < 0.52:
                a = (v - 0.44) / 0.08 * np.pi
                return (ro + ri) / 2 + (ro - ri) / 2 * np.cos(a), z1 - 0.003 + 0.003 * np.sin(a)
            if v < 0.94:
                u = 1 - (v - 0.52) / 0.42
                return ri * (F0 + (1 - F0) * np.sin(u * np.pi / 2)), zf + (z1 - 0.003 - zf) * u
            return ri * F0 * (1 - (v - 0.94) / 0.06), zf
        nm = f"bowlshell_{o['id']}"         # (not "hd_": model_fix hides the decals of bodies with hd_ geoms)
        _revolve(spec, nm, prof, nt=nt, nz=60, two_sided=True)
        b.add_geom(name=nm, type=G.mjGEOM_MESH, meshname=nm, rgba=[rgba[0], rgba[1], rgba[2], 1.0], **_vis())


PLATE = r"\bplate|saucer|\bdish\b|platter|coaster plate"
MUG = r"\bmug\b|\bcup\b|teacup|coffee cup|tea cup"
NO_HANDLE = r"paper|plastic|measuring|medicine|espresso|tumbler|glass|travel|sippy|pen cup|pencil cup|holder|yogurt"
GLASS = r"glass(?!es)|tumbler|\bjar\b|clear|transparent|wine|beaker|vase|carafe"


def _has_sam(scene, oid):
    return (ROOT / "assets" / "sam3d" / scene["name"] / f"{oid}.glb").exists()


def _plate_rims(spec, scene, nt=96):
    """Plates / saucers (a flat cylinder 'puck' with the photo on top) get a turned lip: a low rim that rises from
    the well to the edge, so they read as plates (visual only)."""
    import mujoco
    G = mujoco.mjtGeom
    for o in scene["objects"]:
        sh = o["shape"]
        t = (o.get("category", "") + " " + o.get("label", "")).lower()
        if sh["type"] != "cylinder" or sh.get("axis", "z") != "z" or sh["height"] > 0.05 or \
                not re.search(PLATE, t) or _has_sam(scene, o["id"]):
            continue
        b = spec.body(o["id"])
        if b is None:
            continue
        r, h = sh["radius"], sh["height"]
        g = b.first_geom()
        rgba = list(g.rgba) if g is not None else [0.9, 0.9, 0.9, 1]
        lip = min(0.008, 0.25 * h + 0.003)
        rin = r * 0.78

        def prof(v, r=r, rin=rin, h=h, lip=lip):
            # inner well edge -> up to the rim top -> rounded over -> down the outer edge
            if v < 0.6:
                u = v / 0.6
                return rin + (r - 0.004 - rin) * u, h / 2 + 0.0008 + lip * np.sin(u * np.pi / 2)
            if v < 0.8:
                a = (v - 0.6) / 0.2 * np.pi / 2
                return r - 0.004 + 0.004 * np.sin(a), h / 2 + 0.0008 + lip * np.cos(a)
            u = (v - 0.8) / 0.2
            return r, h / 2 + 0.0008 - (h - 0.001) * u
        nm = f"platerim_{o['id']}"
        _revolve(spec, nm, prof, nt=nt, nz=30, two_sided=True)
        b.add_geom(name=nm, type=G.mjGEOM_MESH, meshname=nm, rgba=[rgba[0], rgba[1], rgba[2], 1.0], **_vis())


def _mug_handles(spec, scene):
    """Mug compounds (a body cylinder + a box handle bar + two box connectors, all colliding) -> the boxes are drawn
    invisible and a smooth C-shaped handle (a chain of capsules) is drawn in their place.  Visual only: the box
    parts keep colliding exactly as before."""
    import mujoco
    G = mujoco.mjtGeom
    for o in scene["objects"]:
        sh = o["shape"]
        txt = (o.get("category", "") + " " + o.get("label", "")).lower()
        if sh["type"] != "compound" or not re.search(MUG, txt) or _has_sam(scene, o["id"]):
            continue
        parts = sh["parts"]
        body = next((p_ for p_ in parts if p_["type"] == "cylinder"), None)
        hnd = next((p_ for p_ in parts if p_.get("name") == "handle" and p_["type"] == "box"), None)
        b = spec.body(o["id"])
        if body is None or hnd is None or b is None:
            continue
        r, h = body["radius"], body["height"]
        hp = np.asarray(hnd.get("pos", (0, 0, 0)), float)
        d = hp[:2] / max(np.linalg.norm(hp[:2]), 1e-9)                       # body-frame handle direction
        boxes = [p_ for p_ in parts if p_["type"] == "box"]
        zs = [abs(p_.get("pos", (0, 0, 0))[2]) for p_ in boxes if p_ is not hnd]
        zh = max(zs) if zs else hnd["size"][2] / 2
        reach = float(np.linalg.norm(hp[:2])) + hnd["size"][0] / 2 - r      # how far the handle sticks out
        rh = max(0.0035, min(0.0065, hnd["size"][1] / 2 * 0.8))
        rgba = list(hnd.get("rgba") or body.get("rgba") or o.get("rgba") or [0.9, 0.9, 0.9, 1])[:3]
        # hide the box parts (they still collide)
        g = b.first_geom()
        while g is not None:
            if g.type == G.mjGEOM_BOX:
                g.rgba = [g.rgba[0], g.rgba[1], g.rgba[2], 0.0]
            g = b.next_geom(g)
        zc = float(body.get("pos", (0, 0, 0))[2]) if body.get("pos") else 0.0
        th = np.linspace(-np.pi / 2, np.pi / 2, 9)
        pts = [(r - 0.003 + max(0.012, reach - rh) * np.cos(q), zc + zh * 1.1 * np.sin(q)) for q in th]
        for i in range(len(pts) - 1):
            (e0, z0), (e1, z1) = pts[i], pts[i + 1]
            b.add_geom(name=f"mughandle_{o['id']}_{i}", type=G.mjGEOM_CAPSULE, size=[rh, 0, 0],
                       fromto=[d[0] * e0, d[1] * e0, z0, d[0] * e1, d[1] * e1, z1],
                       rgba=[*rgba, 1.0], **_vis())


def glassify(m, sc):
    """Glass / clear objects: every visible geom of the body drawn see-through with a hard highlight (visual only)."""
    import mujoco
    for o in sc["objects"]:
        t = (o.get("category", "") + " " + o.get("label", "")).lower()
        if not re.search(GLASS, t) or re.search(r"sunglass|eyeglass|glasses case", t):
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_rgba[g][3] == 0:
                continue
            k = m.geom_matid[g]
            if k >= 0 and m.mat_texid[k].max() >= 0:
                m.geom_rgba[g] = [1.0, 1.0, 1.0, 0.55]          # keep the texture (contents), let the table through
            else:
                c = m.geom_rgba[g][:3] if k < 0 else m.mat_rgba[k][:3]
                m.geom_rgba[g] = [min(1, c[0] * 0.9 + 0.08), min(1, c[1] * 0.95 + 0.08), min(1, c[2] + 0.1), 0.38]
            if k >= 0:
                m.mat_specular[k], m.mat_shininess[k] = 0.9, 0.95


def top_outline(sc):
    """-> (outline (N,2) around the table centre, top thickness) of the dressed top (for the dark studio's edge
    light).  Plain rectangle / ellipse when the scene is not dressed."""
    (sx, sy) = sc["table"]["size"]
    if enabled(sc, auto=True):
        if sc.get("name") == "t6_06":
            return _rounded_rect(sx, sy, 0.075, 0.015), 0.035
        cfg = config(sc)
        t = cfg["top"]
        return safe_outline(sc, t.get("shape", ("rrect", 0.02, 0.02)))[0], t.get("thick", 0.03)
    if (sc.get("visual") or {}).get("table_top") in ("round", "oval"):
        return outline_for(("ellipse",), sx, sy), 0.03
    return _rounded_rect(sx, sy, 0.002, 0.002), 0.04


# ------------------------------------------------------------------------------------------------ the wrapper
def _wrap_cfg(sc, bs, cfg):
    import mujoco
    G = mujoco.mjtGeom
    top_png = _top_texture(sc, cfg)
    wall_png, floor_png = _room_textures(cfg, sc["name"])
    t = cfg["top"]

    def bs2(scene):
        scene = copy.deepcopy(scene)
        scene["_hd_flip"] = cfg.get("flip", ())
        if top_png is not None:
            scene.setdefault("visual", {})["table_texture"] = str(top_png)
        spec = bs(scene)
        (cx, cy), (sx, sy) = scene["table"]["center"], scene["table"]["size"]
        wb = spec.worldbody
        th = t.get("thick", 0.03)

        def mat(name, rgb, spec_=0.1, shin=0.1, refl=0.0):
            spec.add_material(name=name, rgba=_c(rgb) if len(rgb) == 3 else rgb, specular=spec_, shininess=shin,
                              reflectance=refl)
            return name
        legm = cfg.get("legs", {})
        metal = legm.get("metal")
        mat("hd_leg", legm.get("rgb", HONEY), 0.5 if metal else 0.15, 0.5 if metal else 0.25)
        ap = cfg.get("apron")
        if ap:
            mat("hd_apron", ap["rgb"], 0.5 if ap.get("metal") else 0.15, 0.5 if ap.get("metal") else 0.25)
        mats = {"black": mat("hd_black", (14, 14, 15), 0.3, 0.4), "blade": mat("hd_blade", (176, 180, 188), 0.9, 0.85),
                "handle": mat("hd_handle", (230, 180, 30), 0.35, 0.5)}
        for oid, rgb in cfg.get("tmpl_rgb", {}).items():
            mats[oid] = mat(f"hd_t_{oid}", rgb, 0.35, 0.5)
        spec.add_texture(name="hd_floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(floor_png))
        fm = spec.add_material(name="hd_floor_mat", texrepeat=[1 / 1.2, 1 / 1.2], texuniform=True,
                               specular=0.15, shininess=0.3, reflectance=0.0)
        fm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hd_floor_tex"
        spec.add_texture(name="hd_wall_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(wall_png))
        wm = spec.add_material(name="hd_wall_mat", texrepeat=[1, 1], texuniform=False, specular=0.05, shininess=0.05,
                               emission=cfg.get("wall_glow", 0.35))   # bounce light: the wall faces away from the lamps
        wm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hd_wall_tex"

        # hide the flat textured plane / ellipse, the collision box's look and the stock legs
        g = wb.first_geom()
        floor = None
        while g is not None:
            if g.type == G.mjGEOM_PLANE and g.pos[2] < -0.5:
                floor = g
            elif getattr(g, "material", "") == "tablemat":
                g.rgba = [0, 0, 0, 0]
                g.material = ""
            elif g.type == G.mjGEOM_BOX and g.pos[2] < -0.1 and g.size[0] < 0.06 and g.size[1] < 0.06 \
                    and abs(g.pos[0] - cx) < sx / 2 + 0.01 and abs(g.pos[1] - cy) < sy / 2 + 0.01:
                g.rgba = [0, 0, 0, 0]
            g = wb.next_geom(g)
        if floor is not None:
            floor.material, floor.rgba = "hd_floor_mat", [1, 1, 1, 1]
        tb = spec.body("table")
        if tb is not None:
            tg = tb.first_geom()
            while tg is not None:
                tg.rgba = [0, 0, 0, 0]
                tg = tb.next_geom(tg)

        # the top
        outline, _ = safe_outline(scene, t.get("shape", ("rrect", 0.02, 0.02)))
        uv = (lambda p: [0.5 + p[0] / sx, 0.5 + p[1] / sy])
        topmat = "tablemat" if top_png is not None or spec.material("tablemat") is not None else "hd_leg"
        if spec.material("tablemat") is None:
            topmat = mat("hd_top", t.get("rgb", (190, 180, 165)), *t.get("spec", (0.2, 0.3)))
        _prism(spec, "hd_top", outline, -th, 0.0006, uv=uv)
        wb.add_geom(type=G.mjGEOM_MESH, meshname="hd_top", pos=[cx, cy, 0], material=topmat, **_vis())
        if t.get("edge"):
            e = t["edge"]
            mat("hd_edge", e["rgb"], 0.2, 0.3)
            c0 = outline.mean(0)
            big = c0 + (outline - c0) * [(sx + 0.006) / sx, (sy + 0.006) / sy]
            _prism(spec, "hd_edge", big, -e.get("h", th), -0.0015)
            wb.add_geom(type=G.mjGEOM_MESH, meshname="hd_edge", pos=[cx, cy, 0], material="hd_edge", **_vis())
        if t.get("rim"):                          # rattan tray rim: a tube following the outline
            r = t["rim"]["r"]
            mat("hd_rim", t["rim"]["rgb"], 0.15, 0.25)
            c0 = outline.mean(0)
            ring = c0 + (outline - c0) * [(sx + 2 * r) / sx, (sy + 2 * r) / sy]
            for k in range(len(ring)):
                a_, b_ = ring[k], ring[(k + 1) % len(ring)]
                if np.linalg.norm(b_ - a_) < 1e-4:
                    continue
                for z, rr in ((0.003, r), (-th - 0.006, r * 0.7)):
                    wb.add_geom(type=G.mjGEOM_CAPSULE, size=[rr, 0, 0], material="hd_rim",
                                fromto=[cx + a_[0], cy + a_[1], z, cx + b_[0], cy + b_[1], z], **_vis())

        # legs
        style, w = legm.get("style", "square"), legm.get("w", 0.045)
        ix, iy = legm.get("inset", (0.05, 0.05))
        z1, z0 = -th, FLOOR_Z
        legs = [(cx + kx * (sx / 2 - ix), cy + ky * (sy / 2 - iy)) for kx in (-1, 1) for ky in (-1, 1)]
        if style == "tapered":
            sp = legm.get("splay", 0.05)
            for (lx, ly) in legs:
                dx, dy = np.sign(lx - cx) * sp * 0.6, np.sign(ly - cy) * sp
                wb.add_geom(type=G.mjGEOM_CYLINDER, size=[w, 0, 0], material="hd_leg",
                            fromto=[lx, ly, z1, lx + dx * 0.5, ly + dy * 0.5, (z1 + z0) / 2], **_vis())
                wb.add_geom(type=G.mjGEOM_CAPSULE, size=[w * 0.72, 0, 0], material="hd_leg",
                            fromto=[lx + dx * 0.5, ly + dy * 0.5, (z1 + z0) / 2, lx + dx, ly + dy, z0 + w * 0.7],
                            **_vis())
        else:
            for (lx, ly) in legs:
                if style in ("round", "rattan"):
                    wb.add_geom(type=G.mjGEOM_CYLINDER, size=[w, (z1 - z0) / 2, 0], pos=[lx, ly, (z0 + z1) / 2],
                                material="hd_leg", **_vis())
                    if style == "rattan":           # cane bindings
                        for zz in (z1 - 0.05, z1 - 0.07, z0 + 0.14, z0 + 0.16):
                            wb.add_geom(type=G.mjGEOM_CYLINDER, size=[w * 1.12, 0.006, 0], pos=[lx, ly, zz],
                                        rgba=[0.55, 0.38, 0.2, 1], **_vis())
                else:
                    wb.add_geom(type=G.mjGEOM_BOX, size=[w / 2, w / 2, (z1 - z0) / 2], pos=[lx, ly, (z0 + z1) / 2],
                                material="hd_leg", **_vis())
            if style == "rattan":                   # stretchers between the legs
                zs = z0 + 0.15
                for kx in (-1, 1):
                    x = cx + kx * (sx / 2 - ix)
                    wb.add_geom(type=G.mjGEOM_CAPSULE, size=[w * 0.75, 0, 0], material="hd_leg",
                                fromto=[x, cy - sy / 2 + iy, zs, x, cy + sy / 2 - iy, zs], **_vis())
                for ky in (-1, 1):
                    y = cy + ky * (sy / 2 - iy)
                    wb.add_geom(type=G.mjGEOM_CAPSULE, size=[w * 0.75, 0, 0], material="hd_leg",
                                fromto=[cx - sx / 2 + ix, y, zs + 0.03, cx + sx / 2 - ix, y, zs + 0.03], **_vis())
        if ap:
            h = ap["h"]
            ax, ay = sx / 2 - ix, sy / 2 - iy
            for ky in (-1, 1):
                wb.add_geom(type=G.mjGEOM_BOX, size=[ax, 0.009, h / 2], pos=[cx, cy + ky * ay, z1 - h / 2],
                            material="hd_apron", **_vis())
            for kx in (-1, 1):
                wb.add_geom(type=G.mjGEOM_BOX, size=[0.009, ay, h / 2], pos=[cx + kx * ax, cy, z1 - h / 2],
                            material="hd_apron", **_vis())
        sh_ = cfg.get("shelf")
        if sh_:
            zs = z0 + sh_["z"]
            ax, ay = sx / 2 - ix, sy / 2 - iy
            mat("hd_shelf", sh_["rgb"], 0.15, 0.25)
            for kx in (-1, 1):
                wb.add_geom(type=G.mjGEOM_BOX, size=[0.012, ay, 0.025], pos=[cx + kx * ax, cy, zs],
                            material="hd_shelf", **_vis())
            n = max(4, int(2 * ax / 0.05))
            for i in range(n):
                x = cx - ax + 0.02 + (2 * ax - 0.04) * i / (n - 1)
                wb.add_geom(type=G.mjGEOM_BOX, size=[0.016, ay, 0.008], pos=[x, cy, zs + 0.028],
                            material="hd_shelf", **_vis())

        # room: back wall behind the far edge, skirting
        for side, grp in (cfg.get("walls", (("+x", 1),)) if _room() else ()):
            vk = dict(_vis())
            vk["group"] = grp
            if side == "+x":       # behind the table's far edge (as in the photo)
                wx, sg = cx + sx / 2 + cfg.get("wall_gap", 0.18), 1
            else:                  # behind the robot (the poster view looks from across the table)
                wx, sg = -0.85, -1
            wb.add_geom(type=G.mjGEOM_BOX, size=[0.01, 2.6, 1.1], pos=[wx + 0.01 * sg, cy, FLOOR_Z + 1.1],
                        material="hd_wall_mat", **vk)
            wb.add_geom(type=G.mjGEOM_BOX, size=[0.012, 2.6, 0.045], pos=[wx - 0.002 * sg, cy, FLOOR_Z + 0.045],
                        rgba=_c(cfg.get("skirt", (230, 226, 218))), **vk)
        lt = dict(WINDOW, **cfg.get("light", {}))
        if _room():
            wb.add_light(pos=[cx + lt["pos"][0], cy + lt["pos"][1], lt["pos"][2]], dir=lt["dir"],
                         diffuse=lt["diffuse"], specular=lt["specular"], castshadow=True,
                         type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        _templates(spec, scene, mats)
        _soft_templates(spec, scene, cfg)
        _bowl_shells(spec, scene)
        _plate_rims(spec, scene)
        _mug_handles(spec, scene)
        return spec
    return bs2


def _model_fix_cfg(m, sc, cfg):
    import mujoco
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "tablemat")
    if k >= 0:
        m.mat_specular[k], m.mat_shininess[k] = cfg["top"].get("spec", (0.2, 0.3))
    sam = cfg.get("sam_mat", {})
    for o in sc["objects"]:
        look = sam.get(o["id"]) or look_of(o)
        km = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"sam3d_{o['id']}_mat")
        if km >= 0 and look:
            m.mat_specular[km], m.mat_shininess[km] = MAT_LOOK[look]
        # template objects: the decals of the box go
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        has_t = any(m.geom_bodyid[g] == b and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                    .startswith("hd_") for g in range(m.ngeom))
        if has_t:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") \
                        .startswith("hd_"):
                    m.geom_rgba[g][3] = 0.0
    _decal_retint(m, sc, cfg)
    hide_bowl_bases(m, sc)
    _flatten(m, sc, cfg.get("flatten", ()))
    for oid, how in cfg.get("tex_fix", {}).items():      # SAM 3D colour casts
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, f"sam3d_{oid}_tex")
        if t < 0:
            continue
        a, h, w, c = m.tex_adr[t], m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
        px = m.tex_data[a:a + h * w * c].reshape(-1, c)
        f = px[:, :3].astype(float)
        L = f @ [0.299, 0.587, 0.114]
        sat, gain = how
        px[:, :3] = np.clip((L[:, None] + (f - L[:, None]) * sat) * gain, 0, 255).astype(np.uint8)
    m.vis.headlight.ambient = [0.30, 0.29, 0.27]
    m.vis.headlight.diffuse = [0.20, 0.195, 0.185]
    hide_bowl_bases(m, sc)
    glassify(m, sc)


def hide_bowl_bases(m, sc):
    """Bowls with a turned shell: the primitive's flat base disc (radius r) would stick out of the curved foot; draw
    it invisible (it still collides).  The contents decal is a separate geom and stays."""
    import mujoco
    for o in sc["objects"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        if b < 0 or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"bowlshell_{o['id']}") < 0:
            continue
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER and \
                    (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_rgba[g][3] = 0.0


def _flatten(m, sc, ids, keep=0.006):
    """Thin photo-decal items (gloves, a cloth pad, keys): the bounding box's side decals show table pixels and the
    raised top reads as a card.  Hide the side decals and draw the top decal just above the table.  Visual geoms
    only (decals carry no mass and do not collide)."""
    import mujoco
    for o in sc["objects"]:
        if o["id"] not in ids or o["shape"].get("type") != "box":
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        Hh = o["shape"]["size"][2]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                m.geom_rgba[g][3] = 0.0                    # the primitive's plain sides
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or \
                    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("hd_", "sam3d_")):
                continue
            mid = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            Vb = V @ R.reshape(3, 3).T + m.geom_pos[g]
            ext = Vb.max(0) - Vb.min(0)
            if ext[2] > 0.5 * max(ext[0], ext[1]):
                m.geom_rgba[g][3] = 0.0
            elif Vb[:, 2].mean() > 0:
                m.geom_pos[g][2] -= max(0.0, Hh - keep)


def _decal_retint(m, sc, cfg):
    """Photo decals of box objects carry a margin of the photo's table around the object.  Give those table-coloured
    texels the same colour shift as the new procedural top, so the margins melt into it instead of reading as cards."""
    import mujoco
    from PIL import Image
    top = _top_texture(sc, cfg)
    src = (sc.get("visual") or {}).get("table_texture")
    if top is None or not src:
        return
    old = np.asarray(Image.open(ROOT / src).convert("RGB"), float).reshape(-1, 3)
    med = np.median(old, 0)
    spread = np.maximum(np.median(np.abs(old - med), 0) * 1.4826, 6.0)
    new = np.asarray(Image.open(top).convert("RGB"), float).reshape(-1, 3).mean(0)
    gain = new / np.maximum(med, 1)
    for t in range(m.ntex):
        if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t) or "").startswith("dtex"):
            continue
        a, h, w, c = m.tex_adr[t], m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
        px = m.tex_data[a:a + h * w * c].reshape(-1, c)
        f = px[:, :3].astype(float)
        dd = np.sqrt((((f - med) / spread) ** 2).sum(1) / 3)
        k = np.clip((3.0 - dd) / 1.5, 0, 1)[:, None]          # 1 inside ~1.5 sigma, 0 beyond 3 sigma
        px[:, :3] = np.clip(f * (1 - k) + f * gain * k, 0, 255).astype(np.uint8)


def wrap(sc, bs, auto=False):
    """base_spec wrapper adding the set dressing (hand-tuned scenes; every scene with auto=True)."""
    if not enabled(sc, auto):
        return bs
    if sc.get("name") == "t6_06":
        return _wrap06(sc, bs, poster=auto)
    cfg = config(sc)
    if auto:                  # poster/card renders: the far-side 3/4 view needs the wall behind the robot instead
        cfg = dict(cfg, walls=(("+x", 5), ("-x", 4)))
    return _wrap_cfg(sc, bs, cfg)


def model_fix(m, sc, auto=False):
    if not enabled(sc, auto):
        return
    if sc.get("name") == "t6_06":
        return _model_fix06(m, sc)
    _model_fix_cfg(m, sc, config(sc))


# ------------------------------------------------------------------------------------------------ automatic pass
# by-eye corrections where the colour statistics pick the wrong class (all: keep the photo texture)
AUTO_OVERRIDE = {"t6_29": "photo", "t6_30": "photo", "t6_40": "photo", "t6_26": "photo", "t6_46": "photo"}
# tops whose photo texture carries inpainting blobs -> a procedural material matched to the photo by eye
# (visual only; keys besides "spec"/"edge"/"rim"/"legs" go to _top_texture).  TOP_FIX=0 turns it off.
STEEL_LEGS = dict(style="square", w=0.03, rgb=(150, 152, 156), metal=True)
TOP_FIX = {
    "t6_08": dict(kind="wood", style="strips", base=(218, 182, 134), dark=(178, 134, 88), strip_m=0.11,
                  period_m=0.03, contrast=0.6, spec=(0.12, 0.2)),                       # worn pine workbench
    "t6_18": dict(kind="wood", style="strips", base=(168, 150, 128), dark=(112, 94, 76), strip_m=0.095,
                  period_m=0.03, contrast=1.0, spec=(0.04, 0.08)),                      # weathered garden slats
    "t6_19": dict(kind="wood", style="slab", base=(184, 148, 128), dark=(146, 108, 90), period_m=0.02,
                  contrast=0.6, spec=(0.35, 0.5), legs=STEEL_LEGS),                     # wood-look laminate
    "t6_20": dict(kind="brushed", base=(176, 176, 174), spec=(0.65, 0.6), legs=STEEL_LEGS),   # stainless bench
    "t6_25": dict(kind="patchwork", base=(232, 220, 206), ink=(196, 92, 98), spec=(0.0, 0.02)),  # tablecloth
    "t6_26": dict(kind="felt", base=(46, 84, 50), spec=(0.0, 0.02),
                  rim=dict(r=0.014, rgb=(122, 74, 40))),                                # green baize, wood rim
    "t6_29": dict(kind="wood", style="slab", base=(168, 104, 58), dark=(104, 60, 30), period_m=0.016,
                  contrast=0.8, spec=(0.35, 0.5)),                                      # honey oak
    "t6_30": dict(kind="brushed", base=(212, 216, 220), slat_m=0.075, spec=(0.5, 0.5), legs=STEEL_LEGS),
    "t6_31": dict(kind="wood", style="strips", base=(170, 124, 82), dark=(100, 66, 40), strip_m=0.19,
                  period_m=0.025, contrast=1.0, spec=(0.1, 0.2)),                       # farmhouse planks
    "t6_36": dict(kind="laminate", base=(64, 84, 68), spec=(0.5, 0.6)),                # green gloss paint
    "t6_37": dict(kind="terrazzo", base=(196, 190, 182), spec=(0.15, 0.25)),            # terrazzo
    "t6_40": dict(kind="tiletop", base=(232, 230, 224), grout=(192, 188, 180), tile_m=0.15, spec=(0.45, 0.6)),
    "t6_41": dict(kind="rust", base=(112, 96, 86), rust=(150, 88, 52), dark=(64, 56, 52), spec=(0.05, 0.1)),
    "t6_46": dict(kind="stone", base=(64, 64, 62), spec=(0.2, 0.3)),                   # dark slate
    "t6_48": dict(kind="laminate", base=(196, 210, 206), spec=(0.8, 0.9), legs=STEEL_LEGS,
                  edge=dict(rgb=(170, 196, 190), h=0.012)),                             # glass over chrome
}
FLAT = r"key|coaster|cable|glove|map\b|paper|card|sunglass|envelope|note|receipt|ticket|photo|leaflet|cord|hair tie|" \
       r"rubber band|bracelet|necklace|sticker|bookmark|wallet|puzzle|piece|letter|postcard|flyer|coupon|label|" \
       r"placemat|\bmat\b|napkin|tissue|bandage|pad\b"


def _photo_regions(sc):
    """Median colours of the photo's wall (top band) and floor (bottom corners)."""
    from PIL import Image
    a = np.asarray(Image.open(ROOT / sc["photo"]["file"]).convert("RGB"), float)
    h, w = a.shape[:2]
    wall = np.median(a[:int(0.10 * h), int(0.2 * w):int(0.8 * w)].reshape(-1, 3), 0)
    fl = np.concatenate([a[int(0.92 * h):, :int(0.12 * w)].reshape(-1, 3), a[int(0.92 * h):, int(0.88 * w):].reshape(-1, 3)])
    return wall, np.median(fl, 0)


def _relum(c, L, desat=0.0):
    c = np.asarray(c, float)
    l0 = float(c @ [0.299, 0.587, 0.114])
    c = c + (l0 - c) * desat
    return tuple(int(v) for v in np.clip(c * L / max(l0, 1.0), 0, 255))


def _woodish(c):
    R, G, B = c
    sat = (max(c) - min(c)) / max(max(c), 1)
    L = 0.299 * R + 0.587 * G + 0.114 * B
    return R > G > B and sat >= 0.28 and R - B > (25 if L < 100 else 40) and (R - G) >= 0.35 * (R - B) \
        and (G - B) >= 0.15 * (R - B)                    # not red paint


def auto_config(sc):
    """Generic set dressing from the photo's colour statistics and the scene's labels (Tier-2 poster renders)."""
    from PIL import Image
    name = sc["name"]
    vis = sc.get("visual") or {}
    (sx, sy) = sc["table"]["size"]
    wall_c, floor_c = _photo_regions(sc)
    kind, top = "photo", {}
    if vis.get("table_texture"):
        a = np.asarray(Image.open(ROOT / vis["table_texture"]).convert("RGB"), float)
        med = np.median(a.reshape(-1, 3), 0)
        L = float(med @ [0.299, 0.587, 0.114])
        sat = (med.max() - med.min()) / max(med.max(), 1)
        g = np.asarray(Image.open(ROOT / vis["table_texture"]).convert("L").resize((256, 160)), float)
        hp = float(np.abs(np.diff(g, axis=1)).mean())          # fine pattern: cloth, slats, stone, glass
        if hp > 3.3:
            kind = "photo"
        elif _woodish(med):
            kind = "wood"
            base = tuple(int(v) for v in np.clip(med * 1.08, 0, 255))
            top = dict(kind="wood", style="slab", base=base, dark=tuple(int(v) for v in med * [0.70, 0.62, 0.56]),
                       period_m=0.014, contrast=0.8)
        elif (sat < 0.2 and L > 150) or sat >= 0.3 or (sat < 0.12 and L < 140):
            kind = "laminate"
            top = dict(kind="laminate", base=tuple(int(v) for v in np.clip(med * 1.08, 0, 255)))
        kind = AUTO_OVERRIDE.get(name, kind)
    if kind == "photo":
        top = dict(kind="photo")
    fix = TOP_FIX.get(name) if os.environ.get("TOP_FIX", "1") != "0" else None
    if fix:                     # by-eye material for the tops whose photo texture shows inpainting blobs
        top = {k: v for k, v in fix.items() if k not in ("spec", "edge", "rim", "legs")}
        kind = top["kind"]
    shape = ("ellipse",) if vis.get("table_top") in ("round", "oval") else ("rrect", 0.025, 0.025)
    top.update(shape=shape, thick=0.03, seed=int(name[3:]) if name[3:].isdigit() else 0,
               spec=(0.3, 0.45) if kind != "photo" else (0.0, 0.0))
    if fix:
        top["spec"] = fix.get("spec", top["spec"])
        for k in ("edge", "rim"):
            if k in fix:
                top[k] = fix[k]
    if kind == "wood":
        legs = dict(style="square", w=0.045, rgb=tuple(int(v) for v in np.asarray(top["base"]) * 0.75))
    elif kind == "laminate" and float(np.asarray(top["base"]) @ [0.299, 0.587, 0.114]) > 150:
        legs = dict(style="square", w=0.03, rgb=(120, 122, 126), metal=True)
    else:
        legs = dict(style="square", w=0.045, rgb=(66, 58, 52))
    if shape[0] == "ellipse":
        legs["inset"] = (sx * 0.22, sy * 0.22)
    else:
        legs["inset"] = (0.05, 0.05)
    if fix and "legs" in fix:
        legs = dict(fix["legs"], inset=legs["inset"])
    wood_floor = _woodish(floor_c)
    floor = ("planks", _relum(floor_c, 172, 0.35), _relum(floor_c, 128, 0.25)) if wood_floor else \
        ("tiles", _relum(floor_c, 186, 0.6), 3)
    wall = _relum(wall_c, 226, 0.75)
    # thin flat photo-decal items: drawn flat (only when nothing rests on them)
    from rehearsal.scene import footprint_polygon
    fps = {o["id"]: np.asarray(footprint_polygon(o)) for o in sc["objects"]}

    def under(o):
        lo, hi = fps[o["id"]].min(0), fps[o["id"]].max(0)
        return any((f.max(0) > lo).all() and (f.min(0) < hi).all() for k, f in fps.items() if k != o["id"])
    from photo_skin import is_cloth
    flat = tuple(o["id"] for o in sc["objects"] if o["shape"]["type"] == "box" and o["shape"]["size"][2] <= 0.05
                 and re.search(FLAT, (o.get("category", "") + " " + o.get("label", "")).lower())
                 and not is_cloth(o) and not under(o))
    return dict(top=top, legs=legs, wall=wall, skirt=_relum(wall_c, 215, 0.5), floor=floor, flatten=flat,
                walls=(("+x", 5), ("-x", 4)), auto=True, kind=kind)
