"""SAM 3D Objects look-alike meshes inside MuJoCo renders (visual only; physics keeps the primitives).

    from sam3d_mj import add_sam3d_visuals
    spec = render_tables.base_spec(sc)
    names = add_sam3d_visuals(spec, sc)          # -> object ids that got a mesh
    m = render_tables.finish(spec)

For every object of `sc` with assets/sam3d/<sc.name>/<obj>.glb + <obj>.json (status ok, not "use": false; made by
scripts/sam3d_assets.py / scripts/sam3d_tables.py) this adds one visual-only mesh geom to the object's body
(contype=0, conaffinity=0, density=0, group 1), placed exactly like rehearsal/export3d.py's `visual_mesh`
(uniform scale, sitting on the primitive's bottom face; same function, so the web viewer and MuJoCo agree).
With hide_primitive=True the object's own geoms get alpha 0: they still collide and carry the mass, they are just
not drawn.  The planner, physics checks and the shipped scene JSON never see the meshes.

Colour: MuJoCo has no vertex colours, so the GLB's vertex colours are baked into a per-face texture atlas: every
face gets its own 2x2-texel cell holding the face's mean colour, the three corners of the face point at the cell
centre (userfacetexcoord), and faces are laid out in Morton order of their centroids so that mipmapping only ever
blends spatially neighbouring faces.  At 18k faces per object (a few mm each) this reads as smooth colour, keeps
MuJoCo's smooth normals (vertices stay shared), and costs one geom + one 192-270 px texture per object.

Works with photo_skin.dress too: pass a base_spec that calls add_sam3d_visuals; dress's decal pass skips geoms
with alpha 0 and mesh geoms, so the SAM 3D objects get no decals while the other objects keep theirs and cloth
stays cloth.

CLI (compare renders for the hero tables; project .venv):
    .venv/bin/python scripts/sam3d_mj.py [NN ...]   -> footage/tables_sim/hero/sam3d/NN_compare.png
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rehearsal.export3d import ASSETS, visual_meshes  # noqa: E402

TEX = ROOT / "out" / "tables_v6" / "_mj_tex" / "sam3d"
G = mujoco.mjtGeom
_DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_NC = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}


def load_glb(path: Path):
    """Minimal GLB reader for trimesh-exported meshes -> (verts (N,3) float, faces (M,3) int, rgb (N,3) uint8)."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    binc = b[20 + n + 8:]

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        k = _NC[a["type"]]
        arr = np.frombuffer(binc, _DT[a["componentType"]], a["count"] * k, off)
        return arr.reshape(-1, k) if k > 1 else arr

    p = j["meshes"][0]["primitives"][0]
    V = acc(p["attributes"]["POSITION"]).astype(float)
    F = acc(p["indices"]).reshape(-1, 3).astype(int)
    C = acc(p["attributes"]["COLOR_0"])[:, :3] if "COLOR_0" in p["attributes"] else np.full((len(V), 3), 180, np.uint8)
    if C.dtype != np.uint8:
        C = (np.clip(C, 0, 1) * 255).astype(np.uint8)
    return V, F, C


def _morton(P):
    q = ((P - P.min(0)) / max(np.ptp(P, 0).max(), 1e-9) * 1023).astype(np.uint64)

    def spread(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        return (x | (x << 2)) & 0x09249249
    return spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)


def bake(glb: Path, colour_gain: float = 1.0):
    """GLB -> (V, F reordered, texcoords (M,2), texture png path).  Cached on the GLB's mtime.
    Colours are multiplied by the sidecar's per-channel orientation.rgb_gain (photo / reprojected mesh inside the
    SAM mask, measured by sam3d_assets.py; clipped here to 0.7-1.4: unstable on dark objects) and by colour_gain."""
    V, F, C = load_glb(glb)
    side = glb.with_suffix(".json")
    rgb_gain = np.ones(3)
    if side.exists():
        rgb_gain = np.clip(np.asarray((json.loads(side.read_text()).get("orientation") or {}).get("rgb_gain", [1, 1, 1]), float), 0.7, 1.4)
    order = np.argsort(_morton(V[F].mean(1)), kind="stable")
    F = F[order]
    fc = C[F].astype(float).mean(1) * rgb_gain * colour_gain
    M = len(F)
    W = int(np.ceil(np.sqrt(M)))
    cell = 2
    img = np.zeros((W * cell, W * cell, 3), np.uint8)
    r, c = np.divmod(np.arange(M), W)
    blk = np.clip(fc, 0, 255).astype(np.uint8)
    for dy in range(cell):
        for dx in range(cell):
            img[r * cell + dy, c * cell + dx] = blk
    # unused tail cells: repeat the last colour (keeps mipmaps clean)
    if M < W * W:
        rr, cc = np.divmod(np.arange(M, W * W), W)
        for dy in range(cell):
            for dx in range(cell):
                img[rr * cell + dy, cc * cell + dx] = blk[-1]
    # texcoord convention (photo_skin.py): image row r, column c <-> (u = c / W, v = r / H)
    T = np.c_[(c * cell + cell / 2) / (W * cell), (r * cell + cell / 2) / (W * cell)]
    TEX.mkdir(parents=True, exist_ok=True)
    key = int(colour_gain * 100) + int(1000 * rgb_gain.sum())
    f = TEX / f"{glb.parent.name}_{glb.stem}_{int(glb.stat().st_mtime)}_{key}.png"
    if not f.exists():
        Image.fromarray(img).save(f)
    return V, F, T, f


def add_sam3d_visuals(spec: mujoco.MjSpec, scene: dict, hide_primitive: bool = True, assets: Path = ASSETS,
                      colour_gain: float = 1.0, emission: float = 0.3) -> list[str]:
    """Add visual-only SAM 3D mesh geoms to the object bodies of `spec` (built from `scene`); see module doc."""
    done = []
    for oid, vm in visual_meshes(scene, assets).items():
        try:
            body = spec.body(oid)
        except Exception:
            body = None
        if body is None:
            continue
        glb = assets / scene["name"] / f"{oid}.glb"
        V, F, T, png = bake(glb, colour_gain)
        Vb = V * vm["scale"] + np.asarray(vm["pos"], float)      # GLB frame -> body frame (quat is identity)
        tag = f"sam3d_{oid}"
        spec.add_texture(name=tag + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
        mat = spec.add_material(name=tag + "_mat")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tag + "_tex"
        # the splat colours already hold the photo's shading: some emission keeps sides facing away from the
        # render's top light from going grey
        mat.specular, mat.shininess, mat.reflectance, mat.emission = 0.12, 0.15, 0.0, emission
        mesh = spec.add_mesh(name=tag + "_mesh")
        mesh.uservert = Vb.ravel().tolist()
        mesh.userface = F.ravel().tolist()
        mesh.usertexcoord = T.ravel().tolist()
        mesh.userfacetexcoord = np.repeat(np.arange(len(F)), 3).tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        if hide_primitive:
            g = body.first_geom()
            while g is not None:
                g.rgba = [g.rgba[0], g.rgba[1], g.rgba[2], 0.0]
                g = body.next_geom(g)
        body.add_geom(name=tag, type=G.mjGEOM_MESH, meshname=tag + "_mesh", material=tag + "_mat",
                      contype=0, conaffinity=0, density=0, group=1)
        done.append(oid)
    return done


# ---------------------------------------------------------------- compare renders
def _label(im: np.ndarray, text: str) -> np.ndarray:
    from PIL import ImageDraw, ImageFont
    p = Image.fromarray(im)
    d = ImageDraw.Draw(p)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([0, 0, 14 + 17 * len(text), 46], fill=(0, 0, 0))
    d.text((10, 7), text, fill=(255, 255, 255), font=font)
    return np.asarray(p)


def _fit(im, w, h):
    return np.asarray(Image.fromarray(im).convert("RGB").resize((w, h), Image.LANCZOS))


def compare(nn: str, out_dir: Path = ROOT / "footage" / "tables_sim" / "hero" / "sam3d", dressed: bool = True):
    import render_tables as RT
    f = ROOT / "scenes" / f"t6_{nn}.json"
    sc = json.loads(f.read_text())
    got = []

    def spec_fn(s):
        sp = RT.base_spec(s)
        got[:] = add_sam3d_visuals(sp, s)
        return sp
    if dressed:
        from photo_skin import dress
        m, d = dress(sc, spec_fn, RT.finish, RT.settle)
    else:
        m = RT.finish(spec_fn(sc))
        d = RT.settle(m, sc)
    view = RT.render_view(m, d, sc)
    pcam = RT.render_photo(m, d, sc)
    # primitives: the same model and state with the meshes hidden and the primitives shown again (one cloth settle;
    # out/tables_v6/render/NN_*.png can be from an older calibration).  The meshed objects show as plain primitives.
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        if b in got:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if nm.startswith("sam3d_"):
                m.geom_rgba[g][3] = 0.0
                if m.geom_matid[g] >= 0:
                    m.mat_rgba[m.geom_matid[g]][3] = 0.0
            else:
                m.geom_rgba[g][3] = 1.0
    pv0, pp0 = RT.render_view(m, d, sc), RT.render_photo(m, d, sc)
    photo = np.asarray(Image.open(ROOT / sc["photo"]["file"]).convert("RGB"))
    W, H = 800, 600
    pv, pp = _fit(pv0, W, H), _fit(pp0, W, H)
    top = [_label(_fit(photo, W, H), "photo"), _label(pv, "primitives (physics)"),
           _label(_fit(view, W, H), "SAM 3D meshes")]
    bot = [_label(_fit(photo, W, H) // 2 + _fit(pcam, W, H) // 2, "SAM 3D over photo (50%)"),
           _label(pp, "primitives, photo camera"), _label(_fit(pcam, W, H), "SAM 3D, photo camera")]
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    Image.fromarray(grid).save(out_dir / f"{nn}_compare.png")
    Image.fromarray(view).save(out_dir / f"{nn}_view.png")
    Image.fromarray(pcam).save(out_dir / f"{nn}_photo.png")
    print(nn, "meshes:", got, "->", out_dir / f"{nn}_compare.png", flush=True)
    return got


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for nn in args or ["43", "03", "01", "06", "09", "46", "28", "27"]:
        if not (ASSETS / f"t6_{nn}").exists():
            print(nn, "no SAM 3D assets yet", flush=True)
            continue
        try:
            compare(nn, dressed="--plain" not in sys.argv)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(nn, "FAILED", e, flush=True)
