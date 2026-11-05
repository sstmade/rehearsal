"""Visual layer for the batch_tables.py renders: the photo's own pixels on each object, and cloth as cloth.

Pass 1 builds the shipped scene model, settles it and reads where every object ended up.  Pass 2 rebuilds it with
  * photo decals: every face of a box (and the top and visible side of a cylinder) that the photo's camera could
    see gets a thin textured mesh sampled from the photo at that face's 3D points (so a basket shows its weave, a
    book its cover, a bowl the food inside), and
  * cloth: towels, clothes, socks, napkins... are drawn as a MuJoCo flex cloth (a real simulated sheet) that falls
    onto a hidden core the size of the object and drapes over it, textured with the photo of its footprint.
The decals carry no mass and do not collide; the cloth only exists in these renders.  Planning, physics checks
and the shipped scene JSON keep the rigid primitives (a cloth is a thin rigid pad there).

Texture convention (checked with a marker texture): image row r, column c <-> texcoord (u = c / W, v = r / H).
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "out" / "tables_v6" / "_mj_tex"
G = mujoco.mjtGeom

CLOTH = re.compile(r"towel|cloth|napkin|\brag\b|sock|shirt|t-shirt|sweater|hoodie|jumper|clothes|clothing|jeans|"
                   r"pants|trousers|scarf|fabric|bandana|apron|washcloth|knit|tea towel|dish towel|garment|blanket|"
                   r"handkerchief|bib\b|leggings|tank top|shorts")
NOT_CLOTH = re.compile(r"basket|hanger|box|rack|holder|bag\b|pouch|clip|ball|bottle|spray|rod|hook")


# cloth look: sheet half-thickness (m) and whether the sheet ignores its hidden rigid core (render only)
CLOTH_RADIUS = float(os.environ.get("CLOTH_RADIUS", "0.0025"))
CLOTH_SOFT = os.environ.get("CLOTH_SOFT", "0") == "1"
# decals take photo pixels only from inside the object's own SAM outline (no table / neighbour bleed)
MASK_DECALS = os.environ.get("MASK_DECALS", "1") != "0"
# high-res SAM 3.1 outlines (out/tables_v6/masks_hi, scripts/masks_hi.py); MASKS_HI=0 -> perception.mask_px only
MASKS_HI = os.environ.get("MASKS_HI", "1") != "0"
THIN_MASK = re.compile(r"cable|cord|wire|chopstick|\bpens?\b|pencil|marker|brush|needle|straw|skewer|crayon|"
                       r"hair tie|rubber band|\bkeys?\b|spoon|fork|knife|chain")
BOWL_FOOT, BOWL_FILL = 0.62, 0.42   # bowl shell foot radius fraction / contents level (fraction of the inner depth)
CLOTH_SHRINK = {}     # object id -> fraction of the footprint the cloth texture samples (set dressing; default 0.80/0.82)


def is_cloth(o) -> bool:
    name = (o["category"] + " " + o["label"]).lower()
    return bool(CLOTH.search(name)) and not NOT_CLOTH.search(name)


class Photo:
    def __init__(self, sc):
        self.im = np.asarray(Image.open(ROOT / sc["photo"]["file"]).convert("RGB")).astype(np.float32)
        self.sc_objects = sc.get("objects", [])
        self.sc_name = sc.get("name", "")
        c = sc["photo"]["camera"]
        self.K, self.R, self.t = (np.asarray(c[k], float) for k in ("K", "R", "t"))
        self.C = -self.R.T @ self.t
        self.h, self.w = self.im.shape[:2]

    def sample(self, P, fallback):
        """World points (..., 3) -> RGB uint8 (bilinear); points outside the photo take `fallback`."""
        sh = P.shape[:-1]
        pc = P.reshape(-1, 3) @ self.R.T + self.t
        z = pc[:, 2]
        uv = (pc @ self.K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
        u, v = uv[:, 0] - 0.5, uv[:, 1] - 0.5
        ok = (z > 0) & (u >= 0) & (v >= 0) & (u < self.w - 1) & (v < self.h - 1)
        u, v = np.clip(u, 0, self.w - 1.001), np.clip(v, 0, self.h - 1.001)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = (u - u0)[:, None], (v - v0)[:, None]
        I = self.im
        out = (I[v0, u0] * (1 - fu) * (1 - fv) + I[v0, u0 + 1] * fu * (1 - fv) +
               I[v0 + 1, u0] * (1 - fu) * fv + I[v0 + 1, u0 + 1] * fu * fv)
        out[~ok] = np.asarray(fallback[:3], float) * 255
        return np.clip(out, 0, 255).astype(np.uint8).reshape(*sh, 3)

    # ------------------------------------------------------------------ mask-gated sampling
    def _mask(self, oid):
        """SAM outline of object `oid` as a full-image bool mask, eroded 2 px (1 px for thin things) so the edge pixels
        (table, neighbours) drop out.  The high-res outline from out/tables_v6/masks_hi/<scene>.json (same pixel
        frame, with see-through holes) when present, else the scene's perception.mask_px.  None if neither exists."""
        if not hasattr(self, "_masks"):
            self._masks, self._med = {}, {}
            hi = {}
            f = ROOT / "out" / "tables_v6" / "masks_hi" / f"{self.sc_name}.json"
            if MASKS_HI and self.sc_name and f.exists():
                import json
                hi = json.loads(f.read_text())
            for o in self.sc_objects:
                poly = (hi.get("masks") or {}).get(o["id"]) or (o.get("perception") or {}).get("mask_px")
                if not poly or len(poly) < 3:
                    continue
                im = Image.new("L", (self.w, self.h), 0)
                dr = ImageDraw.Draw(im)
                dr.polygon([tuple(map(float, q)) for q in poly], fill=255)
                for hole in (hi.get("holes") or {}).get(o["id"]) or ():
                    if len(hole) >= 3:
                        dr.polygon([tuple(map(float, q)) for q in hole], fill=0)
                thin = THIN_MASK.search((o.get("category", "") + " " + o.get("label", "")).lower())
                im = im.filter(ImageFilter.MinFilter(3 if thin else 5))
                m = np.asarray(im) > 0
                if m.sum() < 30:
                    continue
                self._masks[o["id"]] = m
                self._med[o["id"]] = np.median(self.im[m], 0)
        return self._masks.get(oid)

    def sample_obj(self, P, fallback, oid):
        """Like sample(), but only photo pixels inside this object's own SAM outline are used; texels whose pixel
        falls on the table or on another object are filled from the nearest in-outline texels (or the object's median
        colour when the face sees none of it, e.g. a side the photo cannot see)."""
        img = self.sample(P, fallback)
        if not MASK_DECALS or oid is None or self._mask(oid) is None or img.ndim != 3:
            return img
        M = self._mask(oid)
        sh = P.shape[:-1]
        pc = P.reshape(-1, 3) @ self.R.T + self.t
        uv = (pc @ self.K.T)[:, :2] / np.maximum(pc[:, 2], 1e-6)[:, None]
        u = np.clip(np.round(uv[:, 0] - 0.5).astype(int), 0, self.w - 1)
        v = np.clip(np.round(uv[:, 1] - 0.5).astype(int), 0, self.h - 1)
        ok = (pc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < self.w) & (uv[:, 1] < self.h)
        valid = (ok & M[v, u]).reshape(sh)
        med = self._med[oid]
        if valid.mean() < 0.04:
            return np.clip(np.broadcast_to(med * 0.92, img.shape), 0, 255).astype(np.uint8).copy()
        out = img.astype(np.float32)
        out[~valid] = 0
        w = valid.astype(np.float32)
        acc, wt = out * w[..., None], w.copy()
        # push valid colour outward: repeated 3x3 averaging of known texels into unknown ones
        for _ in range(6):                 # only close small gaps; long smears read as streaks
            if wt.min() > 0:
                break
            S = np.zeros_like(acc); N = np.zeros_like(wt)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    S += np.roll(np.roll(acc, dy, 0), dx, 1)
                    N += np.roll(np.roll(wt, dy, 0), dx, 1)
            grow = (wt == 0) & (N > 0)
            acc[grow] = S[grow] / N[grow][:, None]
            wt[grow] = 1.0
        acc[wt == 0] = med * 0.95
        # soften the seam a little toward the object's median colour where texels were filled
        filled = ~valid
        acc[filled] = acc[filled] * 0.85 + med * 0.15
        return np.clip(acc, 0, 255).astype(np.uint8)

    def faces_camera(self, p, n, min_cos=0.0):
        v = self.C - p
        return float(np.dot(n, v)) > min_cos * float(np.linalg.norm(v)) * float(np.linalg.norm(n))


def _quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class Skin:
    """Collects decal meshes / cloth for one scene, then adds them to a fresh spec."""

    def __init__(self, sc, photo: Photo):
        self.sc, self.ph, self.n = sc, photo, 0
        self.decals = []        # (body name, verts (N,3) body frame, texcoords (N,2), faces, rgb image)
        self.cloths = []        # dict

    def _tex(self, img):
        self.n += 1
        f = TEX / f"{self.sc['name']}_{self.n:03d}.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(f)
        return f

    # ------------------------------------------------------------------ geometry helpers (pass-1 poses)
    def grid_decal(self, body, Pb, to_world, fallback, res):
        """Pb: (nv, nu, 3) grid of body-frame points; texture sampled at the same points, res = (rows, cols)."""
        nv, nu = Pb.shape[:2]
        rows, cols = res
        # texture: bilinear-resample the grid to res, then sample the photo there
        gv, gu = np.linspace(0, nv - 1, rows), np.linspace(0, nu - 1, cols)
        iv, iu = np.minimum(gv.astype(int), nv - 2), np.minimum(gu.astype(int), nu - 2)
        fv, fu = (gv - iv)[:, None, None], (gu - iu)[None, :, None]
        Pt = (Pb[iv][:, iu] * (1 - fv) * (1 - fu) + Pb[iv + 1][:, iu] * fv * (1 - fu) +
              Pb[iv][:, iu + 1] * (1 - fv) * fu + Pb[iv + 1][:, iu + 1] * fv * fu)
        img = self.ph.sample_obj(to_world(Pt), fallback, body)
        V = Pb.reshape(-1, 3)
        T = np.stack(np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv)), -1).reshape(-1, 2)
        F = []
        for j in range(nv - 1):
            for i in range(nu - 1):
                a, b, c, d = j * nu + i, j * nu + i + 1, (j + 1) * nu + i, (j + 1) * nu + i + 1
                F += [a, b, d, a, d, c]
        self.decals.append((body, V, T, F, img))

    def box(self, body, gpos, gR, size, to_world, R_body, fallback, eps=0.0008, skip_bottom=True):
        hx, hy, hz = size
        for ax in range(3):
            for sgn in (1, -1):
                nl = np.zeros(3); nl[ax] = sgn
                nb = gR @ nl
                nw = R_body @ nb
                if skip_bottom and nw[2] < -0.3:
                    continue
                c_b = gpos + gR @ (nl * np.array([hx, hy, hz]))
                if not self.ph.faces_camera(to_world(c_b[None])[0], nw, 0.3):   # edge-on faces would smear
                    continue
                a1, a2 = (ax + 1) % 3, (ax + 2) % 3
                e1, e2 = np.zeros(3), np.zeros(3)
                e1[a1], e2[a2] = (hx, hy, hz)[a1], (hx, hy, hz)[a2]
                if sgn < 0:
                    e1 = -e1                       # keep the triangle winding facing outward
                if np.linalg.norm(e1) < 0.004 or np.linalg.norm(e2) < 0.004:
                    continue
                s1, s2 = np.linspace(-1, 1, 2), np.linspace(-1, 1, 2)
                P = np.array([[c_b + nb * eps + gR @ (e1 * a + e2 * b) for a in s1] for b in s2])
                L1, L2 = 2 * np.linalg.norm(e1), 2 * np.linalg.norm(e2)
                k = float(np.clip(700 / max(L1, L2, 1e-3), 0, 2500))
                res = (int(np.clip(L2 * k, 8, 256)), int(np.clip(L1 * k, 8, 256)))
                self.grid_decal(body, P, to_world, fallback, res)

    def cylinder(self, body, gpos, gR, r, hh, to_world, R_body, fallback, top_dz=0.0, eps=0.0008, side=True):
        # top disc (as a radial grid so it can be a regular mesh grid)
        nt = 24
        th = np.linspace(0, 2 * np.pi, nt + 1)
        ztop = np.array([0, 0, 1.0])
        nw = R_body @ gR @ ztop
        c_b = gpos + gR @ (ztop * hh)
        if nw[2] > -0.3 and self.ph.faces_camera(to_world(c_b[None])[0], nw):
            rings = np.linspace(0.0, 1, 5)          # from the centre (no pin-hole)
            P = np.array([[c_b + gR @ np.array([r * q * np.cos(t), r * q * np.sin(t), eps]) for t in th]
                          for q in rings])
            Ps = P.copy()
            Ps[..., 2] += top_dz                       # sample a little higher (bowl contents)
            self._radial(body, P, Ps, to_world, fallback)
        if not side or hh < 0.01:
            return
        ang = np.linspace(0, 2 * np.pi, 49)
        vis = []
        for t in ang:
            nl = np.array([np.cos(t), np.sin(t), 0.0])
            p = gpos + gR @ (nl * r)
            vis.append(self.ph.faces_camera(to_world(p[None])[0], R_body @ gR @ nl, 0.25))
        vis = np.array(vis)
        if not vis.any():
            return
        # the visible arc (rotate so it is contiguous)
        k0 = int(np.argmin(vis)) if not vis.all() else 0
        idx = [(k0 + i) % 48 for i in range(49)]
        sel = [i for i in idx if vis[i]]
        if len(sel) < 3:
            return
        ts = ang[sel]
        ts = np.unwrap(ts)
        P = np.array([[gpos + gR @ np.array([(r + eps) * np.cos(t), (r + eps) * np.sin(t), z]) for t in ts]
                      for z in np.linspace(hh, -hh, 2)])
        circ = r * (ts[-1] - ts[0])
        kk = float(np.clip(700 / max(circ, 2 * hh, 1e-3), 0, 2500))
        self.grid_decal(body, P, to_world, fallback, (int(np.clip(2 * hh * kk, 8, 256)),
                                                      int(np.clip(circ * kk, 8, 256))))

    def _radial(self, body, P, Ps, to_world, fallback):
        """Disc as (rings x angles) grid; texture is a square top view sampled at Ps."""
        nr, na = P.shape[:2]
        c, rmax = Ps[0, 0], np.linalg.norm(Ps[-1, 0] - Ps[0, 0])
        e1 = (Ps[-1, 0] - c) / max(rmax, 1e-9)
        e3 = np.cross(e1, Ps[-1, na // 4] - c); e3 /= max(np.linalg.norm(e3), 1e-9)
        e2 = np.cross(e3, e1)
        res = int(np.clip(rmax * 2 * 700, 16, 256))
        g = np.linspace(-rmax, rmax, res)
        Q = c + g[None, :, None] * e1 + g[:, None, None] * e2
        img = self.ph.sample_obj(to_world(Q), fallback, body)
        V = P.reshape(-1, 3)
        T = np.array([[(np.dot(p - P[0, 0], e1) / rmax + 1) / 2, (np.dot(p - P[0, 0], e2) / rmax + 1) / 2]
                      for p in V])
        F = []
        for j in range(nr - 1):
            for i in range(na - 1):
                a, b, cc, d = j * na + i, j * na + i + 1, (j + 1) * na + i, (j + 1) * na + i + 1
                F += [a, b, d, a, d, cc]
        self.decals.append((body, V, T, F, img))

    # ------------------------------------------------------------------ apply to a fresh spec
    def apply(self, spec, hide_bodies):
        for k, (body, V, T, F, img) in enumerate(self.decals):
            f = self._tex(img)
            tn, mn, me = f"dtex{k}", f"dmat{k}", f"dmesh{k}"
            spec.add_texture(name=tn, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(f))
            mat = spec.add_material(name=mn)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tn
            mat.specular, mat.shininess = 0.1, 0.1
            mesh = spec.add_mesh(name=me)
            mesh.uservert = np.asarray(V, float).ravel().tolist()
            mesh.usertexcoord = np.asarray(T, float).ravel().tolist()
            F = list(map(int, F))
            F = F + [v for t in range(0, len(F), 3) for v in (F[t], F[t + 2], F[t + 1])]   # two-sided
            mesh.userface = F
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            spec.body(body).add_geom(type=G.mjGEOM_MESH, meshname=me, material=mn, contype=0, conaffinity=0,
                                     group=1, density=0)
        for name in hide_bodies:                          # cloth cores: still collide, not drawn
            b = spec.body(name)
            g = b.first_geom()
            while g is not None:
                g.rgba = [0, 0, 0, 0]
                g = b.next_geom(g)
        if self.cloths:
            # cloth (contype/conaffinity 2) touches only the table and the objects, not the arm or the floor
            def _aff(b):
                g = b.first_geom()
                while g is not None:
                    if g.contype:
                        g.conaffinity = int(g.conaffinity) | 2
                    g = b.next_geom(g)
            _aff(spec.body("table"))
            for o in self.sc["objects"]:
                if CLOTH_SOFT and o["id"] in hide_bodies:
                    continue          # soft towel: the sheet rests on the table, not on its hidden rigid core
                _aff(spec.body(o["id"]))
        for k, c in enumerate(self.cloths):
            f = self._tex(c["img"])
            child = mujoco.MjSpec.from_string(f'''<mujoco><asset>
<texture name="t" type="2d" file="{f}"/><material name="m" texture="t" specular="0.05" shininess="0.05"/></asset>
<worldbody><flexcomp name="c" type="grid" count="{c['nx']} {c['ny']} 1" spacing="{c['sx']:.5f} {c['sy']:.5f} 0.01"
 radius="{CLOTH_RADIUS}" dim="2" material="m" mass="{c['mass']:.3f}">
<contact condim="3" selfcollide="none" contype="2" conaffinity="2" solref="0.004 1"/><edge equality="true" damping="0.02"/></flexcomp>
</worldbody></mujoco>''')
            fr = spec.worldbody.add_frame(pos=c["pos"], quat=c["quat"])
            spec.attach(child, frame=fr, prefix=f"cloth{k}_")


def collect(sc, m, d, photo):
    """Pass-1 model/data -> Skin with decals and cloth for every object."""
    sk = Skin(sc, photo)
    hide = []
    for o in sc["objects"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        if bid < 0:
            continue
        pb, Rb = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()

        def to_world(P, pb=pb, Rb=Rb):
            return np.asarray(P) @ Rb.T + pb
        fb = o.get("rgba", [0.7, 0.7, 0.7, 1])
        if is_cloth(o):
            hide.append(o["id"])
            # footprint of the core in the world (all its geoms' AABBs)
            gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            top = max(d.geom_xpos[g][2] + (m.geom_size[g][2] if m.geom_type[g] == G.mjGEOM_BOX else
                                           m.geom_size[g][1]) for g in gids)
            sh = o["shape"]
            L, W = (sh["size"][:2] if "size" in sh else (2 * sh.get("radius", 0.05),) * 2)
            hgt = max(top, 0.008)
            Lc, Wc = L + 1.4 * hgt + 0.01, W + 1.4 * hgt + 0.01
            if CLOTH_SOFT:                    # a flat soft sheet the size of the towel itself (no drape margin)
                Lc, Wc = L * 1.02, W * 1.02
            nx = int(np.clip(round(Lc / 0.018) + 1, 6, 16))     # ~2 cm cells: wrinkles stay, cost stays low
            ny = int(np.clip(round(Wc / 0.018) + 1, 6, 16))
            yaw = float(np.arctan2(Rb[1, 0], Rb[0, 0]))
            cz = (0.03 if CLOTH_SOFT else top + 0.012)
            ex, ey = np.array([np.cos(yaw), np.sin(yaw), 0]), np.array([-np.sin(yaw), np.cos(yaw), 0])
            # texture: the photo at the top of the pile, over the cloth's (flat) extent
            res_x, res_y = int(np.clip(Lc * 900, 32, 384)), int(np.clip(Wc * 900, 32, 384))
            gx, gy = np.linspace(-Lc / 2, Lc / 2, res_x), np.linspace(-Wc / 2, Wc / 2, res_y)
            P = (np.array([pb[0], pb[1], top]) + gx[None, :, None] * ex + gy[:, None, None] * ey)
            # the draped sheet ends up smaller than its flat extent: sample a slightly shrunken footprint
            shrink = 0.80 if CLOTH_SOFT else 0.82    # sample inside the towel, so no table pixels on its rim
            shrink = CLOTH_SHRINK.get(o["id"], shrink)
            P = np.array([pb[0], pb[1], top]) + (P - np.array([pb[0], pb[1], top])) * np.array([shrink, shrink, 1])
            sk.cloths.append({"pos": [float(pb[0]), float(pb[1]), float(cz)],
                              "quat": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
                              "nx": nx, "ny": ny, "sx": Lc / (nx - 1), "sy": Wc / (ny - 1),
                              "mass": float(np.clip(L * W * 0.4, 0.01, 0.3)), "img": photo.sample_obj(P, fb, o["id"]),
                              "id": o["id"]})
            continue
        sh_type = o["shape"]["type"]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_rgba[g][3] == 0:
                continue
            gpos, gR = m.geom_pos[g].copy(), _quat_mat(m.geom_quat[g])
            col = m.geom_rgba[g]
            if m.geom_type[g] == G.mjGEOM_BOX:
                if sh_type == "bowl":
                    continue                               # thin wall segments: keep their photo colour
                sk.box(o["id"], gpos, gR, m.geom_size[g].copy(), to_world, Rb, col)
            elif m.geom_type[g] == G.mjGEOM_CYLINDER:
                r, hh = m.geom_size[g][0], m.geom_size[g][1]
                if sh_type == "bowl":
                    # the contents disc at the fill level inside the curved shell (hero_dress._bowl_shells, BOWL_FOOT)
                    H_ = float(o["shape"].get("height", 0.06))
                    zf, z1 = -H_ / 2 + 0.008, H_ / 2 - 0.003
                    zfood = zf + BOWL_FILL * (z1 - zf)
                    rr = (r - 0.006) * (BOWL_FOOT + (1 - BOWL_FOOT) * np.sin(BOWL_FILL * np.pi / 2)) * 0.97
                    lift = zfood - (gpos[2] + hh)
                    sk.cylinder(o["id"], gpos + np.array([0, 0, lift]), gR, min(rr, r * 0.97), hh, to_world, Rb, col,
                                top_dz=max(0.0, 0.5 * H_ - (zfood - zf)), side=False)
                else:
                    sk.cylinder(o["id"], gpos, gR, r, hh, to_world, Rb, col)
    return sk, hide


def dress(sc, base_spec, finish, settle, cloth_steps=450, seed=0):
    """-> (model, data) settled, with decals and cloth."""
    photo = Photo(sc)
    m1 = finish(base_spec(sc))
    d1 = settle(m1, sc)
    sk, hide = collect(sc, m1, d1, photo)
    spec = base_spec(sc)
    sk.apply(spec, hide)
    m = finish(spec)
    d = mujoco.MjData(m)
    d.qpos[:m1.nq] = d1.qpos
    d.ctrl[:m1.nu] = d1.ctrl
    # wrinkles: start each cloth vertex at a slightly different height
    rng = np.random.default_rng(seed)
    if m.nq > m1.nq:
        extra = d.qpos[m1.nq:]
        for j in range(m1.njnt, m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and abs(m.jnt_axis[j][2]) > 0.9:
                d.qpos[m.jnt_qposadr[j]] += rng.uniform(0, 0.012)
        del extra
    mujoco.mj_forward(m, d)
    for _ in range(cloth_steps if m.nq > m1.nq else 1):
        mujoco.mj_step(m, d)
    return m, d
