# exec'd by build_refined.py (globals shared): the cloth as a Mesh Cache driven sheet
FL = D["flex"][SRC].astype(np.float32)                                         # (NF, nv, 3) world coords
nv = int(M["flex_vertnum"][0])
E = M["flex_elem"].reshape(-1, 3)[:int(M["flex_elemnum"][0])].astype(int)
ta = int(M["flex_texcoordadr"][0]) if "flex_texcoordadr" in M else 0
TC = M["flex_texcoord"][ta:ta + nv]
pc2 = EXP / f"{KEY}_towel_flex.pc2"
with open(pc2, "wb") as f:
    f.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, nv, 0.0, 1.0, NF))
    f.write(FL.astype("<f4").tobytes())
me = bpy.data.meshes.new("towel")
me.from_pydata(FL[0].tolist(), [], E.tolist())
me.update()
# consistent upward normals
for p in me.polygons:
    pass
uvl = me.uv_layers.new(name="UVMap")
loops_v = np.empty(len(me.loops), np.int64)
me.loops.foreach_get("vertex_index", loops_v)
T2 = np.c_[TC[loops_v, 0], 1.0 - TC[loops_v, 1]]
uvl.data.foreach_set("uv", T2.astype(np.float32).ravel())
me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
towel = bpy.data.objects.new("towel_flex", me)
C_OBJ.objects.link(towel)
mc = towel.modifiers.new("flex_cache", "MESH_CACHE")
mc.cache_format = "PC2"
mc.filepath = str(pc2)
mc.frame_start = 0.0
mc.play_mode = "SCENE"
mc.forward_axis, mc.up_axis = "POS_Y", "POS_Z"
sub = towel.modifiers.new("subdiv", "SUBSURF")
sub.levels, sub.render_levels = 2, 3
sub.uv_smooth = "PRESERVE_BOUNDARIES"
sol = towel.modifiers.new("thickness", "SOLIDIFY")
sol.thickness = 0.0042
sol.offset = -1.0
sol.use_even_offset = False
sol.thickness_clamp = 2.0           # folded cloth: no solidify spikes on degenerate faces
sol.use_rim = True
sol.use_quality_normals = True
P0 = FL[0][:, :2] - FL[0][:, :2].mean(0)
_, _, Vt = np.linalg.svd(P0, full_matrices=False)
ext = (P0 @ Vt.T).max(0) - (P0 @ Vt.T).min(0)                          # towel extent along its own axes
L_t, W_t = float(max(ext)), float(min(ext))
# which UV axis runs along the long side
du = np.ptp(TC[:, 0]) / max(np.ptp(P0 @ Vt[0]), 1e-6)
if np.corrcoef(TC[:, 0], P0 @ Vt[0])[0, 1] ** 2 < 0.5:
    L_t, W_t = W_t, L_t
towel.data.materials.append(simple("prim_towel", lin(tuple(float(v) * 255 for v in M["flex_rgba"][0][:3])), rough=0.95)
                            if PRIM else towel_mat(L_t, W_t))
# keep the sheet's normals pointing up
up = sum(p.normal.z for p in me.polygons)
if up < 0:
    me.flip_normals()
