"""Blender: object mask (white = bodies named in --bodies, black elsewhere) of one shot frame + projected point.
  blender -b X.blend -P mask_render.py -- --shots S.json --frame N --bodies bucket_1 --res 1600x900 --out M.png
     [--point x,y,z]   (prints POINT_PX u v in output pixels)
"""
import json, sys
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
argv = sys.argv[sys.argv.index("--") + 1:]
arg = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
scn = bpy.context.scene
w, h = (int(v) for v in arg("--res").split("x"))
scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = w, h, 100
scn.frame_set(int(arg("--frame")))
bodies = {f"B_{b}" for b in arg("--bodies").split(",")}
def under(o):
    while o is not None:
        if o.name in bodies:
            return True
        o = o.parent
    return False
white = bpy.data.materials.new("mw"); white.use_nodes = True
black = bpy.data.materials.new("mb"); black.use_nodes = True
for m, c in ((white, (1, 1, 1, 1)), (black, (0, 0, 0, 1))):
    nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs[0].default_value = c; e.inputs[1].default_value = 1.0
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(e.outputs[0], o.inputs[0])
n_white = 0
for ob in scn.objects:
    if ob.type not in ("MESH", "CURVE"):
        continue
    mat = white if under(ob) else black
    n_white += mat is white
    ob.data.materials.clear(); ob.data.materials.append(mat)
    ob.modifiers.clear() if False else None
for ob in scn.objects:
    if ob.type == "LIGHT":
        ob.hide_render = True
scn.world.use_nodes = True
bg = next(n for n in scn.world.node_tree.nodes if n.type == "BACKGROUND"); bg.inputs[0].default_value = (0, 0, 0, 1)
for l in list(bg.outputs[0].links):
    pass
scn.render.engine = "CYCLES"; scn.cycles.samples = 4; scn.cycles.use_denoising = False
scn.view_settings.view_transform = "Standard"; scn.view_settings.look = "None"; scn.view_settings.exposure = 0
scn.render.use_motion_blur = False
try:
    scn.camera.data.dof.use_dof = False
except Exception:
    pass
scn.render.filepath = arg("--out"); scn.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("MASK_WHITE_OBJECTS", n_white)
if arg("--point"):
    p = Vector([float(v) for v in arg("--point").split(",")])
    co = world_to_camera_view(scn, scn.camera, p)
    print("POINT_PX", co.x * w, (1 - co.y) * h, co.z)
