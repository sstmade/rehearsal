"""Hero stills from the built scene (06_cycles.blend).

blender -b footage/tables_sim/hero/exec_blender/06_cycles.blend -P scripts/blender06/render_stills.py -- \
        [--samples 256] [--only NAME] [--res 1920x1080] [--out DIR]
"""
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(n, d=None):
    return argv[argv.index(n) + 1] if n in argv else d


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(arg("--out", str(ROOT / "footage/tables_sim/hero/exec_blender")))
scn = bpy.context.scene

# name: (frame, camera position, look-at target, lens mm, f-stop, focus point (None = target))
STILLS = {
    # grasping the first onion, the towel (MuJoCo flex cloth) sharp in the foreground, gripper just behind
    "still_towel_gripper_closeup": (268, (0.58, 0.80, 0.30), (0.44, 0.22, 0.06), 30, 4.0, (0.53, 0.42, 0.01)),
    # the gripper lowering the last onion into the steel bowl, low and tight
    "still_bowl_release_macro": (606, (0.24, -0.52, 0.20), (0.545, -0.245, 0.10), 70, 2.4, None),
    # the hero camera at the moment the third onion goes in
    "still_hero_cam": (606, None, None, None, 4.0, None),
    # wide studio shot: the whole rig on the glowing grid
    "still_studio_wide": (420, (-1.45, -1.55, 0.95), (0.30, 0.0, -0.05), 32, 5.6, (0.38, 0.0, 0.10)),
}


def main():
    cy = scn.cycles
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for dv in prefs.devices:
        dv.use = dv.type != "CPU"
    cy.device = "GPU"
    cy.samples = int(arg("--samples", 256))
    cy.adaptive_threshold = float(arg("--thresh", 0.012))
    res = arg("--res", "1920x1080").split("x")
    scn.render.resolution_x, scn.render.resolution_y = int(res[0]), int(res[1])
    scn.render.use_overwrite = True
    scn.render.use_motion_blur = False
    only = arg("--only")
    hero = scn.camera
    for nm, (fr, pos, tgt, lens, fstop, foc) in STILLS.items():
        if only and nm not in only.split(","):
            continue
        scn.frame_set(fr)
        if pos is None:
            scn.camera = hero
            hero.data.dof.aperture_fstop = fstop
        else:
            cd = bpy.data.cameras.new(nm)
            cd.lens, cd.sensor_fit, cd.sensor_width = lens, "HORIZONTAL", 36.0
            cd.clip_start = 0.01
            o = bpy.data.objects.new(nm, cd)
            scn.collection.objects.link(o)
            o.location = pos
            d = Vector(tgt) - Vector(pos)
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = d.normalized().to_track_quat("-Z", "Y")
            cd.dof.use_dof = True
            cd.dof.aperture_fstop = fstop
            fp = Vector(foc if foc is not None else tgt)
            cd.dof.focus_distance = (fp - Vector(pos)).dot(d.normalized())
            scn.camera = o
        scn.render.filepath = str(OUT / f"{nm}.png")
        bpy.ops.render.render(write_still=True)
        print("STILL", nm, flush=True)


main()
