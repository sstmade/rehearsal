"""Convert a BOP-format frame (scene_camera / scene_gt / scene_gt_info / depth) into a Rehearsal
ground-truth JSON with camera-frame AND table-frame object positions.

Run with:  .venv/bin/python -I datasets/_tools/bop_to_json.py <config.json>
config: {"dataset": "hope", "license": ..., "source_url": ..., "names": {id: name},
         "models_info": path, "out_dir": path, "frames": [{"raw_dir", "im_id", "out_stem"}]}

Table plane: RANSAC on the depth map (pixels outside the GT object boxes), then refined by least squares
on inliers. Table frame (meters):
  origin = intersection of the camera optical axis with the plane,
  +z     = plane normal pointing towards the camera,
  +x     = camera +x (image right) projected onto the plane,
  +y     = z cross x (roughly "away from the camera" / image up).
Saved: T_table_from_cam (4x4), homography_px_to_table (3x3, pixel -> table xy in m, valid for points ON the plane).
"""
import json, os, shutil, sys
import numpy as np
from PIL import Image

rng = np.random.default_rng(0)


def fit_plane(P, bottoms, iters=3000, thr=0.008, tol=0.015):
    """RANSAC over depth triplets. Score = (#GT objects whose lowest bbox corner lies within `tol` of the
    plane AND that are not below it, #depth inliers). `bottoms`: list of (8,3) corner arrays per object."""
    best, best_key = None, (-1, -1)
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ s[0]
        if d < 0:
            n, d = -n, -d
        hmin = np.array([(c @ n + d).min() for c in bottoms])
        n_obj = int((np.abs(hmin) < tol).sum()) - int((hmin < -0.03).sum())
        inl = np.abs(P @ n + d) < thr
        key = (n_obj, int(inl.sum()))
        if key > best_key:
            best_key, best = key, inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    d = -n @ c
    if d < 0:  # make normal point towards camera (camera origin has signed distance d > 0)
        n, d = -n, -d
    return n, d, best_key[1] / len(P)


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    minfo = json.load(open(cfg["models_info"]))
    names = {int(k): v for k, v in cfg["names"].items()}
    os.makedirs(cfg["out_dir"], exist_ok=True)
    for fr in cfg["frames"]:
        raw, im = fr["raw_dir"], fr["im_id"]
        cam = json.load(open(os.path.join(raw, "scene_camera.json")))[str(int(im))]
        gts = json.load(open(os.path.join(raw, "scene_gt.json")))[str(int(im))]
        infos = json.load(open(os.path.join(raw, "scene_gt_info.json")))[str(int(im))]
        K = np.array(cam["cam_K"]).reshape(3, 3)
        dimg = np.array(Image.open(os.path.join(raw, f"depth_{im}.png"))).astype(np.float64)
        depth_m = dimg * cam.get("depth_scale", 1.0) / 1000.0
        H, W = depth_m.shape
        # exclude object boxes; use region spanning the objects (expanded) to avoid walls
        mask = depth_m > 0.1
        boxes = [i["bbox_obj"] for i in infos if i["visib_fract"] > 0]
        roi = np.zeros_like(mask)
        for b in boxes:  # ring around each object box (where its support surface is likely visible)
            ex_, ey_ = b[2] // 2 + 10, b[3] // 2 + 10
            roi[max(0, b[1] - ey_ // 2):min(H, b[1] + b[3] + ey_), max(0, b[0] - ex_):min(W, b[0] + b[2] + ex_)] = True
        for b in boxes:
            roi[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] = False
        corners_all = []
        for g in gts:
            mi = minfo[str(g["obj_id"])]
            R = np.array(g["cam_R_m2c"]).reshape(3, 3); t = np.array(g["cam_t_m2c"]) / 1000.0
            mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]]) / 1000.0
            sz = np.array([mi["size_x"], mi["size_y"], mi["size_z"]]) / 1000.0
            cn = np.array([[mn[0] + sz[0] * a, mn[1] + sz[1] * b, mn[2] + sz[2] * c]
                           for a in (0, 1) for b in (0, 1) for c in (0, 1)])
            corners_all.append(cn @ R.T + t)
        vv, uu = np.nonzero(mask & roi)
        step = max(1, len(uu) // 40000)
        uu, vv = uu[::step], vv[::step]
        z = depth_m[vv, uu]
        P = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
        n, d, inl_frac = fit_plane(P, corners_all)
        # table frame
        ez = n
        ex = np.array([1.0, 0, 0]) - n[0] * n; ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        origin = np.array([0, 0, 1.0]) * (-d / n[2])  # optical axis hits plane
        R_tc = np.stack([ex, ey, ez])  # rows: table axes in cam coords -> table_from_cam rotation
        t_tc = -R_tc @ origin
        T = np.eye(4); T[:3, :3] = R_tc; T[:3, 3] = t_tc
        # homography: table (x,y,0) -> cam point = R_tc^T [x,y,0] + origin -> pixel
        R_ct = R_tc.T
        Hm = K @ np.stack([R_ct[:, 0], R_ct[:, 1], origin], 1)
        Hinv = np.linalg.inv(Hm); Hinv /= Hinv[2, 2]
        cam_height = d  # distance camera->plane
        tilt = float(np.degrees(np.arccos(abs(n @ np.array([0, 0, -1.0])))))  # 0 = looking straight down
        objs = []
        for g, inf in zip(gts, infos):
            oid = g["obj_id"]; mi = minfo[str(oid)]
            R = np.array(g["cam_R_m2c"]).reshape(3, 3); t = np.array(g["cam_t_m2c"]) / 1000.0
            mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]]) / 1000.0
            sz = np.array([mi["size_x"], mi["size_y"], mi["size_z"]]) / 1000.0
            c_cam = R @ (mn + sz / 2) + t
            corners = np.array([[mn[0] + sz[0] * a, mn[1] + sz[1] * b, mn[2] + sz[2] * c]
                                for a in (0, 1) for b in (0, 1) for c in (0, 1)])
            cc = corners @ R.T + t
            h = cc @ n + d  # height above plane per corner
            c_tab = R_tc @ c_cam + t_tc
            up_axis = int(np.argmax(np.abs(R_tc @ R)[2]))  # model axis most aligned with table normal
            objs.append({
                "obj_id": oid, "name": names[oid],
                "visib_fract": round(inf["visib_fract"], 3),
                "bbox_obj_px": inf["bbox_obj"], "bbox_visib_px": inf["bbox_visib"],
                "center_cam_m": [round(v, 4) for v in c_cam],
                "center_table_m": [round(v, 4) for v in c_tab],
                "table_xy_m": [round(c_tab[0], 4), round(c_tab[1], 4)],
                "bottom_height_above_plane_m": round(float(h.min()), 4),
                "top_height_above_plane_m": round(float(h.max()), 4),
                "on_table": bool(abs(h.min()) < 0.02),
                "size_m_model_axes": [round(v, 4) for v in sz],
                "model_axis_closest_to_table_normal": "xyz"[up_axis],
                "R_model_to_cam": np.round(R, 6).tolist(), "t_model_to_cam_m": np.round(t, 5).tolist(),
            })
        img_name = f"{fr['out_stem']}.png"
        shutil.copy(os.path.join(raw, f"rgb_{im}.png"), os.path.join(cfg["out_dir"], img_name))
        shutil.copy(os.path.join(raw, f"depth_{im}.png"), os.path.join(cfg["out_dir"], f"{fr['out_stem']}_depth.png"))
        out = {
            "image": img_name, "depth": f"{fr['out_stem']}_depth.png",
            "depth_units": f"uint16 * {cam.get('depth_scale', 1.0)} = millimeters",
            "dataset": cfg["dataset"], "license": cfg["license"], "source_url": cfg["source_url"],
            "source_member": fr["member"], "width": W, "height": H,
            "camera": {"K": K.round(4).tolist(), "model": "pinhole, no distortion (BOP convention)"},
            "gt_frame": "BOP: object pose = model->camera (OpenCV camera: x right, y down, z forward); "
                        "positions here converted to meters; center_* = model 3D-bbox center",
            "table_plane": {
                "method": "RANSAC on GT depth around (not inside) GT object boxes, hypotheses scored first by #GT objects resting on the plane, then LSQ refit on depth inliers (derived by Rehearsal scout, not part of the dataset)",
                "normal_cam": np.round(n, 5).tolist(), "d_m": round(float(d), 4),
                "equation": "normal_cam . X_cam + d = 0 (normal points to camera)",
                "inlier_fraction": round(float(inl_frac), 3),
                "camera_height_above_plane_m": round(float(cam_height), 4),
                "camera_tilt_from_vertical_deg": round(tilt, 1),
                "T_table_from_cam": np.round(T, 6).tolist(),
                "homography_px_to_table": np.round(Hinv, 9).tolist(),
                "table_frame": "origin = optical axis ∩ plane; +z = plane normal (towards camera); "
                               "+x = image-right projected on plane; +y = z × x (away from camera)",
            },
            "objects": objs,
        }
        with open(os.path.join(cfg["out_dir"], f"{fr['out_stem']}.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        bh = [o["bottom_height_above_plane_m"] for o in objs]
        print(fr["out_stem"], f"tilt={tilt:.0f}deg h={cam_height:.2f}m inl={inl_frac:.2f} n_obj={len(objs)} "
              f"on_table={sum(o['on_table'] for o in objs)} bottom_h med={sorted(bh)[len(bh)//2]:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
