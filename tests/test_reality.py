"""POST /api/reality with a monkeypatched reality_check (no API calls)."""
import json

import mujoco
import pytest

import rehearsal.reality as reality
from rehearsal.pipeline import map_ids, pick_attempt, sim_final_from_traj
from rehearsal.scene import build_model, object_height

from .conftest import DESK_FIXTURE

MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64


def fake_traj(spec, moved=None):
    """One 'full' trajectory row: every object resting on the table at pos (or moved[id] = (x, y, z))."""
    m = build_model(spec)
    row = [3.0] + [0.0] * (7 * (m.nbody - 1))
    for o in spec["objects"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        x, y, z = (moved or {}).get(o["id"], (*o["pos"], object_height(o) / 2))
        row[1 + 7 * (b - 1): 4 + 7 * (b - 1)] = [x, y, z]
        row[4 + 7 * (b - 1)] = 1.0
    return [row]


@pytest.fixture
def rehearsed(root):
    """out/desk_01 with a 2-attempt report (attempt 2 succeeded) and its trajectory."""
    spec = json.loads(DESK_FIXTURE.read_text())
    for o in spec["objects"]:
        o["perception"] = {"box_px": [10, 10, 20, 20]}
    out = root / "out" / "desk_01"
    out.mkdir(parents=True)
    rep = {"scene": "desk_01", "scene_spec": spec, "attempts": [
        {"attempt": 1, "success": False}, {"attempt": 2, "success": True}]}
    (out / "report.json").write_text(json.dumps(rep))
    # attempt 2: mug ends up inside the bowl, 1 cm above the table
    (out / "attempt_2.traj.json").write_text(json.dumps(fake_traj(spec, {"mug_1": (0.42, -0.22, 0.06)})))
    (out / "attempt_1.traj.json").write_text(json.dumps(fake_traj(spec)))
    return spec


def test_helpers(rehearsed):
    sf = sim_final_from_traj(rehearsed, fake_traj(rehearsed, {"mug_1": (0.42, -0.22, 0.06)}))
    assert sf["mug_1"] == {"xy": [0.42, -0.22], "z_base": 0.01}
    assert sf["book_large"]["z_base"] == 0.0 and sf["book_large"]["xy"] == rehearsed["objects"][3]["pos"]
    assert pick_attempt({"attempts": [{"attempt": 1, "success": True}, {"attempt": 2, "success": False}]}, None) == 1
    with pytest.raises(ValueError):
        pick_attempt({"attempts": [{"attempt": 1, "success": False}]}, None)
    twin = {"name": "photo_x", "objects": [{"id": "cup", "pos": [0.43, 0.21]}, {"id": "far", "pos": [2, 2]}]}
    assert map_ids(rehearsed, twin) == {"cup": "mug_1"}


def _fake_check(calls, root):
    def reality_check(video_path, scene, sim_final, data="private", out_dir=None, **kw):
        calls.append(dict(video=video_path, scene=scene["name"], sim_final=sim_final, data=data, out_dir=out_dir))
        gif = root / "fake.reality.gif"
        gif.write_bytes(b"GIF89a fake")
        rows = [{"id": k, "sim_xy": v["xy"], "observed_xy": v["xy"], "delta_cm": 0.5, "status": "ok"}
                for k, v in sim_final.items()]
        return {"per_object": rows, "verdict": "consistent: every object ended within 3 cm of the simulation",
                "overlay": str(gif), "tol_cm": 3.0, "frames": 100, "camera": {"source": "test"}}
    return reality_check


def test_reality_endpoint(server, root, rehearsed, monkeypatch):
    c, srv = server
    calls = []
    monkeypatch.setattr(reality, "reality_check", _fake_check(calls, root))
    s, r = c.multipart("/api/reality", {"video": ("clip.mp4", MP4, "video/mp4"), "scene": "desk_01"})
    assert s == 200, r
    v = srv.mgr.wait(r["job"], 30)
    assert v["state"] == "done" and v["kind"] == "reality", v
    res = v["result"]
    assert res["attempt"] == 2 and res["tol_cm"] == 3.0 and res["verdict"].startswith("consistent")
    assert res["overlay"] == f"/uploads/{r['job']}_reality.gif" and c.get(res["overlay"])[0] == 200
    assert calls[0]["data"] == "private" and calls[0]["video"].endswith(f"{r['job']}.mp4")
    assert calls[0]["sim_final"]["mug_1"] == {"xy": [0.42, -0.22], "z_base": 0.01}
    rep = c.get("/api/report?scene=desk_01")[1]
    assert rep["reality"]["verdict"] == res["verdict"] and rep["reality"]["attempt"] == 2

    # explicit attempt + public + a separate perceived twin with its own ids
    twin = json.loads(json.dumps(rehearsed))
    twin["name"] = "photo_twin"
    for o in twin["objects"]:
        o["id"] = "t_" + o["id"]
        o["pos"] = [o["pos"][0] + 0.01, o["pos"][1]]
    (root / "scenes" / "photo_twin.json").write_text(json.dumps(twin))
    s, r = c.multipart("/api/reality", {"video": ("c.mp4", MP4, "video/mp4"), "scene": "desk_01",
                                        "attempt": "1", "privacy": "public", "twin": "photo_twin"})
    v = srv.mgr.wait(r["job"], 30)
    assert v["state"] == "done", v
    assert calls[1]["scene"] == "photo_twin" and calls[1]["data"] == "public"
    assert calls[1]["sim_final"]["t_mug_1"]["z_base"] == 0.0          # attempt 1: mug never moved
    assert v["result"]["id_map"]["t_mug_1"] == "mug_1"


def test_reality_needs_box_px(server, root, rehearsed, monkeypatch):
    c, srv = server
    monkeypatch.setattr(reality, "reality_check", _fake_check([], root))
    rp = root / "out" / "desk_01" / "report.json"
    rep = json.loads(rp.read_text())
    for o in rep["scene_spec"]["objects"]:
        o.pop("perception")
    rp.write_text(json.dumps(rep))
    s, r = c.multipart("/api/reality", {"video": ("c.mp4", MP4, "video/mp4"), "scene": "desk_01"})
    v = srv.mgr.wait(r["job"], 30)
    assert v["state"] == "error" and "box_px" in v["error"]


def test_reality_rejects_bad_requests(server, root, rehearsed):
    c, _ = server
    vid = ("c.mp4", MP4, "video/mp4")
    assert c.multipart("/api/reality", {"video": vid, "scene": "../x"})[0] == 400
    assert c.multipart("/api/reality", {"video": vid, "scene": "nope"})[0] == 404
    assert c.multipart("/api/reality", {"video": vid, "scene": "desk_01", "attempt": "7"})[0] == 400
    assert c.multipart("/api/reality", {"video": vid, "scene": "desk_01", "attempt": "-1"})[0] == 400
    assert c.multipart("/api/reality", {"video": vid, "scene": "desk_01", "privacy": "x"})[0] == 400
    assert c.multipart("/api/reality", {"video": vid, "scene": "desk_01", "twin": "nope"})[0] == 404
    assert c.multipart("/api/reality", {"video": ("x.mp4", b"not a video at all", "video/mp4"),
                                        "scene": "desk_01"})[0] == 400
    assert c.multipart("/api/reality", {"scene": "desk_01"})[0] == 400
    s, _, _ = c.raw("POST", "/api/reality", b"x", {"Content-Type": "multipart/form-data; boundary=x",
                                                 "Content-Length": str(31 * 1024 * 1024)})
    assert s == 413
