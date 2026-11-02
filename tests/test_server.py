import io
import json

import pytest
from PIL import Image

import rehearsal.perception as perception


def test_scenes_gallery_scene(server, root):
    c, _ = server
    assert c.get("/api/scenes") == (200, ["desk_01"])
    s, g = c.get("/api/gallery")
    assert s == 200 and g[0]["scene"] == "desk_01" and g[0]["has_report"] is False
    s, png = c.get(g[0]["thumb"])
    assert s == 200 and png[:4] == b"\x89PNG"
    s, spec = c.get("/api/scene?name=desk_01")
    assert s == 200 and spec["name"] == "desk_01"
    assert c.get("/api/scene?name=nope")[0] == 404
    assert c.get("/api/report?scene=desk_01") == (200, None)
    assert c.get("/api/status")[1]["running"] is False
    assert c.get("/api/job?id=zzz")[0] == 404
    assert c.get("/api/nope")[0] == 404
    s, html = c.get("/")
    assert s == 200 and b"<title>" in html


def test_budget(server):
    c, _ = server
    s, b = c.get("/api/budget")
    assert s == 200 and {"spent", "cap", "remaining"} <= set(b)


def test_post_scene(server, root):
    c, _ = server
    spec = json.loads((root / "scenes" / "desk_01.json").read_text())
    spec["name"] = "edited_1"
    assert c.post("/api/scene", spec) == (200, {"name": "edited_1"})
    assert (root / "scenes" / "edited_1.json").exists()
    spec["objects"][0]["pos"] = [9, 9]
    s, e = c.post("/api/scene", spec)
    assert s == 400 and "not on the table" in e["error"]
    spec["name"] = "../evil"
    assert c.post("/api/scene", spec)[0] == 400
    s, _, _ = c.raw("POST", "/api/scene", b"{not json", {"Content-Type": "application/json"})
    assert s == 400


@pytest.mark.parametrize("path", [
    "/out/../secret.txt", "/out/%2e%2e/secret.txt", "/photos/..%2fsecret.txt",
    "/web/../../etc/passwd", "/uploads/../secret.txt", "/out//etc/passwd", "/secret.txt",
    "/scenes/desk_01.json", "/out/.hidden", "/web/%00index.html",
])
def test_static_traversal_rejected(server, path):
    c, _ = server
    s, _, body = c.raw("GET", path)
    assert s in (400, 404) and b"top secret" not in body


@pytest.mark.parametrize("q", ["/api/scene?name=../secret", "/api/scene?name=..%2Fscenes%2Fdesk_01",
                               "/api/report?scene=../../x", "/api/scene?name=a/b"])
def test_name_traversal_rejected(server, q):
    c, _ = server
    assert c.get(q)[0] == 400


def test_rehearse_validation(server):
    c, _ = server
    assert c.post("/api/rehearse", {"scene": "../x", "instruction": "hi"})[0] == 400
    assert c.post("/api/rehearse", {"scene": "nope", "instruction": "hi"})[0] == 404
    assert c.post("/api/rehearse", {"scene": "desk_01", "instruction": ""})[0] == 400
    assert c.post("/api/rehearse", {"scene": "desk_01", "instruction": "x", "stress": 10**6})[0] == 400


def test_rehearse_offline_desk01(server, root):
    """Full pipeline, offline agent, small stress test."""
    c, srv = server
    s, r = c.post("/api/rehearse", {"scene": "desk_01", "instruction": "把杯子放进碗里，然后把书摞整齐",
                                    "stress": 4, "look": 0})
    assert s == 200 and r["started"]
    st = c.get("/api/status")[1]
    assert st["running"] and st["job"] == r["job"]
    v = srv.mgr.wait(r["job"], 300)
    assert v["state"] == "done", v["error"] or v["log"][-20:]
    res = v["result"]
    assert res["scene"] == "desk_01" and res["success"] is True and res["stress"]["n"] == 4
    assert res["agent"] == "offline-stand-in + search (hybrid)" and res["planner"] == "hybrid"
    assert res["status"] == "rehearsed" and res["assumptions"]
    assert any("data=private" in line for line in v["log"])  # fixture has no privacy field
    assert v["progress"]["fraction"] == 1.0
    s, rep = c.get("/api/report?scene=desk_01")
    assert s == 200 and rep["stress_test"]["n"] == 4 and rep["active_perception"] is None
    assert rep["status"] == "rehearsed" and rep["intent"]["status"] == "goals"
    assert rep["intent"]["gate"]["verdict"] == "pass" and rep["intent"]["gate"]["preview"]["moves"]
    for f in ["attempt_1.gif", "scene3d.json", "report.md", "stress_trails.json"]:
        assert (root / "out" / "desk_01" / f).exists(), f
    assert c.get("/out/desk_01/attempt_1.gif")[0] == 200
    g = c.get("/api/gallery")[1][0]
    assert g["has_report"] and "/out/desk_01/thumb.png" in g["thumb"]
    st = c.get("/api/status")[1]
    assert st["running"] is False and st["state"] == "done"


def _png(w=64, h=48):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, "PNG")
    return buf.getvalue()


def test_perceive_with_mock(server, root, monkeypatch):
    c, srv = server
    calls = {}
    desk = json.loads((root / "scenes" / "desk_01.json").read_text())

    def fake(path, privacy, corners=None):
        calls.update(path=path, privacy=privacy, corners=corners)
        overlay = root / "overlay_tmp.png"
        overlay.write_bytes(_png())
        return dict(desk, name="whatever"), str(overlay)

    monkeypatch.setattr(perception, "perceive_photo", fake, raising=False)
    corners = [[1, 2], [3, 4], [5, 6], [7, 8]]
    s, r = c.multipart("/api/perceive", {"photo": ("desk.png", _png(), "image/png"),
                                         "paper_corners": json.dumps(corners)})
    assert s == 200, r
    v = srv.mgr.wait(r["job"], 10)
    assert v["state"] == "done", v
    res = v["result"]
    assert res["scene"] == f"photo_{r['job']}" and res["valid"] and res["scene_spec"]["privacy"] == "private"
    assert calls["privacy"] == "private" and calls["corners"] == corners
    assert calls["path"].endswith(f"uploads/{r['job']}.png")
    assert (root / "scenes" / f"{res['scene']}.json").exists()
    s, png = c.get(res["overlay"])
    assert s == 200 and png[:4] == b"\x89PNG"
    assert res["scene"] in c.get("/api/scenes")[1]

    # invalid perception output is returned for editing but not saved
    def off_table(path, privacy, corners=None):
        bad = json.loads(json.dumps(desk))
        bad["objects"][0]["pos"] = [7, 7]
        return bad, None

    monkeypatch.setattr(perception, "perceive_photo", off_table, raising=False)
    s, r = c.multipart("/api/perceive", {"photo": ("d.png", _png(), "image/png"), "privacy": "public"})
    v = srv.mgr.wait(r["job"], 10)
    assert v["state"] == "done" and v["result"]["valid"] is False and v["result"]["overlay"] is None
    assert not (root / "scenes" / f"{v['result']['scene']}.json").exists()


def test_perceive_missing_function(server, monkeypatch):
    c, srv = server
    monkeypatch.delattr(perception, "perceive_photo", raising=False)
    s, r = c.multipart("/api/perceive", {"photo": ("d.png", _png(), "image/png")})
    v = srv.mgr.wait(r["job"], 10)
    assert v["state"] == "error" and "perceive_photo" in v["error"]


def test_perceive_rejects_bad_uploads(server):
    c, _ = server
    assert c.multipart("/api/perceive", {"photo": ("x.png", b"not an image", "image/png")})[0] == 400
    assert c.multipart("/api/perceive", {"other": "1"})[0] == 400
    assert c.multipart("/api/perceive", {"photo": ("d.png", _png(), "image/png"),
                                         "paper_corners": "[[1,2]]"})[0] == 400
    assert c.multipart("/api/perceive", {"photo": ("d.png", _png(), "image/png"), "privacy": "x"})[0] == 400
    # size limit is enforced from Content-Length before reading the body
    s, _, body = c.raw("POST", "/api/perceive", b"x" * 10,
                       {"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(16 * 1024 * 1024)})
    assert s == 413
