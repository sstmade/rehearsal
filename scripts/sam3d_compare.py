"""photo | physics shapes | SAM 3D meshes, all three through the photo's own camera (K, R, t).

Renders the twin with the web viewer itself (web/replay.js, Three.js: the GLBs keep their vertex colours) in headless
Chromium, from the calibrated photo camera, robot hidden, and writes <out>.sam3d_compare.png. Needs a running
`python -m rehearsal.server` and an up-to-date out/<scene>/scene3d.json (`python -m rehearsal.export3d ...`) for the
`visual_mesh` entries. Also prints the per-object photo IoU of each oriented mesh (from assets/sam3d/<scene>/*.json).

    .venv/bin/python scripts/sam3d_compare.py aigen_kitchen photos/ai_gen/aigen_kitchen [--port 8786] [--title "..."]
"""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]

JS = """async ({ name, W, H, meshes }) => {
  const { Replay, shapeBBox } = await import('/web/replay.js');
  const spec = await (await fetch(`/api/scene?name=${name}`)).json();
  const desc = await (await fetch(`/out/${name}/scene3d.json?nc=${Date.now()}`)).json();
  const host = document.createElement('div');
  host.style.cssText = `position:fixed;left:0;top:0;width:${W}px;height:${H}px;`;
  document.body.innerHTML = ''; document.body.appendChild(host);
  const r = new Replay(host, {}); r.renderer.setAnimationLoop(null); r.renderer.setPixelRatio(1);
  r.setTheme('light');
  await r.loadSpec(spec, null, null);
  r.grid.visible = false;
  if (meshes) {
    const byName = Object.fromEntries(desc.bodies.filter((b) => b.visual_mesh).map((b) => [b.name, b.visual_mesh]));
    const list = spec.objects.map((o, k) => {
      const vm = byName[o.id]; if (!vm) return null;
      const bb = shapeBBox(o.shape);   // spec bodies sit on the table (origin at the bottom), recorded ones at the bbox centre
      return { id: 1001 + k, name: o.id, visual_mesh: { ...vm, pos: [vm.pos[0], vm.pos[1], vm.pos[2] + bb[2] / 2] } };
    }).filter(Boolean);
    await r._loadVisualMeshes(list);
    r.setMeshMode(true);
  }
  const p = spec.photo;
  r.setCameraKRt(p.camera.K, p.camera.R, p.camera.t, p.width, p.height);
  r.renderOnce();
  return r.renderer.domElement.toDataURL('image/png');
}"""


async def shots(name, port, w, h):
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=angle", "--ignore-gpu-blocklist"])
        out = []
        for meshes in (False, True):
            pg = await b.new_page(viewport={"width": w, "height": h})
            pg.on("console", lambda m: print("console:", m.text) if m.type == "error" else None)
            await pg.goto(f"http://127.0.0.1:{port}/web/index.html?still")
            url = await pg.evaluate(JS, {"name": name, "W": w, "H": h, "meshes": meshes})
            out.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
            await pg.close()
        await b.close()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out_stem")
    ap.add_argument("--port", type=int, default=8786)
    ap.add_argument("--photo-title", default="Photo (AI-generated test image)")
    a = ap.parse_args()
    spec = json.loads((ROOT / "scenes" / f"{a.scene}.json").read_text())
    ph = spec["photo"]
    W, H = ph["width"], ph["height"]
    scale = min(1.0, 960 / W)
    w, h = round(W * scale), round(H * scale)
    phys, mesh = asyncio.run(shots(a.scene, a.port, W, H))
    photo = Image.open(ROOT / ph["file"]).convert("RGB")
    panels = [im.resize((w, h), Image.LANCZOS) for im in (photo, phys, mesh)]
    titles = [a.photo_title, "Twin: physics shapes (what MuJoCo simulates)",
              "Twin: Meta SAM 3D Objects meshes (visual only)"]
    bar, gap = 26, 6
    out = Image.new("RGB", (3 * w + 2 * gap, h + bar), (14, 16, 20))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, t) in enumerate(zip(panels, titles)):
        x = i * (w + gap)
        out.paste(im, (x, bar))
        d.text((x + 6, 5), t, fill=(230, 232, 236), font=font)
    out.save(f"{a.out_stem}.sam3d_compare.png")
    print(f"wrote {a.out_stem}.sam3d_compare.png")
    for o in spec["objects"]:
        side = ROOT / "assets" / "sam3d" / a.scene / f"{o['id']}.json"
        if side.exists():
            m = json.loads(side.read_text())
            print(f"  {o['id']:20} status {m.get('status')} use {m.get('use', True)} "
                  f"photo IoU {m.get('orientation', {}).get('iou')} shape_err {m.get('orientation', {}).get('shape_err')}")
        else:
            print(f"  {o['id']:20} primitive (no mesh)")


if __name__ == "__main__":
    main()
