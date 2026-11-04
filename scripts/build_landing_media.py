"""Compress the landing-page media into web/app/media/ (public test tables only; never the user's own desk).

    .venv/bin/python scripts/build_landing_media.py

Sources (footage/, all from the 50 AI-generated test tables):
  scenes/t6_NN.json photo + dark/stills/NN_view.png -> media/t/NN_p.jpg, NN_v.jpg   (source photo vs its dark-studio twin)
  tables_sim/hero/exec_dark/NN.mp4 (+ .json)        -> media/clips/NN.mp4 + NN.jpg  (recorded executions, real verdicts)
  tables_sim/hero/exec_blender/06_cycles.mp4        -> media/hero_06_cycles.mp4     (Cycles re-render of the 06 run)
  tables_sim/hero/exec_primitive/06.mp4             -> media/06_primitive.mp4       (the same run, physics primitives)
  manifest.json with each clip's instruction + verdict, read from the run's JSON.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FOOT = ROOT.parent / "footage" / "tables_sim"
OUT = ROOT / "web" / "app" / "media"
CLIPS = ["06", "07", "41", "08", "31", "21", "43", "16", "44", "24", "45", "09", "50", "47"]   # 7 sent, 7 not sent


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def enc(src: Path, dst: Path, width: int, crf: int, ss: float = 0, t: float | None = None):
    if dst.is_file() and dst.stat().st_mtime > src.stat().st_mtime:
        return
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-ss", str(ss), "-i", str(src)]
    if t:
        cmd += ["-t", str(t)]
    cmd += ["-vf", f"scale={width}:-2:flags=lanczos", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(dst)]
    subprocess.run(cmd, check=True)


def poster(src: Path, dst: Path, width: int, at: float):
    if dst.is_file():
        return
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", str(at), "-i", str(src), "-frames:v", "1",
                    "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)], check=True)


def main():
    (OUT / "t").mkdir(parents=True, exist_ok=True)
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    tables = []
    for i in range(1, 51):
        n = f"{i:02d}"
        sf = ROOT / "scenes" / f"t6_{n}.json"
        spec = json.loads(sf.read_text()) if sf.is_file() else {}
        ph = ROOT / spec.get("photo", {}).get("file", "-")          # the source photo (AI-generated test table)
        vw = FOOT / "dark" / "stills" / f"{n}_view.png"
        if not (ph.is_file() and vw.is_file()):
            continue
        for src, suf in ((ph, "p"), (vw, "v")):
            dst = OUT / "t" / f"{n}_{suf}.jpg"
            if not dst.is_file():
                im = Image.open(src).convert("RGB")
                im.thumbnail((560, 560), Image.LANCZOS)
                im.save(dst, quality=80, optimize=True, progressive=True)
        scene = json.loads((ROOT / "scenes" / f"t6_{n}.json").read_text()) if (ROOT / "scenes" / f"t6_{n}.json").is_file() else {}
        tables.append({"n": n, "title": scene.get("title", ""), "objects": len(scene.get("objects", []))})
    clips = []
    for n in CLIPS:
        src = FOOT / "hero" / "exec_dark" / f"{n}.mp4"
        meta = FOOT / "hero" / "exec_dark" / f"{n}.json"
        if not src.is_file():
            continue
        d = json.loads(meta.read_text()) if meta.is_file() else {}
        enc(src, OUT / "clips" / f"{n}.mp4", 720, 30, ss=1.0, t=11)
        poster(src, OUT / "clips" / f"{n}.jpg", 720, 4.0)
        clips.append({"n": n, "instruction": d.get("instruction"), "planned": bool(d.get("planned")),
                      "success": bool(d.get("success")), "error": (d.get("execution_error") or "")[:140]})
    enc(FOOT / "hero" / "exec_blender" / "06_cycles.mp4", OUT / "hero_06_cycles.mp4", 1600, 27)
    poster(FOOT / "hero" / "exec_blender" / "06_cycles.mp4", OUT / "hero_06_cycles.jpg", 1600, 12.0)
    enc(FOOT / "hero" / "exec_primitive" / "06.mp4", OUT / "06_primitive.mp4", 1280, 28)
    enc(FOOT / "hero" / "exec_dark" / "06.mp4", OUT / "06_dark.mp4", 1280, 28)
    (OUT / "manifest.json").write_text(json.dumps({"tables": tables, "clips": clips}, indent=1))
    tot = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{len(tables)} tables, {len(clips)} clips, {tot / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
