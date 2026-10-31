"""Minimal Meta API probes for docs/meta-platform.md (metaspec role, allowance $0.50).

Every paid call reserves and settles through rehearsal.meta_client's budget ledger
(_reserve/_settle), so the shared cap and ledger stay correct. Inputs are synthetic only;
Muse Spark runs in Contributor mode (muse-spark-1.3-contributor). Raw responses are saved
next to this file as probe_<name>.json / .txt.

    .venv/bin/python -m experiments.meta_probe.probe tools_json      # chat.completions: tools + json_schema
    .venv/bin/python -m experiments.meta_probe.probe responses_tool  # Responses: tools + text.format + summary
    .venv/bin/python -m experiments.meta_probe.probe sam_image       # SAM masks on a 160x120 synthetic image
    .venv/bin/python -m experiments.meta_probe.probe sam_video       # SAM concept tracking on a 50-frame clip
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

from rehearsal import meta_client as mc

HERE = Path(__file__).resolve().parent
MODEL = "muse-spark-1.3-contributor"


def save(name: str, obj):
    p = HERE / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1, default=str))
    print("saved", p.name)


def chat_cost(model: str, prompt: int, cached: int, completion: int) -> float:
    p_in, p_c, p_out = mc.TOKEN_PRICES[model]
    return ((prompt - cached) * p_in + cached * p_c + completion * p_out) / 1e6


SIM_TOOL_PARAMS = {
    "type": "object",
    "properties": {"actions": {"type": "array", "items": {
        "type": "object",
        "properties": {"obj": {"type": "string"}, "dest": {"type": "string"},
                       "relation": {"type": "string", "enum": ["in", "on"]}},
        "required": ["obj", "dest", "relation"], "additionalProperties": False}}},
    "required": ["actions"], "additionalProperties": False,
}
FINAL_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["ship", "revise"]},
                   "actions": SIM_TOOL_PARAMS["properties"]["actions"],
                   "reason": {"type": "string"}},
    "required": ["verdict", "actions", "reason"], "additionalProperties": False,
}
SYSTEM = ("You plan pick-and-place for a robot arm. Before answering you MUST call simulate(actions) "
          "to test your plan in a physics simulator, then give the final answer in the required JSON.")
USER = "Scene: mug_1 (ceramic, fragile), bowl_1 (wide), box_1. Task: put the mug in the bowl."
FAKE_SIM = {"success": False, "events": ["t=2.1s mug_1 slipped from gripper during transit (friction 0.31)"],
            "hint": "lower transit_height or slow down"}


def probe_tools_json():
    """Does muse-spark-1.3 accept tools + response_format json_schema in one chat.completions call,
    call the tool first, then answer in the schema after the tool result?"""
    cl = mc.client()
    tools = [{"type": "function", "function": {"name": "simulate", "description":
              "Run the plan in the MuJoCo digital twin; returns success and an event log.",
              "parameters": SIM_TOOL_PARAMS, "strict": True}}]
    rf = {"type": "json_schema", "json_schema": {"name": "Final", "schema": FINAL_SCHEMA, "strict": True}}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    out, total = {}, 0.0
    for turn in (1, 2):
        mc._reserve(0.01, "metaspec.probe_tools_json")
        cost, entry = 0.0, {"caller": "metaspec.probe_tools_json", "model": MODEL, "kind": "chat", "images": 0}
        try:
            r = cl.chat.completions.create(model=MODEL, messages=msgs, tools=tools, response_format=rf,
                                           reasoning_effort="low", prompt_cache_key="rehearsal-probe")
            u = r.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            cost = chat_cost(MODEL, u.prompt_tokens, cached, u.completion_tokens)
            entry.update(prompt_tokens=u.prompt_tokens, cached_tokens=cached, completion_tokens=u.completion_tokens)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out[f"turn{turn}_error"] = entry["error"]
            mc._settle(0.01, 0.0, entry)
            break
        mc._settle(0.01, cost, entry)
        total += cost
        out[f"turn{turn}"] = r.model_dump()
        m = r.choices[0].message
        if turn == 1:
            if not m.tool_calls:
                out["note"] = "model answered without calling the tool"
                break
            msgs.append(m.model_dump(exclude_none=True))
            for tc in m.tool_calls:
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(FAKE_SIM)})
    out["cost_usd"] = total
    save("probe_tools_json.json", out)
    print("cost", round(total, 6))


def probe_responses_tool():
    """Responses API: function tool + text.format json_schema + reasoning summary, then function_call_output."""
    cl = mc.client()
    tools = [{"type": "function", "name": "simulate", "description":
              "Run the plan in the MuJoCo digital twin; returns success and an event log.",
              "parameters": SIM_TOOL_PARAMS, "strict": True}]
    fmt = {"format": {"type": "json_schema", "name": "Final", "schema": FINAL_SCHEMA, "strict": True}}
    out, total, prev = {}, 0.0, None
    inp = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    for turn in (1, 2):
        mc._reserve(0.01, "metaspec.probe_responses_tool")
        cost, entry = 0.0, {"caller": "metaspec.probe_responses_tool", "model": MODEL, "kind": "responses", "images": 0}
        try:
            kw = dict(model=MODEL, input=inp, tools=tools, text=fmt,
                      reasoning={"effort": "low", "summary": "concise"})
            if prev:
                kw["previous_response_id"] = prev
            r = cl.responses.create(**kw)
            u = r.usage
            cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            cost = chat_cost(MODEL, u.input_tokens, cached, u.output_tokens)
            entry.update(prompt_tokens=u.input_tokens, cached_tokens=cached, completion_tokens=u.output_tokens)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out[f"turn{turn}_error"] = entry["error"]
            mc._settle(0.01, 0.0, entry)
            break
        mc._settle(0.01, cost, entry)
        total += cost
        out[f"turn{turn}"] = r.model_dump()
        calls = [it for it in r.output if it.type == "function_call"]
        if turn == 1:
            if not calls:
                out["note"] = "no function_call on turn 1"
                break
            prev = r.id
            inp = [{"type": "function_call_output", "call_id": c.call_id, "output": json.dumps(FAKE_SIM)}
                   for c in calls]
    out["cost_usd"] = total
    save("probe_responses_tool.json", out)
    print("cost", round(total, 6))


# ---------------------------------------------------------------- SAM
def synth_image(w=160, h=120, cx=50, cy=60, r=22):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(im)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 30, 30))
    d.rectangle([105, 40, 140, 85], fill=(30, 60, 220))
    return im


def png_url(im) -> str:
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def sam_stream(content: list, metadata: dict | None, caller: str, price: float):
    cl = mc.client()
    mc._reserve(price, caller)
    text, usage, entry = "", None, {"caller": caller, "model": "sam-3.1", "kind": "sam", "images": 1}
    try:
        kw = dict(model="sam-3.1", input=[{"type": "message", "role": "user", "content": content}])
        if metadata:
            kw["metadata"] = metadata
        with cl.responses.stream(**kw) as stream:
            for ev in stream:
                if ev.type == "response.output_text.delta":
                    text += ev.delta
                elif ev.type == "response.completed":
                    usage = ev.response.usage.model_dump() if ev.response.usage else None
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        mc._settle(price, price, entry)
        raise
    mc._settle(price, price, entry)
    return text, usage


REC = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*?)data=(\d+),(\d+),(.)([^|]*)\|>")


def decode_records(text: str):
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    frames = {}
    for line in text.strip().splitlines():
        fm = re.match(r"<(\d+)f>", line)
        recs = []
        for m in REC.finditer(line):
            oid, box, _, H, W, enc, payload = m.groups()
            b = dict(kv.split("=") for kv in box.split(";"))
            mask = SegmentationMask(encoding="lossless" if enc == "~" else "one_bit",
                                    payload=enc + payload, width=int(W), height=int(H))
            raster = np.frombuffer(decode_mask_to_raster(mask), np.uint8).reshape(int(H), int(W))
            recs.append({"id": int(oid), "box": {k: float(v) for k, v in b.items()}, "mask_hw": [int(H), int(W)],
                         "encoding": enc, "mask_area_px": int(raster.sum()),
                         "mask_fill_of_box": round(float(raster.mean()), 3), "_raster": raster})
        frames[int(fm.group(1)) if fm else -1] = recs
    return frames


def probe_sam_image():
    im = synth_image()
    url = png_url(im)
    res = {}
    # 1) concept prompt, default (lossless) masks, with confidence
    if (HERE / "probe_sam_points.txt").exists() and "--reuse" in sys.argv:  # decode saved output, no call
        t1, u1 = (HERE / "probe_sam_concept.txt").read_text(), None
        t2, u2 = (HERE / "probe_sam_points.txt").read_text(), None
        return _decode_sam_image(t1, u1, t2, u2)
    t1, u1 = sam_stream([{"type": "input_text", "text": "red circle"}, {"type": "input_image", "image_url": url}],
                        {"include_confidence": "true"}, "metaspec.probe_sam_concept", mc.SAM_PER_IMAGE)
    # 2) point prompt on both shapes (always one_bit per docs)
    pt = "<0f>1<|point;x=50;y=60;w=160;h=120|>,2<|point;x=122;y=62;w=160;h=120|>"
    t2, u2 = sam_stream([{"type": "input_text", "text": pt}, {"type": "input_image", "image_url": url}],
                        None, "metaspec.probe_sam_points", mc.SAM_PER_IMAGE)
    save("probe_sam_concept.txt", t1); save("probe_sam_points.txt", t2)
    return _decode_sam_image(t1, u1, t2, u2)


def _decode_sam_image(t1, u1, t2, u2):
    res = {}
    for name, t, u in (("concept", t1, u1), ("points", t2, u2)):
        fr = decode_records(t)
        for recs in fr.values():
            for r in recs:
                ras = r.pop("_raster")
                ys, xs = np.nonzero(ras)
                # mask raster is the box crop: offset by box x1,y1 to get image coords
                r["mask_centroid_img"] = [round(float(xs.mean() + r["box"]["x1"]), 1),
                                          round(float(ys.mean() + r["box"]["y1"]), 1)] if len(xs) else None
        res[name] = {"frames": fr, "usage": u}
    res["truth"] = {"red_circle": {"center": [50, 60], "r": 22, "area_px": round(np.pi * 22 ** 2)},
                    "blue_rect": {"x1": 105, "y1": 40, "x2": 140, "y2": 85, "area_px": 36 * 46}}
    save("probe_sam_image_decoded.json", res)
    print(json.dumps(res, indent=1, default=str)[:3000])


def probe_sam_video():
    import imageio.v2 as iio
    frames = []
    for i in range(50):  # 50 frames = the billing minimum; red disc slides right
        frames.append(np.asarray(synth_image(cx=30 + i * 1.6)))
    p = HERE / "synthetic_50f.mp4"
    iio.mimwrite(p, frames, fps=25, codec="libx264", macro_block_size=8,
                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    url = "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()
    price = 50 * 0.20 / 1000
    t, u = sam_stream([{"type": "input_text", "text": "red circle"}, {"type": "input_video", "video_url": url}],
                      {"mask_encoding": "one_bit"}, "metaspec.probe_sam_video", price)
    save("probe_sam_video.txt", t)
    fr = decode_records(t)
    track = {f: [(r["id"], r["box"]["x1"], r["box"]["x2"]) for r in recs] for f, recs in sorted(fr.items())}
    summary = {"n_frames_returned": len(fr), "ids": sorted({r["id"] for recs in fr.values() for r in recs}),
               "first": track.get(min(track)) if track else None, "last": track.get(max(track)) if track else None,
               "usage": u}
    save("probe_sam_video_decoded.json", summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    {"tools_json": probe_tools_json, "responses_tool": probe_responses_tool,
     "sam_image": probe_sam_image, "sam_video": probe_sam_video}[sys.argv[1]]()
    print(mc.status())
