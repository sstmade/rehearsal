"""Human-looking UI driving for screen captures (Playwright).

Headless recordings show no system cursor, so a drawn cursor (an arrow plus a click ripple) is injected into the
page and follows the real mouse events. Moves follow an eased cubic bezier (400-700 ms, ~60 steps per second), a
click waits a short beat after arriving, and typing runs at an even ~42 ms per character.
"""
from __future__ import annotations

import math
import random
import time

CURSOR_JS = r"""
(() => {
  if (window.__fakeCursor) return; window.__fakeCursor = 1;
  const add = () => {
    const c = document.createElement('div');
    c.id = '__cur';
    c.innerHTML = '<svg width="22" height="22" viewBox="0 0 22 22"><path d="M3 2 L3 17 L7 13.2 L9.8 19.5 L12.4 18.4 L9.7 12.2 L15 12.2 Z" fill="#fff" stroke="#111" stroke-width="1.3" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, {position: 'fixed', left: '0px', top: '0px', zIndex: 2147483647, pointerEvents: 'none',
      transform: 'translate(-3px,-2px)', filter: 'drop-shadow(0 1px 2px rgba(0,0,0,.55))', willChange: 'left, top'});
    document.documentElement.appendChild(c);
    addEventListener('mousemove', (e) => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    addEventListener('mousedown', (e) => {
      const r = document.createElement('div');
      Object.assign(r.style, {position: 'fixed', left: (e.clientX - 16) + 'px', top: (e.clientY - 16) + 'px', width: '32px', height: '32px',
        borderRadius: '50%', border: '2px solid rgba(255,255,255,.85)', zIndex: 2147483646, pointerEvents: 'none',
        transition: 'transform .45s cubic-bezier(.2,.7,.1,1), opacity .45s', transform: 'scale(.3)', opacity: '1'});
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.25)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 600);
    }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', add); else add();
})();
"""


class Driver:
    def __init__(self, page, start=(960, 540)):
        self.pg = page
        self.x, self.y = start
        page.add_init_script(CURSOR_JS)

    def _ease(self, t):
        return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2

    def move(self, x, y, ms=None):
        x0, y0 = self.x, self.y
        dist = math.hypot(x - x0, y - y0)
        if dist < 2:
            return
        ms = ms or max(400, min(700, 300 + dist * 0.45))
        # a gentle arc: control points offset perpendicular to the path
        nx, ny = -(y - y0) / dist, (x - x0) / dist
        bow = min(80, dist * 0.12) * (1 if (int(x0 + y0) % 2) else -1)
        c1 = (x0 + (x - x0) * 0.3 + nx * bow, y0 + (y - y0) * 0.3 + ny * bow)
        c2 = (x0 + (x - x0) * 0.75 + nx * bow * 0.5, y0 + (y - y0) * 0.75 + ny * bow * 0.5)
        n = max(12, int(ms / 16))
        t0 = time.time()
        for i in range(1, n + 1):
            t = self._ease(i / n)
            u = 1 - t
            px = u ** 3 * x0 + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * x
            py = u ** 3 * y0 + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * y
            self.pg.mouse.move(px, py)
            lag = t0 + ms / 1000 * i / n - time.time()
            if lag > 0.001:
                self.pg.wait_for_timeout(lag * 1000)   # (not time.sleep: keeps the screencast events flowing)
        self.x, self.y = x, y

    def to(self, selector, ms=None):
        el = self.pg.wait_for_selector(selector, state="visible", timeout=600000)
        el.scroll_into_view_if_needed()
        b = el.bounding_box()
        tx = b["x"] + b["width"] * (0.5 + random.uniform(-0.12, 0.12))
        ty = b["y"] + b["height"] * (0.5 + random.uniform(-0.15, 0.15))
        self.move(tx, ty, ms)
        return el

    def click(self, selector, ms=None, pause=180):
        self.to(selector, ms)
        self.pg.wait_for_timeout(pause)
        self.pg.mouse.down(); self.pg.wait_for_timeout(70); self.pg.mouse.up()

    def type(self, text, per_char=42):
        for ch in text:
            self.pg.keyboard.type(ch)
            self.pg.wait_for_timeout(per_char)

    def park(self, x, y):
        """Rest the cursor out of the way (e.g. over the empty panel) while something plays."""
        self.move(x, y)


# ---------------------------------------------------------------------------------------------------------------
# 60 fps capture. Playwright's built-in recorder writes ~25 fps, so the page is screencast over CDP and every frame
# is piped into ffmpeg on a constant 60 fps clock (a frame that did not change is repeated, as a screen would show it).
GPU_ARGS = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]   # headless default is SwiftShader (~6 fps)
FFMPEG = str(__import__("pathlib").Path(__file__).resolve().parent.parent /
             ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")


class Recorder60:
    def __init__(self, page, out_path, fps=60, quality=88):
        import subprocess
        self.pg, self.fps, self.out = page, fps, str(out_path)
        self.ff = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg",
                                    "-framerate", str(fps), "-i", "-", "-c:v", "h264_videotoolbox", "-b:v", "16M",
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.out], stdin=subprocess.PIPE)
        self.cdp = page.context.new_cdp_session(page)
        self.t0 = None; self.slot = 0; self.prev = None; self.frames = 0; self.dupes = 0
        self.cdp.on("Page.screencastFrame", self._on)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": quality, "maxWidth": 1920, "maxHeight": 1080,
                                                "everyNthFrame": 1})

    def _on(self, ev):
        import base64
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        except Exception:  # noqa: BLE001
            pass
        ts = ev["metadata"].get("timestamp")
        data = base64.b64decode(ev["data"])
        if self.t0 is None:
            self.t0 = ts
        target = int(round((ts - self.t0) * self.fps))
        while self.prev is not None and self.slot < target:
            self.ff.stdin.write(self.prev); self.slot += 1; self.dupes += 1
        self.prev = data; self.frames += 1

    def stop(self, tail_s=0.5):
        try:
            self.cdp.send("Page.stopScreencast")
        except Exception:  # noqa: BLE001
            pass
        if self.prev is not None:
            for _ in range(int(tail_s * self.fps) + 1):
                self.ff.stdin.write(self.prev); self.slot += 1
        self.ff.stdin.close(); self.ff.wait()
        return {"frames_unique": self.frames, "slots": self.slot, "seconds": round(self.slot / self.fps, 1)}
