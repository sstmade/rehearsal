// Attempt wall: every candidate the planner simulated, replayed live in its own tile from its recorded MuJoCo
// trajectory (the primitive twin, physics look). One offscreen WebGL renderer draws the tiles round-robin into 2D
// canvases, so 50 tiles cost one GL context. Nothing here is synthesised: a tile exists only for a recorded run.
import { Studio } from "/app/viewer.js";

const TW = 520, TH = 340;

export class AttemptWall {
  constructor() {
    this.host = document.createElement("div");
    Object.assign(this.host.style, { position: "fixed", left: "-9999px", top: "0", width: TW + "px", height: TH + "px", pointerEvents: "none" });
    document.body.appendChild(this.host);
    this.studio = new Studio(this.host);
    this.rp = this.studio.rp;
    this.rp.renderer.setPixelRatio(1);
    this.rp.renderer.setAnimationLoop(null);      // driven by draw() below
    this.tiles = [];
    this.ready = null;
    this.k = 0;
    this.last = performance.now();
    const loop = () => { this.draw(); this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
  }
  async load(twin) {
    if (this.base === twin.base && this.ready) return this.ready;
    this.base = twin.base;
    this.ready = (async () => {
      await this.studio.load(twin);
      this.rp.pause(); this.rp.frames = [];
      // close, steep view on the objects (not the arm): what moved, what fell, what was hit
      const [cx, cy] = twin.table.center, [sx, sy] = twin.table.size, k = Math.max(0.6, Math.max(sx, sy) / 0.85);
      this.studio.rp.flyTo({ pos: [cx + 0.12 * k, cy - 0.62 * k, 0.78 * k], target: [cx, cy, 0] }, 0);
      this.rp.camera.aspect = TW / TH; this.rp.camera.updateProjectionMatrix();
      this.rp.renderer.setSize(TW, TH, false);
    })();
    return this.ready;
  }
  clear() { this.tiles = []; }
  // tile: {canvas, frames, hits}
  add(canvas, frames, hits) {
    const t = { cv: canvas, ctx: canvas.getContext("2d"), frames, hits: hits || [], t: frames[0][0], visible: true,
      dur: frames[frames.length - 1][0], t0: frames[0][0], hold: 0 };
    // a failed attempt opens on its failure moment (first contact), holds it, then loops from the start
    if (t.hits.length) { t.t = Math.min(t.dur, Math.max(t.t0, t.hits[0].t)); t.hold = 2.2; }
    canvas.width = TW; canvas.height = TH;
    const io = new IntersectionObserver((es) => { t.visible = es[0].isIntersecting; });
    io.observe(canvas);
    this.tiles.push(t);
    return t;
  }
  draw() {
    const now = performance.now(), dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (!this.ready || !this.tiles.length || !this.rp.bodies.length) return;
    const live = this.tiles.filter((t) => t.visible && t.cv.isConnected);
    for (const t of this.tiles) {                       // every tile keeps its own clock
      if (t.hold > 0) { t.hold -= dt; if (t.hold <= 0) t.t = t.t0; continue; }
      t.t += dt * 1.6;
      if (t.t >= t.dur) { t.t = t.dur; t.hold = 1.2; }
    }
    const per = Math.min(live.length, 5);
    const rp = this.rp;
    for (let i = 0; i < per; i++) {
      const t = live[(this.k + i) % live.length];
      rp.frames = t.frames; rp.t0 = t.t0; rp.duration = t.dur; rp.hits = t.hits; rp.t = t.t;
      rp._apply();
      rp.renderer.render(rp.scene, rp.camera);
      t.ctx.drawImage(rp.renderer.domElement, 0, 0, TW, TH);
    }
    this.k = (this.k + per) % Math.max(live.length, 1);
  }
}
