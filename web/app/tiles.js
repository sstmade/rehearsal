// Attempt wall: every candidate the planner simulated, replayed live in its own tile from its recorded MuJoCo
// trajectory (the primitive twin, physics look). One offscreen WebGL renderer draws the tiles round-robin into 2D
// canvases, so 50 tiles cost one GL context. Nothing here is synthesised: a tile exists only for a recorded run.
import { Studio } from "/app/viewer.js";

const TW = 480, TH = 300;

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
      const v = this.studio.views.hero;
      // a little tighter than the main view
      const pos = v.pos.map((p, i) => v.target[i] + (p - v.target[i]) * 0.6);
      this.studio.rp.flyTo({ pos, target: v.target }, 0);
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
