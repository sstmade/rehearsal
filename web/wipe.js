// Real ↔ twin wipe for the DROID side-by-side clips (docs/media/droid → web/media/droid).
// Each clip is 1920×540: the filmed robot on the left half, Rehearsal's MuJoCo replay of the same recorded joints,
// rendered through the same calibrated camera, on the right. Drawing the left half up to the seam and the right
// half after it shows one continuous scene whose right side is the twin. One video decode per player, drawn to a
// canvas, so both halves stay frame-locked. The burned-in labels (top and bottom 48 px) are cropped away.
const SRC_W = 960, SRC_H = 540, CROP_Y = 48, CROP_H = 444;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

export class Wipe {
  constructor(root, { auto = false, onEpisodeEnd = null } = {}) {
    this.root = root;
    this.canvas = root.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d", { alpha: false });
    this.canvas.width = SRC_W; this.canvas.height = CROP_H;
    this.auto = auto && !reduce;
    // data-mid / data-amp: centre and swing of the auto sweep, as fractions of the visible width
    this.mid = +(root.dataset.mid || 0.5); this.amp = +(root.dataset.amp || 0.3);
    this.s = 0.5; this.manual = false; this.visible = false; this.raf = 0; this.t0 = performance.now();
    this.video = document.createElement("video");
    Object.assign(this.video, { muted: true, loop: true, playsInline: true, preload: "auto" });
    this.video.setAttribute("muted", ""); this.video.setAttribute("playsinline", "");
    this.poster = null; this.blobs = new Map(); this.wantPlay = !reduce;
    // seam + handle (DOM, so it stays crisp at any scale)
    this.seam = document.createElement("div"); this.seam.className = "wipe-seam"; this.seam.setAttribute("aria-hidden", "true");
    this.seam.innerHTML = `<i></i>`;
    root.appendChild(this.seam);
    this.tagL = root.querySelector(".wipe-tag.l"); this.tagR = root.querySelector(".wipe-tag.r");
    // keyboard / screen-reader control: a real range input, visually hidden
    this.range = document.createElement("input");
    Object.assign(this.range, { type: "range", min: 0, max: 100, value: 50, className: "sr wipe-range" });
    this.range.setAttribute("aria-label", "Seam between the real footage and the Rehearsal replay");
    this.range.oninput = () => { this.manual = true; this.s = this.range.value / 100; this.draw(); };
    root.appendChild(this.range);
    this._bindDrag();
    new ResizeObserver(() => { this.tagW = this.tagRW = 0; this._layout(); this.draw(); }).observe(root);
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; if (this.visible && !this.seen) { this.seen = true; if (this.pending && this.wantPlay) this._loadVideo(this.pending); } this._sync(); }, { threshold: 0.05, rootMargin: "200px 0px" }).observe(root);
    document.addEventListener("visibilitychange", () => this._sync());
    this.video.addEventListener("loadeddata", () => this.draw());
  }
  async load(ep) {
    this.ep = ep;
    const poster = new Image();
    poster.src = `/web/media/droid/${ep}.jpg`;
    poster.onload = () => { if (this.ep === ep) { this.poster = poster; this.draw(); } };
    if (reduce && !this.wantPlay) { this.video.removeAttribute("src"); this.video.load(); return; }
    await this._loadVideo(ep);
  }
  async _loadVideo(ep) {
    if (!this.seen) { this.pending = ep; return; }   // don't download clips the viewer never scrolls to
    // fetched as a blob: the local server has no Range support, and a blob URL makes looping and seeking reliable
    try {
      let url = this.blobs.get(ep);
      if (!url) {
        const r = await fetch(`/web/media/droid/${ep}.mp4`);
        if (!r.ok) throw new Error(r.status);
        url = URL.createObjectURL(await r.blob());
        this.blobs.set(ep, url);
      }
      if (this.ep !== ep) return;
      this.video.src = url;
      const start = +(this.root.dataset.start || 0);
      if (start) this.video.addEventListener("loadedmetadata", () => { this.video.currentTime = start; }, { once: true });
      this._sync();
    } catch (err) { console.warn("wipe: clip unavailable", ep, err); }
  }
  setPlaying(on) {
    this.wantPlay = on;
    if (on && !this.video.getAttribute("src") && this.ep) this._loadVideo(this.ep);
    this._sync();
  }
  _sync() {
    const run = this.visible && !document.hidden;
    if (run && this.wantPlay && this.video.getAttribute("src")) this.video.play().catch(() => {});
    else this.video.pause();
    if (run && !this.raf) this.raf = requestAnimationFrame(this._frame);
    if (!run && this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
  }
  _frame = (now) => {
    this.raf = 0;
    if (this.auto && !this.manual) {
      const [a, b] = this._visibleRange();
      const narrow = this.L.w < 600, m = narrow ? 0.5 : this.mid, am = narrow ? 0.28 : this.amp;
      const mid = a + (b - a) * m, amp = (b - a) * am;
      this.s = mid + amp * Math.sin((now - this.t0) / 1000 * (2 * Math.PI / 11));
    }
    this.draw();
    if (this.visible && !document.hidden) this.raf = requestAnimationFrame(this._frame);
  };
  // object-fit: cover + object-position from CSS → which part of the canvas is on screen
  _layout() {
    const r = this.root.getBoundingClientRect();
    const pos = getComputedStyle(this.canvas).objectPosition.split(" ")[0];
    const px = pos.endsWith("%") ? parseFloat(pos) / 100 : 0.5;
    const scale = Math.max(r.width / SRC_W, r.height / CROP_H);
    this.L = { w: r.width, h: r.height, dispW: SRC_W * scale, offX: (r.width - SRC_W * scale) * px };
  }
  _visibleRange() {
    if (!this.L) this._layout();
    const { w, dispW, offX } = this.L;
    return [Math.max(0, -offX / dispW), Math.min(1, (w - offX) / dispW)];
  }
  draw() {
    const { ctx } = this;
    if (!this.placed && !this.manual) { const [a, b] = this._visibleRange(); this.s = a + (b - a) * (this.L.w < 600 ? 0.5 : this.mid); this.placed = true; }
    const src = this.video.readyState >= 2 ? this.video : this.poster;
    if (!src) return;
    const k = (src.videoWidth || src.naturalWidth) / (2 * SRC_W);   // posters may be scaled
    const x = Math.round(this.s * SRC_W);
    if (x > 0) ctx.drawImage(src, 0, CROP_Y * k, x * k, CROP_H * k, 0, 0, x, CROP_H);
    if (x < SRC_W) ctx.drawImage(src, (SRC_W + x) * k, CROP_Y * k, (SRC_W - x) * k, CROP_H * k, x, 0, SRC_W - x, CROP_H);
    if (!this.L) this._layout();
    const sx = this.L.offX + this.s * this.L.dispW;
    this.seam.style.transform = `translate3d(${sx}px,0,0)`;
    // tags hug the seam but never leave the frame
    if (this.tagL) { const w = this.tagW || (this.tagW = this.tagL.offsetWidth); this.tagL.style.transform = `translate3d(${sx - 14 - w}px,0,0)`; this.tagL.style.opacity = sx - 14 - w < 12 ? 0 : 1; }
    if (this.tagR) { const w = this.tagRW || (this.tagRW = this.tagR.offsetWidth); this.tagR.style.transform = `translate3d(${sx + 14}px,0,0)`; this.tagR.style.opacity = sx + 14 + w > this.L.w - 12 ? 0 : 1; }
    if (!this.manual) this.range.value = Math.round(this.s * 100);
  }
  _bindDrag() {
    const set = (e) => {
      if (!this.L) this._layout();
      const r = this.root.getBoundingClientRect();
      this.s = Math.min(1, Math.max(0, (e.clientX - r.left - this.L.offX) / this.L.dispW));
      this.range.value = Math.round(this.s * 100);
      this.draw();
    };
    this.root.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button, a, input")) return;
      this.manual = true; this.root.classList.add("dragging"); set(e);
      this.root.setPointerCapture(e.pointerId);
      const move = (ev) => set(ev);
      const up = () => { this.root.classList.remove("dragging"); this.root.removeEventListener("pointermove", move); };
      this.root.addEventListener("pointermove", move);
      this.root.addEventListener("pointerup", up, { once: true });
      this.root.addEventListener("pointercancel", up, { once: true });
    });
  }
}
