// Perception HUD over the photo, in the opener video's language (scripts/opener/compose.py): thin SAM 3.1 outline
// traced around each object, corner brackets that settle in, a dark mono chip "01 LABEL 0.93" with an accent bar.
// Coordinates come from the scene file (perception.mask_px / box_px, photo pixels). Nothing here is invented:
// objects without a mask get brackets only.
const ACC = [64, 156, 255];
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const easeOut = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);

export class PerceptionHUD {
  constructor(canvas, img) {
    this.cv = canvas; this.img = img; this.ctx = canvas.getContext("2d");
    this.objs = []; this.t0 = 0; this.raf = 0; this.hover = null; this.onReveal = null; this.revealed = 0;
    new ResizeObserver((es) => {
      const c = es[0].contentRect;
      if (Math.abs(c.width - (this.w || 0)) < 1 && Math.abs(c.height - (this.h || 0)) < 1) return;
      cancelAnimationFrame(this.rz); this.rz = requestAnimationFrame(() => this.resize(c.width, c.height));
    }).observe(img);
  }
  resize(w, h) {
    if (w === undefined) { w = this.img.clientWidth; h = this.img.clientHeight; }
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (!w || !h) return;
    this.w = w; this.h = h;
    const r = { width: w, height: h };
    this.cv.width = Math.round(r.width * dpr); this.cv.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw(this.lastT || 0);
  }
  // info: scene info from /api/app/scene
  set(info) {
    cancelAnimationFrame(this.raf);
    this.W = info.width || this.img.naturalWidth; this.H = info.height || this.img.naturalHeight;
    const objs = info.objects.filter((o) => o.box_px || o.mask_px);
    // reveal order: left-to-right, top-to-bottom bands, like a scan
    objs.sort((a, b) => cx(a) - cx(b));
    const n = objs.length, step = Math.min(0.26, 3.2 / Math.max(n, 1));
    this.objs = objs.map((o, i) => {
      let poly = o.mask_px && o.mask_px.length > 2 ? o.mask_px : null;
      if (poly && poly.length > 140) { const k = poly.length / 140; poly = Array.from({ length: 140 }, (_, i) => poly[Math.floor(i * k)]); }
      let len = 0; const cum = [0];
      if (poly) for (let k = 1; k <= poly.length; k++) { const a = poly[k - 1], b = poly[k % poly.length]; len += Math.hypot(b[0] - a[0], b[1] - a[1]); cum.push(len); }
      const bb = o.box_px || bbox(poly);
      return { ...o, poly, len, cum, bb, start: 0.35 + i * step, num: String(info.objects.indexOf(o) + 1).padStart(2, "0") };
    });
    this.revealed = 0;
    this.resize();
  }
  play() {
    this.t0 = performance.now();
    const loop = () => {
      const t = (performance.now() - this.t0) / 1000;
      this.draw(t);
      const k = this.objs.filter((o) => t >= o.start + 0.5).length;
      if (k !== this.revealed) { this.revealed = k; this.onReveal && this.onReveal(k, this.objs); }
      if (t < this.endTime() + 0.2 || this.hover) this.raf = requestAnimationFrame(loop);
    };
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(loop);
  }
  finish() { this.t0 = performance.now() - (this.endTime() + 1) * 1000; this.draw(this.endTime() + 1); this.revealed = this.objs.length; this.onReveal && this.onReveal(this.revealed, this.objs); }
  endTime() { return this.objs.length ? this.objs[this.objs.length - 1].start + 1.6 : 0; }
  setHover(id) { this.hover = id; this.draw(this.lastT || this.endTime() + 1); }

  draw(t) {
    this.lastT = t;
    const g = this.ctx, s = this.w / (this.W || 1);
    g.clearRect(0, 0, this.w, this.h);
    if (!this.objs.length || !s) return;
    const P = (p) => [p[0] * s, p[1] * s];
    const placed = [];
    for (const o of this.objs) {
      const u = t - o.start;
      if (u < 0) continue;
      const hi = this.hover === o.id, lo = this.hover && !hi;
      const fade = lo ? 0.35 : 1;
      // fill flash, then a faint tint that stays
      if (o.poly) {
        const flash = u < 0.9 ? easeOut(u / 0.5) * (1 - clamp((u - 0.5) / 0.4, 0, 1)) : 0;
        const a = (0.05 + 0.2 * flash + (hi ? 0.14 : 0)) * fade;
        g.beginPath(); o.poly.forEach((p, k) => { const [x, y] = P(p); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.closePath();
        g.fillStyle = rgba(ACC, a); g.fill();
        // traced outline: two heads from the first point, both directions
        const f = easeOut((u - 0.08) / 0.75);
        if (f > 0) {
          g.lineJoin = "round";
          g.lineWidth = 5; g.strokeStyle = rgba(ACC, 0.18 * fade);          // soft glow, no shadowBlur (slow)
          strokePart(g, o, f / 2, P, false); strokePart(g, o, f / 2, P, true);
          g.lineWidth = hi ? 2 : 1.4; g.strokeStyle = rgba([160, 205, 255], 0.95 * fade);
          strokePart(g, o, f / 2, P, false); strokePart(g, o, f / 2, P, true);
          if (f < 1) {             // glowing heads
            for (const rev of [false, true]) {
              const p = pointAt(o, rev ? o.len * (1 - f / 2) : o.len * f / 2); const [x, y] = P(p);
              g.fillStyle = "rgba(220,235,255,.95)"; g.beginPath(); g.arc(x, y, 2.2, 0, 7); g.fill();
            }
          }
        }
      }
      // corner brackets, settling in from 1.25x
      const ba = easeOut(u / 0.35) * fade, sc = 1 + 0.25 * (1 - easeOut(u / 0.45));
      const [x0, y0] = P([o.bb[0], o.bb[1]]), [x1, y1] = P([o.bb[2], o.bb[3]]);
      brackets(g, x0, y0, x1, y1, sc, rgba([255, 255, 255], 0.78 * ba), hi ? 2 : 1.4);
      // chip
      const ca = easeOut((u - 0.45) / 0.3) * fade;
      if (ca > 0.01) {
        const typed = clamp((u - 0.45) / 0.45, 0, 1), confT = clamp((u - 0.9) / 0.3, 0, 1);
        drawChip(g, o, x0, y0, x1, y1, ca, typed, confT, placed, this.w, this.h, hi);
      }
    }
  }
}

function cx(o) { const b = o.box_px || bbox(o.mask_px); return (b[0] + b[2]) / 2 + (b[1] + b[3]) * 0.15; }
function bbox(poly) {
  let a = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [x, y] of poly || []) a = [Math.min(a[0], x), Math.min(a[1], y), Math.max(a[2], x), Math.max(a[3], y)];
  return a;
}
function pointAt(o, d) {
  const poly = o.poly, cum = o.cum;
  d = ((d % o.len) + o.len) % o.len;
  let k = 1; while (k < cum.length - 1 && cum[k] < d) k++;
  const a = poly[k - 1], b = poly[k % poly.length], seg = cum[k] - cum[k - 1] || 1, u = (d - cum[k - 1]) / seg;
  return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
}
function strokePart(g, o, frac, P, rev) {
  const n = Math.max(8, Math.min(48, Math.round(48 * frac * 2))), L = o.len * frac;
  if (L <= 0) return;
  g.beginPath();
  for (let i = 0; i <= n; i++) {
    const d = rev ? o.len - L * i / n : L * i / n;
    const [x, y] = P(pointAt(o, d));
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
}
function brackets(g, x0, y0, x1, y1, s, col, w) {
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hw = (x1 - x0) / 2 * s + 5, hh = (y1 - y0) / 2 * s + 5;
  const L = Math.min(14, 0.3 * Math.min(hw, hh) + 4);
  const X0 = cx - hw, X1 = cx + hw, Y0 = cy - hh, Y1 = cy + hh;
  g.strokeStyle = col; g.lineWidth = w; g.lineCap = "round";
  for (const [px, py, dx, dy] of [[X0, Y0, 1, 1], [X1, Y0, -1, 1], [X0, Y1, 1, -1], [X1, Y1, -1, -1]]) {
    g.beginPath(); g.moveTo(px, py + dy * L); g.lineTo(px, py); g.lineTo(px + dx * L, py); g.stroke();
  }
}
const MONO = '"SF Mono", ui-monospace, Menlo, monospace';
function drawChip(g, o, x0, y0, x1, y1, a, typed, confT, placed, W, H, hi) {
  const lab = (o.label || o.id).toUpperCase().slice(0, 28);
  const conf = typeof o.conf === "number" ? o.conf.toFixed(2) : "";
  g.font = `500 10px ${MONO}`; const wn = g.measureText(o.num).width;
  g.font = `500 12px ${MONO}`; const wl = g.measureText(lab).width;
  g.font = `500 10px ${MONO}`; const wc = conf ? g.measureText(conf).width : 0;
  const pad = 9, gap = 7, h = 22, w = pad + wn + gap + wl + (conf ? gap + wc : 0) + pad;
  let x = clamp(x0, 4, W - w - 4), y = y0 - h - 7;
  if (y < 4) { y = y0 + 8; x = clamp(x0 + 8, 4, W - w - 4); }
  // avoid chips already placed
  for (let tries = 0; tries < 6; tries++) {
    const hit = placed.find((r) => x < r[0] + r[2] + 3 && x + w + 3 > r[0] && y < r[1] + r[3] + 3 && y + h + 3 > r[1]);
    if (!hit) break;
    y = hit[1] + h + 4;
  }
  y = clamp(y, 4, H - h - 4);
  placed.push([x, y, w, h]);
  g.globalAlpha = a;
  roundRect(g, x, y, w, h, 3);
  g.fillStyle = hi ? "rgba(14,24,40,.86)" : "rgba(9,11,15,.66)"; g.fill();
  g.strokeStyle = hi ? rgba(ACC, 0.8) : "rgba(255,255,255,.22)"; g.lineWidth = 1; g.stroke();
  g.fillStyle = rgba(ACC, 1); g.fillRect(x, y + 5, 2, h - 10);
  let cx = x + pad; const cy = y + h / 2 + 0.5;
  g.textBaseline = "middle";
  g.font = `500 10px ${MONO}`; g.fillStyle = "rgba(140,150,165,1)"; g.fillText(o.num, cx, cy); cx += wn + gap;
  g.font = `500 12px ${MONO}`; g.fillStyle = "#fff"; g.fillText(lab.slice(0, Math.round(lab.length * typed)), cx, cy); cx += wl + gap;
  if (conf && confT > 0) { g.globalAlpha = a * confT; g.font = `500 10px ${MONO}`; g.fillStyle = "rgba(188,220,255,1)"; g.fillText(conf, cx, cy); }
  g.globalAlpha = 1;
}
function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
