// Three.js replay of a recorded MuJoCo rollout. World is z-up, as in MuJoCo.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const TCP_OFFSET = 0.1034;
const HIT_WINDOW = 0.9;      // seconds a collided object stays highlighted
const GOOD = 0x22a861, CRITICAL = 0xe5484d, ACCENT = 0x3d8bff, UNCERTAIN = 0xf5a524;
const ROBOT_RE = /^(link\d|hand|left_finger|right_finger)$/;

// ---------------------------------------------------------------- shared asset cache
// meshes.bin is ~10 MB: fetch and parse it once per base, share it across viewers.
const assetCache = new Map();
export function loadAssets(base, onProgress) {
  if (!assetCache.has(base)) {
    assetCache.set(base, (async () => {
      const desc = await fetch(`${base}/scene3d.json`).then((r) => { if (!r.ok) throw new Error(`scene3d.json: HTTP ${r.status}`); return r.json(); });
      const buf = await fetchWithProgress(`${base}/meshes.bin`, onProgress);
      return { desc, data: new Float32Array(buf), geoCache: new Map() };
    })());
    assetCache.get(base).catch(() => assetCache.delete(base));
  }
  return assetCache.get(base);
}

async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get("Content-Length") || 0;
  if (!r.body || !total || !onProgress) return r.arrayBuffer();
  const reader = r.body.getReader(), out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length;
    onProgress(got / total);
  }
  return out.buffer;
}

export class Replay {
  constructor(container, { onTime, onHit, interactive = true, photoLook = false } = {}) {
    this.el = container;
    this.onTime = onTime || (() => {});
    this.onHit = onHit || (() => {});
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.bg = new THREE.Color(0x0d0f14);
    this.scene.background = this.bg;
    this.scene.fog = new THREE.Fog(0x0d0f14, 3.2, 6.5);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enabled = interactive;
    this.resetCamera();

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2622, photoLook ? 1.6 : 1.1));
    const key = new THREE.DirectionalLight(photoLook ? 0xfff1dc : 0xffffff, 2.2);
    key.position.set(1.2, -0.8, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.5, far: 5 });
    key.shadow.bias = -0.0004;
    key.target.position.set(0.45, 0, 0);
    this.scene.add(key, key.target);
    const rim = new THREE.DirectionalLight(0x9fc3ff, 0.8);
    rim.position.set(-1.5, 1.2, 1.0);
    this.scene.add(rim);

    this.bodies = [];          // index = MuJoCo body id
    this.bodyByName = {};
    this.tcpLine = null;
    this.trailGroup = new THREE.Group();
    this.scene.add(this.trailGroup);
    this.markGroup = new THREE.Group();
    this.scene.add(this.markGroup);
    this.frames = [];
    this.t = 0; this.playing = false; this.speed = 1;
    this.hits = []; this.firedHits = new Set();
    this.trailAnim = null;
    this.tween = null;
    this.labels = document.createElement("div");
    this.labels.className = "hud";
    container.appendChild(this.labels);

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    this.resize();
    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  resetCamera() {
    this.tween = null;
    this.fixedProjection = false;
    this.camera.position.set(1.75, 1.05, 0.95);
    this.controls.target.set(0.45, 0.0, 0.08);
    this.camera.fov = 38;
    this.resize();
    this.controls.update();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    if (this.fixedProjection) this._applyK();
    else this.camera.updateProjectionMatrix();
  }

  setBackground(hex, transparent = false) {
    this.bg.setHex(hex);
    this.scene.background = transparent ? null : this.bg;
    this.scene.fog = transparent ? null : new THREE.Fog(hex, 3.2, 6.5);
  }

  _clearBodies() {
    for (const b of this.bodies) if (b) this.scene.remove(b);
    this.bodies = []; this.bodyByName = {};
  }

  async loadScene(base, onProgress) {
    const { desc, data, geoCache } = await loadAssets(base, onProgress);
    this._clearBodies();
    for (const b of desc.bodies) this._addBody(b.id, b.name);
    for (const gd of desc.geoms) this._addGeom(gd, data, geoCache);
  }

  // Build a twin from a scene spec: robot meshes from a recorded scene (any), table and objects
  // from primitives so the user can edit positions live. homeFrame: one trajectory frame.
  async loadSpec(spec, robotBase, homeFrame) {
    this._clearBodies();
    this.frames = []; this.hits = [];
    if (this.tcpLine) { this.scene.remove(this.tcpLine); this.tcpLine = null; }
    if (robotBase) {
      try {
        const { desc, data, geoCache } = await loadAssets(robotBase);
        const robotIds = new Set(desc.bodies.filter((b) => ROBOT_RE.test(b.name)).map((b) => b.id));
        for (const b of desc.bodies) if (b.id === 0 || robotIds.has(b.id)) this._addBody(b.id, b.name);
        for (const gd of desc.geoms) if (robotIds.has(gd.body) || gd.type === "plane") this._addGeom(gd, data, geoCache);
        if (homeFrame) {
          for (const id of robotIds) {
            const g = this.bodies[id], i = 1 + 7 * (id - 1);
            g.position.set(homeFrame[i], homeFrame[i + 1], homeFrame[i + 2]);
            g.quaternion.set(homeFrame[i + 4], homeFrame[i + 5], homeFrame[i + 6], homeFrame[i + 3]);
          }
        }
      } catch (e) { console.warn("robot meshes unavailable", e); }
    }
    const t = spec.table;
    const tb = this._addBody(1000, "table");
    tb.position.set(t.center[0], t.center[1], -0.02);
    this._mesh(tb, new THREE.BoxGeometry(t.size[0], t.size[1], 0.04), [0.62, 0.5, 0.38, 1]);
    spec.objects.forEach((o, k) => {
      const g = this._addBody(1001 + k, o.id);
      this._objectMeshes(g, o);
      this.placeObject(o.id, o.pos, o.shape, o.yaw);
    });
    this.markUncertain(spec.objects);
  }

  _objectMeshes(g, o) {
    const s = o.shape, rgba = o.rgba || [0.7, 0.7, 0.7, 1];
    if (s.type === "box") this._mesh(g, new THREE.BoxGeometry(...s.size), rgba);
    else if (s.type === "bowl") {
      const pts = [];
      for (let i = 0; i <= 12; i++) { const a = (i / 12) * Math.PI / 2; pts.push(new THREE.Vector2(s.radius * Math.sin(a) + 0.001, -s.height / 2 + s.height * (1 - Math.cos(a)))); }
      this._mesh(g, new THREE.LatheGeometry(pts, 40).rotateX(Math.PI / 2), rgba, true);
    } else this._mesh(g, new THREE.CylinderGeometry(s.radius, s.radius, s.height, 40).rotateX(Math.PI / 2), rgba);
  }

  placeObject(id, pos, shape, yaw = 0) {
    const g = this.bodyByName[id];
    if (!g) return;
    const h = shape.type === "box" ? shape.size[2] : shape.height;
    g.position.set(pos[0], pos[1], h / 2);
    g.rotation.set(0, 0, yaw || 0);
    const ring = this.markGroup.getObjectByName("u:" + id);
    if (ring) ring.position.set(pos[0], pos[1], 0.002);
  }

  // amber ring = 2σ position uncertainty for low-confidence detections
  markUncertain(objects) {
    this.markGroup.clear();
    for (const o of objects) {
      const p = o.perception;
      if (!p || p.confidence !== "low") continue;
      const r = Math.max(0.02, 2 * (p.sigma_m || 0.012)) + (o.shape.radius || Math.max(...o.shape.size.slice(0, 2)) / 2);
      const ring = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.006, 64),
        new THREE.MeshBasicMaterial({ color: UNCERTAIN, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
      ring.name = "u:" + o.id;
      ring.position.set(o.pos[0], o.pos[1], 0.002);
      this.markGroup.add(ring);
    }
  }

  _addBody(id, name) {
    const g = new THREE.Group();
    g.userData = { name, materials: [] };
    this.bodies[id] = g;
    this.bodyByName[name] = g;
    this.scene.add(g);
    return g;
  }

  _mesh(body, geo, rgba, doubleSide = false) {
    const [r, g, b, a] = rgba;
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), roughness: 0.55, metalness: 0.08,
      transparent: a < 1, opacity: a, side: doubleSide ? THREE.DoubleSide : THREE.FrontSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    body.add(mesh);
    body.userData.materials.push(mat);
    return mesh;
  }

  _addGeom(gd, data, cache) {
    const geo = this._geometry(gd, data, cache);
    if (!geo) return;
    const body = this.bodies[gd.body];
    if (!body) return;
    const mesh = this._mesh(body, geo, gd.rgba);
    mesh.position.set(...gd.pos);
    mesh.quaternion.set(gd.quat[1], gd.quat[2], gd.quat[3], gd.quat[0]);
    mesh.castShadow = gd.type !== "plane";
  }

  _geometry(gd, data, cache) {
    const s = gd.size;
    switch (gd.type) {
      case "box": return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      case "sphere": return new THREE.SphereGeometry(s[0], 24, 16);
      case "cylinder": return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 40).rotateX(Math.PI / 2);
      case "capsule": return new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 24).rotateX(Math.PI / 2);
      case "plane": return new THREE.PlaneGeometry(8, 8);
      case "mesh": {
        const key = gd.mesh.join(":");
        if (!cache.has(key)) {
          const [off, n] = gd.mesh;
          const geo = new THREE.BufferGeometry();
          geo.setAttribute("position", new THREE.BufferAttribute(data.subarray(off, off + n), 3));
          geo.setAttribute("normal", new THREE.BufferAttribute(data.subarray(off + n, off + 2 * n), 3));
          cache.set(key, geo);
        }
        return cache.get(key);
      }
    }
    return null;
  }

  // frames: [[t, x,y,z,qw,qx,qy,qz (body 1), ...], ...]; events: sim events with t
  setTrajectory(frames, events, hitPairs) {
    this.frames = frames;
    this.duration = frames.length ? frames[frames.length - 1][0] : 0;
    this.t0 = frames.length ? frames[0][0] : 0;
    this.hits = hitPairs || [];
    this.firedHits = new Set();
    if (this.tcpLine) this.scene.remove(this.tcpLine);
    const pts = frames.map((f) => this._tcpAt(f));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    geo.setDrawRange(0, 0);
    this.tcpLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.9 }));
    this.scene.add(this.tcpLine);
    this.seek(this.t0);
  }

  _tcpAt(f) {
    const hand = this.bodyByName.hand;
    const i = 1 + 7 * (this.bodies.indexOf(hand) - 1);
    const p = new THREE.Vector3(f[i], f[i + 1], f[i + 2]);
    const q = new THREE.Quaternion(f[i + 4], f[i + 5], f[i + 6], f[i + 3]);
    return p.add(new THREE.Vector3(0, 0, 1).applyQuaternion(q).multiplyScalar(TCP_OFFSET));
  }

  // runs: [{success, trails: {obj: [[x,y,z], ...]}}]; animate: universes fan out over `animate` seconds
  setTrails(runs, { animate = 0 } = {}) {
    this.trailGroup.clear();
    const lines = [];
    runs.forEach((run, k) => {
      const color = run.success ? GOOD : CRITICAL;
      for (const pts of Object.values(run.trails)) {
        const geo = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
          color, transparent: true, opacity: run.success ? 0.3 : 0.9 }));
        const end = new THREE.Mesh(new THREE.SphereGeometry(run.success ? 0.004 : 0.008, 12, 8),
          new THREE.MeshBasicMaterial({ color }));
        end.position.set(...pts[pts.length - 1]);
        this.trailGroup.add(line, end);
        lines.push({ line, end, n: pts.length, delay: (k / Math.max(1, runs.length)) * animate * 0.5 });
      }
    });
    if (animate) {
      for (const l of lines) { l.line.geometry.setDrawRange(0, 0); l.end.visible = false; }
      this.trailAnim = { lines, t: 0, dur: animate * 0.5 };
    } else this.trailAnim = null;
  }

  showTrails(on) { this.trailGroup.visible = on; }
  showRobot(on) {
    for (const b of this.bodies) if (b && this._isRobot(b)) b.visible = on;
  }
  _isRobot(b) { return ROBOT_RE.test(b.userData.name); }

  play() { if (this.t >= this.duration) { this.t = this.t0; this.firedHits = new Set(); } this.playing = true; }
  pause() { this.playing = false; }
  seek(t) {
    this.t = Math.max(this.t0, Math.min(t, this.duration));
    for (const h of this.hits) if (h.t > this.t) this.firedHits.delete(h);
    this._apply();
  }

  // Smooth camera move. pos/target: [x,y,z]
  flyTo(pos, target, dur = 1.6) {
    this.fixedProjection = false;
    this.camera.updateProjectionMatrix();
    this.tween = { p0: this.camera.position.clone(), t0: this.controls.target.clone(),
      p1: new THREE.Vector3(...pos), t1: new THREE.Vector3(...target), u: 0, dur };
  }
  orbit(on, speed = 0.6) { this.controls.autoRotate = on; this.controls.autoRotateSpeed = speed; }

  // OpenCV convention: x_cam = R · X_world + t; camera looks along +z, y down. K: 3x3 intrinsics.
  setCameraKRt(K, R, t, width, height) {
    this.tween = null;
    const Rm = new THREE.Matrix3().set(...R[0], ...R[1], ...R[2]);
    const Rt = Rm.clone().transpose();
    const C = new THREE.Vector3(...t).applyMatrix3(Rt).negate();
    // three camera axes in world: x = R^T e_x, y = -R^T e_y, z = -R^T e_z
    const e = Rt.elements; // column-major
    const xw = new THREE.Vector3(e[0], e[1], e[2]), yw = new THREE.Vector3(e[3], e[4], e[5]).negate(), zw = new THREE.Vector3(e[6], e[7], e[8]).negate();
    const m = new THREE.Matrix4().makeBasis(xw, yw, zw);
    this.camera.position.copy(C);
    this.camera.quaternion.setFromRotationMatrix(m);
    this.camera.up.copy(yw);
    this.controls.target.copy(C.clone().add(zw.clone().multiplyScalar(-0.8)));
    this.controls.enabled = false;
    this.K = { K, width, height };
    this.fixedProjection = true;
    this._applyK();
    this.camera.updateMatrixWorld();
  }

  _applyK() {
    const { K, width: W, height: H } = this.K;
    const n = 0.01, f = 50, fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
    const P = new THREE.Matrix4().set(
      2 * fx / W, 0, 1 - 2 * cx / W, 0,
      0, 2 * fy / H, 2 * cy / H - 1, 0,
      0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
      0, 0, -1, 0);
    this.camera.projectionMatrix.copy(P);
    this.camera.projectionMatrixInverse.copy(P).invert();
  }

  releaseCamera() {
    this.fixedProjection = false;
    this.camera.up.set(0, 0, 1);
    this.controls.enabled = true;
    this.resetCamera();
  }

  renderNow() { this.renderer.render(this.scene, this.camera); }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.25);
    if (this.playing && this.frames.length) {
      this.t += dt * this.speed;
      if (this.t >= this.duration) { this.t = this.duration; this.playing = false; }
      this._apply();
    }
    if (this.tween) {
      const tw = this.tween;
      tw.u = Math.min(1, tw.u + dt / tw.dur);
      const k = tw.u < 0.5 ? 4 * tw.u ** 3 : 1 - (-2 * tw.u + 2) ** 3 / 2;
      this.camera.position.lerpVectors(tw.p0, tw.p1, k);
      this.controls.target.lerpVectors(tw.t0, tw.t1, k);
      if (tw.u >= 1) this.tween = null;
    }
    if (this.trailAnim) {
      const a = this.trailAnim;
      a.t += dt;
      let done = true;
      for (const l of a.lines) {
        const u = Math.max(0, Math.min(1, (a.t - l.delay) / a.dur));
        l.line.geometry.setDrawRange(0, Math.ceil(u * l.n));
        l.end.visible = u >= 1;
        if (u < 1) done = false;
      }
      if (done) this.trailAnim = null;
    }
    if (!this.fixedProjection) this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._placeLabels();
  }

  _apply() {
    const F = this.frames;
    if (!F.length) return;
    // frames are evenly spaced; find bracketing pair
    const step = F.length > 1 ? F[1][0] - F[0][0] : 1;
    const k = Math.min(F.length - 2, Math.max(0, Math.floor((this.t - this.t0) / step)));
    const a = F[k], b = F[Math.min(k + 1, F.length - 1)];
    const u = b[0] > a[0] ? Math.min(1, Math.max(0, (this.t - a[0]) / (b[0] - a[0]))) : 0;
    const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
    for (let id = 1; id < this.bodies.length; id++) {
      const g = this.bodies[id], i = 1 + 7 * (id - 1);
      if (!g || i + 6 >= a.length) continue;
      g.position.set(a[i] + (b[i] - a[i]) * u, a[i + 1] + (b[i + 1] - a[i + 1]) * u, a[i + 2] + (b[i + 2] - a[i + 2]) * u);
      qa.set(a[i + 4], a[i + 5], a[i + 6], a[i + 3]);
      qb.set(b[i + 4], b[i + 5], b[i + 6], b[i + 3]);
      g.quaternion.copy(qa.slerp(qb, u));
    }
    if (this.tcpLine) this.tcpLine.geometry.setDrawRange(0, k + 1);
    this._highlight();
    for (const h of this.hits) {
      if (this.t >= h.t && !this.firedHits.has(h)) { this.firedHits.add(h); if (this.playing) this.onHit(h); }
    }
    this.onTime(this.t, this.duration);
  }

  bodyPosition(name) {
    const b = this.bodyByName[name];
    return b ? b.position.toArray() : null;
  }

  _highlight() {
    const active = new Map();
    for (const h of this.hits) {
      const age = this.t - h.t;
      if (age >= 0 && age < HIT_WINDOW) active.set(h.obj, Math.max(active.get(h.obj) || 0, 1 - age / HIT_WINDOW));
    }
    for (const b of this.bodies) {
      if (!b) continue;
      const k = active.get(b.userData.name) || 0;
      for (const m of b.userData.materials) {
        m.emissive.setHex(CRITICAL);
        m.emissiveIntensity = k * 1.6;
      }
    }
    this.activeHits = this.hits.filter((h) => this.t >= h.t && this.t - h.t < HIT_WINDOW * 1.6);
  }

  _placeLabels() {
    const hits = this.activeHits || [];
    const w = this.el.clientWidth, h = this.el.clientHeight;
    const html = hits.map((hit) => {
      const body = this.bodyByName[hit.obj];
      if (!body) return "";
      const p = body.position.clone().add(new THREE.Vector3(0, 0, 0.12)).project(this.camera);
      const x = (p.x * 0.5 + 0.5) * w, y = (-p.y * 0.5 + 0.5) * h;
      return `<div class="hit-label" style="left:${x}px;top:${y}px">${hit.text}</div>`;
    }).join("");
    if (html !== this._lastLabels || hits.length) { this.labels.innerHTML = html; this._lastLabels = html; }
  }

  // world point -> pixel in a W×H image, for camera-calibrated photos
  static projectKRt(K, R, t, X) {
    const c = [0, 1, 2].map((i) => R[i][0] * X[0] + R[i][1] * X[1] + R[i][2] * X[2] + t[i]);
    return [K[0][0] * c[0] / c[2] + K[0][2], K[1][1] * c[1] / c[2] + K[1][2]];
  }

  snapshot() { this.renderNow(); return this.renderer.domElement.toDataURL("image/png"); }

  dispose() { this.renderer.setAnimationLoop(null); this._ro.disconnect(); this.renderer.dispose(); this.renderer.forceContextLoss?.(); this.el.innerHTML = ""; }
}
