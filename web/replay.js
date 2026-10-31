// Three.js replay of a recorded MuJoCo rollout. World is z-up, as in MuJoCo.
// Capabilities: body playback with slerp, TCP path, collision highlight + labels, stress-test trails
// (with staggered "fan out" reveal), object focus halos, camera fly-to / slow orbit, and a
// "phone camera" snapshot with projected object boxes (used for the synthetic stand-in photo).
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const TCP_OFFSET = 0.1034;
const HIT_WINDOW = 0.9;      // seconds a collided object stays highlighted
const GOOD = 0x22a06b, CRITICAL = 0xe5484d, ACCENT = 0x4f7cff, WARN = 0xf5a524;
const ROBOT_RE = /^(link\d|hand|left_finger|right_finger)$/;
const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

export const VIEWS = {
  home:    { pos: [1.62, -1.30, 0.92], target: [0.50, 0.04, 0.10] },
  twin:    { pos: [1.40, -1.15, 1.05], target: [0.55, 0.02, 0.04] },
  top:     { pos: [0.64, -0.02, 2.05], target: [0.62, 0.0, 0.0] },
  cinema:  { pos: [1.42, -1.12, 0.70], target: [0.52, 0.04, 0.10] },
  universe:{ pos: [1.25, -1.05, 1.25], target: [0.52, 0.0, 0.05] },
};

export class Replay {
  constructor(container, { onTime, onEnd } = {}) {
    this.el = container;
    this.onTime = onTime || (() => {});
    this.onEnd = onEnd || (() => {});
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 4.5;
    this.setTheme("dark");
    this.resetCamera();

    this.hemi = new THREE.HemisphereLight(0xdfe8ff, 0x2a2622, 1.1);
    this.scene.add(this.hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
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

    // Soft ground grid, gives scale and the "lab bench" feel.
    this.grid = new THREE.GridHelper(6, 60, 0x3a3f4a, 0x262a33);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.z = -0.749;
    this.grid.material.transparent = true; this.grid.material.opacity = 0.35;
    this.scene.add(this.grid);

    this.bodies = [];          // index = MuJoCo body id
    this.bodyByName = {};
    this.tcpLine = null;
    this.trailGroup = new THREE.Group();
    this.scene.add(this.trailGroup);
    this.haloGroup = new THREE.Group();
    this.scene.add(this.haloGroup);
    this.frames = [];
    this.t = 0; this.playing = false; this.speed = 1; this.duration = 0; this.t0 = 0;
    this.hits = [];
    this.focus = new Map();    // name -> {color, k}
    this.flight = null;
    this.trailReveal = Infinity;
    this.labels = document.createElement("div");
    this.labels.className = "hud";
    container.appendChild(this.labels);

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    // only render while on screen: several viewers share the page
    this.onScreen = true;
    this._io = new IntersectionObserver(([e]) => { this.onScreen = e.isIntersecting; this.clock.getDelta(); });
    this._io.observe(container);
    this.resize();
    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  setTheme(mode) {
    const dark = mode === "dark";
    const bg = dark ? 0x0d0f14 : 0xe9ecf2;
    this.scene.background = new THREE.Color(bg);
    this.scene.fog = new THREE.Fog(bg, 3.0, 6.5);
    if (this.grid) {
      this.grid.material.color.setHex(dark ? 0x3a3f4a : 0xb8bfcc);
      this.grid.material.opacity = dark ? 0.35 : 0.5;
    }
    if (this.hemi) this.hemi.groundColor.setHex(dark ? 0x2a2622 : 0x8c8477);
  }

  resetCamera() { this.flyTo("home", 0); }

  // Smoothly move the camera to a named view or {pos, target}; dur in seconds.
  flyTo(view, dur = 1.4) {
    const v = typeof view === "string" ? VIEWS[view] : view;
    if (!v) return;
    this.autoRotate = false;
    const from = { pos: this.camera.position.clone(), target: this.controls.target.clone() };
    const to = { pos: new THREE.Vector3(...v.pos), target: new THREE.Vector3(...v.target) };
    if (dur <= 0) {
      this.camera.position.copy(to.pos); this.controls.target.copy(to.target); this.controls.update();
      this.flight = null; return;
    }
    this.flight = { from, to, t: 0, dur };
  }

  orbit(on, speed = 0.6) { if (this.noOrbit) on = false; this.autoRotate = on; this.controls.autoRotate = on; this.controls.autoRotateSpeed = speed; }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    if (this.fixedProjection) this._applyK();
    else this.camera.updateProjectionMatrix();
  }

  async loadScene(base, onProgress) {
    const descP = fetch(`${base}/scene3d.json`).then((r) => r.json());
    const res = await fetch(`${base}/meshes.bin`);
    const total = +res.headers.get("Content-Length") || 0;
    let buf;
    if (res.body && onProgress) {
      const reader = res.body.getReader(); const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length; onProgress(got, total);
      }
      buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      buf = buf.buffer;
    } else buf = await res.arrayBuffer();
    const desc = await descP;
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
    for (const gd of desc.geoms) {
      const geo = this._geometry(gd, data, meshCache);
      if (!geo) continue;
      const [r, g, b, a] = gd.rgba;
      const body = this.bodies[gd.body];
      const isRobot = ROBOT_RE.test(body.userData.name);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b), roughness: isRobot ? 0.45 : 0.6, metalness: isRobot ? 0.15 : 0.05,
        transparent: a < 1, opacity: a });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...gd.pos);
      mesh.quaternion.set(gd.quat[1], gd.quat[2], gd.quat[3], gd.quat[0]);
      mesh.castShadow = gd.type !== "plane";
      mesh.receiveShadow = true;
      body.add(mesh);
      body.userData.materials.push(mat);
    }
    this.objectNames = desc.bodies.map((b) => b.name).filter((n) => !ROBOT_RE.test(n) && !/^(world|table)$/.test(n));
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

  // runs: [{success, trails: {obj: [[x,y,z], ...]}}]. Each run becomes one child group so it can be revealed.
  setTrails(runs) {
    this.trailGroup.clear();
    this.trailRuns = runs || [];
    for (const run of this.trailRuns) {
      const color = run.success ? GOOD : CRITICAL;
      const g = new THREE.Group();
      for (const pts of Object.values(run.trails)) {
        const geo = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
        g.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: run.success ? 0.3 : 0.9 })));
        const end = new THREE.Mesh(new THREE.SphereGeometry(run.success ? 0.004 : 0.0075, 12, 8),
          new THREE.MeshBasicMaterial({ color }));
        end.position.set(...pts[pts.length - 1]);
        g.add(end);
      }
      this.trailGroup.add(g);
    }
    this.revealTrails(this.trailReveal);
  }

  // Show only the first n runs (for the "universes fan out" animation).
  revealTrails(n) {
    this.trailReveal = n;
    this.trailGroup.children.forEach((g, i) => { g.visible = i < n; });
  }

  showTrails(on) { this.trailGroup.visible = on; }
  showRobot(on) { for (const b of this.bodies) if (b && ROBOT_RE.test(b.userData.name)) b.visible = on; }

  // Persistent tinted glow on named objects (accent = selected, warn = uncertain, good/bad).
  setFocus(map) {
    this.focus = new Map(Object.entries(map || {}));
    this.haloGroup.clear();
    for (const [name, kind] of this.focus) {
      const body = this.bodyByName[name];
      if (!body) continue;
      const box = new THREE.Box3().setFromObject(body);
      const size = box.getSize(new THREE.Vector3());
      const r = Math.max(size.x, size.y) * 0.72 + 0.02;
      const color = { accent: ACCENT, warn: WARN, good: GOOD, bad: CRITICAL }[kind] || ACCENT;
      const ring = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.008, 48),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
      ring.position.set(box.min.x + size.x / 2, box.min.y + size.y / 2, 0.003);
      ring.userData.body = body; ring.userData.dz = ring.position.z - body.position.z;
      ring.userData.ox = ring.position.x - body.position.x; ring.userData.oy = ring.position.y - body.position.y;
      this.haloGroup.add(ring);
    }
  }

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
    const dt = this.clock.getDelta();
    if (this.flight) {
      const f = this.flight; f.t += dt;
      const u = ease(Math.min(1, f.t / f.dur));
      this.camera.position.lerpVectors(f.from.pos, f.to.pos, u);
      this.controls.target.lerpVectors(f.from.target, f.to.target, u);
      if (u >= 1) this.flight = null;
    }
    if (this.playing && this.frames.length) {
      this.t += dt * this.speed;
      if (this.t >= this.duration) { this.t = this.duration; this.playing = false; this._apply(); this.onEnd(); }
      else this._apply();
    }
    this.controls.update();
    for (const ring of this.haloGroup.children) {
      const b = ring.userData.body;
      ring.position.set(b.position.x + ring.userData.ox, b.position.y + ring.userData.oy, ring.position.z);
      ring.material.opacity = 0.55 + 0.35 * Math.sin(performance.now() / 350);
    }
    this.renderer.render(this.scene, this.camera);
    this._placeLabels();
  }

  _apply() {
    const F = this.frames;
    if (!F.length) return;
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
      const name = b.userData.name;
      const hit = active.get(name) || 0;
      const focus = this.focus.get(name);
      for (const m of b.userData.materials) {
        if (hit > 0) { m.emissive.setHex(CRITICAL); m.emissiveIntensity = hit * 1.6; }
        else if (focus) { m.emissive.setHex({ accent: ACCENT, warn: WARN, good: GOOD, bad: CRITICAL }[focus] || ACCENT); m.emissiveIntensity = 0.28; }
        else { m.emissiveIntensity = 0; }
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
      const p = body.position.clone().add(new THREE.Vector3(0, 0, 0.14)).project(this.camera);
      const x = (p.x * 0.5 + 0.5) * w, y = (-p.y * 0.5 + 0.5) * h;
      return `<div class="hit-label" style="left:${x}px;top:${y}px">${hit.text}</div>`;
    }).join("");
    if (html !== this._lastLabels || hits.length) { this.labels.innerHTML = html; this._lastLabels = html; }
  }

  // Project a world point with the live camera -> {x, y} in CSS px of the container.
  project(p) {
    const v = new THREE.Vector3(...p).project(this.camera);
    return { x: (v.x * 0.5 + 0.5) * this.el.clientWidth, y: (-v.y * 0.5 + 0.5) * this.el.clientHeight, z: v.z };
  }

  snapshot() { return this.renderer.domElement.toDataURL("image/png"); }

  // Render the current scene from a "phone camera" at w×h and return {dataUrl, boxes:{name:[x1,y1,x2,y2]}}.
  // Used to synthesise a stand-in photo for scenes that were hand-authored (no real photo).
  snapshotFrom({ pos, target, fov = 58 }, w = 1280, h = 960, { hideRobot = true } = {}) {
    const cam = new THREE.PerspectiveCamera(fov, w / h, 0.01, 50);
    cam.up.set(0, 0, 1); cam.position.set(...pos); cam.lookAt(new THREE.Vector3(...target)); cam.updateMatrixWorld();
    const size = this.renderer.getSize(new THREE.Vector2());
    const vis = new Map();
    if (hideRobot) for (const b of this.bodies) if (b && ROBOT_RE.test(b.userData.name)) { vis.set(b, b.visible); b.visible = false; }
    const trailsVis = this.trailGroup.visible, haloVis = this.haloGroup.visible, tcpVis = this.tcpLine ? this.tcpLine.visible : true;
    this.trailGroup.visible = false; this.haloGroup.visible = false; if (this.tcpLine) this.tcpLine.visible = false;
    this.renderer.setSize(w, h, false);
    this.renderer.render(this.scene, cam);
    const dataUrl = this.renderer.domElement.toDataURL("image/jpeg", 0.9);
    this.renderer.setSize(size.x, size.y, false);
    for (const [b, v] of vis) b.visible = v;
    this.trailGroup.visible = trailsVis; this.haloGroup.visible = haloVis; if (this.tcpLine) this.tcpLine.visible = tcpVis;
    const boxes = {};
    for (const name of this.objectNames || []) {
      const body = this.bodyByName[name];
      const box = new THREE.Box3().setFromObject(body);
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
      for (let i = 0; i < 8; i++) {
        const c = new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).project(cam);
        const x = (c.x * 0.5 + 0.5) * w, y = (-c.y * 0.5 + 0.5) * h;
        x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x); y2 = Math.max(y2, y);
      }
      boxes[name] = [x1, y1, x2, y2].map((v) => Math.round(v));
    }
    return { dataUrl, boxes, width: w, height: h };
  }
}
