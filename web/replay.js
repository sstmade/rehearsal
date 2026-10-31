// Three.js replay of a recorded MuJoCo rollout. World is z-up, as in MuJoCo.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const TCP_OFFSET = 0.1034;
const HIT_WINDOW = 0.9;      // seconds a collided object stays highlighted
const GOOD = 0x0ca30c, CRITICAL = 0xd03b3b, ACCENT = 0x3987e5;

export class Replay {
  constructor(container, { onTime } = {}) {
    this.el = container;
    this.onTime = onTime || (() => {});
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1013);
    this.scene.fog = new THREE.Fog(0x0f1013, 3.2, 6.5);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.resetCamera();

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2622, 1.1));
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

    this.bodies = [];          // index = MuJoCo body id
    this.bodyByName = {};
    this.tcpLine = null;
    this.trailGroup = new THREE.Group();
    this.scene.add(this.trailGroup);
    this.frames = [];
    this.t = 0; this.playing = false; this.speed = 1;
    this.hits = [];
    this.labels = document.createElement("div");
    this.labels.className = "hud";
    container.appendChild(this.labels);

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  resetCamera() {
    this.camera.position.set(1.75, 1.05, 0.95);
    this.controls.target.set(0.45, 0.0, 0.08);
    this.controls.update();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async loadScene(base) {
    const [desc, buf] = await Promise.all([
      fetch(`${base}/scene3d.json`).then((r) => r.json()),
      fetch(`${base}/meshes.bin`).then((r) => r.arrayBuffer()),
    ]);
    for (const b of this.bodies) if (b) this.scene.remove(b);
    this.bodies = []; this.bodyByName = {};
    const data = new Float32Array(buf);
    const meshCache = new Map();
    for (const b of desc.bodies) {
      const g = new THREE.Group();
      g.userData = { name: b.name, materials: [] };
      this.bodies[b.id] = g;
      this.bodyByName[b.name] = g;
      this.scene.add(g);
    }
    for (const gd of desc.geoms) {
      const geo = this._geometry(gd, data, meshCache);
      if (!geo) continue;
      const [r, g, b, a] = gd.rgba;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b), roughness: 0.55, metalness: 0.08,
        transparent: a < 1, opacity: a });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...gd.pos);
      mesh.quaternion.set(gd.quat[1], gd.quat[2], gd.quat[3], gd.quat[0]);
      mesh.castShadow = gd.type !== "plane";
      mesh.receiveShadow = true;
      const body = this.bodies[gd.body];
      body.add(mesh);
      body.userData.materials.push(mat);
    }
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

  // runs: [{success, trails: {obj: [[x,y,z], ...]}}]
  setTrails(runs) {
    this.trailGroup.clear();
    for (const run of runs) {
      const color = run.success ? GOOD : CRITICAL;
      for (const pts of Object.values(run.trails)) {
        const geo = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
        this.trailGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
          color, transparent: true, opacity: run.success ? 0.28 : 0.85 })));
        const end = new THREE.Mesh(new THREE.SphereGeometry(run.success ? 0.004 : 0.007, 12, 8),
          new THREE.MeshBasicMaterial({ color }));
        end.position.set(...pts[pts.length - 1]);
        this.trailGroup.add(end);
      }
    }
  }

  showTrails(on) { this.trailGroup.visible = on; }
  showRobot(on) {
    for (const b of this.bodies) if (b && this._isRobot(b)) b.visible = on;
  }
  _isRobot(b) { return /^(link\d|hand|left_finger|right_finger)$/.test(b.userData.name); }

  play() { if (this.t >= this.duration) this.t = this.t0; this.playing = true; }
  pause() { this.playing = false; }
  seek(t) { this.t = Math.max(this.t0, Math.min(t, this.duration)); this._apply(); }

  tick() {
    const dt = this.clock.getDelta();
    if (this.playing && this.frames.length) {
      this.t += dt * this.speed;
      if (this.t >= this.duration) { this.t = this.duration; this.playing = false; }
      this._apply();
    }
    this.controls.update();
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
      g.position.set(a[i] + (b[i] - a[i]) * u, a[i + 1] + (b[i + 1] - a[i + 1]) * u, a[i + 2] + (b[i + 2] - a[i + 2]) * u);
      qa.set(a[i + 4], a[i + 5], a[i + 6], a[i + 3]);
      qb.set(b[i + 4], b[i + 5], b[i + 6], b[i + 3]);
      g.quaternion.copy(qa.slerp(qb, u));
    }
    if (this.tcpLine) this.tcpLine.geometry.setDrawRange(0, k + 1);
    this._highlight();
    this.onTime(this.t, this.duration);
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
    this.labels.innerHTML = hits.map((hit) => {
      const body = this.bodyByName[hit.obj];
      if (!body) return "";
      const p = body.position.clone().add(new THREE.Vector3(0, 0, 0.12)).project(this.camera);
      const x = (p.x * 0.5 + 0.5) * w, y = (-p.y * 0.5 + 0.5) * h;
      return `<div class="hit-label" style="left:${x}px;top:${y}px">${hit.text}</div>`;
    }).join("");
  }

  snapshot() { return this.renderer.domElement.toDataURL("image/png"); }
}
