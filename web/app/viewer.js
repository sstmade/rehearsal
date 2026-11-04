// The 3D twin viewer: rehearsal's Replay (web/replay.js: MuJoCo bodies + recorded trajectories) dressed as a
// simulation studio: near-black void, a glowing floor grid, a cool rim light. Two looks:
//   physics: the collision primitives physics simulates, flat colours + edges;
//   visual:  photo colours, the photo-rectified tabletop and SAM 3D meshes where they exist (visual only).
import * as THREE from "/web/vendor/three.module.min.js";
import { Replay } from "/web/replay.js";

const BG = 0x050608;
const PALETTE = [0x5b8def, 0xe0a458, 0x5fc8a0, 0xc97b9a, 0x9a86e8, 0xd8c56a, 0x5fb3c9, 0xe07a5f, 0x8fb86b, 0xb98fd6];
const ROBOT_RE = /^(link\d|hand|left_finger|right_finger)$/;

export class Studio {
  constructor(el, { onTime, onEnd } = {}) {
    this.el = el;
    this.rp = new Replay(el, { onTime: (t, d) => onTime && onTime(t, d), onEnd: () => onEnd && onEnd() });
    const rp = this.rp;
    rp.scene.background = new THREE.Color(BG);
    rp.scene.fog = new THREE.Fog(BG, 3.2, 8.5);
    rp.renderer.toneMappingExposure = 0.92;
    rp.grid.visible = false;
    rp.hemi.intensity = 0.55;
    rp.hemi.groundColor.setHex(0x0a0b0e);
    // cool rim from behind the scene + a faint warm fill
    rp.scene.children.filter((c) => c.isDirectionalLight && c.color.getHex() === 0x9fc3ff).forEach((l) => rp.scene.remove(l));
    const rim = new THREE.DirectionalLight(0x8fb8ff, 1.25); rim.position.set(-1.6, 1.5, 1.3); rp.scene.add(rim);
    const rim2 = new THREE.DirectionalLight(0x6f98ff, 0.5); rim2.position.set(-0.4, -1.8, 0.6); rp.scene.add(rim2);
    this.grid = glowGrid(); rp.scene.add(this.grid);
    const sh = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.55 }));
    sh.position.z = -0.7495; sh.receiveShadow = true; rp.scene.add(sh);
    this.look = "physics";
    this.edges = [];
    this.tableTex = null;
    this.views = null;
    this.tags = document.createElement("div"); this.tags.className = "hud"; el.appendChild(this.tags);
    this.tagIds = [];
    const loop = () => { this._placeTags(); requestAnimationFrame(loop); }; loop();
  }

  async load(twin, onProgress, info) {
    const rp = this.rp;
    this.proj = null;
    await rp.loadScene(twin.base, onProgress);
    // the MuJoCo floor plane is replaced by the studio grid + shadow catcher
    for (const b of rp.bodies) if (b) b.traverse((m) => { if (m.isMesh && m.geometry && m.geometry.type === "PlaneGeometry") m.visible = false; });
    this.objects = rp.objectNames.slice();
    this.edges.forEach((e) => e.parent && e.parent.remove(e)); this.edges = [];
    this.objects.forEach((name, i) => {
      const g = rp.bodyByName[name]; if (!g) return;
      const col = new THREE.Color(PALETTE[i % PALETTE.length]);
      for (const m of g.userData.materials) { m.userData.orig = m.color.clone(); m.userData.flat = col; }
      for (const mesh of g.userData.physics) {
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 25),
          new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28 }));
        mesh.add(e); this.edges.push(e);
      }
    });
    // the arm: matte white/grey like the opener's Franka (no glossy blue bloom)
    for (const b of rp.bodies) {
      if (!b || !ROBOT_RE.test(b.userData.name)) continue;
      for (const m of b.userData.materials) {
        const c = m.color, l = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
        c.setScalar(l > 0.5 ? 0.62 : Math.max(0.08, l * 0.9)); m.roughness = 0.78; m.metalness = 0.0;
      }
    }
    // static structure (table top, legs, robot stand): dark slate in the physics view
    this.statics = [];
    for (const b of rp.bodies) {
      if (!b || ROBOT_RE.test(b.userData.name) || this.objects.includes(b.userData.name)) continue;
      for (const mesh of b.userData.physics) {
        if (!mesh.visible) continue;
        const m = mesh.material; m.userData.orig = new THREE.Color(0x3a3f49); m.userData.flat = new THREE.Color(0x1d2129);
        this.statics.push(m);
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 25),
          new THREE.LineBasicMaterial({ color: 0x6f8fb8, transparent: true, opacity: 0.35 }));
        mesh.add(e); this.edges.push(e);
      }
    }
    // tabletop photo texture (visual look)
    if (this.tableTex) { rp.scene.remove(this.tableTex); this.tableTex = null; }
    if (twin.table_texture) {
      const tex = await new THREE.TextureLoader().loadAsync(twin.table_texture).catch(() => null);
      if (tex) {
        tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
        const [cx, cy] = twin.table.center, [sx, sy] = twin.table.size;
        const m = new THREE.Mesh(new THREE.PlaneGeometry(sy, sx), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }));
        m.rotation.z = -Math.PI / 2; m.position.set(cx, cy, 0.0007); m.receiveShadow = true;
        this.tableTex = m; rp.scene.add(m);
      }
    }
    this.views = viewsFor(twin.table);
    this.setLook(this.look);
    this.camera("hero", 0);
    // the twin settles under gravity: 1.5 s of real physics, no robot motion
    const settle = await fetch(twin.settle).then((r) => r.json()).catch(() => null);
    if (settle && settle.length) {
      rp.setTrajectory(settle, [], []);
      if (info && info.camera && info.photo) {
        rp.t = settle[0][0]; rp._apply(); rp.scene.updateMatrixWorld(true);
        await this._projectPhoto(info, twin).catch((e) => console.warn("photo projection", e));
        this.setLook(this.look);
      }
      if (rp.tcpLine) rp.tcpLine.visible = false;
      rp.speed = 1; rp.play();
    }
  }

  setLook(look) {
    this.look = look;
    if (this.proj) { cancelAnimationFrame(this._mraf); this._sweepTo(look === "physics" ? this.proj.lo : this.proj.hi, this._instant ? 0 : undefined); return; }
    const rp = this.rp, phys = look === "physics";
    rp.setMeshMode(!phys);
    for (const name of this.objects || []) {
      const g = rp.bodyByName[name]; if (!g) continue;
      for (const m of g.userData.materials) {
        if (m.userData.visual) continue;
        if (phys && m.userData.flat) { m.color.copy(m.userData.flat); m.roughness = 0.9; m.metalness = 0; }
        else if (m.userData.orig) { m.color.copy(m.userData.orig); m.roughness = 0.6; }
      }
    }
    if (this.proj) return;
    for (const m of this.statics || []) m.color.copy(phys ? m.userData.flat : m.userData.orig);
    for (const e of this.edges) e.visible = phys;
    if (this.tableTex) this.tableTex.visible = !phys;
  }

  // ---- photo projection: each object samples the photo through the calibrated photo camera (K, R, t), at its rest
  // pose, so the texture is glued to the object and travels with it. A SAM-mask ID map decides which object owns each
  // pixel, so an object never picks up the pixels of the one in front of it. Faces the camera never saw keep the
  // object's perceived colour.
  async _projectPhoto(info, twin) {
    const rp = this.rp, cam = info.camera, W = info.width, H = info.height;
    const photo = await new THREE.TextureLoader().loadAsync(info.photo);
    photo.colorSpace = THREE.SRGBColorSpace; photo.anisotropy = 4;
    // ID map: object k+1 in red, far objects first so nearer masks win
    const sc = Math.min(1, 1024 / W), cv = document.createElement("canvas");
    cv.width = Math.round(W * sc); cv.height = Math.round(H * sc);
    const g = cv.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, cv.width, cv.height);
    const order = info.objects.map((o, k) => ({ o, k, y: (o.box_px || [0, 0, 0, 0])[3] })).sort((a, b) => a.y - b.y);
    for (const { o, k } of order) {
      g.fillStyle = `rgb(${k + 1},0,0)`; g.beginPath();
      if (o.mask_px && o.mask_px.length > 2) { o.mask_px.forEach(([x, y], i) => (i ? g.lineTo(x * sc, y * sc) : g.moveTo(x * sc, y * sc))); }
      else if (o.box_px) { const [x1, y1, x2, y2] = o.box_px; g.rect(x1 * sc, y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc); }
      g.closePath(); g.fill();
    }
    const idTex = new THREE.CanvasTexture(cv);
    idTex.minFilter = idTex.magFilter = THREE.NearestFilter; idTex.generateMipmaps = false; idTex.colorSpace = THREE.NoColorSpace;
    const K = cam.K, R = cam.R;
    const shared = {
      uPhoto: { value: photo }, uIdMap: { value: idTex },
      uK: { value: new THREE.Vector4(K[0][0], K[1][1], K[0][2], K[1][2]) },
      uR: { value: new THREE.Matrix3().set(R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]) },
      uT: { value: new THREE.Vector3(...cam.t) }, uWH: { value: new THREE.Vector2(W, H) },
      uSweep: { value: -10 }, uScan: { value: 0 },
    };
    const [cx] = twin.table.center, [sx] = twin.table.size;
    this.proj = { shared, lo: cx - sx / 2 - 0.35, hi: cx + sx / 2 + 0.45, mats: [] };
    info.objects.forEach((o, k) => {
      const grp = rp.bodyByName[o.id]; if (!grp) return;
      for (const mesh of grp.userData.physics) {
        const m = mesh.material;
        if (m.userData.visual) continue;
        const own = { uId: { value: k + 1 }, uRest: { value: mesh.matrixWorld.clone() },
          uFlat: { value: m.userData.flat || m.color }, uOrig: { value: m.userData.orig || m.color.clone() } };
        m.onBeforeCompile = (sh) => {
          Object.assign(sh.uniforms, shared, own);
          sh.vertexShader = "uniform mat4 uRest; varying vec3 vRP; varying vec3 vRN;\n" + sh.vertexShader.replace("#include <project_vertex>",
            "#include <project_vertex>\n vRP = (uRest * vec4(transformed, 1.0)).xyz; vRN = normalize(mat3(uRest) * objectNormal);");
          sh.fragmentShader = PROJ_FRAG_DECL + sh.fragmentShader
            .replace("#include <color_fragment>", "#include <color_fragment>\n" + PROJ_FRAG_COLOR)
            .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\n totalEmissiveRadiance += vec3(0.25, 0.62, 1.0) * scanGlow * 1.6;");
        };
        m.customProgramCacheKey = () => "rehearsal-proj";
        m.needsUpdate = true;
        this.proj.mats.push(m);
      }
    });
    // the scan plane that sweeps across the table during the morph
    if (this.scanPlane) rp.scene.remove(this.scanPlane);
    const [, sy] = twin.table.size, cy = twin.table.center[1];
    const pm = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      uniforms: { uA: { value: 0 } },
      vertexShader: "varying vec2 vU; void main(){ vU = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
      fragmentShader: "uniform float uA; varying vec2 vU; void main(){ float e = smoothstep(0.0, 0.08, vU.x) * smoothstep(1.0, 0.92, vU.x);" +
        " float top = 1.0 - vU.y; float a = (0.10 + 0.9 * pow(top, 6.0) * 0.0 + 0.55 * exp(-pow((vU.y) / 0.04, 2.0))) * e * uA;" +
        " gl_FragColor = vec4(vec3(0.25, 0.62, 1.0) * (0.6 + 0.4 * (1.0 - vU.y)), a * (1.0 - 0.75 * vU.y)); }" });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(sy + 0.3, 0.55), pm);
    plane.rotation.set(Math.PI / 2, Math.PI / 2, 0);
    plane.position.set(this.proj.lo, cy, 0.27);
    plane.visible = false; plane.renderOrder = 5;
    this.scanPlane = plane; rp.scene.add(plane);
    shared.uSweep.value = this.look === "physics" ? this.proj.lo : this.proj.hi;
  }
  // animate the reveal front to x = target (primitives behind it, textured photo twin in front)
  _sweepTo(target, dur) {
    const P = this.proj, u = P.shared.uSweep, from = u.value;
    if (dur === undefined) dur = Math.abs(target - from) > 0.2 ? (target > from ? 3.2 : 1.0) : 0;
    const toVis = target > from, rp = this.rp, t0 = performance.now();
    if (this.tableTex && !this.tableTex.material.transparent) { this.tableTex.material.transparent = true; this.tableTex.material.needsUpdate = true; }
    rp.setMeshMode(false);                                  // SAM 3D meshes swap in after the sweep
    const step = () => {
      const k = dur ? Math.min(1, (performance.now() - t0) / (dur * 1000)) : 1;
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      u.value = from + (target - from) * e;
      const f = (u.value - P.lo) / (P.hi - P.lo);           // 0 = all primitive, 1 = all textured
      P.shared.uScan.value = k < 1 ? Math.sin(Math.PI * k) : 0;
      if (this.scanPlane) { this.scanPlane.visible = k < 1 && dur > 0; this.scanPlane.position.x = u.value; this.scanPlane.material.uniforms.uA.value = P.shared.uScan.value; }
      for (const ed of this.edges) { ed.visible = f < 0.999; ed.material.opacity = (ed.userData.op ??= ed.material.opacity) * (1 - f); }
      if (this.tableTex) { this.tableTex.material.opacity = f; this.tableTex.visible = f > 0.001; }
      for (const m of this.statics || []) m.color.copy(m.userData.flat).lerp(m.userData.orig, f);
      if (k < 1) this._mraf = requestAnimationFrame(step);
      else { rp.setMeshMode(toVis || f > 0.5); if (this.onMorph) this.onMorph(f > 0.5 ? "visual" : "physics"); }
    };
    step();
  }

  camera(name, dur = 1.2) { if (this.views && this.views[name]) this.rp.flyTo(this.views[name], dur); }

  async playRun(trajUrl, hits, focus) {
    const frames = await fetch(trajUrl).then((r) => r.json());
    const rp = this.rp;
    rp.setTrajectory(frames, [], hits || []);
    if (rp.tcpLine) rp.tcpLine.visible = true;
    rp.setFocus(focus || {});
    rp.speed = this.speed || 1;
    rp.play();
    return frames.length ? frames[frames.length - 1][0] : 0;
  }
  setFocus(map) { this.rp.setFocus(map || {}); }
  setTags(ids, labels) { this.tagIds = ids || []; this.tagLabels = labels || {}; this._tagEls = null; }
  _placeTags() {
    const rp = this.rp;
    if (!this.tagIds.length) { if (this.tags.innerHTML) this.tags.innerHTML = ""; return; }
    if (!this._tagEls) {
      this.tags.innerHTML = this.tagIds.map((id) => `<div class="obj-tag" data-id="${id}">${esc(this.tagLabels[id] || id)}</div>`).join("");
      this._tagEls = [...this.tags.children];
    }
    const pts = [];
    for (const el of this._tagEls) {
      const g = rp.bodyByName[el.dataset.id]; if (!g) continue;
      const box = new THREE.Box3().setFromObject(g);
      const p = rp.project([(box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, box.max.z + 0.03]);
      pts.push({ el, x: p.x, y: p.y, z: p.z, w: el.offsetWidth || 80 });
    }
    pts.sort((a, b) => b.y - a.y);
    const placed = [];
    for (const q of pts) {
      let y = q.y;
      for (let k = 0; k < 8; k++) {
        const hit = placed.find((r) => Math.abs(r.x - q.x) < (r.w + q.w) / 2 + 4 && Math.abs(r.y - y) < 20);
        if (!hit) break;
        y = hit.y - 21;
      }
      placed.push({ x: q.x, y, w: q.w });
      q.el.style.left = q.x + "px"; q.el.style.top = y + "px"; q.el.style.opacity = q.z < 1 ? 1 : 0;
    }
  }
}

const PROJ_FRAG_DECL = `uniform sampler2D uPhoto; uniform sampler2D uIdMap; uniform vec4 uK; uniform mat3 uR; uniform vec3 uT; uniform vec2 uWH;
uniform float uId; uniform float uSweep; uniform float uScan; uniform vec3 uFlat; uniform vec3 uOrig; varying vec3 vRP; varying vec3 vRN;
float scanGlow = 0.0;
`;
const PROJ_FRAG_COLOR = `{
  float coord = vRP.x + 0.12 * vRP.z;
  float rv = 1.0 - smoothstep(uSweep - 0.035, uSweep + 0.005, coord);    // 1 = behind the front: refined
  vec3 pc = uR * vRP + uT;
  vec2 pp = vec2(uK.x * pc.x / pc.z + uK.z, uK.y * pc.y / pc.z + uK.w) / uWH;
  float inImg = step(0.001, pc.z) * step(0.0, pp.x) * step(pp.x, 1.0) * step(0.0, pp.y) * step(pp.y, 1.0);
  vec2 tuv = vec2(pp.x, 1.0 - pp.y);
  float own = 1.0 - step(0.5, abs(texture2D(uIdMap, tuv).r * 255.0 - uId));
  vec3 camPos = -(transpose(uR) * uT);
  float facing = dot(normalize(vRN), normalize(camPos - vRP));
  float w = inImg * own * smoothstep(0.02, 0.3, facing);
  vec3 photoC = texture2D(uPhoto, tuv).rgb;
  vec3 refined = mix(uOrig, photoC, w);
  diffuseColor.rgb = mix(uFlat, refined, rv);
  scanGlow = uScan * exp(-pow((coord - uSweep) / 0.014, 2.0));
}
`;

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function viewsFor(table) {
  const [cx, cy] = table.center, [sx, sy] = table.size;
  const k = Math.min(1.8, Math.max(0.95, Math.max(sx, sy) / 0.85));
  const tx = cx * 0.82;
  const kt = Math.min(1.6, Math.max(1.1, Math.max(sx, sy) / 0.8));
  return {
    // the table is the subject: a 3/4 view from the side, the arm visible at the left but secondary
    hero: { pos: [cx + 0.7 * kt, cy - 1.0 * kt, 0.92 * kt], target: [cx - 0.02, cy + 0.02, -0.11] },
    front: { pos: [tx + 1.55 * k, cy - 1.45 * k, 1.05 * k], target: [tx * 0.8, cy, -0.06] },
    top: { pos: [tx + 0.02, cy, 2.3 * k], target: [tx, cy, 0] },
    side: { pos: [tx * 0.7, cy - 2.25 * k, 0.5 * k], target: [tx * 0.7, cy, 0.02] },
  };
}

function glowGrid() {
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uColor: { value: new THREE.Color(0x3d7fd6) }, uMinor: { value: 0.1 }, uMajor: { value: 0.5 } },
    vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix*viewMatrix*w; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uMinor; uniform float uMajor; varying vec3 vW;
      float line(vec2 p, float s, float w){ vec2 g = abs(fract(p/s - 0.5) - 0.5) / fwidth(p/s); return 1.0 - min(min(g.x, g.y)/w, 1.0); }
      void main(){
        float d = length(vW.xy - vec2(0.5, 0.0));
        float fade = exp(-d*d*0.55);
        float a = line(vW.xy, uMinor, 1.0)*0.16 + line(vW.xy, uMajor, 1.3)*0.42;
        float glow = exp(-d*d*3.0)*0.06;
        gl_FragColor = vec4(uColor*(1.2+glow*6.0), (a*fade + glow));
      }`,
  });
  mat.extensions = { derivatives: true };
  const m = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), mat);
  m.position.z = -0.749;
  m.renderOrder = -1;
  return m;
}
