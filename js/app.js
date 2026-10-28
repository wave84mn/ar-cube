// AR Cube viewer: camera -> marker tracking -> three.js overlay of a building model on the cube or the table card.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Tracker, projectedBox } from './cv/tracker.js';
import { OneEuro, matToQuat, mulv } from './cv/pose.js';

const VERSION = '0.1 · 27 Oct 2026';
const $ = s => document.querySelector(s);
const D2R = Math.PI / 180;
const params = new URLSearchParams(location.search);

// ---------------- settings (per-viewer convenience only) ----------------
const DEFAULTS = { plinth: 'stone', size: 1, upright: true, shadows: true, debug: false, fov: 66 };
const settings = Object.assign({}, DEFAULTS, readStore('arcube.settings') || {});
if (params.has('debug')) settings.debug = true;
if (params.has('fov')) settings.fov = +params.get('fov');
function readStore(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } }
function saveSettings() { try { localStorage.setItem('arcube.settings', JSON.stringify(settings)); } catch { /* private mode */ } }

// ---------------- DOM ----------------
const video = $('#cam'), glCanvas = $('#gl'), dbgCanvas = $('#dbg');
const dbg = dbgCanvas.getContext('2d');
const proc = document.createElement('canvas');
const pctx = proc.getContext('2d', { willReadFrequently: true });
const disp = document.createElement('canvas');          // the exact frame that is tracked is the frame that is shown
const dctx = disp.getContext('2d', { alpha: false });
$('#ver').textContent = VERSION;

// ---------------- three.js ----------------
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true, powerPreference: 'high-performance' });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.autoClear = false;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera();
camera.matrixAutoUpdate = false;
const bgScene = new THREE.Scene();
const bgCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
let videoTex = null;
const bgMat = new THREE.MeshBasicMaterial({ toneMapped: false, depthTest: false, depthWrite: false });
bgScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat));
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;
scene.add(new THREE.HemisphereLight(0xfff6ea, 0x6d655a, 0.55));
const sun = new THREE.DirectionalLight(0xfff3e0, 2.4);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.6;
scene.add(sun, sun.target);

const CV2GL = new THREE.Matrix4().set(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
const anchors = { cube: new THREE.Group(), card: new THREE.Group() };
for (const a of Object.values(anchors)) { a.matrixAutoUpdate = false; a.visible = false; scene.add(a); }

// holder = where the building stands (rotates between cube faces); inside: turf + yaw group + model
const holder = new THREE.Group();
const yawGroup = new THREE.Group();
const fitGroup = new THREE.Group();
holder.add(yawGroup); yawGroup.add(fitGroup);

let S = 70;                           // cube edge (mm), from layout
let plinth, turf, shadowCatcher;

// ---------------- state ----------------
let layout, tracker, models = [], current = null, modelRoot = null;
const loader = new GLTFLoader();
loader.setMeshoptDecoder(MeshoptDecoder);
const view = { W: 0, H: 0, vw: 0, vh: 0, s: 1, ox: 0, oy: 0 };
const T = {
  cube: { filtT: new OneEuro(3, 2.0, 0.3, 5.0), filtQ: new OneEuro(4, 2.0, 40, 5.0), raw: null, seen: -10, lostAt: -10, acquired: false, base: 0, q: null, prevQ: null },
  card: { filtT: new OneEuro(3, 2.0, 0.3, 5.0), filtQ: new OneEuro(4, 2.0, 40, 5.0), raw: null, seen: -10, lostAt: -10, acquired: false, base: 0, q: null, prevQ: null },
};
let active = null;                    // 'cube' | 'card' | null
const place = { Q: new THREE.Quaternion(), from: new THREE.Quaternion(), to: new THREE.Quaternion(), t0: 0, dur: 0.45,
  yaw: 0, yawFrom: 0, yawTo: 0, yt0: 0, needFace: true, appear: 0 };
let frameNo = 0, lastRes = null, fps = 0, procMs = 0, lastFrameT = 0;
const timing = { capture: 0, read: 0, track: 0, render: 0 };
const sensor = { ok: false, beta: 0, gamma: 0, t: 0 };

// ---------------- start ----------------
$('#go').addEventListener('click', start);
async function start() {
  const err = $('#err');
  err.style.display = 'none';
  const sensorP = enableSensors();          // must run inside the tap (iOS permission prompt)
  if (!navigator.mediaDevices?.getUserMedia) {
    return fail('This browser cannot open the camera here. Open the page in Safari (iPhone) or Chrome (Android), over https.');
  }
  $('#go').disabled = true; $('#go').textContent = 'Starting…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    $('#go').disabled = false; $('#go').textContent = 'Start camera';
    const name = e && e.name;
    return fail(name === 'NotAllowedError' ? 'Camera access was blocked. Allow the camera for this site in the browser settings and try again.'
      : name === 'NotFoundError' ? 'No camera was found on this device.'
      : 'The camera could not start (' + (e && e.message || e) + '). If you opened this link inside another app, open it in Safari or Chrome instead.');
  }
  await sensorP;
  videoTex = new THREE.CanvasTexture(disp);
  videoTex.colorSpace = THREE.SRGBColorSpace;
  videoTex.generateMipmaps = false; videoTex.minFilter = THREE.LinearFilter;
  bgMat.map = videoTex; bgMat.needsUpdate = true;
  $('#start').classList.add('hidden');
  for (const id of ['#hud', '#dock', '#hint']) $(id).hidden = false;
  keepAwake();
  resize();
  runLoop();
}
function fail(msg) { const e = $('#err'); e.textContent = msg; e.style.display = 'block'; }

async function enableSensors() {
  try {
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
      const r = await DeviceOrientationEvent.requestPermission();
      if (r !== 'granted') return;
    }
    window.addEventListener('deviceorientation', e => {
      if (e.beta == null || e.gamma == null) return;
      sensor.ok = true; sensor.beta = e.beta; sensor.gamma = e.gamma; sensor.t = performance.now();
    });
  } catch { /* no sensors: fall back to the screen-up guess */ }
}

let wakeLock = null;
async function keepAwake() {
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* optional */ }
}
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && video.srcObject) keepAwake(); });

// world-up direction in OpenCV camera coordinates (x right, y down, z forward), from the tilt sensor
function upInCamera() {
  if (!sensor.ok || performance.now() - sensor.t > 1500) return null;
  const b = sensor.beta * D2R, g = sensor.gamma * D2R;
  const ux = -Math.cos(b) * Math.sin(g), uy = Math.sin(b), uz = Math.cos(b) * Math.cos(g);
  const th = ((screen.orientation && screen.orientation.angle) ?? window.orientation ?? 0) * D2R;
  const sx = ux * Math.cos(th) - uy * Math.sin(th), sy = ux * Math.sin(th) + uy * Math.cos(th);
  return [sx, -sy, -uz];
}

// ---------------- data ----------------
init();
async function init() {
  try {
    const [lay, mc] = await Promise.all([fetch('data/layout.json').then(r => r.json()), fetch('models/models.json').then(r => r.json())]);
    layout = lay; S = layout.edge;
    tracker = new Tracker(layout);
    models = mc.models;
    buildPlinth();
    buildModelChips();
    const want = params.get('model');
    selectModel((models.find(m => m.id === want) || models[0]).id);
  } catch (e) {
    fail('Could not load the app data: ' + e.message);
  }
  bindSettingsUI();
}

// ---------------- plinth / turf / shadow catcher ----------------
function noiseTexture(base, vary, size = 256, specks = 0) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const n = (Math.random() - 0.5) * vary;
    const sp = specks && Math.random() < specks ? -vary * 1.5 : 0;
    img.data[i * 4] = base[0] + n + sp; img.data[i * 4 + 1] = base[1] + n + sp; img.data[i * 4 + 2] = base[2] + n + sp; img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  // soft blur by drawing scaled copies
  g.globalAlpha = 0.5; g.drawImage(c, 1, 0); g.drawImage(c, 0, 1); g.globalAlpha = 1;
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
function grassTexture() {
  const size = 256, c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d');
  g.fillStyle = '#5f7f3a'; g.fillRect(0, 0, size, size);
  for (let i = 0; i < 9000; i++) {
    const h = 70 + Math.random() * 40, s = 35 + Math.random() * 25, l = 25 + Math.random() * 22;
    g.strokeStyle = `hsl(${h},${s}%,${l}%)`; g.lineWidth = 1;
    const x = Math.random() * size, y = Math.random() * size, a = Math.random() * Math.PI, L = 2 + Math.random() * 4;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * L, y + Math.sin(a) * L); g.stroke();
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
function buildPlinth() {
  const k = 1.025;   // a little larger than the real cube so its edges stay hidden
  const stone = new THREE.MeshStandardMaterial({ map: noiseTexture([214, 207, 194], 22, 256, 0.01), roughness: 0.88, metalness: 0 });
  plinth = new THREE.Mesh(new THREE.BoxGeometry(S * k, S * k, S * k), stone);
  plinth.receiveShadow = true;
  anchors.cube.add(plinth);
  const soil = new THREE.MeshStandardMaterial({ map: noiseTexture([104, 78, 54], 30, 128), roughness: 0.95 });
  const grass = new THREE.MeshStandardMaterial({ map: grassTexture(), roughness: 0.95 });
  const tg = new THREE.BoxGeometry(S * k, 2.2, S * k);
  turf = new THREE.Mesh(tg, [soil, soil, grass, soil, soil, soil]);
  turf.position.y = 1.1;            // slab sits on the base face, top at +2.2 mm
  turf.receiveShadow = true;
  holder.add(turf);
  shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(320, 320), new THREE.ShadowMaterial({ opacity: 0.32 }));
  shadowCatcher.rotation.x = -Math.PI / 2;
  shadowCatcher.receiveShadow = true;
  anchors.card.add(shadowCatcher);
  applyPlinthStyle();
}
function applyPlinthStyle() {
  if (!plinth) return;
  plinth.visible = settings.plinth !== 'none';
  turf.visible = settings.plinth === 'grass' && active !== 'card';
  updateHolderOffset();
}
function updateHolderOffset() {
  // model sits on the turf if shown
  fitGroup.position.y = turf && turf.visible ? 2.2 : 0;
}

// ---------------- models ----------------
const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M5 21V9l7-5 7 5v12"/><path d="M9 21v-5h6v5M3 21h18M9 11h.01M15 11h.01"/></svg>';
function buildModelChips() {
  const nav = $('#models');
  nav.innerHTML = '';
  for (const m of models) {
    const b = document.createElement('button');
    b.className = 'chip'; b.dataset.id = m.id; b.setAttribute('aria-pressed', 'false');
    const th = document.createElement('span'); th.className = 'th';
    if (m.thumb) th.style.backgroundImage = `url("${m.thumb}")`; else th.innerHTML = ICON;
    const txt = document.createElement('span');
    const bb = document.createElement('b'); bb.textContent = m.name;
    const sm = document.createElement('small'); sm.textContent = m.subtitle || '';
    txt.append(bb, sm);
    const ld = document.createElement('i'); ld.className = 'ld';
    b.append(th, txt, ld);
    b.addEventListener('click', () => selectModel(m.id));
    nav.append(b);
  }
}
const cache = new Map();
async function selectModel(id) {
  const m = models.find(x => x.id === id);
  if (!m) return;
  current = m;
  for (const b of document.querySelectorAll('.chip')) b.setAttribute('aria-pressed', String(b.dataset.id === id));
  const chip = document.querySelector(`.chip[data-id="${CSS.escape(id)}"]`);
  try {
    let root = cache.get(id);
    if (!root) {
      chip && chip.classList.add('loading');
      const gltf = await loader.loadAsync(m.file);
      root = prepareModel(gltf.scene, m);
      cache.set(id, root);
    }
    if (current !== m) return;      // user switched while loading
    if (modelRoot) fitGroup.remove(modelRoot);
    modelRoot = root;
    fitGroup.add(root);
    applySize();
    place.yaw = place.yawTo = place.yawFrom = (m.yaw || 0) * D2R;
    place.needFace = true; place.appear = 0;
    updateInfo();
    try { const u = new URL(location.href); u.searchParams.set('model', id); history.replaceState(null, '', u); } catch { /* ignore */ }
  } catch (e) {
    toast('Could not load ' + m.name + ': ' + e.message);
  } finally {
    chip && chip.classList.remove('loading');
  }
}
function prepareModel(scene3d, m) {
  const wrap = new THREE.Group();
  let tris = 0;
  scene3d.traverse(o => {
    if (o.isMesh) {
      o.castShadow = true; o.receiveShadow = true;
      const g = o.geometry; tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const mt of mats) { if (mt.map) mt.map.anisotropy = 4; }
    }
  });
  if (m.upAxis === 'z') scene3d.rotation.x = -Math.PI / 2;
  scene3d.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(scene3d);
  const size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
  scene3d.position.set(-c.x, -box.min.y, -c.z);
  wrap.add(scene3d);
  const foot = Math.max(size.x, size.z);
  wrap.userData = { foot, height: size.y, tris: Math.round(tris), unit: 1 / foot };
  return wrap;
}
function applySize() {
  if (!modelRoot) return;
  const u = modelRoot.userData;
  const fit = (current && current.fit) || 0.94;
  const footMM = active === 'card' ? (layout.card ? 110 : S) : S * fit;
  const s = u.unit * footMM * settings.size;
  modelRoot.scale.setScalar(s);
  const r = Math.max(u.foot, u.height) * s;
  const cam = sun.shadow.camera;
  cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r; cam.near = 1; cam.far = 6 * r + 800;
  cam.updateProjectionMatrix();
}
function updateInfo() {
  if (!current) return;
  $('#infoTitle').textContent = current.name;
  $('#infoDesc').textContent = current.description || '';
  const kv = $('#infoKv'); kv.innerHTML = '';
  const add = (k, v) => { const dt = document.createElement('dt'); dt.textContent = k; const dd = document.createElement('dd'); dd.textContent = v; kv.append(dt, dd); };
  for (const [k, v] of Object.entries(current.facts || {})) add(k, v);
  if (modelRoot) add('Triangles', modelRoot.userData.tris.toLocaleString('en'));
}

// ---------------- layout / projection ----------------
window.addEventListener('resize', resize);
window.addEventListener('orientationchange', () => setTimeout(resize, 300));
video.addEventListener('resize', resize);
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(dpr);
  renderer.setSize(glCanvas.clientWidth || window.innerWidth, glCanvas.clientHeight || window.innerHeight, false);
  dbgCanvas.width = glCanvas.width; dbgCanvas.height = glCanvas.height;
  updateProjection();
}
function intrinsics() {
  const vw = video.videoWidth, vh = video.videoHeight, L = Math.max(vw, vh);
  const f = (L / 2) / Math.tan(settings.fov * D2R / 2);
  return { fx: f, fy: f, cx: vw / 2, cy: vh / 2 };
}
function updateProjection() {
  const W = glCanvas.width, H = glCanvas.height, vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh || !W || !H) return;
  const s = Math.max(W / vw, H / vh);
  const ox = (vw - W / s) / 2, oy = (vh - H / s) / 2;
  Object.assign(view, { W, H, vw, vh, s, ox, oy });
  const K = intrinsics();
  const fx = K.fx * s, fy = K.fy * s, cx = (K.cx - ox) * s, cy = (K.cy - oy) * s;
  const n = 5, f = 20000;
  camera.projectionMatrix.set(2 * fx / W, 0, 1 - 2 * cx / W, 0, 0, 2 * fy / H, 2 * cy / H - 1, 0, 0, 0, -(f + n) / (f - n), -2 * f * n / (f - n), 0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  if (videoTex) { videoTex.repeat.set((W / s) / vw, (H / s) / vh); videoTex.offset.set(ox / vw, oy / vh); }
}

// ---------------- main loop ----------------
function runLoop() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const cb = now => { try { onFrame(now); } catch (e) { console.error(e); } video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else {
    let last = -1;
    const raf = now => { if (video.currentTime !== last) { last = video.currentTime; try { onFrame(now); } catch (e) { console.error(e); } } requestAnimationFrame(raf); };
    requestAnimationFrame(raf);
  }
}

let PROC = 640;                      // processing size (long side), adapted to the phone's speed
function chooseRegion(K, t) {
  const vw = video.videoWidth, vh = video.videoHeight, L = Math.max(vw, vh);
  // tracking: crop around the predicted target
  for (const key of ['cube', 'card']) {
    const st = T[key];
    if (!st.raw || t - st.seen > Math.max(0.25, 2.5 / Math.max(fps, 1)) || !tracker) continue;
    const tg = tracker.targets[key];
    const b = projectedBox(tg, st.raw, K);
    if (!b) continue;
    const pad = 0.35 * Math.max(b.x1 - b.x0, b.y1 - b.y0) + 24;
    const x0 = Math.max(0, Math.floor(b.x0 - pad)), y0 = Math.max(0, Math.floor(b.y0 - pad));
    const x1 = Math.min(vw, Math.ceil(b.x1 + pad)), y1 = Math.min(vh, Math.ceil(b.y1 + pad));
    if (x1 - x0 < 32 || y1 - y0 < 32) continue;
    const cellNative = tg.cell * K.fx / Math.max(50, st.raw.t[2]);
    const scale = Math.min(1, PROC / Math.max(x1 - x0, y1 - y0), 10 / cellNative);
    return { sx: x0, sy: y0, sw: x1 - x0, sh: y1 - y0, scale, radius: clamp(Math.round(cellNative * scale), 2, 7), mode: key };
  }
  // searching: alternate full frame and a sharper centre crop

  if (frameNo % 2 === 0) {
    const scale = PROC / L;
    return { sx: 0, sy: 0, sw: vw, sh: vh, scale, radius: [2, 3, 5][(frameNo >> 1) % 3], mode: 'search' };
  }
  const cw = Math.round(vw / 2), ch = Math.round(vh / 2);
  const scale = Math.min(1, PROC / Math.max(cw, ch));
  return { sx: Math.round((vw - cw) / 2), sy: Math.round((vh - ch) / 2), sw: cw, sh: ch, scale, radius: [3, 5][(frameNo >> 1) % 2], mode: 'centre' };
}
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

let region = null;
function onFrame(nowMs) {
  frameNo++;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !tracker) return render(nowMs / 1000);
  if (vw !== view.vw || vh !== view.vh) updateProjection();
  const t = nowMs / 1000;
  if (lastFrameT) fps = fps * 0.9 + 0.1 / Math.max(1e-3, t - lastFrameT);
  lastFrameT = t;
  const K = intrinsics();
  region = chooseRegion(K, t);
  const w = Math.max(16, Math.round(region.sw * region.scale)), h = Math.max(16, Math.round(region.sh * region.scale));
  if (proc.width !== w || proc.height !== h) { proc.width = w; proc.height = h; }
  const p0 = performance.now();
  if (disp.width !== vw || disp.height !== vh) { disp.width = vw; disp.height = vh; }
  dctx.drawImage(video, 0, 0, vw, vh);
  const p1 = performance.now();
  pctx.drawImage(disp, region.sx, region.sy, region.sw, region.sh, 0, 0, w, h);
  const img = pctx.getImageData(0, 0, w, h);
  const p2 = performance.now();
  const res = tracker.process(img.data, w, h, { x0: region.sx, y0: region.sy, scale: w / region.sw }, K, t, { radius: region.radius });
  const p3 = performance.now();
  procMs = procMs * 0.85 + 0.15 * (p3 - p0);
  timing.capture = timing.capture * 0.85 + 0.15 * (p1 - p0);
  timing.read = timing.read * 0.85 + 0.15 * (p2 - p1);
  timing.track = timing.track * 0.85 + 0.15 * (p3 - p2);
  const cvMs = timing.read + timing.track;
  if (frameNo % 15 === 0) {
    if (cvMs > 30 && PROC > 400) PROC = Math.round(PROC * 0.85);
    else if (cvMs < 14 && PROC < 640) PROC = Math.min(640, Math.round(PROC * 1.1));
  }
  lastRes = res;
  handleResult(res, t);
  if (videoTex) videoTex.needsUpdate = true;   // draw exactly the frame that was tracked
  const r0 = performance.now();
  render(t);
  timing.render = timing.render * 0.85 + 0.15 * (performance.now() - r0);
}

const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpV = new THREE.Vector3();
function handleResult(res, t) {
  for (const key of ['cube', 'card']) {
    const st = T[key], p = res.poses[key];
    const good = p && (p.rms < 3.5 || (p.n >= 2 && p.rms < 6)) && !p.ambiguous;
    if (good) {
      if (t - st.seen > 0.6) { st.filtT.reset(); st.filtQ.reset(); st.prevQ = null; st.acquired = true; }
      st.raw = p; st.seen = t;
      let q = matToQuat(p.R);
      if (st.prevQ && q[0] * st.prevQ[0] + q[1] * st.prevQ[1] + q[2] * st.prevQ[2] + q[3] * st.prevQ[3] < 0) q = q.map(x => -x);
      st.prevQ = q;
      const qf = st.filtQ.filter(q, t), tf = st.filtT.filter(p.t, t);
      const nq = Math.hypot(...qf);
      st.q = qf.map(x => x / nq); st.t = tf;
    }
  }
  const visible = k => T[k].raw && t - T[k].seen < Math.max(0.3, 2.5 / Math.max(fps, 1));
  const next = visible('cube') ? 'cube' : visible('card') ? 'card' : null;
  if (next !== active) switchTarget(next, t);
  for (const key of ['cube', 'card']) {
    const st = T[key], a = anchors[key];
    a.visible = key === active;
    if (key !== active || !st.q) continue;
    tmpQ.set(st.q[0], st.q[1], st.q[2], st.q[3]);
    tmpM.makeRotationFromQuaternion(tmpQ).setPosition(st.t[0], st.t[1], st.t[2]);
    a.matrix.multiplyMatrices(CV2GL, tmpM);
    a.matrixWorldNeedsUpdate = true;
  }
  if (active === 'cube') updateBase(T.cube, t);
  if (active && place.needFace && modelRoot) { faceCamera(false); place.needFace = false; }
  updateStatus(t);
}

function switchTarget(next, t) {
  const prev = active;
  active = next;
  if (next) {
    anchors[next].add(holder);
    applyPlinthStyle(); applySize();
    if (next === 'card') { place.Q.identity(); place.from.identity(); place.to.identity(); place.dur = 0; }
    else if (prev !== 'cube' || t - T.cube.lostAt > 1.5) chooseInitialBase(T.cube);
    if (t - (T[next].lostAt || -10) > 1.5) { place.needFace = true; place.appear = 0; }
  } else if (prev) {
    T[prev].lostAt = t;
  }
}

// ---------------- which cube face is "up" ----------------
const faceN = () => layout.faces.map(f => f.N);
function chooseInitialBase(st) {
  if (!st.raw) return;
  const R = st.raw.R, up = settings.upright ? upInCamera() : null;
  let best = 0, bs = -9;
  faceN().forEach((N, i) => {
    const n = mulv(R, N);
    const sc = up ? n[0] * up[0] + n[1] * up[1] + n[2] * up[2] : (n[2] < 0.15 ? -n[1] : -9);
    if (sc > bs) { bs = sc; best = i; }
  });
  st.base = best;
  const q = quatYto(layout.faces[best].N);
  place.Q.copy(q); place.from.copy(q); place.to.copy(q); place.dur = 0;
}
function updateBase(st, t) {
  if (!settings.upright || !st.raw) return;
  const up = upInCamera();
  if (!up) return;
  const R = st.raw.R;
  const sc = faceN().map(N => { const n = mulv(R, N); return n[0] * up[0] + n[1] * up[1] + n[2] * up[2]; });
  let best = 0;
  for (let i = 1; i < 6; i++) if (sc[i] > sc[best]) best = i;
  if (best !== st.base && sc[best] > sc[st.base] + 0.3) {
    const a = new THREE.Vector3(...layout.faces[st.base].N), b = new THREE.Vector3(...layout.faces[best].N);
    const roll = new THREE.Quaternion().setFromUnitVectors(a, b);
    place.from.copy(place.Q);
    place.to.premultiply(roll);
    place.t0 = t; place.dur = 0.45;
    st.base = best;
  }
}
function quatYto(N) { return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...N)); }

// turn the model about its base so its front (+Z) points at the camera
function faceCamera(animate) {
  const key = active;
  if (!key || !T[key].raw) return;
  const p = T[key].raw;
  // camera position in target coords: -R^T t
  const R = p.R, t = p.t;
  const cam = new THREE.Vector3(-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]));
  // into the holder frame (undo base rotation)
  const inv = place.to.clone().invert();
  cam.applyQuaternion(inv);
  const yaw = Math.atan2(cam.x, cam.z);
  setYaw(yaw, animate);
}
function setYaw(y, animate) {
  const now = performance.now() / 1000;
  let d = y - place.yaw;
  d = Math.atan2(Math.sin(d), Math.cos(d));
  place.yawFrom = place.yaw; place.yawTo = place.yaw + d; place.yt0 = animate ? now : now - 10;
}

// ---------------- render ----------------
const ease = x => x < 0 ? 0 : x > 1 ? 1 : 1 - Math.pow(1 - x, 3);
function render(t = performance.now() / 1000) {
  // placement animation
  const now = performance.now() / 1000;
  if (place.dur > 0) {
    const k = ease((t - place.t0) / place.dur);
    place.Q.slerpQuaternions(place.from, place.to, k);
    if (k >= 1) { place.from.copy(place.to); place.dur = 0; }
  } else place.Q.copy(place.to);
  const ky = ease((now - place.yt0) / 0.5);
  place.yaw = place.yawFrom + (place.yawTo - place.yawFrom) * ky;
  holder.quaternion.copy(place.Q);
  const off = active === 'card' ? 0.3 : S / 2 * (plinth && plinth.visible ? 1.025 : 1) + 0.2;
  holder.position.set(0, off, 0).applyQuaternion(place.Q);
  yawGroup.rotation.y = place.yaw;
  place.appear = Math.min(1, place.appear + 0.12);
  const ap = 0.85 + 0.15 * ease(place.appear);
  yawGroup.scale.setScalar(ap);
  // light from above the building, slightly from the camera side
  if (active) {
    holder.updateMatrixWorld(true);
    const p = new THREE.Vector3().setFromMatrixPosition(holder.matrixWorld);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(holder.getWorldQuaternion(tmpQ));
    const toCam = p.clone().negate().normalize();
    sun.position.copy(p).addScaledVector(up, 500).addScaledVector(toCam, 260).add(new THREE.Vector3(-160, 0, 0));
    sun.target.position.copy(p);
    sun.castShadow = settings.shadows;
  }
  renderer.clear();
  if (videoTex) renderer.render(bgScene, bgCam);
  renderer.render(scene, camera);
  drawDebug();
}

// ---------------- status + debug ----------------
let hintTimer = 0;
function updateStatus(t) {
  const pill = $('#status'), txt = $('#statusText'), hint = $('#hint');
  if (active) {
    pill.classList.add('ok');
    txt.textContent = (current ? current.name : 'Tracking') + (active === 'card' ? ' · card' : '');
    hint.classList.add('off'); hintTimer = t;
  } else {
    pill.classList.remove('ok');
    txt.textContent = 'Looking for the cube…';
    if (t - hintTimer > 1.2) hint.classList.remove('off');
  }
}
function drawDebug() {
  const W = dbgCanvas.width, H = dbgCanvas.height;
  dbg.clearRect(0, 0, W, H);
  if (!settings.debug || !lastRes) return;
  const s = view.s, ox = view.ox, oy = view.oy, map = p => [(p[0] - ox) * s, (p[1] - oy) * s];
  dbg.lineWidth = 2 * (window.devicePixelRatio || 1);
  dbg.font = `${12 * (window.devicePixelRatio || 1)}px ui-monospace, monospace`;
  for (const m of lastRes.markers) {
    const c = m.corners.map(map);
    dbg.strokeStyle = m.refined ? '#6fcf97' : '#f2c94c';
    dbg.beginPath(); dbg.moveTo(c[0][0], c[0][1]); for (let i = 1; i < 4; i++) dbg.lineTo(c[i][0], c[i][1]); dbg.closePath(); dbg.stroke();
    dbg.fillStyle = '#e8a33d'; dbg.fillRect(c[0][0] - 3, c[0][1] - 3, 6, 6);
    dbg.fillStyle = '#fff'; dbg.fillText(String(m.id), (c[0][0] + c[2][0]) / 2, (c[0][1] + c[2][1]) / 2);
  }
  if (region) {
    const a = map([region.sx, region.sy]), b = map([region.sx + region.sw, region.sy + region.sh]);
    dbg.strokeStyle = 'rgba(255,255,255,0.35)'; dbg.setLineDash([6, 6]); dbg.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); dbg.setLineDash([]);
  }
  const p = active && T[active].raw;
  const lines = [
    `fps ${fps.toFixed(0)}  track ${procMs.toFixed(1)} ms  ${region ? region.mode : ''} r${region ? region.radius : ''}`,
    `capture ${timing.capture.toFixed(1)}  read ${timing.read.toFixed(1)}  cv ${timing.track.toFixed(1)}  draw ${timing.render.toFixed(1)} ms`,
    `video ${view.vw}x${view.vh}  proc ${PROC}  fov ${settings.fov}°  markers ${lastRes.markers.length}`,
    p ? `${active}: n ${p.n}  rms ${p.rms.toFixed(2)} px  z ${p.t[2].toFixed(0)} mm  base ${T.cube.base}` : 'no pose',
    `tilt sensor ${sensor.ok ? 'on' : 'off'}`,
  ];
  const lh = 16 * (window.devicePixelRatio || 1), y0 = 64 * (window.devicePixelRatio || 1) + lh;
  dbg.fillStyle = 'rgba(0,0,0,0.55)'; dbg.fillRect(8, y0 - lh, Math.min(W - 16, 420 * (window.devicePixelRatio || 1)), lh * (lines.length + 0.6));
  dbg.fillStyle = '#fff';
  lines.forEach((l, i) => dbg.fillText(l, 16, y0 + i * lh));
}

// ---------------- UI ----------------
function openSheet(id) {
  for (const s of document.querySelectorAll('.sheet')) s.classList.toggle('open', s.id === id);
  $('#scrim').classList.toggle('open', !!id);
}
$('#scrim').addEventListener('click', () => openSheet(null));
$('#btnSettings').addEventListener('click', () => openSheet('sheetSettings'));
$('#btnHelp').addEventListener('click', () => openSheet('sheetHelp'));
$('#btnInfo').addEventListener('click', () => { updateInfo(); openSheet('sheetInfo'); });
$('#btnTurn').addEventListener('click', () => setYaw(place.yawTo + Math.PI / 2, true));
$('#btnFace').addEventListener('click', () => faceCamera(true));
function bindSettingsUI() {
  const seg = $('#segPlinth');
  const paint = () => {
    for (const b of seg.querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.v === settings.plinth));
    $('#rngSize').value = settings.size; $('#sizeVal').textContent = (+settings.size).toFixed(1) + '×';
    $('#chkUpright').checked = settings.upright; $('#chkShadow').checked = settings.shadows; $('#chkDebug').checked = settings.debug;
    $('#rngFov').value = settings.fov; $('#fovVal').textContent = settings.fov + '°';
  };
  seg.addEventListener('click', e => { const v = e.target.closest('button')?.dataset.v; if (!v) return; settings.plinth = v; saveSettings(); applyPlinthStyle(); paint(); });
  $('#rngSize').addEventListener('input', e => { settings.size = +e.target.value; saveSettings(); applySize(); paint(); });
  $('#chkUpright').addEventListener('change', e => { settings.upright = e.target.checked; saveSettings(); });
  $('#chkShadow').addEventListener('change', e => { settings.shadows = e.target.checked; saveSettings(); });
  $('#chkDebug').addEventListener('change', e => { settings.debug = e.target.checked; saveSettings(); });
  $('#rngFov').addEventListener('input', e => { settings.fov = +e.target.value; saveSettings(); updateProjection(); paint(); });
  paint();
}
let toastT = 0;
function toast(msg) { const el = $('#toast'); el.textContent = msg; el.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), 3200); }

// expose a little for automated tests
window.__arcube = { T, get active() { return active; }, get lastRes() { return lastRes; }, place, settings, view, intrinsics, timing, get fps() { return fps; }, get procMs() { return procMs; } };
