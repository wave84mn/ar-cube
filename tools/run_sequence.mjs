// Sequential tracking like the app (ROI from the previous pose, temporal prior) + One Euro smoothing.
// usage: node tests/run_sequence.mjs FRAMEDIR [minCutT betaT minCutQ betaQ]
import fs from 'fs'; import path from 'path'; import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const sharp = require('/home/claude/.npm-global/lib/node_modules/sharp');
import { Tracker, projectedBox } from '../site/js/cv/tracker.js';
import { project, OneEuro, matToQuat, quatToMat } from '../site/js/cv/pose.js';
const [dir, a = '1.0', b = '0.02', c = '1.0', d = '4.0', dc = '1.0'] = process.argv.slice(2);
const layout = JSON.parse(fs.readFileSync(new URL('../site/data/layout.json', import.meta.url)));
const gt = JSON.parse(fs.readFileSync(path.join(dir, 'gt.json')));
const K = gt.K, tracker = new Tracker(layout), tg = tracker.targets.cube;
const fT = new OneEuro(3, +a, +b, +dc), fQ = new OneEuro(4, +c, +d, +dc);
const rawC = [];
let raw = null, seen = -10, prevQ = null, n = 0;
const errRaw = [], errF = [], filt = [];
for (let i = 0; i < gt.frames.length; i++) {
  const fr = gt.frames[i], t = i / 30, Rg = fr.R.flat();
  let roi, hint = {}, pipe = sharp(path.join(dir, fr.file));
  if (raw && t - seen < 0.25) {
    const bx = projectedBox(tg, raw, K), pad = 0.35 * Math.max(bx.x1 - bx.x0, bx.y1 - bx.y0) + 24;
    const x0 = Math.max(0, Math.floor(bx.x0 - pad)), y0 = Math.max(0, Math.floor(bx.y0 - pad)), x1 = Math.min(gt.W, Math.ceil(bx.x1 + pad)), y1 = Math.min(gt.H, Math.ceil(bx.y1 + pad));
    const cell = tg.cell * K.fx / raw.t[2], sc = Math.min(1, 640 / Math.max(x1 - x0, y1 - y0), 10 / cell);
    roi = { x0, y0, scale: sc }; hint.radius = Math.max(2, Math.min(7, Math.round(cell * sc)));
    pipe = pipe.extract({ left: x0, top: y0, width: x1 - x0, height: y1 - y0 }).resize(Math.round((x1 - x0) * sc), Math.round((y1 - y0) * sc), { kernel: 'linear' });
  } else {
    roi = { x0: 0, y0: 0, scale: 640 / gt.W };
    pipe = pipe.resize(640, Math.round(gt.H * 640 / gt.W), { kernel: 'linear' });
  }
  const { data, info } = await pipe.ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const res = tracker.process(new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), info.width, info.height, roi, K, t, hint);
  const p = res.poses.cube;
  if (!p) continue;
  n++; raw = p; seen = t;
  let q = matToQuat(p.R);
  if (prevQ && q.reduce((s, x, k) => s + x * prevQ[k], 0) < 0) q = q.map(x => -x);
  prevQ = q;
  const qf = fQ.filter(q, t), tf = fT.filter(p.t, t), nq = Math.hypot(...qf), Rf = quatToMat(qf.map(x => x / nq));
  const ov = (R, tt) => tg.outline.map(X => { const g = project(Rg, fr.t, K, X), e = project(R, tt, K, X); return [e[0] - g[0], e[1] - g[1]]; });
  const er = ov(p.R, p.t), ef = ov(Rf, tf);
  errRaw.push(er.reduce((s, v) => s + Math.hypot(...v), 0) / 8); errF.push(ef.reduce((s, v) => s + Math.hypot(...v), 0) / 8);
  filt.push(ef); rawC.push(er);
}
const q = (arr, pp) => { const s = arr.slice().sort((x, y) => x - y); return s[Math.floor(pp * (s.length - 1))]; };
// jitter = frame-to-frame change of the corner error (removes slow bias)
const jit = arr => { const d = []; for (let i = 1; i < arr.length; i++) d.push(arr[i].reduce((s, v, k) => s + Math.hypot(v[0] - arr[i - 1][k][0], v[1] - arr[i - 1][k][1]), 0) / 8); return d; };
const jr = jit(filt), jraw = jit(rawC);
console.log(`${path.basename(dir)} filter T(${a},${b}) Q(${c},${d}): tracked ${n}/${gt.frames.length}  raw err med ${q(errRaw, .5).toFixed(2)} p90 ${q(errRaw, .9).toFixed(2)} | filtered med ${q(errF, .5).toFixed(2)} p90 ${q(errF, .9).toFixed(2)} max ${Math.max(...errF).toFixed(1)} | jitter med ${q(jr, .5).toFixed(3)} p90 ${q(jr, .9).toFixed(3)} (raw ${q(jraw, .5).toFixed(3)}/${q(jraw, .9).toFixed(3)}) px dc ${dc}`);
