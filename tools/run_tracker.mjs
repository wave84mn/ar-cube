// Run the browser tracker code on synthetic frames and compare with ground truth.
// usage: node tests/run_tracker.mjs FRAMEDIR [--roi] [--proc 640] [--verbose]
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const sharp = require('/home/claude/.npm-global/lib/node_modules/sharp');
import { Tracker, projectedBox } from '../site/js/cv/tracker.js';
import { project } from '../site/js/cv/pose.js';

const dir = process.argv[2];
const ROI = process.argv.includes('--roi');
const PROC = process.argv.includes('--proc') ? +process.argv[process.argv.indexOf('--proc') + 1] : 640;
const verbose = process.argv.includes('--verbose');
const arg = (k, d) => process.argv.includes(k) ? +process.argv[process.argv.indexOf(k) + 1] : d;
const TARGET_CELL = arg('--cell', 8), RF = arg('--rf', 0.75);
const layout = JSON.parse(fs.readFileSync(new URL('../site/data/layout.json', import.meta.url)));
const gt = JSON.parse(fs.readFileSync(path.join(dir, 'gt.json')));
const K = gt.K, target = gt.target;
const tracker = new Tracker(layout);
const tg = tracker.targets[target];

const flat = R => R.flat();
const rotErr = (A, B) => { let tr = 0; for (let i = 0; i < 9; i++) tr += A[i] * B[i]; return Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180 / Math.PI; };

const rows = [];
let detMs = 0, totMs = 0, nRun = 0;
for (const fr of gt.frames) {
  const Rg = flat(fr.R), tg_ = fr.t;
  let roi = { x0: 0, y0: 0, scale: PROC / Math.max(gt.W, gt.H) };
  const hint = {};
  let pipe = sharp(path.join(dir, fr.file));
  if (ROI) {   // crop around the true outline, as the app does when tracking
    const b = projectedBox(tg, { R: Rg, t: tg_ }, K);
    const pad = 0.35 * Math.max(b.x1 - b.x0, b.y1 - b.y0) + 20;
    const x0 = Math.max(0, Math.floor(b.x0 - pad)), y0 = Math.max(0, Math.floor(b.y0 - pad));
    const x1 = Math.min(gt.W, Math.ceil(b.x1 + pad)), y1 = Math.min(gt.H, Math.ceil(b.y1 + pad));
    const cellNative = tg.cell * K.fx / tg_[2];
    const sc = Math.min(1, PROC / Math.max(x1 - x0, y1 - y0), TARGET_CELL / cellNative);
    roi = { x0, y0, scale: sc };
    hint.radius = Math.max(2, Math.min(7, Math.round(cellNative * sc * RF)));
    pipe = pipe.extract({ left: x0, top: y0, width: x1 - x0, height: y1 - y0 });
    pipe = pipe.resize(Math.round((x1 - x0) * sc), Math.round((y1 - y0) * sc), { kernel: 'linear' });
  } else {
    pipe = pipe.resize(Math.round(gt.W * roi.scale), Math.round(gt.H * roi.scale), { kernel: 'linear' });
  }
  const { data, info } = await pipe.ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  tracker.prev = {};            // independent frames: no temporal prior
  const res = tracker.process(new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), info.width, info.height, roi, K, 0, hint);
  detMs += res.stats.detectMs; totMs += res.stats.totalMs; nRun++;
  // markers that should be readable: face towards camera within 65 deg, all corners inside the image
  let expect = 0, found = 0;
  for (const [id, m] of tg.markers) {
    const plane = tg.planes[m.face];
    const n = [0, 1, 2].map(i => Rg[i * 3] * plane.N[0] + Rg[i * 3 + 1] * plane.N[1] + Rg[i * 3 + 2] * plane.N[2]);
    const c = m.corners.map(X => project(Rg, tg_, K, X));
    const mid = [0, 1, 2].map(i => (c[0][i] + c[2][i]) / 2);
    const ctr = [0, 1, 2].map(i => Rg[i * 3] * m.corners[0][0] + Rg[i * 3 + 1] * m.corners[0][1] + Rg[i * 3 + 2] * m.corners[0][2] + tg_[i]);
    const cosang = -(n[0] * ctr[0] + n[1] * ctr[1] + n[2] * ctr[2]) / Math.hypot(...ctr);
    const inside = c.every(p => p[0] > 2 && p[1] > 2 && p[0] < gt.W - 2 && p[1] < gt.H - 2);
    if (cosang > Math.cos(65 * Math.PI / 180) && inside) {
      expect++;
      if (res.markers.some(d => d.id === id)) found++;
    }
  }
  const est = res.poses[target];
  const row = { file: fr.file, occ: fr.occluded, expect, found, falseIds: res.markers.filter(d => !tg.markers.has(d.id)).length, z: tg_[2] };
  if (est) {
    row.rot = rotErr(Rg, est.R);
    row.trans = Math.hypot(est.t[0] - tg_[0], est.t[1] - tg_[1], est.t[2] - tg_[2]);
    const errs = tg.outline.map(X => { const a = project(Rg, tg_, K, X), b = project(est.R, est.t, K, X); return Math.hypot(a[0] - b[0], a[1] - b[1]); });
    row.px = errs.reduce((s, x) => s + x, 0) / errs.length;
    row.n = est.n; row.rms = est.rms;
  }
  rows.push(row);
  if (verbose) console.log(JSON.stringify(row));
}
const q = (arr, p) => { const s = arr.slice().sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : NaN; };
const withExp = rows.filter(r => r.expect > 0);
const posed = withExp.filter(r => r.px !== undefined);
const px = posed.map(r => r.px), rot = posed.map(r => r.rot);
const bad = posed.filter(r => r.px > 10);
console.log(`${target} ${ROI ? 'ROI' : 'full-frame'} @${PROC}px  frames ${rows.length}`);
console.log(`  markers found ${withExp.reduce((s, r) => s + r.found, 0)} / ${withExp.reduce((s, r) => s + r.expect, 0)} expected readable; ids not in target: ${rows.reduce((s, r) => s + r.falseIds, 0)}`);
console.log(`  pose in ${posed.length}/${withExp.length} frames (${(100 * posed.length / withExp.length).toFixed(1)} %)`);
console.log(`  overlay error px (1280 frame): median ${q(px, 0.5).toFixed(2)}  p90 ${q(px, 0.9).toFixed(2)}  p98 ${q(px, 0.98).toFixed(2)}  max ${Math.max(...px).toFixed(1)}`);
console.log(`  rotation error deg: median ${q(rot, 0.5).toFixed(2)}  p90 ${q(rot, 0.9).toFixed(2)}`);
console.log(`  frames with overlay error > 10 px: ${bad.length}  ${bad.slice(0, 8).map(r => `${r.file}(${r.px.toFixed(0)}px,n=${r.n})`).join(' ')}`);
const missed = withExp.filter(r => r.px === undefined);
console.log(`  frames without pose: ${missed.slice(0, 12).map(r => `${r.file}(exp ${r.expect}, z ${r.z.toFixed(0)}${r.occ ? ', occ' : ''})`).join(' ')}`);
console.log(`  time per frame: detect ${(detMs / nRun).toFixed(1)} ms, total ${(totMs / nRun).toFixed(1)} ms (node, this machine)`);
