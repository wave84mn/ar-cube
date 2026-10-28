// Cube / card tracker: marker detection -> correspondences -> multi-marker pose with disambiguation.
import { MarkerDetector } from './detector.js';
import { refinePose, planarPoseCandidates, reprojErrors, mulv, project } from './pose.js';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export class Tracker {
  constructor(layout, opts = {}) {
    this.layout = layout;
    const S = layout.edge;
    this.targets = {};
    // cube
    const faces = layout.faces.map(f => ({ U: f.U, V: f.V, N: f.N, O: f.N.map(x => x * S / 2) }));
    const cubeMarkers = new Map();
    for (const m of layout.markers) cubeMarkers.set(m.id, { corners: m.corners, face: m.face });
    this.targets.cube = { name: 'cube', markers: cubeMarkers, planes: faces, cell: layout.cell,
      outline: cubeCorners(S) };
    // flat card
    if (layout.card) {
      const cm = new Map();
      for (const m of layout.card.markers) cm.set(m.id, { corners: m.corners, face: 0 });
      const h = layout.card.size / 2;
      this.targets.card = { name: 'card', markers: cm, planes: [{ U: [1, 0, 0], V: [0, 0, -1], N: [0, 1, 0], O: [0, 0, 0] }],
        cell: layout.card.cell, outline: [[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]] };
    }
    const codes = [...layout.markers, ...(layout.card ? layout.card.markers : [])].map(m => ({ id: m.id, cells: m.cells }));
    this.detector = new MarkerDetector(codes, opts.detector || {});
    this.prev = {};
    this.frame = 0;
  }

  // rgba/w/h: processing image; roi: {x0, y0, scale} maps processing px -> video px (xv = x0 + xp / scale);
  // K: intrinsics in video px. Returns {markers, poses: {cube?, card?}, stats}
  process(rgba, w, h, roi, K, time, hint = {}) {
    const t0 = now();
    this.frame++;
    const radii = [2, 3, 5];
    const radius = hint.radius ?? radii[this.frame % radii.length];
    const det = this.detector.detect(rgba, w, h, { radius });
    const t1 = now();
    const markers = det.map(m => ({ id: m.id, hamming: m.hamming, refined: !!m.refined,
      corners: m.corners.map(p => [roi.x0 + p.x / roi.scale, roi.y0 + p.y / roi.scale]) }));
    const poses = {};
    for (const key of Object.keys(this.targets)) {
      const tg = this.targets[key];
      const mine = markers.filter(m => tg.markers.has(m.id));
      if (!mine.length) continue;
      const p = this._solve(tg, mine, K, this.prev[key] && time - this.prev[key].time < 0.5 ? this.prev[key] : null);
      if (p) { p.time = time; poses[key] = p; this.prev[key] = p; }
    }
    return { markers, poses, stats: { detectMs: t1 - t0, totalMs: now() - t0, radius, candidates: this.detector.stats.candidates } };
  }

  _solve(tg, mine, K, prior) {
    let groups = mine.map(m => ({ id: m.id, face: tg.markers.get(m.id).face, obj: tg.markers.get(m.id).corners, img: m.corners }));
    for (let pass = 0; pass < 3; pass++) {
      const obj = groups.flatMap(g => g.obj), img = groups.flatMap(g => g.img);
      const cands = [];
      if (prior) cands.push(refinePose(prior.R, prior.t, obj, img, K));
      const needInit = !prior || cands[0].cost / obj.length > 4;
      if (needInit) {
        // planar candidates from the face with most markers
        const count = new Map();
        for (const g of groups) count.set(g.face, (count.get(g.face) || 0) + 1);
        const faces = [...count.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(e => e[0]);
        for (const f of faces) {
          const gs = groups.filter(g => g.face === f);
          const o = gs.flatMap(g => g.obj), im = gs.flatMap(g => g.img);
          for (const c of planarPoseCandidates(o, im, K, tg.planes[f])) cands.push(refinePose(c.R, c.t, obj, img, K));
        }
      }
      // reject candidates where an observed face points away from the camera
      const ok = cands.filter(c => c.t[2] > 0 && groups.every(g => facing(c, tg.planes[g.face])));
      const pool = ok.length ? ok : cands;
      pool.sort((a, b) => a.cost - b.cost);
      const best = pool[0];
      if (!best) return null;
      // per-marker error, drop the worst outlier and re-solve
      const errs = groups.map(g => { const e = reprojErrors(best.R, best.t, K, g.obj, g.img); return e.reduce((s, x) => s + x, 0) / 4; });
      const sorted = errs.slice().sort((a, b) => a - b), med = sorted[sorted.length >> 1];
      let worst = 0;
      for (let i = 1; i < errs.length; i++) if (errs[i] > errs[worst]) worst = i;
      if (groups.length > 1 && errs[worst] > Math.max(4, 3 * med)) { groups = groups.filter((_, i) => i !== worst); prior = best; continue; }
      const all = reprojErrors(best.R, best.t, K, obj, img);
      const rms = Math.sqrt(all.reduce((s, x) => s + x * x, 0) / all.length);
      const faces = [...new Set(groups.map(g => g.face))];
      // ambiguity margin: how much worse the runner-up (different orientation) is
      let margin = Infinity;
      for (const c of pool.slice(1)) {
        const ang = rotAngle(best.R, c.R);
        if (ang > 5) { margin = Math.min(margin, (c.cost - best.cost) / Math.max(best.cost, 1)); }
      }
      return { R: best.R, t: best.t, rms, n: groups.length, ids: groups.map(g => g.id), faces, ambiguous: margin < 0.5 && groups.length === 1 };
    }
    return null;
  }
}

function facing(pose, plane) {
  const n = mulv(pose.R, plane.N), o = mulv(pose.R, plane.O);
  const c = [o[0] + pose.t[0], o[1] + pose.t[1], o[2] + pose.t[2]];
  return dot(n, c) < 0;
}

function rotAngle(A, B) {   // angle between rotations in degrees
  let tr = 0;
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) tr += A[i * 3 + k] * B[i * 3 + k];
  return Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180 / Math.PI;
}

function cubeCorners(S) {
  const h = S / 2, out = [];
  for (const x of [-h, h]) for (const y of [-h, h]) for (const z of [-h, h]) out.push([x, y, z]);
  return out;
}

// bounding box (video px) of the target outline under a pose; null if behind the camera
export function projectedBox(target, pose, K) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const X of target.outline) {
    const p = project(pose.R, pose.t, K, X);
    if (p[2] <= 0) return null;
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
  }
  return { x0, y0, x1, y1 };
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
