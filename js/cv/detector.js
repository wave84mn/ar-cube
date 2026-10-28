// ArUco marker detector for the AR cube.
// Contour pipeline adapted from js-aruco / js-aruco2 (MIT, (c) 2011 Juan Mellado, (c) 2020 Damiano Falcioni):
// adaptive threshold -> Suzuki border following -> polygon approximation -> convex quads.
// Changes for this project: typed arrays, cell-centre sampling instead of a warped patch, a closed set of
// codes with a strict Hamming limit, and sub-pixel corner refinement by fitting lines to the marker edges.

const NB = [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]];

function popcount32(x) {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

// 6x6 bits (row-major, 1 = white) -> [hi, lo] 32-bit words
function packBits(bits) {
  let hi = 0, lo = 0;
  for (let i = 0; i < 36; i++) {
    if (!bits[i]) continue;
    if (i < 4) hi |= 1 << (3 - i); else lo |= 1 << (35 - i);
  }
  return [hi >>> 0, lo >>> 0];
}

function rotateBits(bits) { // rotate 6x6 grid 90 deg (same convention as js-aruco)
  const out = new Array(36);
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) out[i * 6 + j] = bits[(6 - j - 1) * 6 + i];
  return out;
}

export class MarkerDetector {
  // codes: [{id, cells}] where cells is the 8x8 (1 = black) grid from layout.json
  constructor(codes, opts = {}) {
    this.maxHamming = opts.maxHamming ?? 3;
    this.minSide = opts.minSide ?? 12;
    this.threshC = opts.threshC ?? 7;
    this.dict = codes.map(c => {
      const bits = [];
      for (let i = 1; i < 7; i++) for (let j = 1; j < 7; j++) bits.push(c.cells[i][j] ? 0 : 1);
      return { id: c.id, packed: packBits(bits) };
    });
    this.w = 0; this.h = 0;
  }

  _alloc(w, h) {
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.gray = new Uint8Array(w * h);
    this.blur = new Uint8Array(w * h);
    this.tmp = new Uint16Array(w * h);
    this.bin = new Int32Array((w + 2) * (h + 2));
  }

  // rgba: Uint8ClampedArray (w*h*4). Returns markers [{id, corners:[{x,y}x4], rot, hamming}]
  detect(rgba, w, h, opts = {}) {
    this._alloc(w, h);
    const g = this.gray;
    for (let i = 0, j = 0, n = w * h; j < n; i += 4, j++) g[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
    return this.detectGray(g, w, h, opts);
  }

  detectGray(g, w, h, opts = {}) {
    this._alloc(w, h);
    if (g !== this.gray) this.gray.set(g);
    const radius = opts.radius ?? 2;
    this.stats = { candidates: 0, radius };
    this._boxBlur(this.gray, this.blur, radius);
    const contours = this._findContours(this.gray, this.blur, this.threshC);
    const quads = [];
    const minLen = 4 * this.minSide;
    for (const c of contours) {
      if (c.length < minLen) continue;
      const poly = approxPolyDP(c, c.length * 0.04);
      if (poly.length !== 4 || !isConvex(poly)) continue;
      if (minEdge(poly) < this.minSide) continue;
      // clockwise in image coordinates (y down)
      const dx1 = poly[1].x - poly[0].x, dy1 = poly[1].y - poly[0].y, dx2 = poly[2].x - poly[0].x, dy2 = poly[2].y - poly[0].y;
      if (dx1 * dy2 - dy1 * dx2 < 0) { const t = poly[1]; poly[1] = poly[3]; poly[3] = t; }
      quads.push(poly);
    }
    this.stats.candidates = quads.length;
    const markers = [];
    for (const q of quads) {
      const m = this._decode(q);
      if (m) markers.push(m);
    }
    // keep one detection per id (largest)
    const best = new Map();
    for (const m of markers) {
      m.area = polyArea(m.corners);
      const o = best.get(m.id);
      if (!o || m.area > o.area) best.set(m.id, m);
    }
    const out = [...best.values()];
    if (opts.refine !== false) for (const m of out) this._refine(m);
    return out;
  }

  // separable box blur, radius r (window 2r+1)
  _boxBlur(src, dst, r) {
    const w = this.w, h = this.h, tmp = this.tmp, win = 2 * r + 1;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let s = src[row] * (r + 1);
      for (let x = 1; x <= r; x++) s += src[row + Math.min(x, w - 1)];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = s;
        s += src[row + Math.min(x + r + 1, w - 1)] - src[row + Math.max(x - r, 0)];
      }
    }
    const div = win * win;
    for (let x = 0; x < w; x++) {
      let s = tmp[x] * (r + 1);
      for (let y = 1; y <= r; y++) s += tmp[Math.min(y, h - 1) * w + x];
      for (let y = 0; y < h; y++) {
        dst[y * w + x] = (s / div) | 0;
        s += tmp[Math.min(y + r + 1, h - 1) * w + x] - tmp[Math.max(y - r, 0) * w + x];
      }
    }
  }

  // threshold (pixel darker than local mean by > c) into a 1-px-padded image, then Suzuki border following
  _findContours(gray, blur, c) {
    const w = this.w, h = this.h, W = w + 2, src = this.bin;
    src.fill(0);
    for (let y = 0; y < h; y++) {
      let p = (y + 1) * W + 1, q = y * w;
      for (let x = 0; x < w; x++, p++, q++) src[p] = (gray[q] + c <= blur[q]) ? 1 : 0;
    }
    const deltas = new Int32Array(16);
    for (let i = 0; i < 8; i++) deltas[i] = deltas[i + 8] = NB[i][0] + NB[i][1] * W;
    const contours = [];
    let nbd = 1;
    for (let i = 0; i < h; i++) {
      let pos = (i + 1) * W + 1;
      for (let j = 0; j < w; j++, pos++) {
        const pix = src[pos];
        if (pix === 0) continue;
        let outer = false, hole = false;
        if (pix === 1 && src[pos - 1] === 0) outer = true;
        else if (pix >= 1 && src[pos + 1] === 0) hole = true;
        if (outer || hole) {
          nbd++;
          contours.push(borderFollowing(src, pos, nbd, j, i, hole, deltas));
        }
      }
    }
    return contours;
  }

  // sample cell centres through the quad homography, classify, match against the code set
  _decode(q) {
    const H = squareToQuad(q);
    const g = this.gray, w = this.w, h = this.h;
    const vals = new Float32Array(64);
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        let s = 0, n = 0;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
          const u = (j + 0.3 + 0.2 * b) / 8, v = (i + 0.3 + 0.2 * a) / 8;
          const d = H[6] * u + H[7] * v + 1;
          const x = (H[0] * u + H[1] * v + H[2]) / d, y = (H[3] * u + H[4] * v + H[5]) / d;
          if (x < 0 || y < 0 || x > w - 1.001 || y > h - 1.001) return null;
          s += bilinear(g, w, x, y); n++;
        }
        vals[i * 8 + j] = s / n;
      }
    }
    // threshold: Otsu on the 64 cell means
    const t = otsu64(vals);
    let blackMean = 0, whiteMean = 0, nb = 0, nw = 0;
    for (let k = 0; k < 64; k++) { if (vals[k] <= t) { blackMean += vals[k]; nb++; } else { whiteMean += vals[k]; nw++; } }
    if (!nw || !nb) return null;
    blackMean /= nb; whiteMean /= nw;
    if (whiteMean - blackMean < 25) return null;
    for (let k = 0; k < 8; k++) {
      if (vals[k] > t || vals[56 + k] > t || vals[k * 8] > t || vals[k * 8 + 7] > t) return null; // border must be black
    }
    let bits = [];
    for (let i = 1; i < 7; i++) for (let j = 1; j < 7; j++) bits.push(vals[i * 8 + j] > t ? 1 : 0);
    let best = null;
    for (let rot = 0; rot < 4; rot++) {
      const [hi, lo] = packBits(bits);
      for (const d of this.dict) {
        const dist = popcount32((hi ^ d.packed[0]) >>> 0) + popcount32((lo ^ d.packed[1]) >>> 0);
        if (dist <= this.maxHamming && (!best || dist < best.hamming)) best = { id: d.id, rot, hamming: dist };
      }
      bits = rotateBits(bits);
    }
    if (!best) return null;
    const r = (4 - best.rot) % 4;
    const corners = [0, 1, 2, 3].map(k => ({ x: q[(r + k) % 4].x, y: q[(r + k) % 4].y }));
    return { id: best.id, corners, hamming: best.hamming, contrast: whiteMean - blackMean };
  }

  // fit a line to each marker edge (strongest dark->bright step along the outward normal), intersect neighbours
  _refine(m) {
    const g = this.gray, w = this.w, h = this.h, c = m.corners;
    const cx = (c[0].x + c[1].x + c[2].x + c[3].x) / 4, cy = (c[0].y + c[1].y + c[2].y + c[3].y) / 4;
    const lines = [];
    for (let k = 0; k < 4; k++) {
      const p0 = c[k], p1 = c[(k + 1) % 4];
      const L = Math.hypot(p1.x - p0.x, p1.y - p0.y);
      const dx = (p1.x - p0.x) / L, dy = (p1.y - p0.y) / L;
      let nx = dy, ny = -dx;
      if (((p0.x + p1.x) / 2 - cx) * nx + ((p0.y + p1.y) / 2 - cy) * ny < 0) { nx = -nx; ny = -ny; }
      const R = Math.max(2, Math.min(5, L * 0.06)), step = 0.5, ns = Math.round(2 * R / step) + 1;
      const N = Math.max(6, Math.min(32, Math.round(L / 2)));
      const pts = [];
      const prof = new Float32Array(ns);
      for (let s = 0; s < N; s++) {
        const t = 0.12 + 0.76 * (s + 0.5) / N;
        const px = p0.x + (p1.x - p0.x) * t, py = p0.y + (p1.y - p0.y) * t;
        let ok = true;
        for (let k2 = 0; k2 < ns; k2++) {
          const o = -R + k2 * step, x = px + nx * o, y = py + ny * o;
          if (x < 0 || y < 0 || x > w - 1.001 || y > h - 1.001) { ok = false; break; }
          prof[k2] = bilinear(g, w, x, y);
        }
        if (!ok) continue;
        let bi = -1, bd = 0;
        for (let k2 = 1; k2 < ns - 1; k2++) {
          const d = prof[k2 + 1] - prof[k2 - 1];
          if (d > bd) { bd = d; bi = k2; }
        }
        if (bi < 2 || bi > ns - 3 || bd < 12) continue;
        const dm = prof[bi] - prof[bi - 2], d0 = bd, dp = prof[bi + 2] - prof[bi];
        const den = dm - 2 * d0 + dp;
        const off = den < 0 ? 0.5 * (dm - dp) / den : 0;
        const o = -R + (bi + Math.max(-0.5, Math.min(0.5, off))) * step;
        pts.push([px + nx * o, py + ny * o]);
      }
      if (pts.length < 4) return;     // keep the integer corners
      let line = fitLine(pts);
      const res = pts.filter(p => Math.abs((p[0] - line.x) * line.nx + (p[1] - line.y) * line.ny) < 0.8);
      if (res.length >= 4 && res.length < pts.length) line = fitLine(res);
      lines.push(line);
    }
    const out = [];
    for (let k = 0; k < 4; k++) {
      const a = lines[(k + 3) % 4], b = lines[k];
      const p = intersect(a, b);
      if (!p) return;
      const d = Math.hypot(p.x - c[k].x, p.y - c[k].y);
      if (d > Math.max(2.5, 0.08 * Math.sqrt(m.area))) return;
      out.push(p);
    }
    m.corners = out; m.refined = true;
  }
}

function bilinear(g, w, x, y) {
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, p = y0 * w + x0;
  return (g[p] * (1 - fx) + g[p + 1] * fx) * (1 - fy) + (g[p + w] * (1 - fx) + g[p + w + 1] * fx) * fy;
}

function fitLine(pts) {
  let mx = 0, my = 0;
  for (const p of pts) { mx += p[0]; my += p[1]; }
  mx /= pts.length; my /= pts.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) { const a = p[0] - mx, b = p[1] - my; sxx += a * a; sxy += a * b; syy += b * b; }
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const dx = Math.cos(th), dy = Math.sin(th);
  return { x: mx, y: my, dx, dy, nx: -dy, ny: dx };
}

function intersect(a, b) {
  const den = a.dx * b.dy - a.dy * b.dx;
  if (Math.abs(den) < 1e-6) return null;
  const t = ((b.x - a.x) * b.dy - (b.y - a.y) * b.dx) / den;
  return { x: a.x + a.dx * t, y: a.y + a.dy * t };
}

function otsu64(vals) {
  const v = Array.from(vals).sort((a, b) => a - b);
  let best = v[31], bestScore = -1, total = 0;
  for (const x of v) total += x;
  let sumB = 0;
  for (let i = 0; i < 63; i++) {
    sumB += v[i];
    const wB = i + 1, wF = 64 - wB;
    const mB = sumB / wB, mF = (total - sumB) / wF;
    const score = wB * wF * (mB - mF) * (mB - mF);
    if (score > bestScore) { bestScore = score; best = (v[i] + v[i + 1]) / 2; }
  }
  return best;
}

// homography mapping the unit square (u,v) in [0,1]^2 to quad corners q[0..3] (TL, TR, BR, BL order)
export function squareToQuad(q) {
  const x0 = q[0].x, y0 = q[0].y, x1 = q[1].x, y1 = q[1].y, x2 = q[2].x, y2 = q[2].y, x3 = q[3].x, y3 = q[3].y;
  const px = x0 - x1 + x2 - x3, py = y0 - y1 + y2 - y3;
  if (Math.abs(px) < 1e-9 && Math.abs(py) < 1e-9) {
    return [x1 - x0, x3 - x0, x0, y1 - y0, y3 - y0, y0, 0, 0];
  }
  const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
  const den = dx1 * dy2 - dx2 * dy1;
  const g = (px * dy2 - dx2 * py) / den, hh = (dx1 * py - px * dy1) / den;
  return [x1 - x0 + g * x1, x3 - x0 + hh * x3, x0, y1 - y0 + g * y1, y3 - y0 + hh * y3, y0, g, hh];
}

function borderFollowing(src, pos, nbd, px, py, hole, deltas) {
  const contour = [];
  let s, sEnd, pos1, pos3, pos4;
  s = sEnd = hole ? 0 : 4;
  do {
    s = (s - 1) & 7;
    pos1 = pos + deltas[s];
    if (src[pos1] !== 0) break;
  } while (s !== sEnd);
  if (s === sEnd) {
    src[pos] = -nbd;
    contour.push({ x: px, y: py });
    return contour;
  }
  pos3 = pos;
  let x = px, y = py;
  for (let guard = 0; guard < 200000; guard++) {
    sEnd = s;
    do { pos4 = pos3 + deltas[++s]; } while (src[pos4] === 0);
    s &= 7;
    if (((s - 1) >>> 0) < (sEnd >>> 0)) src[pos3] = -nbd;
    else if (src[pos3] === 1) src[pos3] = nbd;
    contour.push({ x, y });
    x += NB[s][0]; y += NB[s][1];
    if (pos4 === pos && pos3 === pos1) break;
    pos3 = pos4;
    s = (s + 4) & 7;
  }
  return contour;
}

function approxPolyDP(contour, epsilon) {
  const len = contour.length;
  const sliceR = { s: 0, e: 0 };
  let k = 0, startPt, maxDist = 0;
  const eps2 = epsilon * epsilon;
  const poly = [], stack = [];
  for (let i = 0; i < 3; i++) {
    maxDist = 0;
    k = (k + sliceR.s) % len;
    startPt = contour[k];
    if (++k === len) k = 0;
    for (let j = 1; j < len; j++) {
      const pt = contour[k];
      if (++k === len) k = 0;
      const dx = pt.x - startPt.x, dy = pt.y - startPt.y, d = dx * dx + dy * dy;
      if (d > maxDist) { maxDist = d; sliceR.s = j; }
    }
  }
  if (maxDist <= eps2) return [{ x: startPt.x, y: startPt.y }];
  const slice = { s: k, e: (sliceR.s += k) };
  sliceR.s -= sliceR.s >= len ? len : 0;
  sliceR.e = slice.s;
  if (sliceR.e < sliceR.s) sliceR.e += len;
  stack.push({ s: sliceR.s, e: sliceR.e });
  stack.push({ s: slice.s, e: slice.e });
  while (stack.length) {
    const sl = stack.pop();
    const endPt = contour[sl.e % len];
    k = sl.s % len;
    const sPt = contour[k];
    if (++k === len) k = 0;
    let leEps, rs = 0;
    if (sl.e <= sl.s + 1) leEps = true;
    else {
      let md = 0;
      const dx = endPt.x - sPt.x, dy = endPt.y - sPt.y;
      for (let i = sl.s + 1; i < sl.e; i++) {
        const pt = contour[k];
        if (++k === len) k = 0;
        const d = Math.abs((pt.y - sPt.y) * dx - (pt.x - sPt.x) * dy);
        if (d > md) { md = d; rs = i; }
      }
      leEps = md * md <= eps2 * (dx * dx + dy * dy);
    }
    if (leEps) poly.push({ x: sPt.x, y: sPt.y });
    else {
      stack.push({ s: rs, e: sl.e });
      stack.push({ s: sl.s, e: rs });
    }
    if (poly.length > 8) return poly;   // not a quad; stop early
  }
  return poly;
}

function isConvex(c) {
  let orientation = 0;
  const len = c.length;
  let prev = c[len - 1], cur = c[0];
  let dx0 = cur.x - prev.x, dy0 = cur.y - prev.y;
  for (let i = 0, j = 0; i < len; i++) {
    if (++j === len) j = 0;
    prev = cur; cur = c[j];
    const dx = cur.x - prev.x, dy = cur.y - prev.y;
    const a = dx * dy0, b = dy * dx0;
    orientation |= b > a ? 1 : (b < a ? 2 : 3);
    if (orientation === 3) return false;
    dx0 = dx; dy0 = dy;
  }
  return true;
}

function minEdge(p) {
  let m = Infinity;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) m = Math.min(m, Math.hypot(p[i].x - p[j].x, p[i].y - p[j].y));
  return m;
}

function polyArea(p) {
  let a = 0;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) a += p[j].x * p[i].y - p[i].x * p[j].y;
  return Math.abs(a) / 2;
}
