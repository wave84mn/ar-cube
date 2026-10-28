// Pose estimation for a rigid set of markers (cube or flat card).
// Camera model: pinhole, OpenCV convention (x right, y down, z forward), pixels.
// Pose: X_cam = R * X_obj + t, R row-major 3x3 (array of 9), t [3].

export function mat3mul(a, b) {
  const o = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return o;
}
export const transpose = a => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
export const mulv = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => Math.hypot(a[0], a[1], a[2]);

export function expSO3(w) {   // Rodrigues
  const th = norm(w);
  if (th < 1e-12) return [1, -w[2], w[1], w[2], 1, -w[0], -w[1], w[0], 1];
  const k = [w[0] / th, w[1] / th, w[2] / th], c = Math.cos(th), s = Math.sin(th), v = 1 - c;
  return [
    c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
    k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s,
    k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v];
}

export function orthonormalize(R) {   // nearest rotation via polar iteration
  let A = R.slice();
  for (let it = 0; it < 8; it++) {
    const inv = inverse3(transpose(A));
    if (!inv) break;
    A = A.map((x, i) => 0.5 * (x + inv[i]));
  }
  return A;
}

function inverse3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  const s = 1 / det;
  return [A * s, -(b * i - c * h) * s, (b * f - c * e) * s, B * s, (a * i - c * g) * s, -(a * f - c * d) * s, C * s, -(a * h - b * g) * s, (a * e - b * d) * s];
}

export function project(R, t, K, X) {
  const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  return [K.fx * x / z + K.cx, K.fy * y / z + K.cy, z];
}

// Levenberg-Marquardt on reprojection error with Huber weights.
export function refinePose(R0, t0, obj, img, K, opts = {}) {
  let R = R0.slice(), t = t0.slice();
  const n = obj.length, huber = opts.huber ?? 2.0, iters = opts.iters ?? 12;
  let lambda = 1e-3;
  const cost = (R, t) => {
    let c = 0;
    for (let i = 0; i < n; i++) {
      const p = project(R, t, K, obj[i]);
      if (p[2] <= 1) return Infinity;
      const e = Math.hypot(p[0] - img[i][0], p[1] - img[i][1]);
      c += e <= huber ? e * e : 2 * huber * e - huber * huber;
    }
    return c;
  };
  let c0 = cost(R, t);
  for (let it = 0; it < iters; it++) {
    const JTJ = new Float64Array(36), JTr = new Float64Array(6);
    for (let i = 0; i < n; i++) {
      const X = obj[i];
      const rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2], ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2], rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
      const x = rx + t[0], y = ry + t[1], z = rz + t[2];
      const iz = 1 / z, iz2 = iz * iz;
      const u = K.fx * x * iz + K.cx, v = K.fy * y * iz + K.cy;
      const eu = u - img[i][0], ev = v - img[i][1];
      const e = Math.hypot(eu, ev), w = e <= huber ? 1 : huber / e;
      // d(u,v)/d(pc)
      const du = [K.fx * iz, 0, -K.fx * x * iz2], dv = [0, K.fy * iz, -K.fy * y * iz2];
      // d(pc)/d(omega) = -[R X]_x ; d(pc)/dt = I
      const S = [0, rz, -ry, -rz, 0, rx, ry, -rx, 0];   // -[r]_x
      const Ju = [du[0] * S[0] + du[1] * S[3] + du[2] * S[6], du[0] * S[1] + du[1] * S[4] + du[2] * S[7], du[0] * S[2] + du[1] * S[5] + du[2] * S[8], du[0], du[1], du[2]];
      const Jv = [dv[0] * S[0] + dv[1] * S[3] + dv[2] * S[6], dv[0] * S[1] + dv[1] * S[4] + dv[2] * S[7], dv[0] * S[2] + dv[1] * S[5] + dv[2] * S[8], dv[0], dv[1], dv[2]];
      for (let a = 0; a < 6; a++) {
        JTr[a] += w * (Ju[a] * eu + Jv[a] * ev);
        for (let b = a; b < 6; b++) JTJ[a * 6 + b] += w * (Ju[a] * Ju[b] + Jv[a] * Jv[b]);
      }
    }
    for (let a = 0; a < 6; a++) for (let b = 0; b < a; b++) JTJ[a * 6 + b] = JTJ[b * 6 + a];
    let improved = false;
    for (let tries = 0; tries < 6; tries++) {
      const A = Array.from(JTJ);
      for (let a = 0; a < 6; a++) A[a * 6 + a] *= 1 + lambda;
      const d = solve6(A, Array.from(JTr).map(x => -x));
      if (!d) { lambda *= 10; continue; }
      const R1 = mat3mul(expSO3([d[0], d[1], d[2]]), R), t1 = [t[0] + d[3], t[1] + d[4], t[2] + d[5]];
      const c1 = cost(R1, t1);
      if (c1 < c0) {
        R = R1; t = t1; lambda = Math.max(lambda / 10, 1e-7);
        const small = Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]) < 1e-7 && Math.abs(d[3]) + Math.abs(d[4]) + Math.abs(d[5]) < 1e-5;
        const rel = (c0 - c1) / (c0 + 1e-12);
        c0 = c1; improved = !small && rel > 1e-9;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  R = orthonormalize(R);
  return { R, t, cost: c0 };
}

function solve6(A, b) {   // Gaussian elimination with partial pivoting
  const n = 6;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r * n + i]) > Math.abs(A[p * n + i])) p = r;
    if (Math.abs(A[p * n + i]) < 1e-12) return null;
    if (p !== i) {
      for (let c = 0; c < n; c++) { const tmp = A[i * n + c]; A[i * n + c] = A[p * n + c]; A[p * n + c] = tmp; }
      const tb = b[i]; b[i] = b[p]; b[p] = tb;
    }
    for (let r = i + 1; r < n; r++) {
      const f = A[r * n + i] / A[i * n + i];
      if (!f) continue;
      for (let c = i; c < n; c++) A[r * n + c] -= f * A[i * n + c];
      b[r] -= f * b[i];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < n; c++) s -= A[i * n + c] * x[c];
    x[i] = s / A[i * n + i];
  }
  return x;
}

// Jacobi eigen-decomposition of a symmetric n x n matrix (flat array). Returns {values, vectors(columns)}.
function jacobiEigen(Ain, n) {
  const A = Ain.slice(), V = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const tt = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => A[i * n + i]), vectors: V };
}

// Homography (3x3, row-major) from plane points (u,v) to normalised image points (x,y), normalised DLT.
function homographyDLT(uv, xy) {
  const n = uv.length;
  const norm2 = pts => {
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;
    let d = 0;
    for (const p of pts) d += Math.hypot(p[0] - mx, p[1] - my);
    const s = Math.SQRT2 / (d / n || 1);
    return { T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1], p: pts.map(p => [(p[0] - mx) * s, (p[1] - my) * s]) };
  };
  const a = norm2(uv), b = norm2(xy);
  const M = new Array(81).fill(0);
  for (let i = 0; i < n; i++) {
    const [u, v] = a.p[i], [x, y] = b.p[i];
    const r1 = [-u, -v, -1, 0, 0, 0, x * u, x * v, x], r2 = [0, 0, 0, -u, -v, -1, y * u, y * v, y];
    for (let p = 0; p < 9; p++) for (let q = 0; q < 9; q++) M[p * 9 + q] += r1[p] * r1[q] + r2[p] * r2[q];
  }
  const { values, vectors } = jacobiEigen(M, 9);
  let k = 0;
  for (let i = 1; i < 9; i++) if (values[i] < values[k]) k = i;
  const Hn = Array.from({ length: 9 }, (_, i) => vectors[i * 9 + k]);
  // H = Tb^-1 * Hn * Ta
  return mat3mul(mat3mul(inverse3(b.T), Hn), a.T);
}

// Pose candidates from coplanar points of one face. plane: {U,V,N,O} (object frame), points given in object coords.
export function planarPoseCandidates(obj, img, K, plane) {
  const uv = obj.map(X => { const d = [X[0] - plane.O[0], X[1] - plane.O[1], X[2] - plane.O[2]]; return [dot(d, plane.U), dot(d, plane.V)]; });
  const xy = img.map(p => [(p[0] - K.cx) / K.fx, (p[1] - K.cy) / K.fy]);
  const H = homographyDLT(uv, xy);
  let h1 = [H[0], H[3], H[6]], h2 = [H[1], H[4], H[7]], h3 = [H[2], H[5], H[8]];
  const lam = 2 / (norm(h1) + norm(h2));
  let s = lam;
  if (h3[2] * s < 0) s = -s;   // plane in front of the camera
  let r1 = h1.map(x => x * s), r2 = h2.map(x => x * s), tp = h3.map(x => x * s);
  const r3 = cross(r1, r2);
  const Rp = orthonormalize([r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]]);
  // plane frame -> object frame: X = O + u U + v V + w N  => R_obj = Rp * F^T, t = tp - R_obj O
  const F = [plane.U[0], plane.V[0], plane.N[0], plane.U[1], plane.V[1], plane.N[1], plane.U[2], plane.V[2], plane.N[2]];
  const toObj = (Rp_, tp_) => { const R = mat3mul(Rp_, transpose(F)); const RO = mulv(R, plane.O); return { R, t: [tp_[0] - RO[0], tp_[1] - RO[1], tp_[2] - RO[2]] }; };
  const out = [toObj(Rp, tp)];
  // second candidate: mirror the plane normal about the line of sight (planar pose ambiguity)
  const n1 = [Rp[2], Rp[5], Rp[8]], d = tp.map(x => x / norm(tp));
  const c = dot(n1, d), n2 = [2 * c * d[0] - n1[0], 2 * c * d[1] - n1[1], 2 * c * d[2] - n1[2]];
  const axis = cross(n1, n2), sa = norm(axis);
  if (sa > 1e-6) {
    const ang = Math.atan2(sa, dot(n1, n2));
    const Rm = expSO3(axis.map(x => x / sa * ang));
    out.push(toObj(mat3mul(Rm, Rp), tp));
  }
  return out;
}

export function reprojErrors(R, t, K, obj, img) {
  return obj.map((X, i) => { const p = project(R, t, K, X); return Math.hypot(p[0] - img[i][0], p[1] - img[i][1]); });
}

// ---- quaternion helpers + One Euro filter for smoothing ----
export function matToQuat(m) {
  const tr = m[0] + m[4] + m[8];
  let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s; }
  else if (m[0] > m[4] && m[0] > m[8]) { const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2; w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s; }
  else if (m[4] > m[8]) { const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2; w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s; }
  else { const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2; w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s; }
  return [x, y, z, w];
}
export function quatToMat(q) {
  const [x, y, z, w] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
          2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
          2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}

class LowPass { constructor() { this.y = null; } f(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; } }
const alpha = (cutoff, dt) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };

export class OneEuro {   // vector One Euro filter (Casiez et al. 2012)
  constructor(n, minCutoff, beta, dCutoff = 1.0) { this.n = n; this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff; this.reset(); }
  reset() { this.x = Array.from({ length: this.n }, () => new LowPass()); this.dx = Array.from({ length: this.n }, () => new LowPass()); this.prev = null; this.tPrev = null; }
  filter(v, t) {
    if (this.prev === null) { this.prev = v.slice(); this.tPrev = t; v.forEach((x, i) => { this.x[i].f(x, 1); this.dx[i].f(0, 1); }); return v.slice(); }
    const dt = Math.max(1e-3, t - this.tPrev); this.tPrev = t;
    const ad = alpha(this.dCutoff, dt);
    let speed = 0;
    const d = v.map((x, i) => { const dd = this.dx[i].f((x - this.prev[i]) / dt, ad); speed += dd * dd; return dd; });
    speed = Math.sqrt(speed);
    const a = alpha(this.minCutoff + this.beta * speed, dt);
    const out = v.map((x, i) => this.x[i].f(x, a));
    this.prev = v.slice();
    return out;
  }
}
