"""Stand-in 'Kula' massing model (GLB) for testing the viewer before the real model arrives.
Metres, +Y up, front (door) towards +Z. Vertex colours, one material.
"""
import json, struct, math, sys
import numpy as np

rng = np.random.default_rng(3)
V, N, C, I = [], [], [], []


def srgb2lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def quad(p0, p1, p2, p3, col):
    p = [np.asarray(x, float) for x in (p0, p1, p2, p3)]
    n = np.cross(p[1] - p[0], p[3] - p[0]); n /= np.linalg.norm(n)
    b = len(V)
    for q in p: V.append(q); N.append(n); C.append(col)
    I.extend([b, b + 1, b + 2, b, b + 2, b + 3])


def tri(p0, p1, p2, col):
    p = [np.asarray(x, float) for x in (p0, p1, p2)]
    n = np.cross(p[1] - p[0], p[2] - p[0]); n /= np.linalg.norm(n)
    b = len(V)
    for q in p: V.append(q); N.append(n); C.append(col)
    I.extend([b, b + 1, b + 2])


def box(lo, hi, col, faces="xXyYzZ"):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    if "x" in faces: quad((x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0), col)
    if "X" in faces: quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), col)
    if "y" in faces: quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), col)
    if "Y" in faces: quad((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), col)
    if "z" in faces: quad((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), col)
    if "Z" in faces: quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), col)


W, HW = 12.0, 12.3          # wall width, wall height
h = W / 2
MORTAR = srgb2lin((120, 112, 100))
DARK = srgb2lin((28, 26, 24))
STONE = np.array((176, 166, 148))

# openings per wall in (u from wall centre, y) rectangles: (u0, u1, y0, y1)
front_w = [(-4.2, -3.4), (-1.6, -0.8), (0.8, 1.6), (3.4, 4.2)]
openings = {
    "front": [(a, b, 5.0, 6.2) for a, b in front_w] + [(a, b, 8.6, 9.8) for a, b in front_w] + [(-0.75, 0.75, 2.2, 4.4)],
    "back": [(-0.6, 0.6, 9.2, 10.4)],
    "left": [(-2.5, -1.7, 5.2, 6.4), (1.7, 2.5, 5.2, 6.4), (-0.4, 0.4, 9.0, 10.0)],
    "right": [(-2.5, -1.7, 5.2, 6.4), (1.7, 2.5, 5.2, 6.4), (-0.4, 0.4, 9.0, 10.0)],
}
walls = {  # origin, u-axis, outward normal
    "front": (np.array((0, 0, h)), np.array((1, 0, 0)), np.array((0, 0, 1))),
    "back": (np.array((0, 0, -h)), np.array((-1, 0, 0)), np.array((0, 0, -1))),
    "left": (np.array((-h, 0, 0)), np.array((0, 0, 1)), np.array((-1, 0, 0))),
    "right": (np.array((h, 0, 0)), np.array((0, 0, -1)), np.array((1, 0, 0))),
}


def in_opening(name, u0, u1, y0, y1, pad=0.08):
    for a, b, c, d in openings[name]:
        if u1 > a - pad and u0 < b + pad and y1 > c - pad and y0 < d + pad:
            return True
    return False


for name, (O, U, Nn) in walls.items():
    Y = np.array((0, 1, 0))
    P = lambda u, y, d=0.0: O + u * U + y * Y + d * Nn
    # mortar plane
    quad(P(-h, 0), P(h, 0), P(h, HW), P(-h, HW), MORTAR) if False else None
    # mortar background split around openings: simple full quad (openings get dark boxes in front)
    quad(P(-h, 0), P(h, 0), P(h, HW), P(-h, HW), MORTAR)
    # stones: courses with random lengths; quoins at the corners
    y = 0.0
    row = 0
    while y < HW - 0.05:
        ch = rng.uniform(0.34, 0.5) if y > 1.0 else rng.uniform(0.5, 0.75)   # boulder plinth
        y1 = min(HW, y + ch)
        u = -h
        quoin = 1.0 if row % 2 == 0 else 0.6
        first = True
        while u < h - 0.02:
            L = quoin if first else rng.uniform(0.45, 1.1) * (1.3 if y < 1.0 else 1.0)
            first = False
            u1 = min(h, u + L)
            if h - u1 < 0.35: u1 = h
            g = 0.035
            if not in_opening(name, u + g, u1 - g, y + g, y1 - g):
                tint = STONE * rng.uniform(0.82, 1.08) + rng.uniform(-8, 8, 3)
                d = rng.uniform(0.04, 0.09)
                a0, a1, b0, b1 = u + g, u1 - g, y + g, y1 - g
                corner_l, corner_r = (u <= -h + 1e-6), (u1 >= h - 1e-6)
                if corner_l: a0 = u - d
                if corner_r: a1 = u1 + d
                col = srgb2lin(np.clip(tint, 0, 255))
                # stone block: front face + 4 sides
                quad(P(a0, b0, d), P(a1, b0, d), P(a1, b1, d), P(a0, b1, d), col)
                quad(P(a0, b1, d), P(a1, b1, d), P(a1, b1, 0), P(a0, b1, 0), col * 0.95)
                quad(P(a0, b0, 0), P(a1, b0, 0), P(a1, b0, d), P(a0, b0, d), col * 0.8)
                quad(P(a0, b0, 0), P(a0, b0, d), P(a0, b1, d), P(a0, b1, 0), col * 0.85)
                quad(P(a1, b0, d), P(a1, b0, 0), P(a1, b1, 0), P(a1, b1, d), col * 0.85)
            u = u1
        y = y1; row += 1
    # openings: dark recess + light stone frame + sill
    for a, b, c, d_ in openings[name]:
        quad(P(a, c, 0.02), P(b, c, 0.02), P(b, d_, 0.02), P(a, d_, 0.02), DARK)
        fr = srgb2lin((205, 196, 178)); t = 0.12; dd = 0.11
        for (u0, u1, y0, y1) in ((a - t, b + t, d_, d_ + t), (a - t, a, c, d_), (b, b + t, c, d_)):
            quad(P(u0, y0, dd), P(u1, y0, dd), P(u1, y1, dd), P(u0, y1, dd), fr)
            quad(P(u0, y1, dd), P(u1, y1, dd), P(u1, y1, 0), P(u0, y1, 0), fr * 0.9)
            quad(P(u0, y0, 0), P(u0, y0, dd), P(u0, y1, dd), P(u0, y1, 0), fr * 0.85)
            quad(P(u1, y0, dd), P(u1, y0, 0), P(u1, y1, 0), P(u1, y1, dd), fr * 0.85)
        sill = srgb2lin((190, 182, 165))
        quad(P(a - 0.15, c - 0.12, 0.2), P(b + 0.15, c - 0.12, 0.2), P(b + 0.15, c, 0.2), P(a - 0.15, c, 0.2), sill)
        quad(P(a - 0.15, c, 0.2), P(b + 0.15, c, 0.2), P(b + 0.15, c, 0), P(a - 0.15, c, 0), sill * 0.95)
        quad(P(a - 0.15, c - 0.12, 0), P(b + 0.15, c - 0.12, 0), P(b + 0.15, c - 0.12, 0.2), P(a - 0.15, c - 0.12, 0.2), sill * 0.7)

# door leaf (timber) slightly inside the dark recess
wood = srgb2lin((96, 62, 38))
quad((-0.7, 2.2, h + 0.03), (0.7, 2.2, h + 0.03), (0.7, 4.35, h + 0.03), (-0.7, 4.35, h + 0.03), wood)

# front stairs up to the door threshold (2.2 m)
steps = 9; sw = 3.2; run = 0.32; rise = 2.2 / steps
step_col = srgb2lin((168, 160, 146))
for s in range(steps):
    z0 = h + (steps - s - 1) * run; z1 = z0 + run
    y1 = (s + 1) * rise
    box((-sw / 2, 0, h), (sw / 2, y1, z1), step_col * rng.uniform(0.9, 1.05), faces="xXYZ")

# roof: hip (pyramid) with eaves
E, pitch = 0.9, math.radians(38)
r0 = h + E
apex = HW + r0 * math.tan(pitch) - E * math.tan(pitch)
eave_y = HW - E * math.tan(pitch)
roof_col = srgb2lin((128, 82, 62))
corners = [(-r0, eave_y, r0), (r0, eave_y, r0), (r0, eave_y, -r0), (-r0, eave_y, -r0)]
top = (0, apex, 0)
# tile courses as stripes of slightly varied colour
for k in range(4):
    a = np.array(corners[k]); b = np.array(corners[(k + 1) % 4]); tp = np.array(top)
    nstrip = 12
    for s in range(nstrip):
        f0, f1 = s / nstrip, (s + 1) / nstrip
        p0 = a + (tp - a) * f0; p1 = b + (tp - b) * f0; p2 = b + (tp - b) * f1; p3 = a + (tp - a) * f1
        col = roof_col * rng.uniform(0.85, 1.1)
        if s == nstrip - 1: tri(p0, p1, tp, col)
        else: quad(p0, p1, p2, p3, col)
    # fascia
    fy = eave_y - 0.22
    quad((a[0], fy, a[2]), (b[0], fy, b[2]), (b[0], eave_y, b[2]), (a[0], eave_y, a[2]), roof_col * 0.6)
# soffit (underside of the eaves)
soff = srgb2lin((80, 56, 40))
fy = eave_y - 0.22
quad((-r0, fy, -r0), (r0, fy, -r0), (r0, fy, r0), (-r0, fy, r0), soff)
# hip ridges
ridge = srgb2lin((100, 62, 46))
for k in range(4):
    a = np.array(corners[k]); tp = np.array(top)
    d = tp - a; side = np.cross(d, (0, 1, 0)); side = side / np.linalg.norm(side) * 0.12
    quad(a - side + (0, 0.06, 0), a + side + (0, 0.06, 0), tp + side * 0.3 + (0, 0.08, 0), tp - side * 0.3 + (0, 0.08, 0), ridge)

V_ = np.array(V, np.float32); N_ = np.array(N, np.float32); C_ = np.clip(np.array(C, np.float32), 0, 1); I_ = np.array(I, np.uint32)
print("stand-in kula: verts", len(V_), "tris", len(I_) // 3, "bbox", V_.min(0).round(2), V_.max(0).round(2))


def write_glb(path, pos, nrm, col, idx, name):
    col4 = np.concatenate([col, np.ones((len(col), 1), np.float32)], 1)
    bufs = [pos.tobytes(), nrm.tobytes(), col4.astype(np.float32).tobytes(), idx.tobytes()]
    offs, blob = [], b""
    for b in bufs:
        offs.append(len(blob)); blob += b + b"\0" * ((4 - len(b) % 4) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "MT ar-cube stand-in"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"name": "stone", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.92}}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": offs[0], "byteLength": len(bufs[0]), "target": 34962},
                        {"buffer": 0, "byteOffset": offs[1], "byteLength": len(bufs[1]), "target": 34962},
                        {"buffer": 0, "byteOffset": offs[2], "byteLength": len(bufs[2]), "target": 34962},
                        {"buffer": 0, "byteOffset": offs[3], "byteLength": len(bufs[3]), "target": 34963}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(col4), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(blob)


write_glb(sys.argv[1] if len(sys.argv) > 1 else "site/models/kula_standin.glb", V_, N_, C_, I_, "Kula stand-in")
