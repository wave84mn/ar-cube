"""Printable AR cube: rounded white body with flush black marker inlays.

Builds two closed, non-overlapping solids that touch along shared faces:
  body    - rounded cube with a DEPTH-deep recess under every black cell
  markers - the black inlays that fill those recesses
The mesh is built patch by patch (6 flat faces on a shared grid, 12 quarter-cylinder edges,
8 sphere-octant corners) with identical boundary points, then welded.
Outputs (print frame: Z up, top face up, bottom on the bed at z = 0, front face towards -Y):
  out/AR_cube_70mm_body_white.stl, out/AR_cube_70mm_markers_black.stl, out/AR_cube_70mm_2colour.3mf
"""
import json, os, struct, zipfile, sys
import numpy as np
from scipy.spatial import cKDTree

lay = json.load(open("build/layout.json"))
S, R, DEPTH, CELL = lay["edge"], lay["fillet"], lay["inlay_depth"], lay["cell"]
A = S / 2 - R           # half size of the inner (sharp) cube
M = 8                   # segments per quarter arc
H = S / 2

EPS = 0.15   # half-size of the black bridge where two black cells touch only at a corner


def diagonal_contacts(fi):
    """(u, v) points where black cells meet only diagonally (would be non-manifold)."""
    pts = []
    for m in lay["markers"]:
        if m["face"] != fi:
            continue
        c = np.array(m["cells"]); cu, cv = m["center_uv"]
        for i in range(7):
            for j in range(7):
                b = c[i:i + 2, j:j + 2]
                if b[0, 0] == b[1, 1] and b[0, 1] == b[1, 0] and b[0, 0] != b[0, 1]:
                    pts.append((cu - 4 * CELL + (j + 1) * CELL, cv + 4 * CELL - (i + 1) * CELL))
    return pts


# grid breakpoints shared by every face and every edge strip
G = {-A, A}
for m in lay["markers"]:
    cu, cv = m["center_uv"]
    for j in range(9):
        G.add(round(cu - 4 * CELL + j * CELL, 9)); G.add(round(cv - 4 * CELL + j * CELL, 9))
for fi in range(6):
    for u, v in diagonal_contacts(fi):
        for x in (u, v):
            G.add(round(x - EPS, 9)); G.add(round(x + EPS, 9))
G = np.array(sorted(G))
assert G[0] == -A and G[-1] == A


def face_black_grid(fi):
    """black[iu, iv] for grid cell (G[iu],G[iu+1]) x (G[iv],G[iv+1])."""
    n = len(G) - 1
    blk = np.zeros((n, n), bool)
    uc = (G[:-1] + G[1:]) / 2
    for m in lay["markers"]:
        if m["face"] != fi:
            continue
        cu, cv = m["center_uv"]
        for i in range(8):
            for j in range(8):
                if not m["cells"][i][j]:
                    continue
                u0, u1 = cu - 4 * CELL + j * CELL, cu - 4 * CELL + (j + 1) * CELL
                v1 = cv + 4 * CELL - i * CELL; v0 = v1 - CELL
                iu = np.where((uc > u0) & (uc < u1))[0]
                iv = np.where((uc > v0) & (uc < v1))[0]
                blk[np.ix_(iu, iv)] = True
    for u, v in diagonal_contacts(fi):   # bridge: small black square centred on the contact
        iu = np.where((uc > u - EPS) & (uc < u + EPS))[0]
        iv = np.where((uc > v - EPS) & (uc < v + EPS))[0]
        blk[np.ix_(iu, iv)] = True
    return blk


class Builder:
    def __init__(self):
        self.V = []; self.T = []
    def tri(self, p0, p1, p2, want_n):
        p0, p1, p2 = map(np.asarray, (p0, p1, p2))
        n = np.cross(p1 - p0, p2 - p0)
        if np.dot(n, want_n) < 0:
            p1, p2 = p2, p1
        b = len(self.V); self.V += [p0, p1, p2]; self.T.append([b, b + 1, b + 2])
    def quad(self, p0, p1, p2, p3, want_n):  # p0..p3 around the quad
        self.tri(p0, p1, p2, want_n); self.tri(p0, p2, p3, want_n)
    def mesh(self, tol=1e-6):
        if not self.V:
            return np.zeros((0, 3)), np.zeros((0, 3), int)
        V = np.array(self.V, float); T = np.array(self.T)
        tree = cKDTree(V)
        pairs = tree.query_pairs(tol, output_type="ndarray")
        parent = np.arange(len(V))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i, j in pairs:
            ri, rj = find(i), find(j)
            if ri != rj: parent[max(ri, rj)] = min(ri, rj)
        root = np.array([find(i) for i in range(len(V))])
        uniq, inv = np.unique(root, return_inverse=True)
        T2 = inv[T]
        keep = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
        return V[uniq], T2[keep]


def face_point(f, u, v, w):
    return u * np.array(f["U"]) + v * np.array(f["V"]) + w * np.array(f["N"])


def add_rounded_shell(b):
    """12 edge strips + 8 corners (the faces are added separately)."""
    axes = [np.array(v, float) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    # edges
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            for si in (1, -1):
                for sj in (1, -1):
                    nA, nB, e = si * axes[i], sj * axes[j], axes[k]
                    def P(t, kk):
                        d = (M - kk) * nA + kk * nB; d = d / np.linalg.norm(d)
                        return A * nA + A * nB + t * e + R * d
                    for ti in range(len(G) - 1):
                        for kk in range(M):
                            c = P((G[ti] + G[ti + 1]) / 2, kk + 0.5)
                            want = c - ((G[ti] + G[ti + 1]) / 2 * e)
                            b.quad(P(G[ti], kk), P(G[ti + 1], kk), P(G[ti + 1], kk + 1), P(G[ti], kk + 1), want)
    # corners
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                n1, n2, n3 = sx * axes[0], sy * axes[1], sz * axes[2]
                ctr = A * (n1 + n2 + n3)
                def Q(i, j):
                    d = i * n1 + j * n2 + (M - i - j) * n3; d = d / np.linalg.norm(d)
                    return ctr + R * d
                for i in range(M):
                    for j in range(M - i):
                        p0, p1, p2 = Q(i, j), Q(i + 1, j), Q(i, j + 1)
                        b.tri(p0, p1, p2, (p0 + p1 + p2) / 3 - ctr)
                        if i + j + 1 < M:
                            p3 = Q(i + 1, j + 1)
                            b.tri(p1, p3, p2, (p1 + p2 + p3) / 3 - ctr)


def build(with_markers=True):
    body, ink = Builder(), Builder()
    add_rounded_shell(body)
    n = len(G) - 1
    for fi, f in enumerate(lay["faces"]):
        N = np.array(f["N"], float)
        blk = face_black_grid(fi) if with_markers else np.zeros((n, n), bool)
        for iu in range(n):
            for iv in range(n):
                u0, u1, v0, v1 = G[iu], G[iu + 1], G[iv], G[iv + 1]
                if blk[iu, iv]:
                    w = H - DEPTH
                    body.quad(face_point(f, u0, v0, w), face_point(f, u1, v0, w), face_point(f, u1, v1, w), face_point(f, u0, v1, w), N)
                    ink.quad(face_point(f, u0, v0, w), face_point(f, u1, v0, w), face_point(f, u1, v1, w), face_point(f, u0, v1, w), -N)
                    ink.quad(face_point(f, u0, v0, H), face_point(f, u1, v0, H), face_point(f, u1, v1, H), face_point(f, u0, v1, H), N)
                else:
                    body.quad(face_point(f, u0, v0, H), face_point(f, u1, v0, H), face_point(f, u1, v1, H), face_point(f, u0, v1, H), N)
        # walls between black and white neighbours
        U, V = np.array(f["U"], float), np.array(f["V"], float)
        for iu in range(n):
            for iv in range(n):
                for du, dv in ((1, 0), (0, 1)):
                    ju, jv = iu + du, iv + dv
                    if ju >= n or jv >= n or blk[iu, iv] == blk[ju, jv]:
                        continue
                    if du:   # shared edge at u = G[ju], v in [G[iv], G[iv+1]]
                        e0, e1 = (G[ju], G[iv]), (G[ju], G[iv + 1]); toward_j = U
                    else:
                        e0, e1 = (G[iu], G[jv]), (G[iu + 1], G[jv]); toward_j = V
                    to_black = toward_j if blk[ju, jv] else -toward_j
                    q = [face_point(f, e0[0], e0[1], H - DEPTH), face_point(f, e1[0], e1[1], H - DEPTH),
                         face_point(f, e1[0], e1[1], H), face_point(f, e0[0], e0[1], H)]
                    body.quad(*q, to_black)
                    ink.quad(*q, -to_black)
    return body.mesh(), ink.mesh()


def check(name, V, T):
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    und = np.sort(e, 1)
    u, cnt = np.unique(und, axis=0, return_counts=True)
    bad = (cnt != 2).sum()
    # orientation: each directed edge exactly once
    d, dc = np.unique(e, axis=0, return_counts=True)
    dup = (dc != 1).sum()
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / 2
    vol = np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6
    # connected shells
    parent = np.arange(len(V))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a_, b_ in u:
        ra, rb = find(a_), find(b_)
        if ra != rb: parent[ra] = rb
    shells = len({find(i) for i in np.unique(T)})
    chi = len(V) - len(u) + len(T)
    print(f"{name:8s} verts {len(V):6d} tris {len(T):6d} edges!=2 {bad} dup-directed {dup} "
          f"min-area {area.min():.4f} mm2 volume {vol:.1f} mm3 shells {shells} euler {chi}")
    return dict(bad=int(bad), dup=int(dup), minarea=float(area.min()), volume=float(vol), shells=shells, euler=chi)


ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # cube frame -> print frame (Z up, front -> -Y)


def to_print(V):
    P = V @ ROT.T
    P[:, 2] += H
    return P


def write_stl(path, V, T):
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(p1 - p0, p2 - p0); n /= np.linalg.norm(n, axis=1)[:, None]
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n; rec["v"] = np.stack([p0, p1, p2], 1)
    with open(path, "wb") as fh:
        fh.write(b"AR cube - " + os.path.basename(path).encode()[:70].ljust(70, b" "))
        fh.write(struct.pack("<I", len(T))); fh.write(rec.tobytes())


def mesh_xml(V, T):
    vs = "\n".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in V)
    ts = "\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in T)
    return f"<mesh><vertices>\n{vs}\n</vertices><triangles>\n{ts}\n</triangles></mesh>"


def write_3mf(path, parts):
    objs = []
    for i, (name, V, T, colour) in enumerate(parts):
        objs.append(f'<object id="{i + 2}" type="model" name="{name}" pid="1" pindex="{i}">{mesh_xml(V, T)}</object>')
    comps = "".join(f'<component objectid="{i + 2}"/>' for i in range(len(parts)))
    bases = "".join(f'<base name="{n}" displaycolor="{c}"/>' for n, _, _, c in parts)
    nid = len(parts) + 2
    model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<metadata name="Title">AR cube 70 mm (2 colour)</metadata>
<metadata name="Designer">MT project</metadata>
<metadata name="Description">White body + black ArUco MIP_36h12 inlays. Assign filament 1 to the body and filament 2 to the markers.</metadata>
<resources>
<basematerials id="1">{bases}</basematerials>
{chr(10).join(objs)}
<object id="{nid}" type="model" name="AR cube 70mm">
<components>{comps}</components>
</object>
</resources>
<build><item objectid="{nid}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/></build>
</model>'''
    ct = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    (Vb, Tb), (Vi, Ti) = build(True)
    (Vp, Tp), _ = build(False)
    rb = check("body", Vb, Tb); ri = check("markers", Vi, Ti); rp = check("plain", Vp, Tp)
    s = S - 2 * R
    exact = s ** 3 + 6 * s * s * R + 3 * np.pi * s * R * R + 4 / 3 * np.pi * R ** 3
    print(f"plain cube mesh volume {rp['volume']:.1f} vs exact rounded cube {exact:.1f} ({(rp['volume'] / exact - 1) * 100:.3f} %)")
    print(f"body + markers = {rb['volume'] + ri['volume']:.3f} vs plain {rp['volume']:.3f} (difference {rb['volume'] + ri['volume'] - rp['volume']:.6f} mm3)")
    blk_area = sum(sum(sum(r) for r in m["cells"]) for m in lay["markers"]) * CELL * CELL
    nbridge = sum(len(diagonal_contacts(fi)) for fi in range(6))
    print(f"markers volume {ri['volume']:.2f} vs black area x depth {blk_area * DEPTH:.2f} "
          f"+ {nbridge} bridges ({nbridge * 2 * EPS * EPS * DEPTH:.2f} mm3 max)")
    ok = all(r["bad"] == 0 and r["dup"] == 0 and r["shells"] == 1 and r["euler"] == 2 and r["volume"] > 0 for r in (rb, rp)) \
        and ri["bad"] == 0 and ri["dup"] == 0 and ri["volume"] > 0
    print("CHECK", "OK" if ok else "FAILED")
    Pb, Pi = to_print(Vb), to_print(Vi)
    print("print bbox", Pb.min(0).round(4), Pb.max(0).round(4))
    np.savez("build/cube_mesh.npz", Vb=Vb, Tb=Tb, Vi=Vi, Ti=Ti)
    write_stl("out/AR_cube_70mm_body_white.stl", Pb, Tb)
    write_stl("out/AR_cube_70mm_markers_black.stl", Pi, Ti)
    write_3mf("out/AR_cube_70mm_2colour.3mf", [("body (white)", Pb, Tb, "#F4F4F2FF"), ("markers (black)", Pi, Ti, "#1C1C1CFF")])
    for f in sorted(os.listdir("out")):
        print(f, os.path.getsize("out/" + f) // 1024, "KB")
