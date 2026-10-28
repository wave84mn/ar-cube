"""Synthetic camera frames of the AR cube / card with ground-truth poses.
usage: python3 tests/synth.py OUTDIR N [--card] [--seed S] [--size 1280x720] [--video]
Writes OUTDIR/NNNN.png (+ gt.json). Pose convention: X_cam = R X_obj + t (OpenCV camera, mm).
With --video, poses follow a smooth hand-held path instead of independent random draws.
"""
import json, math, os, sys
import numpy as np, cv2

lay = json.load(open("build/layout.json"))
S = lay["edge"]
FACE_PPMM = 20
faces_tex = [cv2.imread(f"build/face_{i}.png", 0) for i in range(6)]


def card_texture(ppmm=8):
    c = lay["card"]; size = c["size"] + 30; n = int(size * ppmm)
    img = np.full((n, n), 255, np.uint8)
    for m in c["markers"]:
        cu, cv_ = m["center_uv"]
        for i in range(8):
            for j in range(8):
                if m["cells"][i][j]:
                    u0 = cu - 4 * c["cell"] + j * c["cell"]; v1 = cv_ + 4 * c["cell"] - i * c["cell"]
                    x0 = int(round((u0 + size / 2) * ppmm)); y0 = int(round((size / 2 - v1) * ppmm))
                    img[y0:y0 + int(round(c["cell"] * ppmm)), x0:x0 + int(round(c["cell"] * ppmm))] = 0
    return img, size, ppmm


def rot_from_euler(rx, ry, rz):
    Rx = cv2.Rodrigues(np.array([rx, 0, 0.]))[0]; Ry = cv2.Rodrigues(np.array([0, ry, 0.]))[0]; Rz = cv2.Rodrigues(np.array([0, 0, rz]))[0]
    return Rz @ Rx @ Ry


def background(rng, W, H):
    base = rng.uniform(40, 200, 3)
    img = np.ones((H, W, 3), np.float32) * base
    # large blurred noise (table / wall texture)
    n = cv2.resize(rng.normal(0, 1, (H // 16 + 1, W // 16 + 1, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img += n * rng.uniform(5, 30)
    # vertical gradient
    img *= np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.8, 1.2), H)[:, None, None]
    # clutter: rectangles, lines, text-like blobs (some dark, high contrast)
    for _ in range(rng.integers(5, 25)):
        col = rng.uniform(0, 255, 3).tolist()
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        k = rng.integers(0, 3)
        if k == 0: cv2.rectangle(img, (x, y), (x + int(rng.uniform(10, 300)), y + int(rng.uniform(10, 300))), col, -1)
        elif k == 1: cv2.line(img, (x, y), (int(rng.uniform(0, W)), int(rng.uniform(0, H))), col, int(rng.uniform(1, 8)))
        else:
            t8 = np.zeros((H, W), np.uint8)
            cv2.putText(t8, "".join(chr(int(c)) for c in rng.integers(65, 90, 6)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.5, 2.5), 255, int(rng.uniform(1, 4)))
            m = (t8 > 0)
            img[m] = col
    return img


def render_planes(img, planes, R, t, K, rng, light):
    """planes: list of (texture, U, V, N, O, half_size_mm, ppmm) -> composite visible ones."""
    H, W = img.shape[:2]
    for tex, U, V, N, O, half, ppmm in planes:
        n_cam = R @ N; c_cam = R @ O + t
        if n_cam @ c_cam >= 0: continue
        corners = [O - half * U + half * V, O + half * U + half * V, O + half * U - half * V, O - half * U - half * V]
        pts = []
        for X in corners:
            p = K @ (R @ X + t); pts.append(p[:2] / p[2])
        n_tex = tex.shape[0]
        src = np.float32([[0, 0], [n_tex, 0], [n_tex, n_tex], [0, n_tex]])
        Hm = cv2.getPerspectiveTransform(src, np.float32(pts))
        shade = 0.35 + 0.65 * max(0.0, -(n_cam @ light))
        black, white = rng.uniform(15, 45), rng.uniform(200, 245)
        albedo = (black + (white - black) * (tex.astype(np.float32) / 255.0))
        # lighting gradient across the face
        g = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.85, 1.05), n_tex, dtype=np.float32)
        albedo = albedo * (g[None, :] if rng.random() < 0.5 else g[:, None])
        warped = cv2.warpPerspective(albedo * shade, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(albedo), Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        tint = np.array([1.0, rng.uniform(0.97, 1.03), rng.uniform(0.92, 1.05)], np.float32)
        img[:] = img * (1 - mask[..., None]) + warped[..., None] * tint * mask[..., None]


def occlude(img, rng, center, scale):
    """skin-coloured 'thumb' ellipse near the target centre."""
    col = np.array([rng.uniform(120, 200), rng.uniform(140, 190), rng.uniform(180, 240)])  # BGR skin
    cx = int(center[0] + rng.uniform(-0.6, 0.6) * scale); cy = int(center[1] + rng.uniform(-0.6, 0.6) * scale)
    ax = (int(scale * rng.uniform(0.12, 0.25)), int(scale * rng.uniform(0.25, 0.5)))
    cv2.ellipse(img, (cx, cy), ax, rng.uniform(0, 180), 0, 360, col.tolist(), -1, cv2.LINE_AA)


def finish(img, rng):
    if rng.random() < 0.6:
        s = rng.uniform(0.3, 1.4); img = cv2.GaussianBlur(img, (0, 0), s)
    if rng.random() < 0.3:   # motion blur
        L = int(rng.uniform(3, 11)); k = np.zeros((L, L), np.float32); k[L // 2, :] = 1 / L
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1); k = cv2.warpAffine(k, M, (L, L)); k /= k.sum()
        img = cv2.filter2D(img, -1, k)
    img = img + rng.normal(0, rng.uniform(1.5, 6), img.shape)
    gain = rng.uniform(0.7, 1.15); img = np.clip(img * gain, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(60, 92))])
    return cv2.imdecode(enc, 1)


def main():
    out = sys.argv[1]; N = int(sys.argv[2])
    card = "--card" in sys.argv; video = "--video" in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    W, H = (int(x) for x in (sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "1280x720").split("x"))
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    f = 0.78 * max(W, H)                         # ~65 deg along the long side
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    if card:
        tex, size, ppmm = card_texture()
        planes = [(tex, np.array([1., 0, 0]), np.array([0, 0, -1.]), np.array([0, 1., 0]), np.zeros(3), size / 2, ppmm)]
    else:
        planes = [(faces_tex[i], np.array(fc["U"], float), np.array(fc["V"], float), np.array(fc["N"], float),
                   np.array(fc["N"], float) * S / 2, S / 2, FACE_PPMM) for i, fc in enumerate(lay["faces"])]
    gts = []
    bg = background(rng, W, H) if video else None
    for i in range(N):
        if video:
            tt = 0.0 if '--static' in sys.argv else i / 30.0
            z = 330 + 110 * math.sin(tt * 0.7)
            R = rot_from_euler(0.5 + 0.35 * math.sin(tt * 0.9), tt * 0.8, 0.15 * math.sin(tt * 1.3))
            if card: R = rot_from_euler(-0.9 + 0.25 * math.sin(tt * 0.6), tt * 0.5, 0.1 * math.sin(tt))
            cx = 0.15 * z * math.sin(tt * 1.1); cy = 0.1 * z * math.cos(tt * 0.8)
            t = np.array([cx, cy, z if not card else z * 1.8])
            light = np.array([0.3, 0.8, 0.5]); light /= np.linalg.norm(light)
            img = bg.copy()
        else:
            if card:
                z = rng.uniform(250, 700); R = rot_from_euler(rng.uniform(-1.4, -0.35), rng.uniform(-math.pi, math.pi), rng.uniform(-0.3, 0.3))
            else:
                z = rng.uniform(170, 600); R = cv2.Rodrigues(rng.normal(0, 1, 3) * rng.uniform(0, math.pi) / 1.0)[0]
            ex, ey = rng.uniform(-0.3, 0.3) * W, rng.uniform(-0.3, 0.3) * H
            t = np.array([ex * z / f, ey * z / f, z])
            light = rng.normal(0, 1, 3); light[2] = -abs(light[2]) - 0.3; light /= np.linalg.norm(light)
            img = background(rng, W, H)
        render_planes(img, planes, R, t, K, rng, -light)
        pc = K @ t; ctr = pc[:2] / pc[2]
        scale_px = f * (S if not card else lay["card"]["size"]) / t[2]
        occ = (not video) and rng.random() < 0.35
        if occ: occlude(img, rng, ctr, scale_px)
        img = finish(img, rng) if not video else finish_video(img, rng)
        cv2.imwrite(os.path.join(out, f"{i:04d}.png"), img)
        gts.append({"file": f"{i:04d}.png", "R": R.tolist(), "t": t.tolist(), "occluded": bool(occ)})
    json.dump({"K": {"fx": f, "fy": f, "cx": W / 2, "cy": H / 2}, "W": W, "H": H, "target": "card" if card else "cube", "frames": gts},
              open(os.path.join(out, "gt.json"), "w"))
    print("wrote", N, "frames to", out)


def finish_video(img, rng):
    img = cv2.GaussianBlur(img, (0, 0), 0.6) + rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
