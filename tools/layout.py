"""Single source of truth for the AR cube marker layout.

Cube frame (same as three.js): +X right, +Y up, +Z front. Units: mm. Origin = cube centre.
Each face carries a 2x2 grid of ArUco MIP_36h12 markers (8x8 cells incl. black border).
Writes build/layout.json (used by the print geometry, the paper PDF and the web viewer)
and build/face_<n>.png images (20 px/mm) for rendering tests.
"""
import json, sys
import numpy as np

S = 70.0          # cube edge
R_FILLET = 2.5    # rounded edge radius (print only)
CELL = 3.5        # marker cell size -> marker = 8 cells = 28 mm
GAP = 4.0         # white gap between the 4 markers of a face
DEPTH = 1.0       # inlay depth (print only)

FACES = [  # name, U (right), V (up), N (out) -- looking at the face from outside
    ("top",    (1, 0, 0),  (0, 0, -1), (0, 1, 0)),
    ("front",  (1, 0, 0),  (0, 1, 0),  (0, 0, 1)),
    ("right",  (0, 0, -1), (0, 1, 0),  (1, 0, 0)),
    ("back",   (-1, 0, 0), (0, 1, 0),  (0, 0, -1)),
    ("left",   (0, 0, 1),  (0, 1, 0),  (-1, 0, 0)),
    ("bottom", (1, 0, 0),  (0, 0, 1),  (0, -1, 0)),
]
MSIZE = 8 * CELL
OFF = GAP / 2 + MSIZE / 2           # marker centre offset from face centre
CARD_W, CARD_CELL = 170.0, 5.0      # flat table card: 170 mm square, 40 mm markers at the corners
CARD_MSIZE = 8 * CARD_CELL
CARD_OFF = CARD_W / 2 - CARD_MSIZE / 2 - 3.0
SLOTS = [(-OFF, OFF), (OFF, OFF), (-OFF, -OFF), (OFF, -OFF)]  # TL, TR, BL, BR on the face


def load_codes():
    js = json.load(open("build/mip36h12_js.json"))
    ids = json.load(open("build/picked_ids.json"))["ids"]
    return js["codes"], ids


def marker_cells(code):
    """8x8 array, 1 = black. Row 0 = top of the marker (towards +V)."""
    g = np.ones((8, 8), int)
    bits = np.array([int(c) for c in code]).reshape(6, 6)  # 1 = white
    g[1:7, 1:7] = 1 - bits
    return g


def build():
    codes, ids = load_codes()
    for f in FACES:
        U, V, N = map(np.array, f[1:])
        assert np.allclose(np.cross(U, V), N), f[0]
    markers = []
    for fi, (name, U, V, N) in enumerate(FACES):
        U, V, N = map(np.array, (U, V, N))
        fc = N * S / 2
        for si, (cu, cv) in enumerate(SLOTS):
            mid = ids[4 * fi + si]
            c = fc + cu * U + cv * V
            h = MSIZE / 2
            corners = [c - h * U + h * V, c + h * U + h * V, c + h * U - h * V, c - h * U - h * V]  # TL TR BR BL
            markers.append({
                "id": mid, "face": fi, "slot": si, "center_uv": [cu, cv],
                "corners": [[round(float(x), 6) for x in p] for p in corners],
                "cells": marker_cells(codes[mid]).tolist(),
            })
    # flat table card: lies on the table (normal +Y), page-up = -Z, page-right = +X, plane Y = 0
    card_ids = json.load(open("build/picked_ids.json"))["card_ids"]
    U, V = np.array((1, 0, 0)), np.array((0, 0, -1))
    card_markers = []
    for si, (sx, sy) in enumerate([(-1, 1), (1, 1), (-1, -1), (1, -1)]):
        cu, cv = sx * CARD_OFF, sy * CARD_OFF
        c = cu * U + cv * V
        h = CARD_MSIZE / 2
        corners = [c - h * U + h * V, c + h * U + h * V, c + h * U - h * V, c - h * U - h * V]
        card_markers.append({"id": card_ids[si], "slot": si, "center_uv": [cu, cv],
                             "corners": [[round(float(x), 6) for x in p] for p in corners],
                             "cells": marker_cells(codes[card_ids[si]]).tolist()})
    layout = {
        "name": "AR cube v1", "units": "mm", "edge": S, "fillet": R_FILLET, "cell": CELL,
        "marker_size": MSIZE, "gap": GAP, "inlay_depth": DEPTH,
        "dictionary": "ARUCO_MIP_36h12",
        "frame": "+X right, +Y up, +Z front; origin at cube centre",
        "faces": [{"name": n, "U": U_, "V": V_, "N": N_} for n, U_, V_, N_ in FACES],
        "markers": markers,
        "card": {"size": CARD_W, "cell": CARD_CELL, "marker_size": CARD_MSIZE, "offset": CARD_OFF,
                 "frame": "card lies flat: plane Y = 0, normal +Y, page right = +X, page up = -Z",
                 "markers": card_markers},
    }
    json.dump(layout, open("build/layout.json", "w"), indent=1)
    return layout


def face_image(layout, fi, ppmm=20, size_mm=None):
    """Greyscale image of one face, white background, black cells; image up = +V."""
    size_mm = size_mm or layout["edge"]
    n = int(round(size_mm * ppmm))
    img = np.full((n, n), 255, np.uint8)
    cell = layout["cell"]
    for m in layout["markers"]:
        if m["face"] != fi:
            continue
        cu, cv = m["center_uv"]
        for i in range(8):
            for j in range(8):
                if not m["cells"][i][j]:
                    continue
                u0 = cu - 4 * cell + j * cell
                v1 = cv + 4 * cell - i * cell
                x0 = int(round((u0 + size_mm / 2) * ppmm)); x1 = int(round((u0 + cell + size_mm / 2) * ppmm))
                y0 = int(round((size_mm / 2 - v1) * ppmm)); y1 = int(round((size_mm / 2 - v1 + cell) * ppmm))
                img[y0:y1, x0:x1] = 0
    return img


if __name__ == "__main__":
    lay = build()
    import cv2
    for fi in range(6):
        cv2.imwrite(f"build/face_{fi}.png", face_image(lay, fi))
    print("markers", len(lay["markers"]), "ids", [m["id"] for m in lay["markers"]])
    print("marker size", MSIZE, "offset", OFF, "white margin to cube edge", S / 2 - OFF - MSIZE / 2)
