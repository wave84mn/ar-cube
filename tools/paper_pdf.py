"""Paper test set (A4): page 1 = flat table card, page 2 = 64 mm fold-up cube net."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, black, white

lay = json.load(open("build/layout.json"))
GREY = Color(0.55, 0.55, 0.55); LIGHT = Color(0.80, 0.80, 0.80); TAB = Color(0.90, 0.90, 0.90)
PW, PH = A4


def draw_cells(c, cells, cu, cv, cell, k, x0, y0, rot180=False):
    """Draw one 8x8 marker; (x0, y0) = page position of face centre (mm); k = scale."""
    for i in range(8):
        j = 0
        while j < 8:
            if not cells[i][j]:
                j += 1; continue
            j1 = j
            while j1 < 8 and cells[i][j1]:
                j1 += 1
            u0 = cu - 4 * cell + j * cell; u1 = cu - 4 * cell + j1 * cell
            v1 = cv + 4 * cell - i * cell; v0 = v1 - cell
            if rot180:
                u0, u1, v0, v1 = -u1, -u0, -v1, -v0
            # tiny overlap so adjacent rectangles don't leave hairline gaps in viewers
            c.rect((x0 + u0 * k) * mm, (y0 + v0 * k) * mm, (u1 - u0) * k * mm, (v1 - v0) * k * mm + 0.02 * mm, stroke=0, fill=1)
            j = j1


def header(c, title, lines):
    c.setFillColor(black); c.setFont("Helvetica-Bold", 13)
    c.drawString(14 * mm, PH - 16 * mm, title)
    c.setFont("Helvetica", 8.5); c.setFillColor(Color(0.25, 0.25, 0.25))
    for n, t in enumerate(lines):
        c.drawString(14 * mm, PH - (22 + 4.2 * n) * mm, t)


def scale_bar(c, x, y, below=False):
    c.setStrokeColor(black); c.setLineWidth(0.6)
    c.line(x * mm, y * mm, (x + 50) * mm, y * mm)
    for t in (0, 50):
        c.line((x + t) * mm, (y - 1.5) * mm, (x + t) * mm, (y + 1.5) * mm)
    c.setFont("Helvetica", 7); c.setFillColor(black)
    if below:
        c.drawString(x * mm, (y - 5) * mm, "must measure 50 mm")
        c.drawString(x * mm, (y - 8.5) * mm, "(print at 100 % / Actual size)")
    else:
        c.drawString((x + 52) * mm, (y - 1) * mm, "must measure 50 mm (print at 100 % / Actual size)")


def card_page(c):
    card = lay["card"]
    header(c, "AR cube - TABLE CARD", [
        "1. Print this page at 100 % (Actual size, no 'fit to page') on plain or matte paper.",
        "2. Lay it flat on a table in good light, open the viewer on your phone and point it at the card.",
        "3. The building stands in the circle. Walk around the card to see every side.",
    ])
    cx, cy = PW / mm / 2, 148.0
    c.setFillColor(black)
    for m in card["markers"]:
        draw_cells(c, m["cells"], m["center_uv"][0], m["center_uv"][1], card["cell"], 1.0, cx, cy)
    c.setStrokeColor(LIGHT); c.setLineWidth(0.8); c.setDash(2, 2)
    c.circle(cx * mm, cy * mm, 30 * mm, stroke=1, fill=0)
    c.setDash()
    c.setFillColor(GREY); c.setFont("Helvetica", 8)
    c.drawCentredString(cx * mm, (cy - 2) * mm, "the building appears here")
    scale_bar(c, 14, 22)
    c.setFont("Helvetica", 6.5); c.setFillColor(GREY)
    c.drawString(14 * mm, 12 * mm, f"AR cube v1 - ArUco MIP_36h12 ids {', '.join(str(m['id']) for m in card['markers'])} - keep the white border clean")


def net_page(c):
    E = 60.0                                   # paper cube edge
    k = E / lay["edge"]
    header(c, "AR cube - PAPER CUBE (60 mm)", [
        "Print at 100 % on thick matte paper or card (160-250 g/m2). Cut along the outer outline (tabs included).",
        "Crease every dashed line, fold the faces up, and glue each grey tab inside the neighbouring face.",
        "Keep the printed faces flat and clean. Any face can point up; the viewer stands the building on top.",
    ])
    x0 = (PW / mm - 3 * E) / 2                 # left edge of the net
    ytop = PH / mm - 45                        # top edge of the 'top' face
    # face centres (page mm) in the net:  . top . / left front right / . bottom . / . back .
    pos = {
        "top": (x0 + 1.5 * E, ytop - 0.5 * E), "left": (x0 + 0.5 * E, ytop - 1.5 * E),
        "front": (x0 + 1.5 * E, ytop - 1.5 * E), "right": (x0 + 2.5 * E, ytop - 1.5 * E),
        "bottom": (x0 + 1.5 * E, ytop - 2.5 * E), "back": (x0 + 1.5 * E, ytop - 3.5 * E),
    }
    names = [f["name"] for f in lay["faces"]]
    T = 9.0                                    # tab depth
    def tab(cxp, cyp, side, label):
        h = E / 2
        if side == "L":  pts = [(cxp - h, cyp - h), (cxp - h - T, cyp - h + T), (cxp - h - T, cyp + h - T), (cxp - h, cyp + h)]
        if side == "R":  pts = [(cxp + h, cyp - h), (cxp + h + T, cyp - h + T), (cxp + h + T, cyp + h - T), (cxp + h, cyp + h)]
        if side == "U":  pts = [(cxp - h, cyp + h), (cxp - h + T, cyp + h + T), (cxp + h - T, cyp + h + T), (cxp + h, cyp + h)]
        p = c.beginPath(); p.moveTo(pts[0][0] * mm, pts[0][1] * mm)
        for q in pts[1:]: p.lineTo(q[0] * mm, q[1] * mm)
        c.setFillColor(TAB); c.setStrokeColor(GREY); c.setLineWidth(0.4)
        c.drawPath(p, stroke=1, fill=1)
        c.setFillColor(GREY); c.setFont("Helvetica", 6)
        mx = sum(q[0] for q in pts) / 4; my = sum(q[1] for q in pts) / 4
        c.saveState(); c.translate(mx * mm, my * mm)
        if side in "LR": c.rotate(90)
        c.drawCentredString(0, -2, "glue - " + label); c.restoreState()
    tab(*pos["top"], "L", "left"); tab(*pos["top"], "R", "right"); tab(*pos["top"], "U", "back")
    tab(*pos["bottom"], "L", "left"); tab(*pos["bottom"], "R", "right")
    tab(*pos["back"], "L", "left"); tab(*pos["back"], "R", "right")
    # faces: outlines (cut = solid grey on the outside edges, fold = dashed light)
    for fi, name in enumerate(names):
        fx, fy = pos[name]
        c.setFillColor(black)
        for m in lay["markers"]:
            if m["face"] == fi:
                draw_cells(c, m["cells"], m["center_uv"][0], m["center_uv"][1], lay["cell"], k, fx, fy, rot180=(name == "back"))
    # fold lines
    c.setStrokeColor(LIGHT); c.setLineWidth(0.5); c.setDash(3, 2)
    h = E / 2
    folds = [((x0 + E, ytop - E), (x0 + 2 * E, ytop - E)), ((x0 + E, ytop - 2 * E), (x0 + 2 * E, ytop - 2 * E)),
             ((x0 + E, ytop - 3 * E), (x0 + 2 * E, ytop - 3 * E)), ((x0 + E, ytop - E), (x0 + E, ytop - 2 * E)),
             ((x0 + 2 * E, ytop - E), (x0 + 2 * E, ytop - 2 * E))]
    for name, sides in (("top", "LRU"), ("bottom", "LR"), ("back", "LR")):
        fx, fy = pos[name]
        if "L" in sides: folds.append(((fx - h, fy - h), (fx - h, fy + h)))
        if "R" in sides: folds.append(((fx + h, fy - h), (fx + h, fy + h)))
        if "U" in sides: folds.append(((fx - h, fy + h), (fx + h, fy + h)))
    for (a, b) in folds:
        c.line(a[0] * mm, a[1] * mm, b[0] * mm, b[1] * mm)
    c.setDash()
    # cut outline of the faces (edges not shared with another face or a tab)
    c.setStrokeColor(GREY); c.setLineWidth(0.4)
    cuts = [((x0, ytop - E), (x0 + E, ytop - E)), ((x0, ytop - 2 * E), (x0 + E, ytop - 2 * E)), ((x0, ytop - E), (x0, ytop - 2 * E)),
            ((x0 + 2 * E, ytop - E), (x0 + 3 * E, ytop - E)), ((x0 + 2 * E, ytop - 2 * E), (x0 + 3 * E, ytop - 2 * E)),
            ((x0 + 3 * E, ytop - E), (x0 + 3 * E, ytop - 2 * E)), ((x0 + E, ytop - 4 * E), (x0 + 2 * E, ytop - 4 * E))]
    for (a, b) in cuts:
        c.line(a[0] * mm, a[1] * mm, b[0] * mm, b[1] * mm)
    scale_bar(c, x0 - 4, ytop - 4 * E + 14, below=True)


if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)
    c = canvas.Canvas("out/AR_cube_paper_test.pdf", pagesize=A4)
    c.setTitle("AR cube - paper test set"); c.setAuthor("MT project")
    card_page(c); c.showPage()
    net_page(c); c.showPage()
    c.save()
    print("wrote out/AR_cube_paper_test.pdf")
