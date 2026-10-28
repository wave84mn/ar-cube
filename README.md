# AR Cube

A printable AR cube plus a phone web viewer. Hold the cube in front of the phone and a building model stands on it and turns with it. A flat paper card works too.

- **Viewer:** `index.html`. Open it over https in Safari (iPhone) or Chrome (Android). Nothing is installed, and the camera image is processed on the phone.
- **Print files:** `print/`
  - `AR_cube_70mm_2colour.3mf`: one object with two parts, *body (white)* and *markers (black)*.
  - `AR_cube_70mm_STL.zip`: the same two parts as STLs.
  - `AR_cube_paper_test.pdf`: page 1 is a flat table card, page 2 is a 60 mm fold-up cube. Print at 100 %.
- **Models:** `models/models.json` lists them; each is a `.glb` (Y up, front towards +Z, any units, auto-scaled to the cube).

## How it works
- Each cube face carries 2×2 ArUco markers (dictionary MIP_36h12, 24 ids, at least 14 bits apart even when rotated). The card uses 4 more ids.
- `js/cv/detector.js` finds the markers. It thresholds the frame, finds square outlines, reads the 6×6 code and refines the corners to sub-pixel accuracy.
- `js/cv/pose.js` and `js/cv/tracker.js` solve one pose from every visible marker. They use Levenberg-Marquardt with robust weights and reject candidates that would put a seen face pointing away from the camera.
- `js/app.js`:
  - Tracks a crop around the cube once it is found. The crop is scaled so each marker cell is about 10 px.
  - Smooths the pose with a One Euro filter.
  - Draws exactly the camera frame that was tracked, so the overlay doesn't lag.
  - Keeps the building upright on whichever face points up, using the phone's tilt sensor.
- `data/layout.json` is the single source for the marker layout. The print geometry, the PDF and the viewer are all generated from it.

## Rebuilding (tools/)
Run from a working folder that contains `build/`, `out/` and `site/`:
`layout.py` → `geometry.py` (3MF/STL) → `paper_pdf.py` → `standin_kula.py`.
Tests: `synth.py` makes synthetic camera frames with known poses, `run_tracker.mjs` and `run_sequence.mjs` score the tracker, and `e2e.cjs` runs the app in Chromium with a fake camera.

Third-party: three.js r184 (MIT, `js/vendor/three/LICENSE`); the contour pipeline is adapted from js-aruco / js-aruco2 (MIT, `js/cv/LICENSE-js-aruco.txt`).
