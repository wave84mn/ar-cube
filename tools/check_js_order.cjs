const sharp = require('/home/claude/.npm-global/lib/node_modules/sharp');
global.CV = require('/home/claude/arcube/vendor_src/js-aruco2/src/cv.js').CV;
const AR = require('/home/claude/arcube/vendor_src/js-aruco2/src/aruco.js').AR;
const lay = require('/home/claude/arcube/build/layout.json');
(async () => {
  const det = new AR.Detector({dictionaryName: 'ARUCO_MIP_36h12', maxHammingDistance: 3});
  let ok = true;
  for (let fi = 0; fi < 6; fi++) {
    // downscale to 7 px/mm and pad with white
    const {data, info} = await sharp(`build/face_${fi}.png`).resize(490, 490).extend({top: 60, bottom: 60, left: 60, right: 60, background: '#fff'}).ensureAlpha().raw().toBuffer({resolveWithObject: true});
    const t0 = Date.now();
    const ms = det.detect({width: info.width, height: info.height, data});
    const dt = Date.now() - t0;
    const face = lay.faces[fi];
    for (const m of lay.markers.filter(m => m.face === fi)) {
      const got = ms.find(x => x.id === m.id);
      if (!got) { console.log('face', fi, 'missing', m.id); ok = false; continue; }
      let maxe = 0;
      m.corners.forEach((p, k) => {
        const u = p[0]*face.U[0]+p[1]*face.U[1]+p[2]*face.U[2];
        const v = p[0]*face.V[0]+p[1]*face.V[1]+p[2]*face.V[2];
        const ex = (u + 35) * 7 + 60, ey = (35 - v) * 7 + 60;
        maxe = Math.max(maxe, Math.hypot(ex - got.corners[k].x, ey - got.corners[k].y));
      });
      if (maxe > 3) { console.log('face', fi, 'id', m.id, 'corner mismatch', maxe.toFixed(1), JSON.stringify(got.corners)); ok = false; }
    }
    console.log('face', fi, 'found', ms.length, 'markers in', dt, 'ms');
  }
  console.log(ok ? 'JS corner order matches layout (TL,TR,BR,BL)' : 'MISMATCH');
})();
