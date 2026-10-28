// Export js-aruco2 ARUCO_MIP_36h12 code list as bit strings
const path = process.argv[2];
global.CV = require(path + '/src/cv.js').CV;
const AR = require(path + '/src/aruco.js').AR;
const d = new AR.Dictionary('ARUCO_MIP_36h12');
require('fs').writeFileSync(process.argv[3], JSON.stringify({nBits: d.nBits, markSize: d.markSize, tau: d.tau, codes: d.codeList}));
console.log('codes', d.codeList.length, 'nBits', d.nBits, 'markSize', d.markSize, 'tau', d.tau);
