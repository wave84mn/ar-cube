import json, numpy as np, cv2
js = json.load(open('build/mip36h12_js.json'))
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12)
match = 0; mism = []
for i, code in enumerate(js['codes']):
    img = cv2.aruco.generateImageMarker(d, i, 8, borderBits=1)
    bits = (img[1:7, 1:7] > 127).astype(int).flatten()
    jsb = np.array([int(c) for c in code])
    if (bits == jsb).all(): match += 1
    else: mism.append(i)
print('ids matching OpenCV:', match, 'of', len(js['codes']), 'first mismatches', mism[:10])
