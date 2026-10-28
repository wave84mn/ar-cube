import json, numpy as np
js = json.load(open('build/mip36h12_js.json'))
codes = np.array([[int(c) for c in s] for s in js['codes']]).reshape(-1, 6, 6)
N = len(codes); flat = codes.reshape(N, -1)
rflat = np.stack([np.rot90(codes, k, axes=(1, 2)) for k in range(4)], 1).reshape(N, 4, -1)
D = np.array([(rflat != flat[i][None, None]).sum(-1).min(1) for i in range(N)]); np.fill_diagonal(D, 99)
selfrot = np.array([min((rflat[i, k] != flat[i]).sum() for k in (1, 2, 3)) for i in range(N)])
white = flat.sum(1)
cube = json.load(open('build/picked_ids.json'))['ids']
sel = list(cube); card = []
for _ in range(4):
    cand = [c for c in range(N) if c not in sel and selfrot[c] >= 14 and abs(white[c] - 18) <= 4]
    best = max(cand, key=lambda c: (D[c, sel].min(), -abs(white[c] - 18), -c))
    card.append(best); sel.append(best)
print('card ids', card, 'min distance to all others', min(D[c, [s for s in sel if s != c]].min() for c in card))
d = json.load(open('build/picked_ids.json')); d['card_ids'] = [int(c) for c in card]
json.dump(d, open('build/picked_ids.json', 'w'))
