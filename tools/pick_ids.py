import json, numpy as np, itertools, random
js = json.load(open('build/mip36h12_js.json'))
codes = np.array([[int(c) for c in s] for s in js['codes']]).reshape(-1, 6, 6)
N = len(codes)
rots = np.stack([np.rot90(codes, k, axes=(1, 2)) for k in range(4)], 1)  # N,4,6,6
flat = codes.reshape(N, -1); rflat = rots.reshape(N, 4, -1)
# pairwise min distance over rotations
D = np.full((N, N), 99)
for i in range(N):
    d = (rflat[:, :, :] != flat[i][None, None, :]).sum(-1).min(1)
    D[i] = d
np.fill_diagonal(D, 99)
selfrot = np.array([min((rflat[i, k] != flat[i]).sum() for k in (1, 2, 3)) for i in range(N)])
white = flat.sum(1)
print('dictionary min dist', D.min(), 'selfrot min', selfrot.min())
ok = np.where((np.abs(white - 18) <= 4) & (selfrot >= 14))[0]
print('candidates', len(ok))
def score(sel):
    sub = D[np.ix_(sel, sel)]
    return sub.min(), (sub == sub.min()).sum()
best = None
rng = random.Random(1)
for trial in range(400):
    sel = [rng.choice(list(ok))]
    while len(sel) < 24:
        rest = [c for c in ok if c not in sel]
        m = [(D[c, sel].min(), -abs(white[c] - 18), rng.random(), c) for c in rest]
        sel.append(max(m)[3])
    # local swaps
    improved = True
    while improved:
        improved = False
        s0 = score(sel)
        for a in range(24):
            for c in ok:
                if c in sel: continue
                t = sel.copy(); t[a] = c
                s1 = score(t)
                if (s1[0] > s0[0]) or (s1[0] == s0[0] and s1[1] < s0[1]):
                    sel, s0, improved = t, s1, True
    s = score(sel)
    if best is None or (s[0] > best[0][0]) or (s[0] == best[0][0] and s[1] < best[0][1]):
        best = (s, sorted(sel))
print('best min distance', best[0][0], 'pairs at min', best[0][1] // 2)
sel = best[1]
print('ids', sel)
print('white bits', [int(white[i]) for i in sel])
print('self-rotation distance', [int(selfrot[i]) for i in sel])
json.dump({'ids': [int(i) for i in sel], 'min_pair_distance': int(best[0][0])}, open('build/picked_ids.json', 'w'))
