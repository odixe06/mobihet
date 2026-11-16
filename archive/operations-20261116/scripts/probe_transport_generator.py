"""Train fold -> val fold only; never reads official test. Benign-anchored latent transport vs Core copula.

Core copula (exact fit: 256 bins on [-4,4], mid-rank normal scores, ridge 1e-4) is fitted on a fold-train subset.
Transport: a real benign row x_b -> randomized PIT under the benign marginals -> z_b; Gaussian OT map
T_c(z) = mu_c + A_c (z - mu_0) with A_c the Monge map N(mu_0,S_0) -> N(mu_c,S_c); x = F_c^{-1}(Phi(T_c z_b)).
If z_b ~ N(mu_0,S_0) the output has exactly Core's class-c latent law; per vehicle it stays near local context.
Judge = HGB on real fold-train rows; reports per class how often each synthetic source is judged as its label.
"""
import glob, json, sys, time
import numpy as np, pyarrow.parquet as pq
from scipy.special import ndtri, ndtr
from sklearn.ensemble import HistGradientBoostingClassifier as H

t0 = time.time()
cols = json.load(open('/home/odixe/project/mobihet/knowledge/input_order.json'))
cols = cols if isinstance(cols, list) else (cols.get('order') or cols.get('inputs'))
assert len(cols) == 77
fs = sorted(glob.glob('/home/odixe/project/mobihet-features/data/v4-scaled/train/scenario=*/world=*/part-*.parquet'))
assert len(fs) == 410
rng = np.random.default_rng(20261108)
pick = rng.choice(len(fs), int(sys.argv[1]) if len(sys.argv) > 1 else 80, replace=False)
Xt, Yt, Ct, Xv, Yv, Cv = [], [], [], [], [], []
for i in pick:
    t = pq.ParquetFile(fs[i]).read(columns=cols + ['client_id', 'label', 'fold']).to_pandas()
    fold = t['fold'].astype(str).values
    a = t[fold == 'train'].sample(frac=.25, random_state=int(i)); b = t[fold == 'val']
    Xt.append(a[cols].to_numpy(np.float32)); Yt.append(a['label'].to_numpy(np.int64)); Ct.append(a['client_id'].to_numpy())
    Xv.append(b[cols].to_numpy(np.float32)); Yv.append(b['label'].to_numpy(np.int64)); Cv.append(b['client_id'].to_numpy())
Xt, Yt, Ct, Xv, Yv, Cv = map(np.concatenate, (Xt, Yt, Ct, Xv, Yv, Cv))
print('train', Xt.shape, 'val', Xv.shape, f'{time.time()-t0:.0f}s', flush=True)
C, D, NB = 16, 77, 256
binf = lambda x: np.clip(((x + 4) * (NB / 8)).astype(np.int64), 0, NB - 1)
cdf = np.zeros((C, D, NB)); mass = np.zeros((C, D, NB)); mu = np.zeros((C, D)); S = np.zeros((C, D, D))
r = np.arange(D)
for c in range(C):
    x = Xt[Yt == c]; b = binf(x)
    h = np.stack([np.bincount(b[:, j], minlength=NB) for j in range(D)]).astype(np.float64)
    m = h / h.sum(1, keepdims=True); cd = m.cumsum(1)
    score = ndtri(np.clip(cd - m / 2, 1e-4, 1 - 1e-4))
    z = score[r, b]
    mu[c] = z.mean(0); S[c] = np.cov(z, rowvar=False, bias=True) + 1e-4 * np.eye(D)
    cdf[c], mass[c] = cd, m
L = np.linalg.cholesky(S); Sfull = S

def sqrtm(M):
    w, V = np.linalg.eigh(M); return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

S0h = sqrtm(S[0]); S0ih = np.linalg.inv(S0h)
A = np.stack([S0ih @ sqrtm(S0h @ S[c] @ S0h) @ S0ih for c in range(C)])

def inv_cdf(c, u):
    cd, m = cdf[c], mass[c]
    j = np.stack([np.searchsorted(cd[f], u[:, f]) for f in range(D)], 1).clip(max=NB - 1)
    hi = cd[r, j]; mm = m[r, j]
    frac = np.clip((u - (hi - mm)) / np.maximum(mm, 1e-12), 0, 1)
    return (-4 + (j + frac) * (8 / NB)).astype(np.float32)

def copula(c, n):
    z = mu[c] + rng.standard_normal((n, D)) @ L[c].T
    return inv_cdf(c, np.clip(ndtr(z), 1e-6, 1 - 1e-6))

def transport(c, xb):
    b = binf(xb); lo = cdf[0][r, b] - mass[0][r, b]
    u = np.clip(lo + rng.random(xb.shape) * mass[0][r, b], 1e-6, 1 - 1e-6)   # randomized PIT, benign marginals
    z = ndtri(u); zc = mu[c] + (z - mu[0]) @ A[c].T
    return inv_cdf(c, np.clip(ndtr(zc), 1e-6, 1 - 1e-6))

TAU = float(sys.argv[2]) if len(sys.argv) > 2 else .2
tv = .5 * np.abs(mass - mass[0][None]).sum(-1)                     # (C, D) from pooled per-class hists only
SETS = {c: np.flatnonzero(tv[c] > TAU) for c in range(1, C)}
if len(sys.argv) > 3:   # field-dependency sets measured from benign twins (train fold, one scenario)
    tw = json.load(open(sys.argv[3])); lab = json.load(open('/home/odixe/project/mobihet/knowledge/label_mapping.json'))['label_to_id']
    SETS = {c: np.array([], int) for c in range(1, C)}
    for name, o in tw.items():
        if o.get('changed_frac'): SETS[lab[name]] = np.flatnonzero(np.array(o['changed_frac']) > .5)
print('TAU', TAU, 'sizes', {c: len(v) for c, v in SETS.items()}, flush=True)

def sparse(c, xb):
    S = SETS[c]; K = np.setdiff1d(r, S)
    if len(S) == 0: return xb.copy()
    b = binf(xb); lo = cdf[c][r, b] - mass[c][r, b]
    u = np.clip(lo + rng.random(xb.shape) * mass[c][r, b], 1e-6, 1 - 1e-6)   # randomized PIT under class-c marginals
    z = ndtri(u); Sg = S_full = S
    Skk = Sg_ = None
    cov = Sfull[c]; mk = mu[c][K]; ms = mu[c][S]
    W = np.linalg.solve(cov[np.ix_(K, K)], cov[np.ix_(K, S)])            # (|K|,|S|)
    cm = ms + (z[:, K] - mk) @ W
    cc = cov[np.ix_(S, S)] - cov[np.ix_(S, K)] @ W
    Lc = np.linalg.cholesky(cc + 1e-6 * np.eye(len(S)))
    zs = cm + rng.standard_normal((len(xb), len(S))) @ Lc.T
    full = mu[c] + np.zeros_like(z); full[:, S] = zs
    x = xb.copy(); x[:, S] = inv_cdf(c, np.clip(ndtr(full), 1e-6, 1 - 1e-6))[:, S]
    return x

# judge on real fold-train rows, class-balanced cap
idx = np.concatenate([rng.permutation(np.flatnonzero(Yt == c))[:40000] for c in range(C)])
judge = H(max_iter=300, learning_rate=.1, max_leaf_nodes=63, random_state=0).fit(Xt[idx], Yt[idx])
print('judge fit', f'{time.time()-t0:.0f}s', flush=True)
N = 4000
benign_tr = Xt[Yt == 0]
out = {'files': int(len(pick)), 'train_rows': int(len(Xt)), 'val_rows': int(len(Xv)), 'per_class': []}
print('cls | judged-as-label: val real / copula / transport | judged-benign: val / copula / transport | TV16 vs val: copula / transport')

def tv16(a, b):
    ba = np.clip(((a + 4) * 2).astype(int), 0, 15); bb = np.clip(((b + 4) * 2).astype(int), 0, 15)
    return float(np.mean([.5 * np.abs(np.bincount(ba[:, f], minlength=16) / len(a) - np.bincount(bb[:, f], minlength=16) / len(b)).sum() for f in range(D)]))

for c in range(C):
    xv = Xv[Yv == c]; xv = xv[rng.permutation(len(xv))[:N]]
    xc = copula(c, N); src = benign_tr[rng.integers(len(benign_tr), size=N)]; xt = sparse(c, src) if (c and len(SETS[c])) else (copula(c, N) if c else src)
    pv, pc, pt = (judge.predict(x) for x in (xv, xc, xt))
    top = lambda p: [(int(k), round(float((p == k).mean()), 3)) for k in np.argsort(-np.bincount(p, minlength=C))[:3]]
    print("   top judged copula", top(pc), "transport", top(pt), "val", top(pv), flush=True)
    row = {'class': c, 'val_n': int(len(xv)),
           'as_label': [float((p == c).mean()) for p in (pv, pc, pt)],
           'as_benign': [float((p == 0).mean()) for p in (pv, pc, pt)],
           'tv16_vs_val': [tv16(xc, xv), tv16(xt, xv)]}
    out['per_class'].append(row)
    print(c, '| %.3f %.3f %.3f | %.3f %.3f %.3f | %.4f %.4f' % (*row['as_label'], *row['as_benign'], *row['tv16_vs_val']), flush=True)
for k in ('as_label', 'as_benign', 'tv16_vs_val'):
    out['mean_' + k] = np.mean([r_[k] for r_ in out['per_class']], 0).tolist()
print({k: out[k] for k in out if k.startswith('mean_')}, f'{time.time()-t0:.0f}s')
json.dump(out, open('/home/odixe/project/mobihet/runs/transport-probe/judge.json', 'w'), indent=1)
