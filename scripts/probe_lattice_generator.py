"""Train fold -> val fold only; never reads official test. Core copula vs lattice-snapped copula (synthetic values moved
to the nearest value the scaled feature can take). Judges: HGB and a torch MLP on real fold-train rows (offline instruments,
not part of any method). Lattice here = distinct fold-train values of the subset (deployable form: public scaler mid-ranks)."""
import glob, json, sys, time
import numpy as np, pyarrow.parquet as pq, torch
from scipy.special import ndtri, ndtr
from sklearn.ensemble import HistGradientBoostingClassifier as H
t0 = time.time()
cols = json.load(open('/home/odixe/project/mobihet/knowledge/input_order.json'))['inputs']
fs = sorted(glob.glob('/home/odixe/project/mobihet-features/data/v4-scaled/train/scenario=*/world=*/part-*.parquet'))
rng = np.random.default_rng(20261108); pick = rng.choice(len(fs), 60, replace=False)
Xt, Yt, Xv, Yv = [], [], [], []
for i in pick:
    t = pq.ParquetFile(fs[i]).read(columns=cols + ['label', 'fold']).to_pandas(); fold = t['fold'].astype(str).values
    a = t[fold == 'train'].sample(frac=.25, random_state=int(i)); b = t[fold == 'val']
    Xt.append(a[cols].to_numpy(np.float32)); Yt.append(a['label'].to_numpy(np.int64))
    Xv.append(b[cols].to_numpy(np.float32)); Yv.append(b['label'].to_numpy(np.int64))
Xt, Yt, Xv, Yv = map(np.concatenate, (Xt, Yt, Xv, Yv))
C, D, NB = 16, 77, 256; r = np.arange(D)
binf = lambda x: np.clip(((x + 4) * (NB / 8)).astype(np.int64), 0, NB - 1)
cdf = np.zeros((C, D, NB)); mass = np.zeros((C, D, NB)); mu = np.zeros((C, D)); L = np.zeros((C, D, D))
for c in range(C):
    x = Xt[Yt == c]; b = binf(x)
    h = np.stack([np.bincount(b[:, j], minlength=NB) for j in range(D)]).astype(float)
    m = h / h.sum(1, keepdims=True); cd = m.cumsum(1); z = ndtri(np.clip(cd - m / 2, 1e-4, 1 - 1e-4))[r, b]
    mu[c] = z.mean(0); L[c] = np.linalg.cholesky(np.cov(z, rowvar=False, bias=True) + 1e-4 * np.eye(D)); cdf[c], mass[c] = cd, m
lattice = [np.unique(Xt[:, j]) for j in range(D)]
def copula(c, n):
    u = np.clip(ndtr(mu[c] + rng.standard_normal((n, D)) @ L[c].T), 1e-6, 1 - 1e-6)
    j = np.stack([np.searchsorted(cdf[c][f], u[:, f]) for f in range(D)], 1).clip(max=NB - 1)
    hi = cdf[c][r, j]; mm = mass[c][r, j]
    return (-4 + (j + np.clip((u - (hi - mm)) / np.maximum(mm, 1e-12), 0, 1)) * (8 / NB)).astype(np.float32)
def snap(x, c):
    """nearest lattice value inside the same coarse bin, so the class-c bin mass is unchanged."""
    out = x.copy()
    for f in range(D):
        v = lattice[f]; k = np.clip(np.searchsorted(v, x[:, f]), 1, len(v) - 1)
        lo, hi = v[k - 1], v[k]; near = np.where(np.abs(x[:, f] - lo) <= np.abs(hi - x[:, f]), lo, hi)
        out[:, f] = np.where(binf(near) == binf(x[:, f]), near, x[:, f])
    return out
idx = np.concatenate([rng.permutation(np.flatnonzero(Yt == c))[:40000] for c in range(C)])
hgb = H(max_iter=300, max_leaf_nodes=63, random_state=0).fit(Xt[idx], Yt[idx])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
mlp = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                          torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.BatchNorm1d(256),
                          torch.nn.ReLU(), torch.nn.Linear(256, C)).to(dev)
opt = torch.optim.AdamW(mlp.parameters(), 1e-3, weight_decay=1e-4)
Xi, Yi = torch.tensor(Xt[idx], device=dev), torch.tensor(Yt[idx], device=dev)
for ep in range(8):
    for p in torch.randperm(len(Xi), device=dev).split(512):
        loss = torch.nn.functional.cross_entropy(mlp(Xi[p]), Yi[p]); opt.zero_grad(); loss.backward(); opt.step()
mlp.eval()
mp = lambda x: mlp(torch.tensor(x, device=dev)).argmax(1).cpu().numpy()
print('judges fit', f'{time.time()-t0:.0f}s', flush=True)
N = 4000; res = {}
print('cls | HGB as-label: val / copula / snapped | MLP as-label: val / copula / snapped')
for c in range(C):
    xv = Xv[Yv == c]; xv = xv[rng.permutation(len(xv))[:N]]; xc = copula(c, N); xs = snap(xc, c)
    row = [[float((j(x) == c).mean()) for x in (xv, xc, xs)] for j in (hgb.predict, mp)]
    res[c] = row; print(c, '| %.3f %.3f %.3f | %.3f %.3f %.3f' % (*row[0], *row[1]), flush=True)
m = np.mean([res[c] for c in res], 0); print('mean HGB', m[0].round(3), 'MLP', m[1].round(3), f'{time.time()-t0:.0f}s')
json.dump({'per_class': res, 'mean': m.tolist()}, open('/home/odixe/project/mobihet/runs/transport-probe/lattice.json', 'w'), indent=1)
