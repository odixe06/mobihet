"""H2-T0-lite (docs/research/de-xuat-huong-moi.md, H2): do pooled class-conditional statistics in the 66-feature
input space -- additive sums that never expose a row -- give a client knowledge it cannot get from its own data?

No client's rows ever reach another client (FL rule, owner 2026-10-10): the only cross-client quantity is the
per-class additive sums (n, sum x, sum x x^T) of the other 19 clients, which this script adds up the way an RSU
(or a secure-aggregation round) would.

Diagnostic only: plain MLP (no BatchNorm) instead of DAGSNet, short training, a random test subsample scored
descriptively. Nothing here selects a method or a hyperparameter for the main protocol.

Arms per (target client, seed), same steps and the same 50/50 batch mix:
  a0    local natural only
  a1    local + class-balanced replay of the client's own rows      (control: prior/balancing, no foreign info)
  diag  local + samples from pooled diagonal Gaussians p(x|c) of the other 19 clients
  full  local + samples from pooled full-covariance Gaussians
  mean  local + the pooled class mean itself                         (control: prototype-like point mass)

    python <archive>/scripts/run_local_guarded.py python scripts/probe_h2_input_stats.py   (env nckh)
"""
import json, time
from pathlib import Path
import numpy as np, pyarrow.dataset as ds, torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
DATA = ROOT.parent / 'dataset'
OUT = ROOT / 'papers/probes/h2_input_stats'
FEAT, C = json.loads((ROOT / 'knowledge/meta.json').read_text())['feature_cols'], 16
SCALER = json.loads((DATA / 'centralized/scaler.json').read_text())['features']
TARGETS, SEEDS, ARMS = [0, 5, 10, 15], [42, 43], ['a0', 'a1', 'diag', 'full', 'mean']
LOCAL_ROWS, TEST_P = 300_000, 0.03
STEPS, BATCH, LR = 3000, 512, 1e-3
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batches(path):
    # Only *.parquet (test/ also holds *.stats.json); no readahead: the default buffers GBs (peak 4,6 GB measured).
    files = [str(p) for p in sorted(path.glob('*.parquet'))]
    for b in ds.dataset(files, format='parquet').to_batches(columns=FEAT + ['label'], batch_size=131072,
                                                           batch_readahead=1, fragment_readahead=1):
        x = np.column_stack([b.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in FEAT])
        yield x, b.column('label').to_numpy(zero_copy_only=False).astype(np.int64)


def scan_clients():
    """Per-client per-class sums n, s1, s2 (float64) and a random local sample of each target client."""
    stats, local = {}, {}
    for k in range(20):
        path = DATA / f'fl_client/alpha05/20_client/train/client_id={k:03d}'
        n_rows = ds.dataset([str(p) for p in sorted(path.glob('*.parquet'))], format='parquet').count_rows()
        g = np.random.default_rng([1729, k])
        n, s1, s2 = np.zeros(C), np.zeros((C, 66)), np.zeros((C, 66, 66))
        keep_x, keep_y = [], []
        for x, y in batches(path):
            for c in np.unique(y):
                xc = x[y == c].astype(np.float64)
                n[c] += len(xc); s1[c] += xc.sum(0); s2[c] += xc.T @ xc
            if k in TARGETS:
                m = g.random(len(y)) < LOCAL_ROWS / n_rows
                keep_x.append(x[m]); keep_y.append(y[m])
        stats[k] = (n, s1, s2)
        if k in TARGETS:
            local[k] = (np.concatenate(keep_x), np.concatenate(keep_y))
        print(f'client {k:2d}: {n_rows} rows', flush=True)
    return stats, local


def load_test():
    g, xs, ys = np.random.default_rng(7), [], []
    mu = np.array([SCALER[f]['mean'] for f in FEAT], np.float32)
    sd = np.array([SCALER[f]['std_used'] for f in FEAT], np.float32)
    for x, y in batches(DATA / 'centralized/test'):
        m = g.random(len(y)) < TEST_P
        xs.append((x[m] - mu) / sd); ys.append(y[m])
    return np.concatenate(xs), np.concatenate(ys)


def pooled(stats, exclude):
    n = sum(stats[k][0] for k in stats if k != exclude)
    s1 = sum(stats[k][1] for k in stats if k != exclude)
    s2 = sum(stats[k][2] for k in stats if k != exclude)
    mean = s1 / n[:, None]
    cov = s2 / n[:, None, None] - np.einsum('ci,cj->cij', mean, mean)
    return n, mean, cov


def macro_f1(cm):
    tp = np.diag(cm); p = tp / np.maximum(cm.sum(0), 1); r = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f1.mean()), f1


def run(arm, seed, lx, ly, aux, tx, ty):
    torch.manual_seed(seed); g = torch.Generator(device=dev).manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(66, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, C)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    half = BATCH // 2
    for _ in range(STEPS):
        i = torch.randint(len(ly), (BATCH if aux is None else half,), device=dev, generator=g)
        xb, yb = lx[i], ly[i]
        if aux is not None:
            xa, ya = aux(half, g)
            xb, yb = torch.cat([xb, xa]), torch.cat([yb, ya])
        loss = torch.nn.functional.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval(); pred = []
    with torch.no_grad():
        for j in range(0, len(tx), 65536):
            pred.append(net(tx[j:j + 65536]).argmax(1))
    pred = torch.cat(pred).cpu().numpy()
    cm = np.bincount(ty * C + pred, minlength=C * C).reshape(C, C)
    return macro_f1(cm)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    stats, local = scan_clients()
    tx_np, ty = load_test()
    print(f'scan+test {time.time() - t0:.0f}s, test rows {len(ty)}', flush=True)
    tx = torch.from_numpy(tx_np).to(dev)
    results = {}
    for k in TARGETS:
        lx_np, ly_np = local[k]
        lx, ly = torch.from_numpy(lx_np).to(dev), torch.from_numpy(ly_np).to(dev)
        n, mean, cov = pooled(stats, k)
        mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
        sd_t = torch.tensor(np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0, None)), dtype=torch.float32, device=dev)
        chol = torch.tensor(np.linalg.cholesky(cov + 1e-4 * np.eye(66)), dtype=torch.float32, device=dev)
        order = torch.argsort(ly)
        counts = torch.bincount(ly, minlength=C)
        starts = torch.cumsum(counts, 0) - counts
        own_classes = torch.nonzero(counts).flatten()

        def cls(m, g, pool=None):
            pool = torch.arange(C, device=dev) if pool is None else pool
            return pool[torch.randint(len(pool), (m,), device=dev, generator=g)]

        def own_bal(m, g):
            y = cls(m, g, own_classes)
            u = torch.rand(m, device=dev, generator=g)
            idx = order[starts[y] + (u * counts[y]).long().clamp_max(counts[y] - 1)]
            return lx[idx], y

        def diag(m, g):
            y = cls(m, g); z = torch.randn(m, 66, device=dev, generator=g)
            return mean_t[y] + sd_t[y] * z, y

        def full(m, g):
            y = cls(m, g); z = torch.randn(m, 66, 1, device=dev, generator=g)
            return mean_t[y] + (chol[y] @ z).squeeze(-1), y

        def point(m, g):
            y = cls(m, g)
            return mean_t[y], y

        aux = dict(a0=None, a1=own_bal, diag=diag, full=full, mean=point)
        for seed in SEEDS:
            for arm in ARMS:
                t1 = time.time()
                f, per = run(arm, seed, lx, ly, aux[arm], tx, ty)
                results[f'c{k}_s{seed}_{arm}'] = dict(client=k, seed=seed, arm=arm, f1_macro=f, f1_per_class=per.tolist(),
                                                     local_rows=int(len(ly_np)))
                print(f'client {k} seed {seed} {arm:5s} f1_macro {f:.4f} ({time.time() - t1:.0f}s)', flush=True)
        (OUT / 'results.json').write_text(json.dumps(dict(config=dict(targets=TARGETS, seeds=SEEDS, arms=ARMS,
            local_rows=LOCAL_ROWS, test_p=TEST_P, steps=STEPS, batch=BATCH, lr=LR, test_rows=int(len(ty))),
            results=results), indent=1))
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
