"""Round 13 probe, no DAGSNet training (design §Vòng 13): C gpe rescoring and P3 BN recalibration of audited epoch-6 weights,
P1 teacher coverage of pooled-moment Gaussian rows, P4 RFF ridge teacher-only, P2 urban/highway mode diagnostic.

    python fd_probe13.py --weights <dir with *.pt + expected.json> --parquet <partition root> --out probe13.json [--smoke]
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import fd
from fd import P
from mh.v2_methods import calibrate
from mh.metrics import metrics_from_confusion, per_class_from_confusion

HARD = (0, 4, 7, 13)   # benign, dataReplay, positionMirroring, timeDelayAttack


def summarize(cms):
    mm = [metrics_from_confusion(c) for c in cms]
    pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, P.CLASSES)] for c in cms], axis=0)
    return dict(f1_macro=float(np.mean([m['f1_macro'] for m in mm])), benign_f1=float(pc[0]),
                worst=float(min(m['f1_macro'] for m in mm)), f1_per_class=pc.tolist())


def score_models(R, models, rules, holds=('group', 'old')):
    out = {}
    for hold in holds:
        cms = {rule: [] for rule in rules}
        for k, model in enumerate(models):
            scored = R.score(model, k, hold, rules)
            for rule in rules: cms[rule].append(scored[rule][0])
        out.update({f'{hold}/{rule}': summarize(c) for rule, c in cms.items()})
    return out


def load(path, dev):
    w = torch.load(path, map_location=dev, weights_only=True)
    make = lambda s: (lambda m: (m.load_state_dict(s, strict=True), m.eval())[1])(P.make('dagsnet').to(dev))
    return [make(s) for s in w['private']], make(w['proxy'][0])


def ridge_features(x, kind, omega=None, bias=None):
    if kind == 'linear': return torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)
    m = omega.shape[1]
    return torch.cat([math.sqrt(2 / m) * torch.cos(x @ omega + bias), torch.ones(len(x), 1, device=x.device)], 1)


def p4(R, smoke, log):
    """Pooled / own ridge (G = sum phi phi^T, H = sum phi onehot^T) on every train row; lambda multiplier chosen on old."""
    keep = torch.nonzero(~R.zero_cols).squeeze(1); dev = R.dev; chunk = 65536
    gen = torch.Generator(device=dev).manual_seed(13)
    starts = np.cumsum([0] + R.n); pick = torch.randint(0, int(starts[-1]), (4096,), device=dev, generator=gen).cpu().numpy()
    rows = []
    for k in range(R.K):
        local = pick[(pick >= starts[k]) & (pick < starts[k + 1])] - starts[k]
        if len(local): rows.append(R.X[k][torch.from_numpy(local).to(dev)].float()[:, keep])
    sample = torch.cat(rows); d = torch.cdist(sample, sample); sigma = float(d[torch.triu_indices(len(d), len(d), 1, device=dev).unbind()].median())
    out = dict(sigma=sigma, configs={})
    hold = {h: (R.H[h][0].float()[:, keep], R.H[h][1]) for h in ('group', 'old')}
    for kind, m in ([('linear', 0), ('rff', 32), ('rff', 64)] if smoke else [('linear', 0), ('rff', 128), ('rff', 256), ('rff', 1024)]):
        g2 = torch.Generator(device=dev).manual_seed(1000 + m)
        omega = torch.randn(len(keep), m, device=dev, generator=g2) / sigma if m else None
        bias = 2 * math.pi * torch.rand(m, device=dev, generator=g2) if m else None
        dim = (m or len(keep)) + 1; G = torch.zeros(R.K, dim, dim, dtype=torch.float64, device=dev)
        H = torch.zeros(R.K, dim, 16, dtype=torch.float64, device=dev); t = time.time()
        for k in range(R.K):
            for s in range(0, R.n[k], chunk):
                phi = ridge_features(R.X[k][s:s + chunk].float()[:, keep], kind, omega, bias)
                G[k] += (phi.T @ phi).double(); H[k] += (phi.T @ torch.nn.functional.one_hot(R.Y[k][s:s + chunk], 16).float()).double()
        feats = {h: ridge_features(x, kind, omega, bias).double() for h, (x, _) in hold.items()}
        def f1(W, h):
            pred = (feats[h] @ W).argmax(1); y = hold[h][1]
            return metrics_from_confusion(torch.bincount(y * 16 + pred, minlength=256).view(16, 16).cpu().numpy())
        def solve(g, h, c): return torch.linalg.solve(g + c * torch.trace(g) / len(g) * torch.eye(len(g), dtype=g.dtype, device=dev), h)
        pooled_G, pooled_H = G.sum(0), H.sum(0)
        grid = {c: f1(solve(pooled_G, pooled_H, c), 'old')['f1_macro'] for c in (1e-4, 1e-3, 1e-2)}
        c = max(grid, key=grid.get); W = solve(pooled_G, pooled_H, c); g = f1(W, 'group')
        cm = torch.bincount(hold['group'][1] * 16 + (feats['group'] @ W).argmax(1), minlength=256).view(16, 16).cpu().numpy()
        own = [f1(solve(G[k], H[k], c), 'group')['f1_macro'] for k in range(R.K)]
        name = 'linear' if kind == 'linear' else f'rff{m}'
        out['configs'][name] = dict(dim=dim, lambda_mult=c, old_grid=grid, pooled_group_f1=g['f1_macro'],
                                    pooled_group_benign=float(per_class_from_confusion(cm, P.CLASSES)[0]['f1']),
                                    pooled_per_class=[x['f1'] for x in per_class_from_confusion(cm, P.CLASSES)],
                                    own_group_f1_mean=float(np.mean(own)), own_group_f1_worst=float(min(own)),
                                    bytes_up=4 * (dim * (dim + 1) // 2 + 16 * dim), bytes_down=4 * 16 * dim, sec=time.time() - t)
        log(name, {k: v for k, v in out['configs'][name].items() if k not in ('pooled_per_class', 'old_grid')})
    return out


def p2(R, root, proxy, features, smoke, log):
    """Per (hard class, urban/highway) sums over the partition parquet (holdout rows included: diagnostic only); proxy
    acceptance of single-Gaussian vs 2-mode mixture rows."""
    dev = R.dev; files = sorted(Path(root).glob('train/client_id=*/*.parquet'))[::40 if smoke else 1]
    n = torch.zeros(len(HARD), 2, dtype=torch.float64, device=dev); s1 = torch.zeros(len(HARD), 2, P.D, dtype=torch.float64, device=dev)
    s2 = torch.zeros(len(HARD), 2, P.D, P.D, dtype=torch.float64, device=dev)
    for f in files:
        t = pq.read_table(f, columns=features + ['label', 'scenario'])
        y = torch.from_numpy(t['label'].to_numpy().astype(np.int64)).to(dev)
        urban = torch.from_numpy(pc.starts_with(t['scenario'].cast(pa.string()), 'urban').to_numpy(zero_copy_only=False)).to(dev)
        x = torch.from_numpy(np.stack([t[c].to_numpy().astype(np.float32) for c in features], 1)).to(dev)
        for i, c in enumerate(HARD):
            for mode in (0, 1):
                sel = (y == c) & (urban == bool(mode)); xs = x[sel].double()
                n[i, mode] += len(xs); s1[i, mode] += xs.sum(0); s2[i, mode] += xs.T @ xs
    gen = torch.Generator(device=dev).manual_seed(17); out = {}
    def draw(mu, cov, count):
        L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(P.D, dtype=cov.dtype, device=dev))
        return (mu + (L @ torch.randn(P.D, count, dtype=cov.dtype, device=dev, generator=gen)).T).float()
    for i, c in enumerate(HARD):
        if (n[i] < 2).any(): out[P.CLASSES[c]] = dict(counts=n[i].tolist(), skipped=True); continue
        mu = s1[i] / n[i][:, None]; cov = s2[i] / n[i][:, None, None] - mu[:, :, None] * mu[:, None, :]
        tot = n[i].sum(); mu_all = s1[i].sum(0) / tot; cov_all = s2[i].sum(0) / tot - torch.outer(mu_all, mu_all)
        single = draw(mu_all, cov_all, 4000)
        k_urban = int(round(4000 * float(n[i, 1] / tot))); mix = torch.cat([draw(mu[0], cov[0], 4000 - k_urban), draw(mu[1], cov[1], k_urban)])
        acc = lambda x: float((R.logits(proxy, R.mask(x)).argmax(1) == c).float().mean())
        diff = mu[1] - mu[0]; pooled = (cov[0] * n[i, 0] + cov[1] * n[i, 1]) / tot
        maha = float(torch.sqrt(diff @ torch.linalg.solve(pooled + 1e-4 * torch.eye(P.D, dtype=pooled.dtype, device=dev), diff)))
        out[P.CLASSES[c]] = dict(counts=n[i].tolist(), single_accept=acc(single), mixture_accept=acc(mix), mode_mahalanobis=maha)
        log('P2', P.CLASSES[c], out[P.CLASSES[c]])
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--parquet', required=True)
    p.add_argument('--out', required=True); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    m = fd.Method(R); log('setup rows', sum(R.n), 'rules', R.rules)
    expected = json.loads((Path(a.weights) / 'expected.json').read_text()); result = dict(C={}, P3={}, smoke=a.smoke)
    for name, exp in expected.items():
        privates, proxy = load(Path(a.weights) / f'{name}.pt', R.dev)
        base = score_models(R, privates, ('head', 'gp', 'gpe'))
        if not a.smoke: assert abs(base['group/gp']['f1_macro'] - exp) <= .002, ('gp does not reproduce the audit', name, base['group/gp'], exp)
        result['C'][name] = dict(expected_gp=exp, **base); log('C', name, {k: round(v['f1_macro'], 4) for k, v in base.items()})
        recal = []
        for k, model in enumerate(privates):
            model = copy.deepcopy(model); calibrate(model, R.mask(R.X[k][R.cal[k]].float())); recal.append(model)
        result['P3'][name] = score_models(R, recal, ('gp', 'gpe')); log('P3', name, {k: round(v['f1_macro'], 4) for k, v in result['P3'][name].items()})
        if name == 'gpool_s42':
            g = torch.Generator(device=R.dev).manual_seed(0); labels = torch.arange(16 * 4000, device=R.dev) % 16
            x = R.mask(m.draw(labels, g, 0)).float(); hit = (R.logits(proxy, x).argmax(1) == labels).float()
            accept = [float(hit[labels == c].mean()) for c in range(16)]
            result['P1_coverage'] = dict(accept=accept, gate_fraction=sum(accept[c] for c in fd.P1_CLASSES) / 16)
            log('P1 coverage', [round(v, 3) for v in accept], 'gate', round(result['P1_coverage']['gate_fraction'], 4))
            features = json.loads((P.ROOT / 'knowledge/meta.json').read_text())['feature_cols']
            result['P2'] = p2(R, a.parquet, proxy, features, a.smoke, log)
    result['P4'] = p4(R, a.smoke, log)
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n'); log('done')


if __name__ == '__main__':
    main()
