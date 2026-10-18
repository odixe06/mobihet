"""Probe 16, no DAGSNet training (design §Probe 16). Probe 14 showed the frozen private backbone already supports ~0,84 (last
Linear refit on pooled real rows) vs private gp 0,73: the decision layer is the bottleneck. Here:

A  backbone quality by training regime (epoch 6, s42): DML-m4b, DML-m4b without aggregation (own), Gaussian-m4b (no proxy),
   Gaussian-m4b own moments (no FL): last Linear / head MLP refit on the pooled sample vs own rows (probe 14 procedure).
B  which transmissible data fits the private's last Linear (frozen backbone; ep15 m4b / gpool s42), prior-corrected to pi_g:
   own real | + pooled-moment Gaussian (hard label) | + Gaussian kept where the proxy agrees | + Gaussian relabelled by the proxy |
   + other clients' real rows relabelled by the proxy (oracle inputs) | pooled real (oracle). Bias-only fits on Gaussian rows.
C  proxy head calibration from pooled class moments of the proxy's own 256-d features (CCVR-style; full / diagonal covariance).

    python fd_probe16.py --weights <dir with *.pt + expected.json> --out probe16.json [--smoke]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import fd
from fd import P
from fd_probe14 import C, SIZES, cm, f1, fit_linear, fit_mlp, forward, oracle_bias

A_SETS = ('dml_m4b_e6', 'dml_m4b_own_e6', 'gauss_m4b_e6', 'gauss_m4b_own_e6')
B_SETS = ('m4b_s42', 'gpool_s42')
PER_CLASS = 4000


def models(path, dev):
    w = torch.load(path, map_location=dev, weights_only=True)
    make = lambda s: (lambda m: (m.load_state_dict(s, strict=True), m.eval())[1])(P.make('dagsnet').to(dev))
    return [make(s) for s in w['private']], (make(w['proxy'][0]) if len(w['proxy']) == 1 else None)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    Z = SIZES['smoke' if a.smoke else 'full']; per_class = 40 if a.smoke else PER_CLASS; fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    m = fd.Method(R); dev, K = R.dev, R.K; log('setup rows', sum(R.n))
    g = torch.Generator(device='cpu').manual_seed(14)   # the probe-14 sample (same seed, same order of draws)
    perm = [torch.randperm(R.n[k], generator=g).to(dev) for k in range(K)]; idx = [pm[:Z['S']] for pm in perm]
    XS = torch.cat([R.mask(R.X[k][i].float()) for k, i in enumerate(idx)]); YS = torch.cat([R.Y[k][i] for k, i in enumerate(idx)])
    src = torch.cat([torch.full((len(i),), k, device=dev) for k, i in enumerate(idx)])
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]
    gen = torch.Generator(device=dev).manual_seed(16); yG = torch.arange(C * per_class, device=dev) % C
    XG = R.mask(m.draw(yG, gen, 0)).float(); assert not m.draw_bad
    gp = [R.log_pg - R.log_pi[k] for k in range(K)]
    prior_bias = lambda y: R.log_pg - ((torch.bincount(y, minlength=C).double() + 1) / (len(y) + C)).log().float()
    def linear(h, y, hg): return float(f1(cm(yg, (fit_linear(h, y)(hg) + prior_bias(y)).argmax(1))))
    expected = json.loads((Path(a.weights) / 'expected.json').read_text()); result = dict(smoke=a.smoke, A={}, B={}, C={})
    steps = Z['mlp_epochs'] * -(-len(YS) // 2048)

    for name in A_SETS:
        privates, _ = models(Path(a.weights) / f'{name}.pt', dev); per = {k: [] for k in ('gp', 'linear_pool', 'linear_own', 'mlp_pool')}
        for k, model in enumerate(privates):
            zg, hg, cg = forward(model, Xg, True); _, hs, cs = forward(model, XS, True); own = src == k
            per['gp'].append(float(f1(cm(yg, (zg + gp[k]).argmax(1)))))
            per['linear_pool'].append(linear(hs, YS, hg)); per['linear_own'].append(linear(hs[own], YS[own], hg))
            per['mlp_pool'].append(float(f1(cm(yg, (fit_mlp(cs, YS, steps, k)(cg) + prior_bias(YS)).argmax(1)))))
            del hs, cs, hg, cg
        if not a.smoke: assert abs(np.mean(per['gp']) - expected[name]) <= .002, (name, np.mean(per['gp']), expected[name])
        result['A'][name] = dict(mean={k: float(np.mean(v)) for k, v in per.items()}, worst={k: float(min(v)) for k, v in per.items()}, per_client=per)
        log('A', name, {k: round(v, 4) for k, v in result['A'][name]['mean'].items()}); del privates; torch.cuda.empty_cache()

    for name in B_SETS:
        privates, proxy = models(Path(a.weights) / f'{name}.pt', dev)
        zGp = forward(proxy, XG).argmax(1); acc = zGp == yG; zSp = forward(proxy, XS).argmax(1)
        keys = ('gp', 'own', 'own_gauss', 'own_gauss_kept', 'own_gauss_proxy', 'own_others_proxy', 'pool_real', 'bias_gauss', 'bias_gauss_kept')
        per = {k: [] for k in keys}
        for k, model in enumerate(privates):
            zg, hg, _ = forward(model, Xg, True); _, hs, _ = forward(model, XS, True); zG, hG, _ = forward(model, XG, True)
            own = src == k; ho, yo = hs[own], YS[own]
            per['gp'].append(float(f1(cm(yg, (zg + gp[k]).argmax(1)))))
            per['own'].append(linear(ho, yo, hg))
            per['own_gauss'].append(linear(torch.cat([ho, hG]), torch.cat([yo, yG]), hg))
            per['own_gauss_kept'].append(linear(torch.cat([ho, hG[acc]]), torch.cat([yo, yG[acc]]), hg))
            per['own_gauss_proxy'].append(linear(torch.cat([ho, hG]), torch.cat([yo, zGp]), hg))
            per['own_others_proxy'].append(linear(torch.cat([ho, hs[~own]]), torch.cat([yo, zSp[~own]]), hg))
            per['pool_real'].append(linear(hs, YS, hg))
            b, _ = oracle_bias(zG, yG, gp[k]); per['bias_gauss'].append(float(f1(cm(yg, (zg + b).argmax(1)))))
            b, _ = oracle_bias(zG[acc], yG[acc], gp[k]); per['bias_gauss_kept'].append(float(f1(cm(yg, (zg + b).argmax(1)))))
            del hs, hG, hg
            if k % 5 == 4 or a.smoke: log('B', name, k, {key: round(v[-1], 4) for key, v in per.items()})
        if not a.smoke: assert abs(np.mean(per['gp']) - expected[name]) <= .002, (name, np.mean(per['gp']), expected[name])
        result['B'][name] = dict(mean={k: float(np.mean(v)) for k, v in per.items()}, worst={k: float(min(v)) for k, v in per.items()},
                                 per_client=per, proxy_accept=[float(acc[yG == c].float().mean()) for c in range(C)])
        log('B', name, {k: round(v, 4) for k, v in result['B'][name]['mean'].items()})

        # C: proxy head from pooled class moments of its 256-d features (sums over the sample = what secure aggregation returns)
        zpg, hpg, _ = forward(proxy, Xg, True); _, hps, _ = forward(proxy, XS, True); h = hps.double()
        n = torch.bincount(YS, minlength=C).double(); mu = torch.zeros(C, h.shape[1], dtype=torch.float64, device=dev).index_add_(0, YS, h) / n[:, None]
        out = dict(proxy=float(f1(cm(yg, zpg.argmax(1)))), pool_real=linear(hps, YS, hpg))
        gv = torch.Generator(device=dev).manual_seed(161); yv = torch.arange(C * per_class, device=dev) % C
        for form in ('full', 'diag'):
            feats = []
            for c in range(C):
                d = h[YS == c] - mu[c]
                cov = d.T @ d / max(len(d) - 1, 1) if form == 'full' else torch.diag((d * d).sum(0) / max(len(d) - 1, 1))
                L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(len(cov), dtype=cov.dtype, device=dev))
                feats.append((mu[c] + (L @ torch.randn(len(cov), per_class, dtype=cov.dtype, device=dev, generator=gv)).T).float())
            out[f'ccvr_{form}'] = linear(torch.cat(feats), yv, hpg)
        result['C'][name] = out; log('C', name, {k: round(v, 4) for k, v in out.items()})
        del privates, proxy; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n'); log('done')


if __name__ == '__main__':
    main()
