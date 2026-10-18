"""Probe 19, no DAGSNet training (design §Probe 19). Probe 18: the round-15 private (gkd head 0,803) gains +0,045 from an oracle
16-class logit bias; the shared proxy gains +0,047..+0,066. Can such a bias be estimated with federated statistics?

Biases (16 numbers added to logits), scored on `group`:
  proxy  b_f1 (oracle macro-F1 coordinate ascent on `old`), b_ce (CE-optimal on `old`), b_bce (class-balanced CE on `old`); the
         CE fits are convex in b and federated exactly (each client sums gradient/Hessian over its own temporal holdout rows;
         `old` = the pooled temporal holdout of all clients, i.e. what secure aggregation returns)
  private head / gp; the three proxy biases transferred unchanged (z_k + b_proxy); per-private b_bce / b_f1 on pooled `old`
         (oracle: needs other clients' rows through private k); per-private b_bce on its own train rows (local only, 0 bytes)

    python fd_probe19.py --weights <dir with *.pt + expected.json> --out probe19.json [--smoke]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import fd
from fd_probe14 import C, SIZES, cm, f1, forward, oracle_bias
from fd_probe16 import models

RULE = {'m4b_s42': 'gp', 'frz_s42': 'head', 'gkd_s42': 'head'}


def ce_bias(z, y, balanced):
    """argmin_b of (class-balanced) CE of softmax(z + b) — convex; LBFGS full batch. b is centred (softmax is shift-invariant)."""
    w = 1 / torch.bincount(y, minlength=C).clamp_min(1).float() if balanced else torch.ones(C, device=z.device)
    b = torch.zeros(C, device=z.device, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=1, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = (w[y] * F.cross_entropy(z + b, y, reduction='none')).sum() / w[y].sum(); loss.backward(); return loss
    with torch.enable_grad(): opt.step(closure)
    return (b - b.mean()).detach()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    Z = SIZES['smoke' if a.smoke else 'full']; fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    dev, K = R.dev, R.K; log('setup rows', sum(R.n))
    g = torch.Generator(device='cpu').manual_seed(14)   # the probe-14 own-row sample
    idx = [torch.randperm(R.n[k], generator=g)[:Z['S']].to(dev) for k in range(K)]
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]; Xo, yo = R.mask(R.H['old'][0].float()), R.H['old'][1]
    gp = [R.log_pg - R.log_pi[k] for k in range(K)]; zero = torch.zeros(C, device=dev)
    score = lambda z, b: float(f1(cm(yg, (z + b).argmax(1))))
    expected = json.loads((Path(a.weights) / 'expected.json').read_text()); result = dict(smoke=a.smoke, sets={})
    for name, exp in expected.items():
        privates, proxy = models(Path(a.weights) / f'{name}.pt', dev)
        zpg, zpo = forward(proxy, Xg), forward(proxy, Xo)
        bp = dict(f1=oracle_bias(zpo, yo, zero)[0], ce=ce_bias(zpo, yo, False), bce=ce_bias(zpo, yo, True))
        out = dict(proxy={'none': score(zpg, zero), **{k: score(zpg, b) for k, b in bp.items()}},
                   proxy_bias={k: b.tolist() for k, b in bp.items()})
        per = {k: [] for k in ('gp', 'head', 'proxy_f1', 'proxy_ce', 'proxy_bce', 'own_bce', 'old_bce', 'old_f1')}
        for k, model in enumerate(privates):
            zg, zo = forward(model, Xg), forward(model, Xo); zs = forward(model, R.mask(R.X[k][idx[k]].float())); ys = R.Y[k][idx[k]]
            base = gp[k] if RULE[name] == 'gp' else zero
            per['gp'].append(score(zg, gp[k])); per['head'].append(score(zg, zero))
            for key, b in bp.items(): per[f'proxy_{key}'].append(score(zg, b))
            per['own_bce'].append(score(zg, ce_bias(zs, ys, True)))
            per['old_bce'].append(score(zg, ce_bias(zo, yo, True)))
            per['old_f1'].append(score(zg, oracle_bias(zo, yo, base)[0]))
        if not a.smoke: assert abs(np.mean(per['gp']) - exp) <= .002, (name, np.mean(per['gp']), exp)
        out.update(rule=RULE[name], mean={k: float(np.mean(v)) for k, v in per.items()}, worst={k: float(min(v)) for k, v in per.items()},
                   per_client=per)
        result['sets'][name] = out
        log(name, 'proxy', {k: round(v, 4) for k, v in out['proxy'].items()}, '| private', {k: round(v, 4) for k, v in out['mean'].items()})
        del privates, proxy; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n'); log('done')


if __name__ == '__main__':
    main()
