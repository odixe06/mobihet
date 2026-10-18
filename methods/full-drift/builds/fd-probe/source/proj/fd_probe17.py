"""Probe 17, no DAGSNet training (design §Probe 17). Probe 16: the private decision layer refit on own rows + rows relabelled by
the proxy reaches 0,831 with real rows of other clients (oracle) but 0,767 with pooled-moment Gaussian rows. Can synthetic rows
built on the vehicle from what it already receives (pooled class moments + the aggregated proxy, incl. its averaged BN statistics)
close that gap? Input optimisation only; DAGSNet weights frozen.

Synthesis (all start from pooled-moment Gaussian draws, m4b columns kept at 0, proxy in eval mode, fp32, Adam on x):
  gauss   no refinement
  ce      CE(proxy(x), c) + 0.1 * Mahalanobis_c(x) / D
  di_lo   ce + 0.1 * BN statistic matching (batch mean/var of every BN input vs the proxy's averaged running stats)
  di_hi   ce + 1.0 * BN statistic matching
Batches of 4096 rows with labels drawn from pi_g (so batch statistics are comparable with the running statistics), 16 batches.
Head refit (frozen private backbone, last Linear, prior-corrected to pi_g) on own real rows + synthetic rows labelled by the proxy
(argmax, and soft KL); references: own only, + other clients' real rows labelled by the proxy (oracle), pooled real (oracle).
Realism: proxy acceptance of c, 1-NN distance to pooled real rows (vs held-out real rows), 1-NN label agreement.
Also C of probe 16 with the label order fixed (CCVR on the proxy's pooled 256-d class moments).

    python fd_probe17.py --weights <dir with *.pt + expected.json> --out probe17.json [--smoke]
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
from fd import P
from fd_probe14 import C, SIZES, cm, f1, fit_linear, forward
from fd_probe16 import models

SETS = ('m4b_s42', 'm4b_s43', 'm4b_e6')
SYNTH = dict(gauss=None, ce=(0.1, 0.0), di_lo=(0.1, 0.1), di_hi=(0.1, 1.0))   # (Mahalanobis weight, BN weight)
STEPS, LR, BATCH, BATCHES = 200, 0.05, 4096, 16


def fit_soft(h, target, y_hard, hg_shape):
    """Last Linear fit to soft targets (rows with a one-hot target behave as CE), LBFGS full batch, tiny L2."""
    x = torch.cat([h.float(), torch.ones(len(h), 1, device=h.device)], 1)
    W = torch.zeros(x.shape[1], C, device=h.device, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=200, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = -(target * F.log_softmax(x @ W, 1)).sum(1).mean() + 1e-4 * W[:-1].pow(2).sum(); loss.backward(); return loss
    with torch.enable_grad(): opt.step(closure)
    W = W.detach()
    return lambda f: torch.cat([f.float(), torch.ones(len(f), 1, device=f.device)], 1) @ W


def synthesize(R, m, proxy, kind, labels, gen, log):
    """Pooled-moment Gaussian draws for `labels`, optionally refined against the proxy (see module doc)."""
    x0 = R.mask(m.draw(labels, gen, 0)).float()
    if SYNTH[kind] is None: return x0
    w_maha, w_bn = SYNTH[kind]; t = m.gaussian_state['tables'][0]
    keep = (~R.zero_cols).float(); kept = torch.nonzero(~R.zero_cols).squeeze(1); mu = t['mu'][labels].float()
    sigma = (t['factor'].double() @ t['factor'].double().transpose(1, 2))[:, kept][:, :, kept]   # marginal on the 53 kept columns
    prec = torch.linalg.inv(sigma + 1e-4 * torch.eye(len(kept), dtype=sigma.dtype, device=sigma.device)).float()
    proxy.eval(); stats = []
    def hook(mod, inp, out):
        a = inp[0]; dims = [0] + list(range(2, a.dim()))
        stats.append(((a.mean(dims) - mod.running_mean).square().mean() + (a.var(dims, unbiased=False) - mod.running_var).square().mean()))
    hooks = [mod.register_forward_hook(hook) for mod in proxy.modules() if isinstance(mod, torch.nn.BatchNorm1d)] if w_bn else []
    x = x0.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=LR)
    for step in range(STEPS):   # Mahalanobis to the label's class Gaussian, marginal on the kept columns
        stats.clear(); opt.zero_grad()
        z = proxy(x * keep)[0]; loss = F.cross_entropy(z, labels)
        if w_maha:
            d = (x - mu)[:, kept]
            loss = loss + w_maha * torch.einsum('ni,nij,nj->n', d, prec[labels], d).mean() / len(kept)
        if w_bn: loss = loss + w_bn * torch.stack(stats).mean()
        loss.backward(); opt.step()
    for h in hooks: h.remove()
    return (x.detach() * keep)


def nn_dist(ref, q):
    rn = (ref * ref).sum(1); out_d, out_i = [], []
    for a in range(0, len(q), 2048):
        qq = q[a:a + 2048]; best = torch.full((len(qq),), float('inf'), device=q.device); bi = torch.zeros(len(qq), dtype=torch.long, device=q.device)
        for s in range(0, len(ref), 131072):
            d = rn[s:s + 131072][None] - 2 * qq @ ref[s:s + 131072].T + (qq * qq).sum(1, keepdim=True)
            v, i = d.min(1); better = v < best; best = torch.where(better, v, best); bi = torch.where(better, i + s, bi)
        out_d.append(best.clamp_min(0).sqrt()); out_i.append(bi)
    return torch.cat(out_d), torch.cat(out_i)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    global STEPS, BATCH, BATCHES
    if a.smoke: STEPS, BATCH, BATCHES = 3, 256, 2
    Z = SIZES['smoke' if a.smoke else 'full']; fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    m = fd.Method(R); dev, K = R.dev, R.K; log('setup rows', sum(R.n), 'factor shape', tuple(m.gaussian_state['tables'][0]['factor'].shape))
    g = torch.Generator(device='cpu').manual_seed(14)   # the probe-14 sample
    perm = [torch.randperm(R.n[k], generator=g).to(dev) for k in range(K)]; idx = [pm[:Z['S']] for pm in perm]
    XS = torch.cat([R.mask(R.X[k][i].float()) for k, i in enumerate(idx)]); YS = torch.cat([R.Y[k][i] for k, i in enumerate(idx)])
    src = torch.cat([torch.full((len(i),), k, device=dev) for k, i in enumerate(idx)])
    held = torch.cat([R.mask(R.X[k][pm[Z['S']:Z['S'] + 256]].float()) for k, pm in enumerate(perm)])   # real rows not in XS
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]
    gp = [R.log_pg - R.log_pi[k] for k in range(K)]
    prior_bias = lambda y: R.log_pg - ((torch.bincount(y, minlength=C).double() + 1) / (len(y) + C)).log().float()
    expected = json.loads((Path(a.weights) / 'expected.json').read_text()); result = dict(smoke=a.smoke, synth={}, sets={}, C={})
    base_d, _ = nn_dist(XS, held); result['real_nn_dist_median'] = float(base_d.median())
    for name in SETS:
        privates, proxy = models(Path(a.weights) / f'{name}.pt', dev); zSp = forward(proxy, XS)
        syn = {}
        for kind in SYNTH:
            gen = torch.Generator(device=dev).manual_seed(170); xs, ys = [], []
            for b in range(BATCHES):
                labels = torch.multinomial(R.log_pg.exp(), BATCH, replacement=True, generator=gen)
                xs.append(synthesize(R, m, proxy, kind, labels, gen, log)); ys.append(labels)
            x, y = torch.cat(xs), torch.cat(ys); zp = forward(proxy, x); assert torch.isfinite(zp).all(), kind
            d, i = nn_dist(XS, x)
            syn[kind] = (x, zp)
            result['synth'][f'{name}/{kind}'] = dict(accept=float((zp.argmax(1) == y).float().mean()),
                                                     accept_class=[float((zp.argmax(1) == y)[y == c].float().mean()) if (y == c).any() else None for c in range(C)],
                                                     nn_dist_median=float(d.median()), nn_label_agree=float((YS[i] == zp.argmax(1)).float().mean()),
                                                     proxy_label_counts=torch.bincount(zp.argmax(1), minlength=C).tolist())
            log('synth', name, kind, {k: v for k, v in result['synth'][f'{name}/{kind}'].items() if k not in ('accept_class', 'proxy_label_counts')})
        keys = ['gp', 'own', 'own_others_proxy', 'pool_real'] + [f'{k}_{lab}' for k in SYNTH for lab in ('hard', 'soft')]
        per = {k: [] for k in keys}
        for k, model in enumerate(privates):
            zg, hg, _ = forward(model, Xg, True); _, hs, _ = forward(model, XS, True); own = src == k; ho, yo = hs[own], YS[own]
            lin = lambda h, y: float(f1(cm(yg, (fit_linear(h, y)(hg) + prior_bias(y)).argmax(1))))
            per['gp'].append(float(f1(cm(yg, (zg + gp[k]).argmax(1))))); per['own'].append(lin(ho, yo))
            per['own_others_proxy'].append(lin(torch.cat([ho, hs[~own]]), torch.cat([yo, zSp[~own].argmax(1)])))
            per['pool_real'].append(lin(hs, YS))
            for kind, (x, zp) in syn.items():
                _, hx, _ = forward(model, x, True); lab = zp.argmax(1)
                per[f'{kind}_hard'].append(lin(torch.cat([ho, hx]), torch.cat([yo, lab])))
                target = torch.cat([F.one_hot(yo, C).float(), F.softmax(zp, 1)]); yy = torch.cat([yo, lab])
                per[f'{kind}_soft'].append(float(f1(cm(yg, (fit_soft(torch.cat([ho, hx]), target, yy, None)(hg) + prior_bias(yy)).argmax(1)))))
                del hx
            del hs, hg
            if k % 5 == 4 or a.smoke: log(name, k, {key: round(v[-1], 4) for key, v in per.items()})
        if not a.smoke: assert abs(np.mean(per['gp']) - expected[name]) <= .002, (name, np.mean(per['gp']), expected[name])
        result['sets'][name] = dict(mean={k: float(np.mean(v)) for k, v in per.items()}, worst={k: float(min(v)) for k, v in per.items()}, per_client=per)
        log(name, {k: round(v, 4) for k, v in result['sets'][name]['mean'].items()})
        if name != 'm4b_e6':   # C: CCVR on the proxy's pooled 256-d class moments, labels in block order (probe-16 bug fixed)
            zpg, hpg, _ = forward(proxy, Xg, True); _, hps, _ = forward(proxy, XS, True); h = hps.double()
            n = torch.bincount(YS, minlength=C).double(); mu = torch.zeros(C, h.shape[1], dtype=torch.float64, device=dev).index_add_(0, YS, h) / n[:, None]
            per_class = 40 if a.smoke else 4000; yv = torch.arange(C, device=dev).repeat_interleave(per_class)
            gv = torch.Generator(device=dev).manual_seed(161); out = dict(proxy=float(f1(cm(yg, zpg.argmax(1)))))
            for form in ('full', 'diag'):
                feats = []
                for c in range(C):
                    dd = h[YS == c] - mu[c]
                    cov = dd.T @ dd / max(len(dd) - 1, 1) if form == 'full' else torch.diag((dd * dd).sum(0) / max(len(dd) - 1, 1))
                    L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(len(cov), dtype=cov.dtype, device=dev))
                    feats.append((mu[c] + (L @ torch.randn(len(cov), per_class, dtype=cov.dtype, device=dev, generator=gv)).T).float())
                out[f'ccvr_{form}'] = float(f1(cm(yg, (fit_linear(torch.cat(feats), yv)(hpg) + prior_bias(yv)).argmax(1))))
            result['C'][name] = out; log('C', name, out)
        del privates, proxy, syn; torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(result, indent=1) + '\n')
    log('done')


if __name__ == '__main__':
    main()
