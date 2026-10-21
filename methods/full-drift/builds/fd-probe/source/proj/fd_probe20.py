"""Probe 20, no DAGSNet training: where does the private still disagree with its frozen snapshot teacher, and do the gkd
Gaussian rows reach those places? Audited epoch-15 weights of the headline arm (q3_zl la_kdpx, s42), m4b mask throughout.
Teacher = the received global snapshot (weights 'proxy'[0]); private k = weights 'private'[k].

Row sources, each scored by every private and the teacher (argmax disagreement, per-row KL(teacher || private)):
  H   group holdout rows (unseen receivers). 'teachable' = private wrong and teacher right.
  X   real train rows of the other clients (oracle inputs, probe 15 cov_real); O = the client's own rows
  G   pooled-moment Gaussian rows as gkd draws them (class cycle, tables[0])
  Gk  G ranked by KL: top quarter of a 4x candidate pool (priority by disagreement, same budget)
  Ga  G moved by a few signed-gradient steps that raise KL(teacher || private), Mahalanobis budget to the class Gaussian
  Gm  mixup of two pooled-class draws, class pair uniform, lambda ~ U(0, 1)
  Gq  plain draws, class shares = the classes in the client's own-row disagreement pairs (local, 0 byte; 10 % uniform floor)
  Gp  mixup as Gm, pair (a, b) drawn from the client's own-row (teacher a, private b) disagreement histogram (10 % uniform floor)
  Gr  own real row of class a mixed with a pooled draw of b, (a, b) as Gp
  Gc  round-18 copula draws (pooled marginal histograms + normal-score correlation), class cycle
  Gn5 / Gn1  own real rows + 0.5 / 1.0 x pooled class-covariance noise (rows uniform); Gnp: 0.5, rows of the Gq priority classes
  Gs  own real rows, each visible feature replaced with probability RHO by a copula draw of the row's class;
      Gsp the same on the Gq priority-class rows; Gsg with Gaussian instead of copula values
  Gx3 / Gx5  own row of class a whose features are replaced w.p. 0.3 / 0.5 by those of an own row of class b, (a, b) from the
      own disagreement pairs; Gxu: pairs uniform over the class pairs the client holds (priority control)
Pair histogram = 16 x 16 counts of (teacher argmax, private argmax) over disagreeing rows. cos(src, H) compares where a source
disagrees with where the private loses holdout rows the teacher gets right. Diagnostic only: a utility claim needs training.

    python fd_probe20.py --weights <round_015.pt> --fixture <dir> --out probe20.json [--clients 0,1] [--smoke]
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import fd
from fd import P
from fd_probe14 import C, cm, f1, forward

POOL, STEPS, EPS, RHO = 4, 5, 0.25, 0.3   # Gk candidates per kept row; Ga signed steps, step in class sd; Gs swap rate


def kl_rows(zt, zs):
    return (F.softmax(zt, 1) * (F.log_softmax(zt, 1) - F.log_softmax(zs, 1))).sum(1)


def pairs(zt, zs, weight=None):
    pt, ps = zt.argmax(1), zs.argmax(1); w = (pt != ps).double() if weight is None else weight.double() * (pt != ps)
    return torch.zeros(C * C, dtype=torch.float64, device=zt.device).index_add_(0, pt * C + ps, w)


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


def summary(zt, zs, y=None):
    k = kl_rows(zt, zs); top = k.sort(descending=True).values; n = len(k)
    out = dict(rows=n, disagree=float((zt.argmax(1) != zs.argmax(1)).float().mean()), kl_mean=float(k.mean()),
               kl_p50=float(k.median()), kl_p90=float(top[n // 10]), kl_top10_share=float(top[:max(n // 10, 1)].sum() / k.sum().clamp_min(1e-12)))
    if y is not None:
        out['teacher_label'] = float((zt.argmax(1) == y).float().mean())
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True); p.add_argument('--clients', default=None); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    fixture = Path(a.fixture)
    fd.fixture_path = lambda args: fixture   # the local fixture (no test split: test files may alias old)
    fd.check_full_manifest = lambda *v: None
    fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    m = fd.Method(R); dev, K = R.dev, R.K; log('setup rows', sum(R.n))
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    make = lambda s: (lambda net: (net.load_state_dict(s, strict=True), net.eval())[1])(P.make('dagsnet').to(dev))
    teacher = make(w['proxy'][0])
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]; zgt = forward(teacher, Xg)
    gen = torch.Generator(device='cpu').manual_seed(20)
    per = 256 if a.smoke else 4096
    idx = [torch.randperm(R.n[k], generator=gen)[:per].to(dev) for k in range(K)]
    XS = torch.cat([R.mask(R.X[k][i].float()) for k, i in enumerate(idx)]); src = torch.cat([torch.full((len(i),), k, device=dev) for k, i in enumerate(idx)])
    zSt = forward(teacher, XS)
    # gkd draws: class cycle on the pooled table; Gk keeps the top quarter by KL of a POOL x larger draw (same budget as G)
    n_g = 512 if a.smoke else 16384; gg = torch.Generator(device=dev).manual_seed(200)
    lab = torch.arange(n_g, device=dev) % C; XG = R.mask(m.draw(lab, gg, 0)).float()
    lab4 = torch.arange(POOL * n_g, device=dev) % C; XG4 = R.mask(m.draw(lab4, gg, 0)).float()
    zGt, zG4t = forward(teacher, XG), forward(teacher, XG4)
    t = m.gaussian_state['tables'][0]; kept = torch.nonzero(~R.zero_cols).squeeze(1); keep = (~R.zero_cols).float()
    sd = (t['factor'].double() @ t['factor'].double().transpose(1, 2)).diagonal(dim1=1, dim2=2).sqrt().float()   # (C, D)
    sigma = (t['factor'].double() @ t['factor'].double().transpose(1, 2))[:, kept][:, :, kept]
    prec = torch.linalg.inv(sigma + 1e-4 * torch.eye(len(kept), dtype=sigma.dtype, device=dev)).float()
    maha = lambda x, y: torch.einsum('ni,nij,nj->n', (x - t['mu'][y].float())[:, kept], prec[y], (x - t['mu'][y].float())[:, kept]) / len(kept)
    clients = [int(c) for c in a.clients.split(',')] if a.clients else list(range(K))
    result = dict(weights=os.path.abspath(a.weights), fixture=str(fixture), smoke=a.smoke, pool=POOL, steps=STEPS, eps=EPS,
                  teacher_group_f1=float(f1(cm(yg, zgt.argmax(1)))), G_teacher_label=float((zGt.argmax(1) == lab).float().mean()),
                  G_maha=float(maha(XG, lab).mean()), clients={})
    SRC = ('H', 'O', 'X', 'G', 'Gk', 'Ga', 'Gm', 'Gq', 'Gp', 'Gr', 'Gc', 'Gn5', 'Gn1', 'Gnp', 'Gs', 'Gsp', 'Gsg', 'Gx3', 'Gx5', 'Gxu')
    m.make_copula(); XC = m.draw_copula(lab, gg)   # round 18 cop generator (pooled marginals + normal-score correlation)
    tot = {s: torch.zeros(C * C, dtype=torch.float64, device=dev) for s in SRC}
    YS = torch.cat([R.Y[k][i] for k, i in enumerate(idx)])
    for k in clients:
        private = make(w['private'][k])
        zg = forward(private, Xg); teach = (zg.argmax(1) != yg) & (zgt.argmax(1) == yg)
        hH = torch.zeros(C * C, dtype=torch.float64, device=dev).index_add_(0, yg[teach] * C + zg.argmax(1)[teach], torch.ones(int(teach.sum()), dtype=torch.float64, device=dev))
        zS = forward(private, XS); own = src == k
        zG, zG4 = forward(private, XG), forward(private, XG4)
        hO = pairs(zSt[own], zS[own]); wp = (0.9 * hO / hO.sum().clamp_min(1) + 0.1 / (C * C)).float()   # local priority
        gk = torch.Generator(device=dev).manual_seed(2000 + k)
        def mix(pa, pb, xa=None):
            xa = R.mask(m.draw(pa, gk, 0)).float() if xa is None else xa
            lam = torch.rand(len(pa), 1, device=dev, generator=gk)
            return lam * xa + (1 - lam) * R.mask(m.draw(pb, gk, 0)).float()
        u = torch.randint(0, C * C, (n_g,), device=dev, generator=gk); XM, lM = mix(u // C, u % C), u // C
        wc = (0.9 * (hO.view(C, C).sum(0) + hO.view(C, C).sum(1)) / (2 * hO.sum()).clamp_min(1) + 0.1 / C).float()
        lQ = torch.multinomial(wc, n_g, replacement=True, generator=gk); XQ = R.mask(m.draw(lQ, gk, 0)).float()
        v = torch.multinomial(wp, n_g, replacement=True, generator=gk); XP, lP = mix(v // C, v % C), v // C
        oy, order = YS[own].sort(); ox = XS[own][order]   # Gr: a random own row of class a (absent class: a pooled draw)
        count = torch.bincount(oy, minlength=C); start = count.cumsum(0) - count
        v2 = torch.multinomial(wp, n_g, replacement=True, generator=gk); a2, b2 = v2 // C, v2 % C
        pick = start[a2] + (torch.rand(n_g, device=dev, generator=gk) * count[a2]).long().clamp_max((count[a2] - 1).clamp_min(0))
        base = torch.where((count[a2] > 0)[:, None], ox[pick.clamp_max(len(ox) - 1)], R.mask(m.draw(a2, gk, 0)).float())
        XR = mix(a2, b2, base)
        # on-manifold anchors: own rows + sigma * pooled class-covariance noise (uniform rows / rows of the priority classes);
        # own rows with each visible feature replaced w.p. RHO by a pooled-copula draw of the row's class (Gs)
        def jitter(i, sigma):
            yy = oy[i]; z = torch.randn(len(i), P.D, device=dev, generator=gk)
            return R.mask(ox[i] + sigma * (t['factor'][yy] @ z[:, :, None]).squeeze(-1)), yy
        iu = torch.randint(0, len(oy), (n_g,), device=dev, generator=gk)
        ip = torch.multinomial(wc[oy] / count[oy].clamp_min(1).float(), n_g, replacement=True, generator=gk)
        (XN5, lN5), (XN1, lN1), (XNP, lNP) = jitter(iu, .5), jitter(iu, 1.), jitter(ip, .5)
        def swap(rho): return (torch.rand(n_g, P.D, device=dev, generator=gk) < rho) & ~R.zero_cols
        XSW = torch.where(swap(RHO), m.draw_copula(oy[iu], gk), ox[iu])
        XSP = torch.where(swap(RHO), m.draw_copula(oy[ip], gk), ox[ip])   # Gsp: Gs on the priority-class rows
        XSG = torch.where(swap(RHO), R.mask(m.draw(oy[iu], gk, 0)).float(), ox[iu])   # Gsg: Gs with Gaussian values
        # Gx: crossover of two own rows, class a <- features of class b w.p. rho; (a, b) from the own disagreement pairs
        # (pairs the client holds both classes of); Gxu: same rows, pairs uniform over the pairs it holds (no priority)
        held = (count > 0).float(); both = (held[:, None] * held[None, :]).flatten() * (1 - torch.eye(C, device=dev).flatten())
        def rows_of(c): return start[c] + (torch.rand(len(c), device=dev, generator=gk) * count[c]).long().clamp_max((count[c] - 1).clamp_min(0))
        def cross(weights, rho):
            v = torch.multinomial(weights, n_g, replacement=True, generator=gk); ca, cb = v // C, v % C
            return torch.where(swap(rho), ox[rows_of(cb)], ox[rows_of(ca)]), ca
        wpo = (hO.float() * both + 1e-3 * both)
        (XX3, lX3), (XX5, lX5), (XXU, lXU) = cross(wpo, .3), cross(wpo, .5), cross(both, .3)
        extra = {s: (x, l) for s, x, l in (('Gc', XC, lab), ('Gn5', XN5, lN5), ('Gn1', XN1, lN1), ('Gnp', XNP, lNP), ('Gs', XSW, oy[iu]),
                                           ('Gsp', XSP, oy[ip]), ('Gsg', XSG, oy[iu]), ('Gx3', XX3, lX3), ('Gx5', XX5, lX5), ('Gxu', XXU, lXU))}
        zM, zMt, zQ, zQt, zP, zPt, zR, zRt = (f(x) for x in (XM, XQ, XP, XR) for f in (lambda x: forward(private, x), lambda x: forward(teacher, x)))
        ze = {s: (forward(teacher, x), forward(private, x), l) for s, (x, l) in extra.items()}
        k4 = kl_rows(zG4t, zG4); top = k4.topk(n_g).indices
        x = XG.clone()   # Ga: signed ascent on KL(teacher || private), per-feature step EPS * class sd, clipped to 3 sd of the start
        for _ in range(STEPS):
            x.requires_grad_(True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = kl_rows(teacher(x)[0].float(), private(x)[0].float()).sum()
            g, = torch.autograd.grad(loss, x)
            x = (x.detach() + EPS * sd[lab] * g.sign() * keep)
            x = torch.minimum(torch.maximum(x, XG - 3 * sd[lab]), XG + 3 * sd[lab])
        zat, za = forward(teacher, x), forward(private, x)
        row = dict(head_group_f1=float(f1(cm(yg, zg.argmax(1)))), teachable=float(teach.float().mean()),
                   H=summary(zgt, zg, yg), O=summary(zSt[own], zS[own]), X=summary(zSt[~own], zS[~own]),
                   G=summary(zGt, zG, lab), Gk=summary(zG4t[top], zG4[top], lab4[top]), Ga=summary(zat, za, lab),
                   Gm=summary(zMt, zM, lM), Gq=summary(zQt, zQ, lQ), Gp=summary(zPt, zP, lP), Gr=summary(zRt, zR, a2),
                   Ga_maha=float(maha(x, lab).mean()), **{s: summary(*v) for s, v in ze.items()})
        hs = dict(H=hH, O=hO, X=pairs(zSt[~own], zS[~own]), G=pairs(zGt, zG), Gk=pairs(zG4t[top], zG4[top]), Ga=pairs(zat, za),
                  Gm=pairs(zMt, zM), Gq=pairs(zQt, zQ), Gp=pairs(zPt, zP), Gr=pairs(zRt, zR), **{s: pairs(v[0], v[1]) for s, v in ze.items()})
        row['cos_to_H'] = {s: cos(h, hH) for s, h in hs.items() if s != 'H'}
        row['top_pairs'] = {s: [(int(i) // C, int(i) % C, round(float(h[i] / h.sum().clamp_min(1)), 3)) for i in h.topk(5).indices] for s, h in hs.items()}
        for s, h in hs.items(): tot[s] += h / h.sum().clamp_min(1)
        result['clients'][k] = row
        log(k, 'F1', round(row['head_group_f1'], 4), {s: round(row[s]['disagree'], 4) for s in SRC},
            'cos', {s: round(v, 3) for s, v in row['cos_to_H'].items()})
        del private
        Path(a.out).write_text(json.dumps(result, indent=1) + '\n')
    rows = result['clients'].values()
    result['mean'] = {s: {f: float(np.mean([r[s][f] for r in rows])) for f in ('disagree', 'kl_mean', 'kl_top10_share')} for s in SRC}
    result['mean']['cos_to_H'] = {s: float(np.mean([r['cos_to_H'][s] for r in rows])) for s in SRC[1:]}
    result['mean']['teacher_label'] = {s: float(np.mean([r[s]['teacher_label'] for r in rows])) for s in SRC[3:]}
    result['mean']['Ga_maha'] = float(np.mean([r['Ga_maha'] for r in rows]))
    result['pooled_top_pairs'] = {s: [(int(i) // C, int(i) % C, round(float(h[i] / h.sum()), 3)) for i in h.topk(8).indices] for s, h in tot.items()}
    result['pooled_cos_to_H'] = {s: cos(h, tot['H']) for s, h in tot.items() if s != 'H'}
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n')
    log('mean', json.dumps(result['mean'])); log('pooled cos', result['pooled_cos_to_H']); log('pairs', result['pooled_top_pairs'])


if __name__ == '__main__':
    main()
