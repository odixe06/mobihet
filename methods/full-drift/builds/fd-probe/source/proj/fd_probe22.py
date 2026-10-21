"""Probe 22, no DAGSNet training: two candidate fixes after the round-24 FAIL, measured offline on audited ep15 weights (default
headline xa, q3_zl la_kdpx s42), m4b mask, local pilot fixture (train rows = its 2.5 % sample; group = the audited holdout).

A  pair reliability from own labels (0 byte). xo prioritised (teacher a, private b) disagreement pairs by mass; on group half of
   that mass (benign <-> timeDelay) is a coin flip for the teacher. Per client: rho_own(a, b) = P(y = a | teacher a, private b) on
   its own labelled rows vs rho_group(a, b) on group, and the share of the priority mass where the teacher is right on group under
   xo weights (mass) and reliability weights (mass * max(0, 2 rho_own - 1)).
B  teacher head calibration from secure-summable class statistics of the teacher's h256 features (LDA: class means + one shared
   within-class covariance, 16 x 256 + 256 x 257 / 2 floats per client; CCVR: one covariance per class). Virtual features
   N(mu_c, Sigma) with class shares of the pooled counts re-fit the last Linear from its trained weights. Reported: teacher F1 and
   per class on group (weak6: per-class covariance only for the 6 classes the teacher scores worst on pooled own rows, shared
   covariance for the rest), and the private's error mass the calibrated teacher gets right (new teachable share) vs the trained teacher.

    python fd_probe22.py --weights <round_015.pt> --fixture <dir> --out probe22.json [--smoke]
"""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import fd
from fd import P
from fd_probe14 import C, cm, forward

VIRTUAL, STEPS, BATCH, LR = 200000, 3000, 4096, 1e-3


def f1c(y, pred):
    c = cm(y, pred).double(); tp = c.diagonal()
    return 2 * tp / (c.sum(1) + c.sum(0)).clamp_min(1)


def calibrate(fc, mu, cov, share, gen, dev):
    """Re-fit Linear(256 -> 16) on virtual features; cov is (256, 256) shared or (16, 256, 256) per class."""
    fc = copy.deepcopy(fc).float().train(); opt = torch.optim.Adam(fc.parameters(), lr=LR)
    eye = 1e-4 * torch.eye(mu.shape[1], device=dev, dtype=torch.float64)
    L = torch.linalg.cholesky((cov + eye) if cov.dim() == 2 else cov + eye).float()
    for _ in range(STEPS):
        c = torch.multinomial(share, BATCH, replacement=True, generator=gen)
        z = torch.randn(BATCH, mu.shape[1], device=dev, generator=gen)
        x = mu[c] + (z @ L.T if L.dim() == 2 else (L[c] @ z[:, :, None]).squeeze(-1))
        loss = F.cross_entropy(fc(x), c); opt.zero_grad(); loss.backward(); opt.step()
    return fc.eval()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    fixture = Path(a.fixture)
    fd.fixture_path = lambda args: fixture
    fd.check_full_manifest = lambda *v: None
    fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    dev, K = R.dev, R.K
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    make = lambda s: (lambda net: (net.load_state_dict(s, strict=True), net.eval())[1])(P.make('dagsnet').to(dev))
    gen = torch.Generator(device='cpu').manual_seed(22); gdev = torch.Generator(device=dev).manual_seed(22)
    per = 256 if a.smoke else 16384
    idx = [torch.randperm(R.n[k], generator=gen)[:per].to(dev) for k in range(K)]
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]
    teacher = make(w['proxy'][0]); zgt, hgt, _ = forward(teacher, Xg, feats=True); tg = zgt.argmax(1)
    out = dict(weights=str(Path(a.weights).resolve()), fixture=str(fixture), per_client=per, virtual_steps=STEPS, A={}, B={})

    # A: pair reliability, own labels vs group
    rel_rows, xo_right, rw_right, xo_mass, rw_mass = [], 0., 0., 0., 0.
    stats = dict(n=torch.zeros(C, device=dev, dtype=torch.float64), s=torch.zeros(C, 256, device=dev, dtype=torch.float64),
                 ss=torch.zeros(C, 256, 256, device=dev, dtype=torch.float64), cm=torch.zeros(C, C, device=dev, dtype=torch.long))
    for k in range(K):
        xs, ys = R.mask(R.X[k][idx[k]].float()), R.Y[k][idx[k]]
        zst, hst, _ = forward(teacher, xs, feats=True); private = make(w['private'][k])
        ts, ps = zst.argmax(1), forward(private, xs).argmax(1); pg = forward(private, Xg).argmax(1)
        h = hst.double()   # B: secure-summable statistics of the teacher's features on own rows
        stats['cm'] += cm(ys, ts)   # teacher confusion on own rows (additive): picks the weak classes for 'weak6'
        stats['n'] += torch.bincount(ys, minlength=C).double(); stats['s'].index_add_(0, ys, h)
        for c in range(C): hc = h[ys == c]; stats['ss'][c] += hc.T @ hc
        def table(y_, t_, p_):
            d = t_ != p_; key = t_[d] * C + p_[d]
            tot = torch.bincount(key, minlength=C * C).double(); right = torch.bincount(key[y_[d] == t_[d]], minlength=C * C).double()
            return tot, right
        tot_o, right_o = table(ys, ts, ps); tot_g, right_g = table(yg, tg, pg)
        rho_o = right_o / tot_o.clamp_min(1); rho_g = right_g / tot_g.clamp_min(1)
        ok = (tot_o >= 20) & (tot_g >= 20)
        if ok.sum() >= 3: rel_rows.append(np.corrcoef(rho_o[ok].cpu().numpy(), rho_g[ok].cpu().numpy())[0, 1])
        # priority mass from own-row histogram (xo) vs reliability-weighted; utility proxy = teacher-right share on group pairs
        wx = tot_o / tot_o.sum(); wr = tot_o * (2 * rho_o - 1).clamp_min(0); wr = wr / wr.sum().clamp_min(1e-12)
        xo_right += float((wx * rho_g).sum()); rw_right += float((wr * rho_g).sum())
        top = torch.argsort(-tot_o)[:4]
        out['A'][k] = dict(own_disagree=float((ts != ps).float().mean()), corr_rho=rel_rows[-1] if ok.sum() >= 3 else None,
                           top_pairs=[(int(i) // C, int(i) % C, round(float(tot_o[i] / tot_o.sum()), 3), round(float(rho_o[i]), 3), round(float(rho_g[i]), 3)) for i in top],
                           xo_teacher_right_group=float((wx * rho_g).sum()), rel_teacher_right_group=float((wr * rho_g).sum()))
        del private
    out['A_mean'] = dict(corr_rho_own_group=float(np.nanmean(rel_rows)), xo_teacher_right_group=xo_right / K, rel_teacher_right_group=rw_right / K)
    log('A', out['A_mean'])

    # B: teacher last-layer calibration from pooled (secure-summed) class statistics
    n, mu = stats['n'], stats['s'] / stats['n'][:, None].clamp_min(1)
    within = stats['ss'] - stats['n'][:, None, None] * mu[:, :, None] * mu[:, None, :]
    shared = within.sum(0) / (n.sum() - C); per_class = within / (n[:, None, None] - 1).clamp_min(1)
    share = (n / n.sum()).float(); fc = teacher.head[5]
    base = f1c(yg, tg); out['B']['teacher'] = dict(f1=float(base.mean()), per_class=base.tolist())
    preds = {}
    c_ = stats['cm'].double(); own_f1 = 2 * c_.diagonal() / (c_.sum(0) + c_.sum(1)).clamp_min(1); weak = own_f1.argsort()[:6]
    weak6 = shared.expand(C, -1, -1).clone(); weak6[weak] = per_class[weak]   # own covariance for the 6 weakest, shared for the rest
    out['B']['weak6_classes'] = weak.tolist()
    for name, cov in (('lda', shared), ('ccvr', per_class), ('diag', torch.diag_embed(per_class.diagonal(dim1=1, dim2=2))), ('weak6', weak6)):
        new = calibrate(fc, mu.float(), cov, share, gdev, dev)
        with torch.no_grad(): preds[name] = new(hgt.float()).argmax(1)
        s = f1c(yg, preds[name]); out['B'][name] = dict(f1=float(s.mean()), per_class=s.tolist())
        log('B', name, round(float(s.mean()), 4), 'vs teacher', round(float(base.mean()), 4))
    out['B']['bytes_per_client_fp16'] = dict(lda=2 * (C * 256 + 256 * 257 // 2), ccvr=2 * C * (256 + 256 * 257 // 2),
                                             weak6=2 * (C * 256 + 7 * 256 * 257 // 2))
    # new teachable share of private errors under each teacher (privates 0-4)
    rows = []
    for k in range(5):
        pg = forward(make(w['private'][k]), Xg).argmax(1); wrong = pg != yg
        teachers = dict(teacher=tg, **preds)
        rows.append(dict(**{name: float(((t == yg) & wrong).sum() / wrong.sum()) for name, t in teachers.items()},
                         **{name + '_ceiling': float(f1c(yg, torch.where(wrong & (t == yg), yg, pg)).mean()) for name, t in teachers.items()},
                         private=float(f1c(yg, pg).mean())))
    out['B']['teachable_share_of_private_errors'] = {name: float(np.mean([r[name] for r in rows])) for name in rows[0]}   # + private F1 ceilings
    log('B teachable share', out['B']['teachable_share_of_private_errors'])
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
