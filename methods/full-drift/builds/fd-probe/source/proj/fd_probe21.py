"""Probe 21, no DAGSNet training: of the errors a private shares with its frozen teacher (the mass no KL-to-teacher synthetic
row can fix), which are knowledge-limited (labelled pooled real rows fix them on the same frozen backbone) and which are
feature-limited (nothing on this backbone fixes them)? Audited ep15 weights (default: headline xa, q3_zl la_kdpx s42), m4b mask,
local pilot fixture (train rows = its 2.5 % sample; group = the audited holdout).

Per model (teacher = 'proxy'[0], and private k), on group:
  head        the model as trained
  last        Linear(256 -> 16) re-fitted on its h256 features over pooled labelled train rows of all clients
  refit       the whole head (LN, 544 -> 256, ReLU, -> 16) re-trained from its own weights on the same rows
  bias        head logits + per-class bias fitted on group itself (upper bound of any prior fix; diagnostic only)
plus the per-class F1 of each, so a class whose F1 stays low under 'refit' is feature-limited on this backbone.

    python fd_probe21.py --weights <round_015.pt> --fixture <dir> --out probe21.json [--clients 0,1] [--smoke]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import fd
from fd import P
from fd_probe14 import C, cm, forward

EPOCHS, BATCH, LR = 8, 4096, 1e-3


def f1c(y, pred):
    c = cm(y, pred).double(); tp = c.diagonal()
    return 2 * tp / (c.sum(1) + c.sum(0)).clamp_min(1)


def fit(module, x, y, epochs, gen):
    module = module.float().train(); opt = torch.optim.Adam(module.parameters(), lr=LR)
    for _ in range(epochs):
        for i in torch.randperm(len(x), generator=gen, device='cpu').split(BATCH):
            i = i.to(x.device); loss = F.cross_entropy(module(x[i].float()), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return module.eval()


def bias_oracle(z, y, gen, steps=300):
    b = torch.zeros(C, device=z.device, requires_grad=True); opt = torch.optim.Adam([b], lr=0.05)
    for _ in range(steps):   # smooth macro-F1 surrogate is overkill here: CE on group, then report argmax F1
        loss = F.cross_entropy(z + b, y, weight=1 / torch.bincount(y, minlength=C).clamp_min(1).float()); opt.zero_grad(); loss.backward(); opt.step()
    return (z + b.detach()).argmax(1)


def report(name, model, Xs, ys, Xg, yg, gen):
    zg, hg, cg = forward(model, Xg, feats=True); zs, hs, cs = forward(model, Xs, feats=True)
    last = fit(torch.nn.Linear(256, C).to(Xg.device), hs, ys, EPOCHS * 2, gen)
    head = fit(__import__('copy').deepcopy(model.head), cs, ys, EPOCHS, gen)
    with torch.no_grad():
        preds = dict(head=zg.argmax(1), last=last(hg.float()).argmax(1), refit=head(cg.float()).argmax(1))
    preds['bias'] = bias_oracle(zg, yg, gen)
    return {k: f1c(yg, p).tolist() for k, p in preds.items()}, preds


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True); p.add_argument('--clients', default='0,1,2,3,4'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    fixture = Path(a.fixture)
    fd.fixture_path = lambda args: fixture   # the local fixture (no test split: test files may alias old)
    fd.check_full_manifest = lambda *v: None
    fd.deterministic(); torch.backends.cudnn.deterministic = True
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    dev, K = R.dev, R.K
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    make = lambda s: (lambda net: (net.load_state_dict(s, strict=True), net.eval())[1])(P.make('dagsnet').to(dev))
    gen = torch.Generator(device='cpu').manual_seed(21)
    per = 256 if a.smoke else 16384
    idx = [torch.randperm(R.n[k], generator=gen)[:per].to(dev) for k in range(K)]
    Xs = torch.cat([R.mask(R.X[k][i].float()) for k, i in enumerate(idx)]); ys = torch.cat([R.Y[k][i] for k, i in enumerate(idx)])
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]
    log('pooled train rows', len(ys), 'group rows', len(yg))
    out = dict(weights=str(Path(a.weights).resolve()), fixture=str(fixture), pooled_rows=len(ys), epochs=EPOCHS, models={})
    teacher = make(w['proxy'][0]); res, tp = report('teacher', teacher, Xs, ys, Xg, yg, gen); out['models']['teacher'] = res
    log('teacher', {k: round(sum(v) / C, 4) for k, v in res.items()})
    for k in [int(c) for c in a.clients.split(',')]:
        res, pp = report(f'private{k}', make(w['private'][k]), Xs, ys, Xg, yg, gen)
        shared = (pp['head'] != yg) & (tp['head'] != yg)
        res['shared_fixed_by_teacher_refit'] = float(((tp['refit'] == yg) & shared).sum() / shared.sum().clamp_min(1))
        res['shared_fixed_by_own_refit'] = float(((pp['refit'] == yg) & shared).sum() / shared.sum().clamp_min(1))
        out['models'][f'private{k}'] = res
        log(f'private{k}', {key: round(sum(v) / C, 4) for key, v in res.items() if isinstance(v, list)},
            'shared fixed by teacher refit', round(res['shared_fixed_by_teacher_refit'], 3))
        Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
