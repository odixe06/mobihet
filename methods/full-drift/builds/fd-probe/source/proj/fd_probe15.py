"""Probe 15 (design §Probe 15): one more local epoch (epoch 16 of the locked T50 cosine LR) from the audited epoch-15
`fd_dml_m4b` s42 state, per client, under controlled private objectives. Separates forgetting caused by the local objective
from missing input coverage. Measurement only (oracle rows / proxy initialisation are not methods).

cont         ep15 private, DML exactly (proxy copy co-trains; KD private <- proxy copy, adaptive weight 1/(CE+CE))
kdfix        ep15 private, CE + KD from the frozen global proxy on local rows                         (control of cov_*)
cov_real     kdfix + KD from the frozen proxy on 256 real rows/batch of the OTHER clients (oracle inputs, no labels)
cov_gauss    kdfix + KD from the frozen proxy on 256 pooled-moment Gaussian rows/batch (gpool draw, class cycle)
pinit        private := global proxy, then DML exactly
pinit_ce     private := global proxy, CE on local rows only
pinit_kdfix  private := global proxy, CE + KD from the frozen proxy on local rows
pinit_kdonly private := global proxy, KD only from the frozen proxy on local rows

Private gp / head macro F1 on `group`, agreement with the frozen proxy on `group`, accuracy on own rows, at fixed steps.

    python fd_probe15.py --weights <dir with m4b_s42.pt> --clients 0,5 --variants cont,kdfix --out probe15_0.json [--smoke]
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
from fd_probe14 import cm, f1, forward

VARIANTS = ('cont', 'kdfix', 'cov_real', 'cov_gauss', 'pinit', 'pinit_ce', 'pinit_kdfix', 'pinit_kdonly')
CHECK = (0, 64, 256, 1024, 2048)
EPOCH = 16   # the epoch after the audited weights, on the locked 50-epoch cosine schedule
EXTRA = 256


def kl(student, teacher):
    return F.kl_div(F.log_softmax(student, 1), F.log_softmax(teacher, 1).detach(), log_target=True, reduction='batchmean')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--clients', required=True); p.add_argument('--variants', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); clients = [int(c) for c in a.clients.split(',')]; variants = a.variants.split(',')
    assert set(variants) <= set(VARIANTS), variants
    t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_gpool', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=EXTRA,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    m = fd.Method(R); dev = R.dev; log('setup rows', sum(R.n), 'lr', R.lr(EPOCH))
    w = torch.load(Path(a.weights) / 'm4b_s42.pt', map_location=dev, weights_only=True)
    make = lambda s: (lambda net: (net.load_state_dict(s, strict=True), net)[1])(P.make('dagsnet').to(dev))
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]
    proxy0 = make(w['proxy'][0]).eval(); ref = forward(proxy0, Xg).argmax(1)
    result = dict(smoke=a.smoke, lr=R.lr(EPOCH), clients=clients, variants=variants, proxy_group=float(f1(cm(yg, ref))), runs={})
    g0 = torch.Generator(device='cpu').manual_seed(15)
    for k in clients:
        own = torch.randperm(R.n[k], generator=g0)[:4096].to(dev); Xown, yown = R.mask(R.X[k][own].float()), R.Y[k][own]
        others = [j for j in range(R.K) if j != k]; per = 512 if a.smoke else 50000
        pool = torch.cat([R.mask(R.X[j][torch.randperm(R.n[j], generator=g0)[:per].to(dev)].float()) for j in others])
        bias = R.log_pg - R.log_pi[k]
        def evaluate(net, step):
            z = forward(net, Xg); pred = (z + bias).argmax(1)
            return dict(step=step, gp=float(f1(cm(yg, pred))), head=float(f1(cm(yg, z.argmax(1)))),
                        agree=float((pred == ref).float().mean()), own_acc=float((forward(net, Xown).argmax(1) == yown).float().mean()))
        for v in variants:
            started = time.time()
            private = make(w['proxy'][0] if v.startswith('pinit') else w['private'][k]).train()
            teacher = make(w['proxy'][0]); dml = v in ('cont', 'pinit'); teacher.train(dml)
            models = [private] + ([teacher] if dml else [])
            opt = torch.optim.AdamW([q for net in models for q in net.parameters()], lr=R.lr(EPOCH), weight_decay=0.)
            scaler = torch.amp.GradScaler(); g = R.gen(EPOCH, k, 0)
            torch.manual_seed(int(np.random.SeedSequence([42, EPOCH, k, 7]).generate_state(1)[0]))
            order = torch.randperm(R.n[k], device=dev, generator=g); gx = R.gen(EPOCH, k, 15)
            labels = torch.arange(EXTRA, device=dev) % 16
            curve = [evaluate(private, 0)]; private.train(); step = 0; skips = 0
            for start in range(0, len(order), 512):
                index = order[start:start + 512]; x, y = R.mask(R.X[k][index].float()), R.Y[k][index]
                cov = v in ('cov_real', 'cov_gauss'); xe = None
                if cov:   # one private forward over [local, extra] rows, as gpool trains (BN sees both)
                    xe = pool[torch.randint(0, len(pool), (EXTRA,), device=dev, generator=gx)] if v == 'cov_real' \
                        else R.mask(m.draw(labels, gx, k))
                with torch.autocast('cuda', dtype=torch.float16):
                    zall = private(torch.cat([x, xe]) if cov else x)[0].float(); zs = zall[:len(y)]; ce = F.cross_entropy(zs, y)
                    if dml: zr = teacher(x)[0].float()
                    else:
                        with torch.no_grad(): zr = teacher(x)[0].float()
                    pce = F.cross_entropy(zr, y); weight = 1 / (ce + pce).detach().clamp_min(1e-6); kd = kl(zs, zr)
                    if v in ('cont', 'pinit'): loss = ce + weight * kd + pce + weight * kl(zr, zs.detach())
                    elif v == 'pinit_ce': loss = ce
                    elif v == 'pinit_kdonly': loss = weight * kd
                    else: loss = ce + weight * kd
                    if cov:
                        with torch.no_grad(): zt = teacher(xe)[0].float()
                        loss = loss + weight * kl(zall[len(y):], zt)
                assert torch.isfinite(loss), (k, v, step)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                for net in models: torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                before = scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                skips += scaler.get_scale() < before; step += 1
                if step in CHECK: curve.append(evaluate(private, step)); private.train()
            curve.append(evaluate(private, step)); assert not m.draw_bad
            z = forward(private, Xg); final_cm = cm(yg, (z + bias).argmax(1)).cpu().numpy().tolist()
            result['runs'][f'{v}/{k}'] = dict(curve=curve, steps=step, skipped=int(skips), seconds=time.time() - started, group_cm_gp=final_cm)
            log(v, 'client', k, 'steps', step, [(c['step'], round(c['gp'], 4), round(c['agree'], 3), round(c['own_acc'], 3)) for c in curve])
            Path(a.out).write_text(json.dumps(result, indent=1) + '\n')
        del pool; torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
