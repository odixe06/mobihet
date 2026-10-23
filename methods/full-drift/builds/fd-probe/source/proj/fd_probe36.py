"""P36: teacher (original vs Project48) x support (gaussian, copula, GMM) against the deployed method (original + gaussian) on two audited
seeds (s42 = xa, s43 = ya-v2), and GMM restricted to five rare attack classes, full or diagonal. P32 frozen-head harness.
Input-space supports do not depend on the weights and are built once with the P32/P35 seeds; s42 arms shared with P32/P35 must reproduce them.
Protocol: docs/research/methods/teacher-support-two-seed-2026-10-22.md. No official test.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, basis, decode_stats, encode_stats, fd, fit, forward, P, report, sha, virtual
from fd_probe24 import adapt
from fd_probe32 import copula_fit, copula_draw, summary
from fd_probe35 import gmm_draw, gmm_fit
from mh.metrics import metrics_from_confusion

S5 = ['positionMirroring', 'suddenStop', 'timeDelayAttack', 'constantPositionOffset', 'dataReplay']
ARMS = [('orig_gauss', 'original', 'gaussian'), ('orig_cop', 'original', 'copula'), ('p48_gauss', 'calibrated', 'gaussian'),
        ('p48_cop', 'calibrated', 'copula'), ('p48_gmm8', 'calibrated', 'gmm8'), ('p48_gmm8_att', 'calibrated', 'gmm8_att'),
        ('p48_gmm8d_att', 'calibrated', 'gmm8d_att'), ('p48_copgmm8d', 'calibrated', 'copgmm8d')]


def gmm_fit_diag(u, K, g, iters=100, tol=1e-4, reg=1e-4):
    """gmm_fit (P35) with diagonal covariances: same seeding, Lloyd steps and EM stopping rule."""
    n, d = u.shape; u = u.double()
    centers = u[torch.randint(n, (1,), device=u.device, generator=g)]
    for _ in range(K - 1):
        d2 = torch.cdist(u, centers).min(1).values.square()
        centers = torch.cat([centers, u[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(10):
        one = F.one_hot(torch.cdist(u, centers).argmin(1), K).double(); count = one.sum(0)
        centers = torch.where(count[:, None] > 0, one.T @ u / count.clamp_min(1)[:, None], centers)
    resp, prev = F.one_hot(torch.cdist(u, centers).argmin(1), K).double(), -math.inf
    for it in range(iters):
        nk = resp.sum(0) + 1e-10; weight = nk / n; mu = resp.T @ u / nk[:, None]
        var = (resp.T @ u.square() / nk[:, None] - mu.square()).clamp_min(0) + reg
        quad = u.square() @ (1 / var).T - 2 * u @ (mu / var).T + (mu.square() / var).sum(1)[None]
        logp = weight.log()[None] - .5 * (d * math.log(2 * math.pi) + var.log().sum(1)[None] + quad)
        ll = torch.logsumexp(logp, 1); resp = (logp - ll[:, None]).exp(); mean_ll = float(ll.mean())
        if mean_ll - prev < tol: break
        prev = mean_ll
    return dict(weight=weight, mu=mu, chol=torch.diag_embed(var.sqrt()), iters=it + 1, mean_ll=mean_ll)


def calibrate(weights, own_x, dev):
    """P32 teacher setup for one weights file: original snapshot and its Project48-calibrated head (same seeds)."""
    w = torch.load(weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48); total, own = 0, {}
    for k, (x, y) in enumerate(own_x):
        z, h, _ = forward(teacher, x, feats=True)
        values, _ = encode_stats(h, y, 'project48', proj); total = total + values; own[k] = (x, y, h.float(), z)
    n, mu, cov = decode_stats(total, 'project48', proj)
    bank_h, bank_y = virtual(mu, cov, n / n.sum(), 200000, 23)
    fc = torch.nn.Linear(48, C).to(dev)
    with torch.no_grad(): fc.weight.copy_(teacher.head[5].weight @ proj.float()); fc.bias.copy_(teacher.head[5].bias)
    fc = fit(fc, bank_h, bank_y, 3000, 23); del bank_h, bank_y
    calibrated = copy.deepcopy(teacher)
    with torch.no_grad(): calibrated.head[5].weight.copy_(fc.weight @ proj.float().T); calibrated.head[5].bias.copy_(fc.bias)
    calibrated.requires_grad_(False)
    return w, dict(original=teacher, calibrated=calibrated), own


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', nargs=2, required=True, help='s42 and s43 round_015.pt'); p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True); p.add_argument('--p32', required=True); p.add_argument('--p35', required=True)
    p.add_argument('--steps', type=int, default=1000); p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    p32, p35 = (json.loads(Path(f).read_text()) for f in (a.p32, a.p35))
    assert p32['weights_sha'] == sha(a.weights[0]) == p35['weights_sha']
    names = manifest['features']; zero = [names.index(n) for n in fd.BLOCK + fd.POSITION]
    active = [i for i in range(66) if i not in zero]; classes = manifest['classes']; s5 = [classes.index(c) for c in S5]
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    # Input-space pool and supports (independent of the weights): P32 rows (seed 22), gaussian (24), copula (32), gmm8 (P35, seed 53).
    gen = torch.Generator(device='cpu').manual_seed(22); own_x = []
    input_n, input_s, input_q = torch.zeros(C, device=dev, dtype=torch.float64), torch.zeros(C, 66, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); idx = torch.randperm(len(y), generator=gen)[:16384].to(dev); x, y = x[idx], y[idx]; own_x.append((x, y))
        one = F.one_hot(y, C).double(); input_n += one.sum(0); input_s += one.T @ x.double()
        for c in range(C): xx = x[y == c].double(); input_q[c] += xx.T @ xx
    im = input_s / input_n[:, None]
    ic = (input_q - input_n[:, None, None] * im[:, :, None] * im[:, None, :]) / (input_n[:, None, None] - 1)
    gaussian, labels = virtual(im, ic, torch.ones(C, device=dev) / C, a.bank, 24); gaussian[:, zero] = 0
    pool_x = torch.cat([x for x, _ in own_x]); pool_y = torch.cat([y for _, y in own_x])
    tables, _ = copula_fit(pool_x, pool_y, torch.tensor(active, device=dev)); copula = copula_draw(tables, labels, 32)
    u_pool = torch.asinh(pool_x[:, active]); lo, hi = u_pool.min(0).values, u_pool.max(0).values; cols = torch.tensor(active, device=dev)
    g = torch.Generator(device=dev).manual_seed(53); gmm8 = torch.zeros(a.bank, 66, device=dev); iters = {}
    for c in range(C):   # identical loop to P35 gmm8
        t = gmm_fit(u_pool[pool_y == c], 8, g); slots = torch.where(labels == c)[0]
        gmm8[slots[:, None], cols[None]] = torch.sinh(torch.maximum(torch.minimum(gmm_draw(t, len(slots), g), hi), lo)); iters[classes[c]] = t['iters']
    g = torch.Generator(device=dev).manual_seed(61); gmm8d = torch.zeros(a.bank, 66, device=dev); iters_d = {}
    for c in s5:
        t = gmm_fit_diag(u_pool[pool_y == c], 8, g); slots = torch.where(labels == c)[0]
        gmm8d[slots[:, None], cols[None]] = torch.sinh(torch.maximum(torch.minimum(gmm_draw(t, len(slots), g), hi), lo)); iters_d[classes[c]] = t['iters']
    att = torch.isin(labels, torch.tensor(s5, device=dev))[:, None]
    supports = dict(gaussian=gaussian, copula=copula, gmm8=gmm8, gmm8_att=torch.where(att, gmm8, gaussian),
                    gmm8d_att=torch.where(att, gmm8d, gaussian), copgmm8d=torch.where(att, gmm8d, copula))
    for s, x in supports.items():
        x[:, zero] = 0; assert torch.isfinite(x).all() and torch.isfinite(x.half()).all(), s
    log('supports built; gmm8 iters', iters, 'gmm8d iters', iters_d)
    # Byte accounting per client (fp32 unless stated); R1 budget from the audited xa total at 15 rounds.
    d, K, n5 = len(active), 8, len(s5)
    full_iter = 2 * 4 * n5 * K * (1 + d + d * (d + 1) // 2); diag_iter = 2 * 4 * n5 * K * (1 + 2 * d); lloyd = 10 * 2 * 4 * n5 * K * (d + 1)
    p48_round = 4 * (16 + 16 * 48 + 16 * 48 * 49 // 2) + 4 * (16 * 256 + 16)
    cop_compact = 2 * C * d * 256 + 4 * C * (d + d * (d + 1) // 2) + 4 * C * d * 257 + 4 * C * d * (d + 1) // 2
    xa15 = 5.18e6; budget = dict(r15=1.5 * xa15, r50=1.5 * xa15 * 50 / 15)
    bytes_ = dict(gmm8d_att=lloyd + diag_iter * max(iters_d.values()), gmm8_att=lloyd + full_iter * max(iters[c] for c in S5),
                  p48_per_round=p48_round, p48_15=14 * p48_round, p48_50=49 * p48_round, copula_compact=cop_compact)
    bytes_['stack_p48_cop'] = dict(r15=xa15 + bytes_['p48_15'] + cop_compact, r50=xa15 * 50 / 15 + bytes_['p48_50'] + cop_compact)
    bytes_['stack_p48_copgmm8d'] = {r: v + bytes_['gmm8d_att'] for r, v in bytes_['stack_p48_cop'].items()}
    bytes_['budget'] = budget
    evals = {split: read(split) for split in ('old', 'group')}
    global_counts = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(C)], device=dev)
    pg = ((global_counts + 1) / (global_counts.sum() + C)).log()
    out = dict(weights_sha=[sha(w) for w in a.weights], fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__),
               p32_sha=sha(a.p32), p35_sha=sha(a.p35), protocol='docs/research/methods/teacher-support-two-seed-2026-10-22.md',
               scope='frozen backbone and BN; two audited seeds; density fits on pooled rows (federated EM equals pooled EM given the same init); '
                     'no official test', args=vars(a), gmm_iters=dict(full=iters, diag=iters_d), bytes=bytes_, seeds={})
    predictions, heads = {}, {}; path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    for seed, weights in zip(('s42', 's43'), a.weights):
        w, teachers, own = calibrate(weights, own_x, dev)
        targets, diag = {}, {}
        for arm, t, s in ARMS:
            z = forward(teachers[t], supports[s]); targets[arm] = z.log_softmax(1)
            diag[arm] = dict(class_agreement=float((z.argmax(1) == labels).float().mean()))
        if seed == 's42':   # identical teachers and supports to P32/P35
            for arm, key in (('orig_gauss', 'original_gaussian'), ('p48_gauss', 'calibrated_gaussian'), ('p48_cop', 'calibrated_copula')):
                assert diag[arm]['class_agreement'] == p32['generation_diagnostics'][key]['class_agreement'], arm
            assert abs(diag['p48_gmm8']['class_agreement'] - p35['generation_diagnostics']['gmm8']['class_agreement']) == 0
        out['seeds'][seed] = dict(weights=str(weights), generation=diag, clients={})
        log(seed, 'teachers calibrated', {k: round(v['class_agreement'], 3) for k, v in diag.items()})
        for k in range(20):
            private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
            private.requires_grad_(False)
            x, y, ht, zt = own[k]; _, hp, _ = forward(private, x, feats=True)
            features = {split: (yy, forward(private, xx, feats=True)[1].float()) for split, (xx, yy) in evals.items()}
            counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
            shift = ((counts + 1) / (counts.sum() + C)).log() - pg
            proxy_ce = F.cross_entropy(zt, y).detach()
            syn_h = {s: forward(private, xx, feats=True)[1].float() for s, xx in supports.items()}
            rec = out['seeds'][seed]['clients'][k] = {}
            for arm, t, s in ARMS:
                with torch.no_grad(): real_target = teachers[t].head[5](ht).log_softmax(1)
                head = adapt(private.head[5], hp.float(), y, syn_h[s], targets[arm], real_target, proxy_ce, shift, a.steps, 2400 + k)
                rec[arm] = {}
                for split, (yy, hh) in features.items():
                    with torch.no_grad(): pred = head(hh).argmax(1)
                    rec[arm][split] = report(yy, pred); predictions[f'{seed}_{k}_{arm}_{split}'] = pred.cpu().numpy().astype('uint8')
                heads[f'{seed}_{k}_{arm}'] = {nm: v.detach().cpu() for nm, v in head.state_dict().items()}
            path.write_text(json.dumps(out, indent=1) + '\n')
            log(seed, 'client', k, {arm: round(rec[arm]['group']['metrics']['f1_macro'], 4) for arm, _, _ in ARMS})
            del private
        del w, teachers, own
    for split, (xx, yy) in evals.items(): predictions[split + '_y'] = yy.cpu().numpy().astype('uint8')
    np.savez_compressed(path.with_suffix('.npz'), **predictions); torch.save(heads, path.with_suffix('.pt'))
    saved = np.load(path.with_suffix('.npz')); checked = 0
    for seed, v in out['seeds'].items():
        for k, arms_k in v['clients'].items():
            for arm, splits in arms_k.items():
                for split, rep in splits.items():
                    yy, pred = saved[split + '_y'].astype('int64'), saved[f'{seed}_{k}_{arm}_{split}'].astype('int64')
                    matrix = np.bincount(yy * C + pred, minlength=C * C).reshape(C, C)
                    assert matrix.tolist() == rep['confusion'] and metrics_from_confusion(matrix) == rep['metrics']; checked += 1
    # Readings (locked).
    res = {seed: {arm: summary(v['clients'], arm) for arm, _, _ in ARMS} for seed, v in out['seeds'].items()}
    repro ={arm: res['s42'][arm]['mean'] for arm in ('orig_gauss', 'p48_gauss', 'p48_cop', 'p48_gmm8')}
    expected = dict(orig_gauss=p32['summary']['original_gaussian']['mean'], p48_gauss=p32['summary']['calibrated_gaussian']['mean'],
                    p48_cop=p32['summary']['calibrated_copula']['mean'], p48_gmm8=p35['summary']['gmm8']['mean'])
    reproduced = {arm: repro[arm] == expected[arm] for arm in repro}
    delta = lambda seed, arm, ref: {m: res[seed][arm][m] - res[seed][ref][m] for m in ('mean', 'benign', 'worst')}
    gate_a = {seed: delta(seed, 'p48_cop', 'orig_gauss') for seed in res}
    for v in gate_a.values(): v['pass'] = v['mean'] >= .005 and v['benign'] >= -.005 and v['worst'] >= -.010
    gate_b = {seed: delta(seed, 'p48_copgmm8d', 'p48_cop') for seed in res}
    for v in gate_b.values(): v['pass'] = v['mean'] >= .003 and v['benign'] >= -.005 and v['worst'] >= -.010
    readings = dict(reproduced_s42=reproduced, A=dict(per_seed=gate_a, pass_=all(v['pass'] for v in gate_a.values())),
                    B=dict(per_seed=gate_b, bytes_gmm8d=bytes_['gmm8d_att'], bytes_ok=bytes_['gmm8d_att'] <= 5e5,
                           pass_=all(v['pass'] for v in gate_b.values()) and bytes_['gmm8d_att'] <= 5e5),
                    vs_deployed={seed: {arm: delta(seed, arm, 'orig_gauss') for arm, _, _ in ARMS} for seed in res})
    pcf = lambda seed, arm, c: float(np.mean([v[arm]['group']['per_class'][c] for v in out['seeds'][seed]['clients'].values()]))
    per_class = {seed: {arm: {classes[c]: pcf(seed, arm, c) - pcf(seed, 'orig_gauss', c) for c in range(C)} for arm, _, _ in ARMS} for seed in res}
    out.update(summary=res, readings=readings, per_class_delta_vs_deployed=per_class, predictions_sha=sha(path.with_suffix('.npz')),
               heads_sha=sha(path.with_suffix('.pt')), audit_checks=checked, seconds=time.monotonic() - t0)
    path.write_text(json.dumps(out, indent=1) + '\n'); log('saved', str(path), 'audit checks', checked, json.dumps(readings['A']), json.dumps(readings['B']))


if __name__ == '__main__':
    main()
