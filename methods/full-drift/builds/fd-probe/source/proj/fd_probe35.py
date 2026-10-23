"""P35: which class's samples carry the real-support gain, how far from real rows the head tolerates, and whether a learnable density
model (GMM per class, class-conditional MADE with Gaussian mixtures) reaches the real-support oracle. P32 frozen-head harness, Project48.
Real and copula banks are regenerated with the P32 seeds (checked against P32); Gaussian client 0 must reproduce P32. Density models are fit
on the pooled rows (R3-ineligible capacity oracles) in asinh(z) space; a memorization check compares 1-NN distances with unpooled real rows.
Protocol: docs/research/methods/support-capacity-oracle-2026-10-22.md. No official test.
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
from fd_probe33 import geometry, per_class_rows
from mh.metrics import metrics_from_confusion

ARMS = ['real_cPS', 'cop_rPS', 'real_cB', 'cop_rB', 'real_j05', 'real_j20', 'gmm8', 'gmm32', 'made']
PS = ['positionMirroring', 'suddenStop']


def gmm_fit(u, K, g, iters=100, tol=1e-4, reg=1e-4):
    """Full-covariance GMM by EM (float64): k-means++ seeding, 10 Lloyd steps, then EM. Matmul-only M-step (deterministic on CUDA)."""
    n, d = u.shape; u = u.double()
    centers = u[torch.randint(n, (1,), device=u.device, generator=g)]
    for _ in range(K - 1):
        d2 = torch.cdist(u, centers).min(1).values.square()
        centers = torch.cat([centers, u[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(10):
        one = F.one_hot(torch.cdist(u, centers).argmin(1), K).double(); count = one.sum(0)
        centers = torch.where(count[:, None] > 0, one.T @ u / count.clamp_min(1)[:, None], centers)
    resp, prev, eye = F.one_hot(torch.cdist(u, centers).argmin(1), K).double(), -math.inf, torch.eye(d, device=u.device, dtype=torch.float64)
    for it in range(iters):
        nk = resp.sum(0) + 1e-10; weight = nk / n; mu = resp.T @ u / nk[:, None]
        logp = torch.empty(n, K, device=u.device, dtype=torch.float64); chol = []
        for k in range(K):
            diff = u - mu[k]; cov = (resp[:, k, None] * diff).T @ diff / nk[k] + reg * eye
            L = torch.linalg.cholesky((cov + cov.T) / 2); chol.append(L)
            mahal = torch.linalg.solve_triangular(L, diff.T, upper=False).square().sum(0)
            logp[:, k] = weight[k].log() - .5 * (d * math.log(2 * math.pi) + 2 * L.diagonal().log().sum() + mahal)
        ll = torch.logsumexp(logp, 1); resp = (logp - ll[:, None]).exp(); mean_ll = float(ll.mean())
        if mean_ll - prev < tol: break
        prev = mean_ll
    return dict(weight=weight, mu=mu, chol=torch.stack(chol), iters=it + 1, mean_ll=mean_ll)


def gmm_draw(t, n, g):
    comp = torch.multinomial(t['weight'], n, replacement=True, generator=g)
    z = torch.randn(n, t['mu'].shape[1], device=comp.device, generator=g, dtype=torch.float64)
    return (t['mu'][comp] + (t['chol'][comp] @ z[:, :, None]).squeeze(-1)).float()


class MaskedLinear(torch.nn.Linear):
    def __init__(self, i, o, mask):
        super().__init__(i, o); self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(torch.nn.Module):
    """Class-conditional MADE (Germain et al. 2015) with a mixture of Gaussians per dimension; class enters every hidden layer unmasked."""
    def __init__(self, d, hidden=512, mix=10):
        super().__init__(); self.d, self.mix = d, mix
        deg_in = torch.arange(1, d + 1); deg_h = torch.arange(hidden) % (d - 1) + 1; deg_out = deg_in.repeat_interleave(3 * mix)
        self.l1 = MaskedLinear(d, hidden, (deg_h[:, None] >= deg_in[None]).float())
        self.l2 = MaskedLinear(hidden, hidden, (deg_h[:, None] >= deg_h[None]).float())
        self.l3 = MaskedLinear(hidden, d * 3 * mix, (deg_out[:, None] > deg_h[None]).float())
        self.c1, self.c2 = torch.nn.Linear(C, hidden, bias=False), torch.nn.Linear(C, hidden, bias=False)   # one-hot matmul, no index_add

    def params(self, u, y):
        one = F.one_hot(y, C).float()
        h = F.relu(self.l1(u) + self.c1(one)); h = F.relu(self.l2(h) + self.c2(one))
        o = self.l3(h).view(len(u), self.d, 3, self.mix)
        return o[:, :, 0].log_softmax(-1), o[:, :, 1], o[:, :, 2].clamp(-6, 3)

    def nll(self, u, y):
        logit, mean, logs = self.params(u, y)
        logp = logit - logs - .5 * math.log(2 * math.pi) - .5 * ((u[:, :, None] - mean) / logs.exp()).square()
        return -torch.logsumexp(logp, -1).sum(1).mean()

    @torch.no_grad()
    def sample(self, y, g):
        u = torch.zeros(len(y), self.d, device=y.device)
        for j in range(self.d):
            logit, mean, logs = self.params(u, y)
            comp = torch.multinomial(logit[:, j].exp(), 1, generator=g).squeeze(1)
            pick = lambda t: t[:, j].gather(1, comp[:, None]).squeeze(1)
            u[:, j] = pick(mean) + pick(logs).exp() * torch.randn(len(y), device=y.device, generator=g)
        return u


def made_fit(u, y, held_u, held_y, seed, epochs=30, batch=2048, log=print):
    torch.manual_seed(seed); g = torch.Generator(device=u.device).manual_seed(seed)
    model = MADE(u.shape[1]).to(u.device); opt = torch.optim.Adam(model.parameters(), lr=1e-3); trace = []
    for epoch in range(epochs):
        model.train(); order = torch.randperm(len(u), device=u.device, generator=g)
        for s in range(0, len(u), batch):
            i = order[s:s + batch]; loss = model.nll(u[i], y[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): trace.append(dict(epoch=epoch + 1, train_batch_nll=float(loss), held_nll=float(model.nll(held_u, held_y))))
        if (epoch + 1) % 10 == 0: log('made', trace[-1])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model.eval(), trace


def nn_distance(q, ref, chunk=32768):
    best = torch.full((len(q),), math.inf, device=q.device)
    for s in range(0, len(ref), chunk): best = torch.minimum(best, torch.cdist(q, ref[s:s + chunk]).min(1).values)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--p32', required=True); p.add_argument('--p33b', required=True); p.add_argument('--p34', required=True)
    p.add_argument('--clients', default=','.join(map(str, range(20)))); p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    p32, p33b, p34 = (json.loads(Path(f).read_text()) for f in (a.p32, a.p33b, a.p34))
    assert p32['weights_sha'] == sha(a.weights) == p33b['weights_sha'] == p34['weights_sha']
    saved = np.load(Path(a.p32).with_suffix('.npz'))
    names = manifest['features']; zero = [names.index(n) for n in fd.BLOCK + fd.POSITION]
    active = [i for i in range(66) if i not in zero]; classes = manifest['classes']
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48)
    # P32 setup, unchanged; also keep each client's unsampled rows for the memorization check.
    gen = torch.Generator(device='cpu').manual_seed(22)
    own, rest, total, input_n, input_s, input_q = {}, [], 0, torch.zeros(C, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, device=dev, dtype=torch.float64), torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); perm = torch.randperm(len(y), generator=gen); idx = perm[:16384].to(dev)
        rest.append((x[perm[16384:].to(dev)], y[perm[16384:].to(dev)])); x, y = x[idx], y[idx]
        z, h, _ = forward(teacher, x, feats=True)
        values, _ = encode_stats(h, y, 'project48', proj); total = total + values
        own[k] = (x, y, h.float(), z)
        one = F.one_hot(y, C).double(); input_n += one.sum(0); input_s += one.T @ x.double()
        for c in range(C): xx = x[y == c].double(); input_q[c] += xx.T @ xx
    n, mu, cov = decode_stats(total, 'project48', proj)
    bank_h, bank_y = virtual(mu, cov, n / n.sum(), 200000, 23)
    fc = torch.nn.Linear(48, C).to(dev)
    with torch.no_grad(): fc.weight.copy_(teacher.head[5].weight @ proj.float()); fc.bias.copy_(teacher.head[5].bias)
    fc = fit(fc, bank_h, bank_y, 3000, 23); del bank_h, bank_y
    calibrated = copy.deepcopy(teacher)
    with torch.no_grad(): calibrated.head[5].weight.copy_(fc.weight @ proj.float().T); calibrated.head[5].bias.copy_(fc.bias)
    calibrated.requires_grad_(False)
    im = input_s / input_n[:, None]
    ic = (input_q - input_n[:, None, None] * im[:, :, None] * im[:, None, :]) / (input_n[:, None, None] - 1)
    gaussian, labels = virtual(im, ic, torch.ones(C, device=dev) / C, a.bank, 24); gaussian[:, zero] = 0
    pool_x = torch.cat([own[k][0] for k in range(20)]); pool_y = torch.cat([own[k][1] for k in range(20)])
    rng = np.random.default_rng(27); pool_lab = pool_y.cpu().numpy(); lab = labels.cpu().numpy(); pick = np.empty(len(lab), dtype='int64')
    for c in range(C):
        slots = np.where(lab == c)[0]; pick[slots] = rng.choice(np.where(pool_lab == c)[0], len(slots), replace=True)
    real = pool_x[torch.from_numpy(pick).to(dev)].clone()
    tables, _ = copula_fit(pool_x, pool_y, torch.tensor(active, device=dev)); copula = copula_draw(tables, labels, 32)
    for s, x in (('real', real), ('copula', copula)):
        agree = float((forward(calibrated, x).argmax(1) == labels).float().mean())
        assert agree == p32['generation_diagnostics']['calibrated_' + s]['class_agreement'], (s, agree)
    log('real and copula banks match P32')
    # Supports.
    supports, ps = {}, torch.isin(labels, torch.tensor([classes.index(c) for c in PS], device=dev)); benign = labels == 0
    supports['real_cPS'] = torch.where(ps[:, None], copula, real); supports['cop_rPS'] = torch.where(ps[:, None], real, copula)
    supports['real_cB'] = torch.where(benign[:, None], copula, real); supports['cop_rB'] = torch.where(benign[:, None], real, copula)
    for s, sigma, seed in (('real_j05', .05, 51), ('real_j20', .20, 52)):
        g = torch.Generator(device=dev).manual_seed(seed); x = real.clone()
        x[:, active] += sigma * torch.randn(len(x), len(active), device=dev, generator=g); supports[s] = x
    u_pool = torch.asinh(pool_x[:, active])
    rest_x = torch.cat([r[0] for r in rest]); rest_y = torch.cat([r[1] for r in rest])
    held = per_class_rows(rest_y.cpu().numpy(), np.arange(len(rest_y)), 57); held = torch.from_numpy(np.concatenate(held)).to(dev)
    held_u, held_y = torch.asinh(rest_x[held][:, active]), rest_y[held]
    lo, hi = u_pool.min(0).values, u_pool.max(0).values   # samples bounded to the pooled per-feature range (asinh space)
    fits = {}
    for s, K, seed in (('gmm8', 8, 53), ('gmm32', 32, 54)):
        g = torch.Generator(device=dev).manual_seed(seed); x = torch.zeros(a.bank, 66, device=dev); info = []
        for c in range(C):
            t = gmm_fit(u_pool[pool_y == c], K, g); slots = torch.where(labels == c)[0]
            x[slots[:, None], torch.tensor(active, device=dev)[None]] = torch.sinh(torch.maximum(torch.minimum(gmm_draw(t, len(slots), g), hi), lo))
            info.append(dict(iters=t['iters'], mean_ll=t['mean_ll']))
        supports[s] = x; fits[s] = info; log(s, 'fit', [i['iters'] for i in info])
    model, trace = made_fit(u_pool, pool_y, held_u, held_y, 55, log=log)
    x = torch.zeros(a.bank, 66, device=dev); x[:, active] = torch.sinh(torch.maximum(torch.minimum(model.sample(labels, torch.Generator(device=dev).manual_seed(55)), hi), lo))
    supports['made'] = x; fits['made'] = trace; del model
    for s, x in supports.items():
        x[:, zero] = 0; assert torch.isfinite(x).all() and torch.isfinite(x.half()).all(), s
    # Memorization check in asinh(z) space: samples vs unpooled real rows, both against the pool.
    held_nn = float(nn_distance(held_u, u_pool).median()); copy_check = dict(heldout_median=held_nn)
    for s in ('gmm8', 'gmm32', 'made', 'real_j05', 'real_j20'):
        m = float(nn_distance(torch.asinh(supports[s][:, active]), u_pool).median())
        copy_check[s] = dict(median=m, ratio=m / held_nn, copy=m / held_nn < .8)
    log('memorization', copy_check)
    diagnostics = {}
    for s, x in supports.items():
        z = forward(calibrated, x)
        diagnostics[s] = dict(class_agreement=float((z.argmax(1) == labels).float().mean()), target_mass=z.softmax(1).mean(0).tolist())
    log('teacher agreement', {s: round(d['class_agreement'], 3) for s, d in diagnostics.items()})
    targets = {s: forward(calibrated, x).log_softmax(1) for s, x in supports.items()}
    evals = {split: read(split) for split in ('old', 'group')}
    out = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__),
               p32_sha=sha(a.p32), p33b_sha=sha(a.p33b), p34_sha=sha(a.p34),
               protocol='docs/research/methods/support-capacity-oracle-2026-10-22.md',
               scope='frozen backbone and BN; Project48 teacher; density models fit on pooled rows (R3-ineligible); no official test',
               args=vars(a), fits=fits, memorization=copy_check, generation_diagnostics=diagnostics, clients={})
    predictions, heads = {}, {}; path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    global_counts = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(C)], device=dev)
    pg = ((global_counts + 1) / (global_counts.sum() + C)).log(); picked = np.zeros(len(pool_lab), bool); picked[pick] = True
    owner = np.repeat(np.arange(20), 16384)
    for k in map(int, a.clients.split(',')):
        private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
        private.requires_grad_(False)
        x, y, ht, zt = own[k]; _, hp, _ = forward(private, x, feats=True)
        features = {split: (yy, forward(private, xx, feats=True)[1].float()) for split, (xx, yy) in evals.items()}
        counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
        shift = ((counts + 1) / (counts.sum() + C)).log() - pg
        proxy_ce = F.cross_entropy(zt, y).detach()
        with torch.no_grad(): real_target = calibrated.head[5](ht).log_softmax(1)
        syn_h = {s: forward(private, xx, feats=True)[1].float() for s, xx in supports.items()}
        if k == 0:   # harness identity: the P32 Gaussian arm of client 0, bit for bit
            head = adapt(private.head[5], hp.float(), y, forward(private, gaussian, feats=True)[1].float(),
                         forward(calibrated, gaussian).log_softmax(1), real_target, proxy_ce, shift, a.steps, 2400)
            with torch.no_grad(): pred = head(features['group'][1]).argmax(1).cpu().numpy()
            assert np.array_equal(pred, saved['0_calibrated_gaussian_group'].astype('int64')); log('client 0 gaussian reproduces P32')
        other = per_class_rows(pool_lab, np.where((owner != k) & ~picked)[0], [33, k])
        _, h_pool, _ = forward(private, pool_x, feats=True)
        geo = {s: geometry(h_pool.float(), other, hs, labels) for s, hs in syn_h.items()}
        del h_pool
        out['clients'][k] = dict(geometry=geo)
        for s in ARMS:
            head = adapt(private.head[5], hp.float(), y, syn_h[s], targets[s], real_target, proxy_ce, shift, a.steps, 2400 + k)
            out['clients'][k][s] = {}
            for split, (yy, hh) in features.items():
                with torch.no_grad(): pred = head(hh).argmax(1)
                out['clients'][k][s][split] = report(yy, pred)
                predictions[f'{k}_{s}_{split}'] = pred.cpu().numpy().astype('uint8')
            heads[f'{k}_{s}'] = {nm: v.detach().cpu() for nm, v in head.state_dict().items()}
        path.write_text(json.dumps(out, indent=1) + '\n')
        log('client', k, {s: round(out['clients'][k][s]['group']['metrics']['f1_macro'], 4) for s in ARMS},
            'P32 gaussian/copula/real', [round(p32['clients'][str(k)][f'calibrated_{s}']['group']['metrics']['f1_macro'], 4)
                                          for s in ('gaussian', 'copula', 'real')])
        del private
    for split, (xx, yy) in evals.items(): predictions[split + '_y'] = yy.cpu().numpy().astype('uint8')
    np.savez_compressed(path.with_suffix('.npz'), **predictions); torch.save(heads, path.with_suffix('.pt'))
    saved_now = np.load(path.with_suffix('.npz')); checked = 0
    for k, arms_k in out['clients'].items():
        for s in ARMS:
            for split, rep in arms_k[s].items():
                yy, pred = saved_now[split + '_y'].astype('int64'), saved_now[f'{k}_{s}_{split}'].astype('int64')
                matrix = np.bincount(yy * C + pred, minlength=C * C).reshape(C, C)
                assert matrix.tolist() == rep['confusion'] and metrics_from_confusion(matrix) == rep['metrics']; checked += 1
    # Readings (locked in the protocol).
    res = {s: summary(out['clients'], s) for s in ARMS}
    anchors = {s: p32['summary']['calibrated_' + s] for s in ('gaussian', 'copula', 'real')}
    anchors.update(own_bal=p33b['summary']['own_bal'], pcop=p34['summary']['pcop'])
    p32f = lambda arm, c: float(np.mean([v[f'calibrated_{arm}']['group']['per_class'][c] for v in p32['clients'].values()]))
    pcf = lambda arm, c: float(np.mean([v[arm]['group']['per_class'][c] for v in out['clients'].values()]))
    per_class = {s: {classes[c]: pcf(s, c) - p32f('gaussian', c) for c in range(C)} for s in ARMS}
    for s in ('copula', 'real'): per_class[s] = {classes[c]: p32f(s, c) - p32f('gaussian', c) for c in range(C)}
    g_ps = {s: sum(v[c] for c in PS) for s, v in per_class.items()}
    gauss = anchors['gaussian']; span = g_ps['real'] - g_ps['copula']; gain = anchors['real']['mean'] - gauss['mean']
    readings = dict(
        G_PS=g_ps,
        A=dict(real_cPS_lost=(g_ps['real'] - g_ps['real_cPS']) / span, cop_rPS_recovered=(g_ps['cop_rPS'] - g_ps['copula']) / span,
               real_cB_lost=(g_ps['real'] - g_ps['real_cB']) / span, cop_rB_recovered=(g_ps['cop_rB'] - g_ps['copula']) / span),
        B={s: (res[s]['mean'] - gauss['mean']) / gain for s in ('real_j05', 'real_j20')},
        C={s: dict(mean=res[s]['mean'] - gauss['mean'], G_PS=g_ps[s], benign=res[s]['benign'] - gauss['benign'],
                   worst=res[s]['worst'] - gauss['worst'], copy=copy_check[s]['copy']) for s in ('gmm8', 'gmm32', 'made')})
    for s, v in readings['C'].items():
        v['capable'] = bool(v['mean'] >= .008 and v['G_PS'] >= .065 and v['benign'] >= -.005 and v['worst'] >= -.010 and not v['copy'])
    geometry_mean = {s: {m: float(np.nanmean([[x[m] if x[m] is not None else np.nan for x in v['geometry'][s]] for v in out['clients'].values()]))
                         for m in ('precision', 'recall', 'landing')} for s in ARMS}
    precision = {s: {c: float(np.nanmean([v['geometry'][s][classes.index(c)]['precision'] for v in out['clients'].values()])) for c in PS}
                 for s in ARMS}
    out.update(summary=res, anchors=anchors, per_class_delta_vs_gaussian=per_class, readings=readings, geometry_mean=geometry_mean,
               precision_PS=precision, predictions_sha=sha(path.with_suffix('.npz')), heads_sha=sha(path.with_suffix('.pt')),
               audit_checks=checked, seconds=time.monotonic() - t0)
    path.write_text(json.dumps(out, indent=1) + '\n'); log('saved', str(path), 'audit checks', checked, json.dumps(readings))


if __name__ == '__main__':
    main()
