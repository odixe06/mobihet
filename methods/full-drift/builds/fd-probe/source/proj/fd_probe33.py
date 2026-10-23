"""P33: what the private head learns from a support. Pair accounting of the saved P32 heads, and k-NN precision/recall/landing of every
P32 support inside each private's own feature space (the input of its last Linear). No training, no official test.
Supports are regenerated with the P32 code and seeds and checked against the P32 JSON; saved heads must reproduce the P32 predictions.
Protocol: docs/research/methods/client-learning-diagnosis-2026-10-22.md
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from fd_probe23 import C, basis, decode_stats, encode_stats, fd, fit, forward, P, sha, virtual
from fd_probe32 import GROUPS, copula_fit, copula_draw, hybrid

K_NN, PER_CLASS, ROWS = 3, 256, 16384
TRUSTED = ['positionMirroring', 'constantPositionOffset', 'trafficCongestionSybil', 'dataReplay']
BINS = [(0, 0), (0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]


def prc(ref, syn):
    """k-NN precision/recall (Kynkaanniemi et al. 2019) of syn against ref; None when either side has too few rows."""
    if len(ref) <= K_NN or len(syn) <= K_NN: return None, None
    sr = torch.cdist(syn, ref)
    r_ref = torch.cdist(ref, ref).kthvalue(K_NN + 1, 1).values; r_syn = torch.cdist(syn, syn).kthvalue(K_NN + 1, 1).values
    return float((sr <= r_ref[None]).any(1).float().mean()), float((sr.T <= r_syn[None]).any(1).float().mean())


def per_class_rows(labels, candidates, seed):
    rng = np.random.default_rng(seed); out = []
    for c in range(C):
        idx = np.intersect1d(np.where(labels == c)[0], candidates)
        out.append(np.sort(rng.choice(idx, min(PER_CLASS, len(idx)), replace=False)))
    return out


def geometry(h_ref, ref_rows, h_syn, syn_labels):
    """Per class: precision, recall vs same-class reference; landing = 1-NN in the 16-class reference has the sample's label."""
    ref_rows = [torch.as_tensor(r, device=h_ref.device) for r in ref_rows]
    ref_all = torch.cat([h_ref[r] for r in ref_rows]); ref_lab = torch.cat([torch.full((len(r),), c) for c, r in enumerate(ref_rows)])
    out = []
    for c in range(C):
        syn = h_syn[syn_labels == c]
        pr, rc = prc(h_ref[ref_rows[c]], syn)
        land = float((ref_lab[torch.cdist(syn, ref_all).argmin(1).cpu()] == c).float().mean()) if len(syn) else None
        out.append(dict(precision=pr, recall=rc, landing=land))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--p32', required=True, help='P32 JSON; its .npz/.pt siblings hold predictions and heads')
    p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    p32 = json.loads(Path(a.p32).read_text()); assert p32['weights_sha'] == sha(a.weights)
    saved = np.load(Path(a.p32).with_suffix('.npz')); heads = torch.load(Path(a.p32).with_suffix('.pt'), weights_only=True)
    names = manifest['features']; zero = [names.index(n) for n in fd.BLOCK + fd.POSITION]
    active = [i for i in range(66) if i not in zero]
    groups = {g: [names.index(n) for n in v] for g, v in GROUPS.items()}
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48)
    # P32 setup, unchanged (same rows, Project48 calibration, Gaussian, real, copula and hybrid supports).
    gen = torch.Generator(device='cpu').manual_seed(22)
    own, total, input_n, input_s, input_q = {}, 0, torch.zeros(C, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, device=dev, dtype=torch.float64), torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); idx = torch.randperm(len(y), generator=gen)[:ROWS].to(dev); x, y = x[idx], y[idx]
        z, h, _ = forward(teacher, x, feats=True)
        values, _ = encode_stats(h, y, 'project48', proj); total = total + values
        own[k] = (x, y)
        one = F.one_hot(y, C).double(); input_n += one.sum(0); input_s += one.T @ x.double()
        for c in range(C): xx = x[y == c].double(); input_q[c] += xx.T @ xx
    n, mu, cov = decode_stats(total, 'project48', proj)
    bank_h, bank_y = virtual(mu, cov, n / n.sum(), 200000, 23)
    fc = torch.nn.Linear(48, C).to(dev)
    with torch.no_grad(): fc.weight.copy_(teacher.head[5].weight @ proj.float()); fc.bias.copy_(teacher.head[5].bias)
    fc = fit(fc, bank_h, bank_y, 3000, 23); del bank_h, bank_y
    calibrated = copy.deepcopy(teacher)
    with torch.no_grad(): calibrated.head[5].weight.copy_(fc.weight @ proj.float().T); calibrated.head[5].bias.copy_(fc.bias)
    im = input_s / input_n[:, None]
    ic = (input_q - input_n[:, None, None] * im[:, :, None] * im[:, None, :]) / (input_n[:, None, None] - 1)
    gaussian, labels = virtual(im, ic, torch.ones(C, device=dev) / C, a.bank, 24); gaussian[:, zero] = 0
    pool_x = torch.cat([own[k][0] for k in range(20)]); pool_y = torch.cat([own[k][1] for k in range(20)])
    rng = np.random.default_rng(27); pool_lab = pool_y.cpu().numpy(); lab = labels.cpu().numpy(); pick = np.empty(len(lab), dtype='int64')
    for c in range(C):
        slots = np.where(lab == c)[0]; pick[slots] = rng.choice(np.where(pool_lab == c)[0], len(slots), replace=True)
    real = pool_x[torch.from_numpy(pick).to(dev)].clone()
    tables, _ = copula_fit(pool_x, pool_y, torch.tensor(active, device=dev))
    supports = dict(gaussian=gaussian, copula=copula_draw(tables, labels, 32), real=real)
    for i, (gname, cols) in enumerate(groups.items()):
        supports['hyb_' + gname] = hybrid(real, labels, cols, active, im, ic, 320 + i)
    teachers = dict(original=teacher, calibrated=calibrated)
    for key, d in p32['generation_diagnostics'].items():   # identical supports and teachers, or stop
        t, s = key.split('_', 1)
        agree = float((forward(teachers[t], supports[s]).argmax(1) == labels).float().mean())
        assert agree == d['class_agreement'], (key, agree, d['class_agreement'])
    log('supports and teachers match P32')
    gx, gy = read('group'); tpred = forward(calibrated, gx).argmax(1)
    assert np.array_equal(saved['group_y'], gy.cpu().numpy())
    owner = np.repeat(np.arange(20), ROWS); picked = np.zeros(len(pool_lab), bool); picked[pick] = True
    group_rows = per_class_rows(gy.cpu().numpy(), np.arange(len(gy)), 34)
    arms = list(supports)
    counts = {k: np.array(manifest['counts'][f'train_{k}'], dtype=float) for k in range(20)}
    out = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__), p32_sha=sha(a.p32),
               protocol='docs/research/methods/client-learning-diagnosis-2026-10-22.md', k_nn=K_NN, per_class=PER_CLASS,
               teacher='calibrated (Project48)', scope='frozen P32 heads/supports; no training; no official test', clients={})
    classes = manifest['classes']; trusted = [classes.index(n) for n in TRUSTED]
    pairs = {s: dict(fixed=np.zeros((2, C, C), int), broken=np.zeros((2, C, C), int)) for s in arms if s != 'gaussian'}
    for k in range(20):
        private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
        _, hg, _ = forward(private, gx, feats=True); hg = hg.float()
        preds = {}
        for s in arms:   # saved P32 heads reproduce the saved predictions on these features
            head = torch.nn.Linear(256, C).to(dev); head.load_state_dict(heads[f'{k}_calibrated_{s}'])
            with torch.no_grad(): preds[s] = head(hg).argmax(1)
            assert np.array_equal(preds[s].cpu().numpy(), saved[f'{k}_calibrated_{s}_group'].astype('int64')), (k, s)
        for s in pairs:
            ok = (tpred == gy).long()
            for name, mask, err in (('fixed', (preds['gaussian'] != gy) & (preds[s] == gy), preds['gaussian']),
                                    ('broken', (preds['gaussian'] == gy) & (preds[s] != gy), preds[s])):
                flat = (ok[mask] * C + gy[mask]) * C + err[mask]
                pairs[s][name] += torch.bincount(flat, minlength=2 * C * C).view(2, C, C).cpu().numpy()
        other = per_class_rows(pool_lab, np.where((owner != k) & ~picked)[0], [33, k])
        mine = per_class_rows(pool_lab, np.where(owner == k)[0], [35, k])
        _, h_pool, _ = forward(private, pool_x, feats=True); h_pool = h_pool.float()
        syn = {s: forward(private, xx, feats=True)[1].float() for s, xx in supports.items()}
        syn['own'] = torch.cat([h_pool[r] for r in mine]); own_lab = torch.cat([torch.full((len(r),), c) for c, r in enumerate(mine)])
        geo = {}
        for s, hs in syn.items():
            lab_s = own_lab.to(dev) if s == 'own' else labels
            geo[s] = dict(other=geometry(h_pool, other, hs, lab_s), group=geometry(hg, group_rows, hs, lab_s))
        share = counts[k] / counts[k].sum()
        f1 = {s: p32['clients'][str(k)]['calibrated_' + s]['group']['per_class'] for s in arms}
        out['clients'][k] = dict(local_share=share.tolist(), f1=f1, geometry=geo)
        path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(out, indent=1) + '\n')
        log('client', k, {s: round(float(np.nanmean([v['precision'] or np.nan for v in g['other']])), 3) for s, g in geo.items()})
        del private, h_pool, syn
    # A: pair accounting (gaussian -> support), split by teacher right/wrong on the row.
    summary = dict(pairs={}, bins={}, cells={}, spearman={})
    tr = lambda m: sum(int(m[t, 0]) + int(m[0, t]) for t in trusted)   # (true trusted, err benign) + (true benign, err trusted)
    for s, d in pairs.items():
        fixed, broken = d['fixed'], d['broken']; net = int(fixed.sum() - broken.sum())
        top = sorted(((int(fixed[:, i, j].sum() - broken[:, i, j].sum()), classes[i], classes[j]) for i in range(C) for j in range(C) if i != j),
                     reverse=True)
        summary['pairs'][s] = dict(fixed=int(fixed.sum()), broken=int(broken.sum()), net=net,
                                   net_teacher_right=int(fixed[1].sum() - broken[1].sum()), net_teacher_wrong=int(fixed[0].sum() - broken[0].sum()),
                                   net_trusted_teacher_right=tr(fixed[1]) - tr(broken[1]),
                                   top_net_pairs=[dict(net=v, true=i, error=j) for v, i, j in top[:8]],
                                   worst_net_pairs=[dict(net=v, true=i, error=j) for v, i, j in top[-5:]],
                                   matrices=dict(fixed=fixed.tolist(), broken=broken.tolist()))
    # Delta-F1 contribution to the macro mean by local class share bin (support minus gaussian).
    for s in pairs:
        acc = np.zeros(len(BINS)); cnt = np.zeros(len(BINS), int)
        for k, v in out['clients'].items():
            d = np.array(v['f1'][s]) - np.array(v['f1']['gaussian'])
            for b, (lo, hi) in enumerate(BINS):
                m = (np.array(v['local_share']) == 0) if hi == 0 else ((np.array(v['local_share']) > lo) & (np.array(v['local_share']) <= hi))
                acc[b] += d[m].sum(); cnt[b] += int(m.sum())
        summary['bins'][s] = [dict(bin=list(b), cells=int(c), delta_macro_contribution=float(x / C / 20)) for b, c, x in zip(BINS, cnt, acc)]
    # C: which geometry measure explains the per-class Delta-F1 across supports (means over clients, and pooled client cells).
    for ref in ('other', 'group'):
        rows = []
        for s in pairs:
            for c in range(C):
                cell = dict(support=s, cls=classes[c])
                cell['dF1'] = float(np.mean([v['f1'][s][c] - v['f1']['gaussian'][c] for v in out['clients'].values()]))
                for m in ('precision', 'recall', 'landing'):
                    diff = [(v['geometry'][s][ref][c][m] - v['geometry']['gaussian'][ref][c][m]) for v in out['clients'].values()
                            if v['geometry'][s][ref][c][m] is not None and v['geometry']['gaussian'][ref][c][m] is not None]
                    cell['d' + m] = float(np.mean(diff))
                rows.append(cell)
        summary['cells'][ref] = rows
        summary['spearman'][ref] = {m: float(spearmanr([r['dF1'] for r in rows], [r['d' + m] for r in rows]).statistic)
                                    for m in ('precision', 'recall', 'landing')}
        pooled = {m: ([], []) for m in ('precision', 'recall', 'landing')}
        for v in out['clients'].values():
            for s in pairs:
                for c in range(C):
                    for m in pooled:
                        g, b = v['geometry'][s][ref][c][m], v['geometry']['gaussian'][ref][c][m]
                        if g is not None and b is not None:
                            pooled[m][0].append(v['f1'][s][c] - v['f1']['gaussian'][c]); pooled[m][1].append(g - b)
        summary['spearman'][ref + '_pooled_client_cells'] = {m: float(spearmanr(*pooled[m]).statistic) for m in pooled}
    summary['geometry_mean'] = {ref: {s: {m: float(np.nanmean([[x[m] if x[m] is not None else np.nan for x in v['geometry'][s][ref]]
                                                                for v in out['clients'].values()]))
                                          for m in ('precision', 'recall', 'landing')}
                                      for s in list(supports) + ['own']} for ref in ('other', 'group')}
    out.update(summary=summary, seconds=time.monotonic() - t0)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n'); log('saved', a.out)


if __name__ == '__main__':
    main()
