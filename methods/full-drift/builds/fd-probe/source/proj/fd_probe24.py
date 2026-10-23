"""Teacher x synthetic-support factorial on frozen private backbones and BN, old/group holdouts only.

Diagnostic head adaptation, not a substitute for from-scratch ep15/50-round FL. All arms use identical own rows,
sample streams, 512 real + 256 synthetic rows/step. Synthetic inversion starts at pooled Gaussian noise, never a record.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, D, basis, decode_stats, encode_stats, fd, fit, forward, P, report, sha, virtual


def invert(model, initial, targets, projection, input_scale, feature_scale, zero, steps):
    """Same shared teacher features for both target heads: isolate support from teacher quality."""
    keep = torch.ones(66, device=initial.device); keep[zero] = 0
    result = []
    for start in range(0, len(initial), 512):
        x0, goal = initial[start:start + 512], targets[start:start + 512]
        x = x0.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=.03)
        for _ in range(steps):
            h = model.head[:4](model.combined(x * keep))
            match = ((h @ projection - goal) / feature_scale).square().mean()
            prior = ((x - x0) / input_scale).square().mean()
            opt.zero_grad(set_to_none=True); (match + .01 * prior).backward(); opt.step()
        result.append((x.detach() * keep))
    return torch.cat(result)


def adapt(head, own_h, own_y, syn_h, targets, real_target, proxy_ce, shift, steps, seed, hard=False):
    head = copy.deepcopy(head).float().train().requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
    g = torch.Generator(device=own_h.device).manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(own_y), (512,), device=own_h.device, generator=g)
        j = torch.randint(len(syn_h), (256,), device=own_h.device, generator=g)
        z = head(own_h[i]); ce = F.cross_entropy(z + shift, own_y[i])
        weight = 1 / (ce.detach() + proxy_ce)
        real_kd = F.kl_div(z.log_softmax(1), real_target[i], log_target=True, reduction='batchmean')
        syn_kd = F.cross_entropy(head(syn_h[j]), targets[j]) if hard else \
            F.kl_div(head(syn_h[j]).log_softmax(1), targets[j], log_target=True, reduction='batchmean')
        loss = ce + weight * (real_kd + syn_kd)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert all(torch.isfinite(p).all() for p in head.parameters())
    return head.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--clients', default=','.join(map(str, range(20)))); p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--invert-steps', type=int, default=80); p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    assert manifest['group_overlap'] == 0 and manifest['train_frac'] == .025
    zero = [manifest['features'].index(n) for n in fd.BLOCK + fd.POSITION]
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48)
    gen = torch.Generator(device='cpu').manual_seed(22)
    own, total, input_n, input_s, input_q = {}, 0, torch.zeros(C, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, device=dev, dtype=torch.float64), torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); idx = torch.randperm(len(y), generator=gen)[:16384].to(dev); x, y = x[idx], y[idx]
        z, h, _ = forward(teacher, x, feats=True)
        values, stat_bytes = encode_stats(h, y, 'project48', proj); total = total + values
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
    # Uniform class component schedule, matching gkd. Paired labels/seeds for the two support banks.
    initial, labels = virtual(im, ic, torch.ones(C, device=dev) / C, a.bank, 24)
    target, target_labels = virtual(mu, cov, torch.ones(C, device=dev) / C, a.bank, 24)
    assert torch.equal(labels, target_labels)
    feature_scale = cov.diagonal(dim1=1, dim2=2).mean(0).sqrt().float().clamp_min(.1)
    input_scale = ic.diagonal(dim1=1, dim2=2).mean(0).sqrt().float().clamp_min(.1)
    initial[:, zero] = 0
    revised = invert(teacher, initial, target, proj.float(), input_scale, feature_scale, zero, a.invert_steps)
    log('generated', a.bank, 'rows per support')
    supports = dict(gaussian=initial, inversion=revised)
    teachers = dict(original=teacher, calibrated=calibrated)
    targets = {}
    diagnostics = {}
    for s, x in supports.items():
        for t, model in teachers.items():
            z = forward(model, x)
            targets[t, s] = z.log_softmax(1)
            diagnostics[t + '_' + s] = dict(class_agreement=float((z.argmax(1) == labels).float().mean()),
                                            entropy=float(-(z.softmax(1) * z.log_softmax(1)).sum(1).mean()))
    evals = {split: read(split) for split in ('old', 'group')}
    out = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__),
               scope='frozen backbone and BN; own CE + real/synthetic KL; no online FL or official test',
               args=vars(a), generation_diagnostics=diagnostics, calibration_stats_up_bytes=stat_bytes, clients={})
    predictions = {}; heads = {}; path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    global_counts = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(C)], device=dev)
    pg = ((global_counts + 1) / (global_counts.sum() + C)).log()
    for k in map(int, a.clients.split(',')):
        private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
        private.requires_grad_(False)
        x, y, ht, zt = own[k]; _, hp, _ = forward(private, x, feats=True)
        syn_h = {s: forward(private, xx, feats=True)[1].float() for s, xx in supports.items()}
        features = {split: (yy, forward(private, xx, feats=True)[1].float()) for split, (xx, yy) in evals.items()}
        counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
        shift = ((counts + 1) / (counts.sum() + C)).log() - pg
        proxy_ce = F.cross_entropy(zt, y).detach()
        out['clients'][k] = dict(original={split: report(yy, private.head[5](hh).argmax(1)) for split, (yy, hh) in features.items()})
        arms = [(t, s, False) for t in teachers for s in supports] + [('calibrated', 'inversion', True)]
        for t, s, hard in arms:
            key = t + '_' + s + ('_hard' if hard else '')
            with torch.no_grad(): real_target = teachers[t].head[5](ht).log_softmax(1)
            adapted = adapt(private.head[5], hp.float(), y, syn_h[s], labels if hard else targets[t, s],
                            real_target, proxy_ce, shift, a.steps, 2400 + k, hard)
            out['clients'][k][key] = {}
            for split, (yy, hh) in features.items():
                with torch.no_grad(): pred = adapted(hh).argmax(1)
                out['clients'][k][key][split] = report(yy, pred)
                predictions[f'{k}_{key}_{split}'] = pred.cpu().numpy().astype('uint8')
            heads[f'{k}_{key}'] = {n: v.detach().cpu() for n, v in adapted.state_dict().items()}
        path.write_text(json.dumps(out, indent=1) + '\n')
        log('client', k, {name: round(v['group']['metrics']['f1_macro'], 4) for name, v in out['clients'][k].items()})
        del private
    for split, (xx, yy) in evals.items(): predictions[split + '_y'] = yy.cpu().numpy().astype('uint8')
    np.savez_compressed(path.with_suffix('.npz'), **predictions)
    torch.save(heads, path.with_suffix('.pt'))
    out.update(predictions_sha=sha(path.with_suffix('.npz')), heads_sha=sha(path.with_suffix('.pt')), seconds=time.monotonic() - t0)
    path.write_text(json.dumps(out, indent=1) + '\n'); log('saved', str(path))


if __name__ == '__main__':
    main()
