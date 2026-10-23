"""P33b: is the real-support gain other-client diversity or class balance on real rows? The P32 harness (Project48 teacher), with the
synthetic bank replaced by the client's own sampled rows: own_bal (256 per class, with replacement) and own_nat (4096 at the local
distribution). Zero bytes, R3-eligible. Gaussian of client 0 is re-run and must reproduce the P32 predictions.
Protocol: docs/research/methods/client-learning-diagnosis-2026-10-22.md section 5. No official test.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, basis, decode_stats, encode_stats, fd, fit, forward, P, report, sha, virtual
from fd_probe24 import adapt
from fd_probe32 import summary
from mh.metrics import metrics_from_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--p32', required=True); p.add_argument('--steps', type=int, default=1000); p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    p32 = json.loads(Path(a.p32).read_text()); assert p32['weights_sha'] == sha(a.weights); saved = np.load(Path(a.p32).with_suffix('.npz'))
    names = manifest['features']; zero = [names.index(n) for n in fd.BLOCK + fd.POSITION]
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48)
    # P32 setup, unchanged.
    gen = torch.Generator(device='cpu').manual_seed(22)
    own, total, input_n, input_s, input_q = {}, 0, torch.zeros(C, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, device=dev, dtype=torch.float64), torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); idx = torch.randperm(len(y), generator=gen)[:16384].to(dev); x, y = x[idx], y[idx]
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
    gaussian, _ = virtual(im, ic, torch.ones(C, device=dev) / C, a.bank, 24); gaussian[:, zero] = 0
    gaussian_target = forward(calibrated, gaussian).log_softmax(1)
    evals = {split: read(split) for split in ('old', 'group')}
    out = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__), p32_sha=sha(a.p32),
               protocol='docs/research/methods/client-learning-diagnosis-2026-10-22.md#5', args=vars(a),
               scope='frozen backbone and BN; own rows as the support bank; zero bytes; no official test', clients={})
    predictions = {}; path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    global_counts = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(C)], device=dev)
    pg = ((global_counts + 1) / (global_counts.sum() + C)).log()
    arms = ('own_bal', 'own_nat')
    for k in range(20):
        private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
        private.requires_grad_(False)
        x, y, ht, zt = own[k]; _, hp, _ = forward(private, x, feats=True)
        rng = np.random.default_rng(36 + k); yk = y.cpu().numpy()
        bal = np.concatenate([rng.choice(np.where(yk == c)[0], a.bank // C, replace=True) for c in range(C)])
        banks = dict(own_bal=torch.from_numpy(bal).to(dev), own_nat=torch.from_numpy(rng.choice(len(yk), a.bank, replace=False)).to(dev))
        features = {split: (yy, forward(private, xx, feats=True)[1].float()) for split, (xx, yy) in evals.items()}
        counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
        shift = ((counts + 1) / (counts.sum() + C)).log() - pg
        proxy_ce = F.cross_entropy(zt, y).detach()
        with torch.no_grad(): real_target = calibrated.head[5](ht).log_softmax(1)
        if k == 0:   # harness identity: the P32 Gaussian arm of client 0, bit for bit
            syn = forward(private, gaussian, feats=True)[1].float()
            head = adapt(private.head[5], hp.float(), y, syn, gaussian_target, real_target, proxy_ce, shift, a.steps, 2400 + k)
            with torch.no_grad(): pred = head(features['group'][1]).argmax(1).cpu().numpy()
            assert np.array_equal(pred, saved['0_calibrated_gaussian_group'].astype('int64')); log('client 0 gaussian reproduces P32')
        out['clients'][k] = {}
        for arm in arms:
            idx = banks[arm]
            with torch.no_grad(): target = calibrated.head[5](ht[idx]).log_softmax(1)   # teacher on its own features of these rows
            head = adapt(private.head[5], hp.float(), y, hp[idx].float(), target, real_target, proxy_ce, shift, a.steps, 2400 + k)
            out['clients'][k][arm] = {}
            for split, (yy, hh) in features.items():
                with torch.no_grad(): pred = head(hh).argmax(1)
                out['clients'][k][arm][split] = report(yy, pred)
                predictions[f'{k}_{arm}_{split}'] = pred.cpu().numpy().astype('uint8')
        path.write_text(json.dumps(out, indent=1) + '\n')
        log('client', k, {arm: round(v['group']['metrics']['f1_macro'], 4) for arm, v in out['clients'][k].items()},
            'P32 gaussian/real', round(p32['clients'][str(k)]['calibrated_gaussian']['group']['metrics']['f1_macro'], 4),
            round(p32['clients'][str(k)]['calibrated_real']['group']['metrics']['f1_macro'], 4))
        del private
    for split, (xx, yy) in evals.items(): predictions[split + '_y'] = yy.cpu().numpy().astype('uint8')
    np.savez_compressed(path.with_suffix('.npz'), **predictions)
    out['summary'] = {arm: summary(out['clients'], arm) for arm in arms}
    out['summary'].update({'p32_' + s: p32['summary']['calibrated_' + s] for s in ('gaussian', 'copula', 'real')})
    saved_now = np.load(path.with_suffix('.npz')); checked = 0
    for k, arms_k in out['clients'].items():
        for arm, splits in arms_k.items():
            for split, rep in splits.items():
                yy, pred = saved_now[split + '_y'].astype('int64'), saved_now[f'{k}_{arm}_{split}'].astype('int64')
                matrix = np.bincount(yy * C + pred, minlength=C * C).reshape(C, C)
                assert matrix.tolist() == rep['confusion'] and metrics_from_confusion(matrix) == rep['metrics']; checked += 1
    out.update(predictions_sha=sha(path.with_suffix('.npz')), audit_checks=checked, seconds=time.monotonic() - t0)
    path.write_text(json.dumps(out, indent=1) + '\n'); log('saved', str(path), 'audit checks', checked, out['summary'])


if __name__ == '__main__':
    main()
