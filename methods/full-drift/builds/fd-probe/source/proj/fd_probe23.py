"""Offline CCVR compression screen. Frozen audited backbone, original pilot holdouts, no official test.

Class moments travel as raw fp32 sums through the existing CRC/version codec, then are pooled and centered.
Full covariance is a diagnostic ceiling; projected and Nystrom statistics preserve class-specific covariance.
This probe is not an online-FL utility experiment. A finite virtual bank is shared across controls.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj'))
import fd
from fd import P
from fd_probe14 import cm, forward
from mh.metrics import metrics_from_confusion
from mh.v2_math import pack_vector, unpack

C, D = 16, 256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(y, pred):
    matrix = cm(y, pred)
    cc = matrix.double()
    return dict(metrics=metrics_from_confusion(matrix.cpu().numpy()),
                per_class=(2 * cc.diagonal() / (cc.sum(0) + cc.sum(1)).clamp_min(1)).tolist(),
                confusion=matrix.tolist())


@torch.no_grad()
def basis(weight, width):
    # Includes every original head direction; remaining directions are fixed random complements.
    g = torch.Generator(device='cpu').manual_seed(23)
    extra = torch.randn(D, D, generator=g, dtype=torch.float64).to(weight.device)
    return torch.linalg.qr(torch.cat([weight.double().T, extra], 1), mode='reduced').Q[:, :width]


def encode_stats(h, y, mode, projection):
    h = h.double()
    if mode.startswith('project'): h = h @ projection
    one = F.one_hot(y, C).double()
    n, s = one.sum(0), one.T @ h
    q = torch.stack([h[y == c].T @ h[y == c] for c in range(C)])
    width = h.shape[1]
    if mode.startswith('nystrom'):
        values = torch.cat([n.flatten(), s.flatten(), q.diagonal(dim1=1, dim2=2).flatten(), (q @ projection).flatten()])
    else:
        tri = torch.triu_indices(width, width, device=h.device)
        values = torch.cat([n.flatten(), s.flatten(), q[:, tri[0], tri[1]].flatten()])
    packet = pack_vector('moments', values, base=15, version=15, count=len(y))
    # Consume decoded values, never the source tensors.
    return unpack(packet, h.device, expected_base=15)['values'].double(), len(packet)


def decode_stats(v, mode, projection):
    width = projection.shape[1] if mode.startswith('project') else D
    n, at = v[:C], C
    s = v[at:at + C * width].reshape(C, width); at += C * width
    mu = s / n[:, None].clamp_min(1)
    if mode.startswith('nystrom'):
        diagonal = v[at:at + C * D].reshape(C, D); at += C * D
        sketch = v[at:].reshape(C, D, projection.shape[1])
        cross = sketch - n[:, None, None] * mu[:, :, None] * (mu @ projection)[:, None, :]
        cross = cross / (n[:, None, None] - 1).clamp_min(1)
        small = projection.T @ cross
        small = (small + small.transpose(1, 2)) / 2
        cov = cross @ torch.linalg.pinv(small, hermitian=True, rtol=1e-7) @ cross.transpose(1, 2)
        exact_diag = (diagonal - n[:, None] * mu.square()) / (n[:, None] - 1).clamp_min(1)
        residual = (exact_diag - cov.diagonal(dim1=1, dim2=2)).clamp_min(0)
        cov = cov + torch.diag_embed(residual)
    else:
        tri = torch.triu_indices(width, width, device=v.device)
        q = torch.zeros(C, width, width, dtype=v.dtype, device=v.device)
        q[:, tri[0], tri[1]] = v[at:].reshape(C, -1)
        q[:, tri[1], tri[0]] = q[:, tri[0], tri[1]]
        cov = (q - n[:, None, None] * mu[:, :, None] * mu[:, None, :]) / (n[:, None, None] - 1).clamp_min(1)
    return n, mu, (cov + cov.transpose(1, 2)) / 2


def virtual(mu, cov, share, rows, seed):
    """One seeded finite bank, class-contiguous generation avoids per-row 256x256 factor expansion."""
    gen = torch.Generator(device=mu.device).manual_seed(seed)
    labels = torch.multinomial(share.float(), rows, replacement=True, generator=gen)
    x = torch.empty(rows, mu.shape[1], device=mu.device)
    eigen, vectors = torch.linalg.eigh(cov)
    factors = vectors * eigen.clamp_min(1e-4).sqrt()[:, None, :]
    for c in range(C):
        idx = torch.where(labels == c)[0]
        z = torch.randn(len(idx), mu.shape[1], device=mu.device, generator=gen)
        x[idx] = mu[c].float() + z @ factors[c].float().T
    return x, labels


def fit(fc, x, y, steps, seed):
    fc = copy.deepcopy(fc).float().train()
    opt = torch.optim.Adam(fc.parameters(), lr=1e-3)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(x), (4096,), device=x.device, generator=gen)
        loss = F.cross_entropy(fc(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert all(torch.isfinite(p).all() for p in fc.parameters())
    return fc.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=3000); p.add_argument('--virtual', type=int, default=200000)
    p.add_argument('--per-client', type=int, default=16384)
    p.add_argument('--modes', default='full,shared,diag,project16,project32,project48,nystrom2,nystrom4')
    a = p.parse_args(); started = time.monotonic()
    log = lambda *v: print(round(time.monotonic() - started, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2)
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected, name
    assert manifest['group_overlap'] == 0 and manifest['train_frac'] == .025
    dev = torch.device('cuda'); w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    zero = [manifest['features'].index(n) for n in fd.BLOCK + fd.POSITION]
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev)
        x[:, zero] = 0
        y = torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
        return x, y
    evals = {}
    for split in ('old', 'group'):
        x, y = read(split); z, h, _ = forward(teacher, x, feats=True)
        evals[split] = (y, h.float(), z); del x
    modes = a.modes.split(',')
    projections = {m: basis(teacher.head[5].weight, int(m[7:])) for m in modes if m.startswith('project')}
    sketch_gen = torch.Generator(device='cpu').manual_seed(23)
    omega = torch.linalg.qr(torch.randn(D, 4, generator=sketch_gen, dtype=torch.float64).to(dev), mode='reduced').Q
    projections.update({m: omega[:, :int(m[7:])] for m in modes if m.startswith('nystrom')})
    totals, sizes, own_h, own_y = {}, {}, [], []
    gen = torch.Generator(device='cpu').manual_seed(22)
    for k in range(20):
        x, y = read(f'train_{k}')
        idx = torch.randperm(len(y), generator=gen)[:a.per_client].to(dev)
        _, h, _ = forward(teacher, x[idx], feats=True); y = y[idx]
        own_h.append(h); own_y.append(y)
        for mode in modes:
            wiremode = mode if mode.startswith(('project', 'nystrom')) else 'full'
            v, size = encode_stats(h, y, wiremode, projections.get(mode))
            totals[mode] = totals.get(mode, 0) + v; sizes[mode] = size
        del x
    hs, ys = torch.cat(own_h).float(), torch.cat(own_y)
    log('collected', len(ys), 'rows; teacher group', report(evals['group'][0], evals['group'][2].argmax(1))['metrics']['f1_macro'])
    out = dict(weights=str(Path(a.weights).resolve()), weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'),
               source_sha=sha(__file__), steps=a.steps, virtual=a.virtual, per_client=a.per_client,
               scope='offline frozen-backbone screen, old/group only; no online FL or official test', modes={},
               original={split: report(y, z.argmax(1)) for split, (y, h, z) in evals.items()})
    outpath = Path(a.out); outpath.parent.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        t0 = time.monotonic(); proj = projections.get(mode)
        wiremode = mode if mode.startswith(('project', 'nystrom')) else 'full'
        n, mu, cov = decode_stats(totals[mode], wiremode, proj)
        if mode == 'shared': cov = ((cov * (n - 1)[:, None, None]).sum(0) / (n.sum() - C)).expand_as(cov)
        if mode == 'diag': cov = torch.diag_embed(cov.diagonal(dim1=1, dim2=2))
        bank, labels = virtual(mu, cov, n / n.sum(), a.virtual, 23)
        fc = copy.deepcopy(teacher.head[5])
        if mode.startswith('project'):
            fc = torch.nn.Linear(proj.shape[1], C).to(dev)
            with torch.no_grad():
                fc.weight.copy_(teacher.head[5].weight @ proj.float()); fc.bias.copy_(teacher.head[5].bias)
            residual = float((teacher.head[5].weight - fc.weight @ proj.float().T).detach().abs().max())
            assert residual < 1e-5, residual
        fc = fit(fc, bank, labels, a.steps, 23)
        pred = {}
        for split, (y, h, z) in evals.items():
            with torch.no_grad(): newz = fc(h @ proj.float() if mode.startswith('project') else h)
            pred[split] = report(y, newz.argmax(1))
            pred[split]['teacher_correct_to_wrong'] = int(((z.argmax(1) == y) & (newz.argmax(1) != y)).sum())
            pred[split]['teacher_wrong_to_correct'] = int(((z.argmax(1) != y) & (newz.argmax(1) == y)).sum())
        head_packet = pack_vector('dense', torch.cat([fc.weight.flatten(), fc.bias.flatten()]), base=15, version=15)
        out['modes'][mode] = dict(**pred, stats_up_bytes=sizes[mode], head_down_bytes=len(head_packet),
                                  calibration_exchange_bytes=sizes[mode] + len(head_packet), seconds=time.monotonic() - t0)
        # The real-row refit is an unreachable oracle, used only to distinguish compression loss from Gaussian misspecification.
        oracle = fit(fc, hs @ proj.float() if mode.startswith('project') else hs, ys, a.steps, 24)
        with torch.no_grad():
            y, h, _ = evals['group']
            out['modes'][mode]['pooled_real_oracle_group'] = report(y, oracle(h @ proj.float() if mode.startswith('project') else h).argmax(1))
        outpath.write_text(json.dumps(out, indent=1) + '\n')
        log(mode, 'group', round(pred['group']['metrics']['f1_macro'], 4), 'benign', round(pred['group']['per_class'][0], 4),
            'exchange', out['modes'][mode]['calibration_exchange_bytes'])
        del bank, labels, fc, oracle
    log('saved', str(outpath))


if __name__ == '__main__':
    main()
