"""Teacher head calibration using additive real-label gradient/diagonal Hessian at a common frozen snapshot.

One-query fixed steps and a two-query control. No per-record gradient/activation is communicated; no official test.
Diagonal Hessian is an approximation to full Newton, not sufficient statistics for arbitrary future CE steps.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, D, P, fd, forward, report, sha
from mh.v2_math import pack_vector, unpack


def query(weight, features, labels, importance, number):
    values, message_sizes = [], []
    for h, y, scale in zip(features, labels, importance):
        hh = torch.cat([h.double(), torch.ones(len(h), 1, device=h.device, dtype=torch.float64)], 1)
        p = (hh @ weight.double().T).softmax(1)
        gradient = ((p - F.one_hot(y, C)) * scale[:, None]).T @ hh
        diagonal = (p * (1 - p) * scale[:, None]).T @ hh.square()
        # For any head: softmax Hessian <= .5 I, hh^T <= diag(|h| ||h||_1).
        # This majorizer is independent of the queried head, unlike the Hessian diagonal.
        majorizer = .5 * (scale[:, None] * hh.abs() * hh.abs().sum(1, keepdim=True)).sum(0)
        vector = torch.cat([scale.sum().reshape(1), gradient.flatten(), diagonal.flatten(), majorizer])
        packet = pack_vector('dense', vector, base=number, version=number + 1, count=len(y))
        values.append(unpack(packet, h.device, expected_base=number)['values'].double()); message_sizes.append(len(packet))
    pooled = torch.stack(values).sum(0); size = C * (D + 1)
    return (pooled[1:1 + size].reshape(C, D + 1) / pooled[0],
            pooled[1 + size:1 + 2 * size].reshape(C, D + 1) / pooled[0],
            pooled[1 + 2 * size:] / pooled[0], message_sizes)


def safe_step(weight, gradient, direction, majorizer):
    eta = (gradient * direction).sum() / (majorizer[None] * direction.square()).sum().clamp_min(1e-20)
    return weight - .99 * eta * direction, float(eta)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); t0 = time.monotonic(); fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    population_path = Path(a.weights).parents[3] / 'source/data_manifest.json'
    population = json.loads(population_path.read_text())
    weights = torch.load(a.weights, weights_only=True, map_location=dev)
    model = P.make('dagsnet').to(dev); model.load_state_dict(weights['proxy'][0], strict=True); model.eval().requires_grad_(False)
    zero = [manifest['features'].index(name) for name in fd.BLOCK + fd.POSITION]
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    features, labels, importance = [], [], []; generator = torch.Generator().manual_seed(22)
    for k in range(20):
        x, y = read(f'train_{k}'); index = torch.randperm(len(y), generator=generator)[:16384].to(dev); y = y[index]
        h = forward(model, x[index], feats=True)[1].float()
        sampled = torch.bincount(y, minlength=C).double()
        full = torch.tensor(population['counts'][f'train_{k}'], device=dev, dtype=torch.float64)
        assert not ((full > 0) & (sampled == 0)).any()
        features.append(h); labels.append(y); importance.append((full / sampled.clamp_min(1))[y])
    evals = {split: read(split) for split in ('old', 'group')}
    eval_h = {split: (y, forward(model, x, feats=True)[1].float()) for split, (x, y) in evals.items()}
    initial = torch.cat([model.head[5].weight, model.head[5].bias[:, None]], 1).double()
    gradient, diagonal, majorizer, up_sizes = query(initial, features, labels, importance, 0)
    # Verify actual communicated gradient against central autograd on the same sampled weighted risk.
    check = initial.clone().requires_grad_(True); total = sum(v.sum() for v in importance)
    risk = sum((F.cross_entropy(F.linear(h.double(), check[:, :-1], check[:, -1]), y, reduction='none') * imp).sum()
               for h, y, imp in zip(features, labels, importance)) / total
    reference = torch.autograd.grad(risk, check)[0]
    error = float((reference - gradient).abs().max()); assert error < 1e-6, error
    damping = .01 * diagonal.mean()
    direction = gradient / (diagonal + damping)
    plain, plain_eta = safe_step(initial, gradient, gradient, majorizer)
    first, first_eta = safe_step(initial, gradient, direction, majorizer)
    candidates = dict(original=initial, safe_gradient=plain, safe_diagonal=first)
    second_g, second_h, second_majorizer, second_sizes = query(first, features, labels, importance, 1)
    second, second_eta = safe_step(first, second_g, second_g / (second_h + .01 * second_h.mean()), second_majorizer)
    candidates['safe_diagonal_two_queries'] = second
    result = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), population_sha=sha(population_path), source_sha=sha(__file__),
                  scope='offline frozen teacher; class inclusion weights from audited full train counts; old/group only',
                  primary='safe_diagonal', step_rule='quadratic global majorizer; 0.99 numerical margin',
                  etas=[plain_eta, first_eta, second_eta], gradient_max_abs_error=error, models={})
    for name, matrix in candidates.items():
        packet = pack_vector('dense', matrix.flatten(), base=15, version=15)
        decoded = unpack(packet, dev, expected_base=15)['values'].reshape(C, D + 1)
        res = {}
        for split, (y, h) in eval_h.items(): res[split] = report(y, F.linear(h, decoded[:, :-1], decoded[:, -1]).argmax(1))
        sampled_risk = sum(float((F.cross_entropy(F.linear(h, decoded[:, :-1], decoded[:, -1]), y, reduction='none').double() * imp).sum())
                           for h, y, imp in zip(features, labels, importance)) / float(total)
        assert sampled_risk <= float(risk.detach()) + 1e-6, ('risk increased', name, sampled_risk, float(risk.detach()))
        res['weighted_sample_train_ce'] = sampled_risk
        res.update(up_bytes_per_client=0 if name == 'original' else up_sizes[0] * (2 if name.endswith('two_queries') else 1),
                   down_bytes_per_client=0 if name == 'original' else len(packet) * (2 if name.endswith('two_queries') else 1))
        result['models'][name] = res
        print(name, round(res['group']['metrics']['f1_macro'], 6), round(res['group']['per_class'][0], 6), flush=True)
    result['seconds'] = time.monotonic() - t0
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__': main()
