"""Causal calibration-estimator intervention; keep the frozen probe23/24 implementations and sampled rows.

Compare unweighted client subsamples with inverse-inclusion weighting. No training-harness modification.
Per-class weighting requires every owned class present in the subsample; production must reserve rare-class rows.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

import fd_probe23
import fd_probe24
from mh.v2_math import pack_vector, unpack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=int, choices=(23, 24), required=True)
    parser.add_argument('--weighting', choices=('prior', 'total', 'class'), required=True)
    args, remaining = parser.parse_known_args()
    fixture = Path(remaining[remaining.index('--fixture') + 1])
    out = Path(remaining[remaining.index('--out') + 1])
    manifest = json.loads((fixture / 'manifest.json').read_text())
    module = fd_probe23 if args.probe == 23 else fd_probe24
    original = module.encode_stats
    calls = []

    def weighted(h, y, mode, projection):
        assert mode == 'project48', 'isolate the selected projection; no estimator sweep'
        k = len(calls); assert k < 20
        values, _ = original(h, y, mode, projection)
        full = torch.tensor(manifest['counts'][f'train_{k}'], device=values.device, dtype=values.dtype)
        count = values[:16]
        assert not ((full > 0) & (count == 0)).any(), 'reserve every owned class before production weighting'
        factor = torch.ones_like(count) if args.weighting == 'prior' else \
            torch.ones_like(count) * (full.sum() / count.sum()) if args.weighting == 'total' else full / count.clamp_min(1)
        mean_end = 16 + 16 * 48
        weighted_values = torch.cat([count * factor,
                                     (values[16:mean_end].reshape(16, 48) * factor[:, None]).flatten(),
                                     (values[mean_end:].reshape(16, -1) * factor[:, None]).flatten()])
        packet = pack_vector('moments', weighted_values, base=15, version=15, count=len(y))
        calls.append(dict(client=k, sample_counts=count.tolist(), population_counts=full.tolist(), weights=factor.tolist()))
        return unpack(packet, values.device, expected_base=15)['values'].double(), len(packet)

    module.encode_stats = weighted
    original_virtual = module.virtual
    if args.weighting == 'prior':
        population = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(16)]).double()
        def prior_only(mu, cov, share, rows, seed):
            if rows == 200000: share = (population / population.sum()).to(mu.device)
            return original_virtual(mu, cov, share, rows, seed)
        module.virtual = prior_only
    sys.argv = [f'fd_probe{args.probe}.py', *remaining]
    if args.probe == 23: sys.argv += ['--modes', 'project48']
    module.main()
    assert len(calls) == 20
    result = json.loads(out.read_text())
    result.update(weighting=args.weighting, weighting_source_sha=fd_probe23.sha(__file__), sampling_weights=calls,
                  scope=result['scope'] + '; causal inverse-inclusion weighting intervention, same sampled rows')
    out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__': main()
