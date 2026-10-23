"""Check pooled statistics/CRC and audit saved probe24 predictions -> all ten per-client metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'source/proj'))
from fd_probe23 import basis, decode_stats, encode_stats
from mh.metrics import metrics_from_confusion
from mh.v2_math import pack_vector, unpack


def arithmetic():
    torch.manual_seed(23)
    w = torch.randn(16, 256); projection = basis(w, 48)
    assert float((w.double() - w.double() @ projection @ projection.T).abs().max()) < 1e-10
    # Different means between clients must survive pooling; averaging local covariances would fail this check.
    h = torch.randn(320, 256); h[160:] += 3
    y = torch.arange(320) % 16
    parts = [encode_stats(h[s:s + 160], y[s:s + 160], 'project48', projection)[0] for s in (0, 160)]
    n, mu, cov = decode_stats(sum(parts), 'project48', projection)
    u = h.double() @ projection
    for c in range(16):
        uc = u[y == c]; centered = uc - uc.mean(0)
        assert torch.allclose(mu[c], uc.mean(0), atol=3e-6)
        assert torch.allclose(cov[c], centered.T @ centered / (len(uc) - 1), atol=1e-5)
    packet = pack_vector('moments', torch.ones(10), base=3, version=4, count=7)
    for bad, base in ((packet[:-1] + bytes([packet[-1] ^ 1]), 3), (packet, 2)):
        try: unpack(bad, expected_base=base)
        except AssertionError: pass
        else: raise AssertionError('corruption/base accepted')
    print('PASS: old-head span, cross-client mean covariance, packet CRC/base')


def audit(path):
    value = json.loads(path.read_text()); arrays = np.load(path.with_suffix('.npz'))
    assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == value['predictions_sha']
    assert hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest() == value['heads_sha']
    heads = torch.load(path.with_suffix('.pt'), weights_only=True, map_location='cpu')
    checks = 0
    for client, arms in value['clients'].items():
        for arm, splits in arms.items():
            if arm == 'original': continue
            layer = torch.nn.Linear(256, 16); layer.load_state_dict(heads[f'{client}_{arm}'], strict=True)
            assert all(torch.isfinite(p).all() for p in layer.parameters())
            for split, expected in splits.items():
                y, pred = arrays[split + '_y'].astype('int64'), arrays[f'{client}_{arm}_{split}'].astype('int64')
                matrix = np.bincount(y * 16 + pred, minlength=256).reshape(16, 16)
                assert np.array_equal(matrix, expected['confusion'])
                for key, result in metrics_from_confusion(matrix).items(): assert abs(result - expected['metrics'][key]) < 1e-12
                checks += 1
    print('AUDITED', path, checks, 'prediction/confusion/10-metric checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('paths', nargs='*', type=Path); args = parser.parse_args()
    arithmetic()
    for path in args.paths: audit(path)
