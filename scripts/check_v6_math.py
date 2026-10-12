"""Bounded real-fixture sampler/codec gate; no model training or F1 claim."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mh.dagsnet import build_model, CFG
from mh.v6_geometry import Method, encode, decode, restore_derived
from mh.v6_run import ARMS
from mh.v6_schema import NAMES
from mh.v4_schema import FIRST, PROFILES, STRUCTURAL


def main():
    torch.set_num_threads(1)
    fixture = ROOT / 'papers/pilot/fixture'
    X, Y = [], []
    for k in range(20):
        labels = np.load(fixture / f'train_{k}_y.npy', mmap_mode='r')
        indices = np.random.default_rng(6102+k).choice(len(labels), min(512, len(labels)), replace=False)
        features = np.load(fixture / f'train_{k}_X.npy', mmap_mode='r')
        X.append(torch.from_numpy(features[indices].astype('float32')))
        Y.append(torch.from_numpy(labels[indices].astype('int64')))
    print('Bounded class counts', sum(torch.bincount(y, minlength=16) for y in Y).tolist(), flush=True)
    torch.manual_seed(42)
    init = build_model(dict(CFG, n_features=66, num_classes=16), latent=True)
    results = []
    for arm in ARMS:
        run = SimpleNamespace(X=X, Y=Y, dev=torch.device('cpu'), K=20, init=init,
                              a=SimpleNamespace(arm=arm), bytes=[], n=[len(y) for y in Y])
        try:
            method = Method(run)
            g = torch.Generator().manual_seed(6101)
            labels = torch.arange(4096) % 16
            sample = method.draw(labels, g, 0)
            assert sample.shape == (4096, 66) and torch.isfinite(sample).all()
            if not method.plain:
                zero, one = sample.double() == method.atom0, sample.double() == method.atom1
                assert (zero[:, [FIRST]+sum(PROFILES, [])] | one[:, [FIRST]+sum(PROFILES, [])]).all()
                assert zero[one[:, FIRST]][:, STRUCTURAL].all()
                for group in PROFILES:
                    assert (one[:, group].sum(1) <= 1).all()
            if method.dimension != 66 and method.mode != 'shuffle':
                exact = restore_derived(sample, method.dimension)
                assert torch.allclose(exact, sample, rtol=0, atol=3e-5)
            for kind, packet in method.packets.items():
                decode(packet, kind, method.context, 'cpu')
                damaged = bytearray(packet)
                damaged[-1] ^= 1
                try:
                    decode(bytes(damaged), kind, method.context, 'cpu')
                except AssertionError:
                    pass
                else:
                    raise AssertionError('tampered V6 packet accepted')
                try:
                    decode(packet, kind, b'wrong-context---', 'cpu')
                except AssertionError:
                    pass
                else:
                    raise AssertionError('wrong V6 context accepted')
            row = dict(arm=arm, pass_math=True, dimension=method.dimension,
                       covariance_dimension=len(method.columns), diagnostics=method.diag,
                       bytes_per_client={direction: sum(v[4] for v in run.bytes if v[1] == 0 and v[2] == direction)
                                         for direction in ('up', 'down')}, sampled_rows=len(sample))
        except (AssertionError, torch.linalg.LinAlgError) as error:
            row = dict(arm=arm, pass_math=False, reason=str(error))
        results.append(row)
        print(arm, 'PASS' if row['pass_math'] else 'FAIL', row.get('reason', ''), flush=True)
    out = ROOT / 'papers/v6'
    out.mkdir(exist_ok=True)
    proof = dict(scope='512 uniform seeded real train rows/client; CPU math only; no F1 or two-GPU proof',
                 results=results, source={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (ROOT / 'mh').glob('v6*.py')})
    (out / 'math-probe.json').write_text(json.dumps(proof, indent=2))
    assert all(row['pass_math'] for row in results), 'V6 bounded math gate failed; inspect before pilot'


if __name__ == '__main__':
    main()
