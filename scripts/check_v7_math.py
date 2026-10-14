"""Bounded actual-codec sampler checks and covariance-preserving shuffle control."""
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
from mh.v7_nystrom import Method, ARMS, decode


def covariance(table):
    if 'sd' not in table:
        return table['factor'].double() @ table['factor'].double().transpose(1, 2)
    result = torch.diag_embed(table['sd'].double().square())
    if 'factor' in table:
        result += table['factor'].double() @ table['factor'].double().transpose(1, 2)
    return result


def main():
    torch.set_num_threads(1)
    fx = ROOT / 'papers/pilot/fixture'
    x, y = [], []
    for k in range(20):
        labels = np.load(fx / f'train_{k}_y.npy', mmap_mode='r')
        indices = np.random.default_rng(7602+k).choice(len(labels), 512, replace=False)
        x.append(torch.from_numpy(np.load(fx / f'train_{k}_X.npy', mmap_mode='r')[indices].astype('float32')))
        y.append(torch.from_numpy(labels[indices].astype('int64')))
    init = build_model(dict(CFG, n_features=66, num_classes=16), latent=True)
    results, reference = [], {}
    for arm in ARMS:
        run = SimpleNamespace(dev=torch.device('cpu'), init=init, X=x, Y=y, K=20,
                              n=[512]*20, bytes=[], a=SimpleNamespace(arm=arm, seed=42))
        method = Method(run)
        labels = torch.arange(4096) % 16
        sample = method.draw(labels, torch.Generator().manual_seed(7604), 0)
        assert sample.shape == (4096,66) and torch.isfinite(sample.half()).all()
        for kind, packet in method.packets.items():
            actual = 3 if kind == 'basis' else 2
            size = 528 if actual == 3 else 16*(66+66*67//2 if method.form == 'full' else 66*(method.rank+2))
            decode(packet, actual, size, method.context)
            for damaged, context in ((packet[:-1], method.context),
                                     (packet[:-1]+bytes([packet[-1]^1]), method.context),
                                     (packet, bytes([method.context[0]^1])+method.context[1:])):
                try:
                    decode(damaged, actual, size, context)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('tampered packet/context accepted')
        if method.mode == 'fl':
            reference[method.form] = method.tables[0]
        if method.mode == 'shuffle':
            before, after = covariance(reference[method.form]), covariance(method.tables[0])
            assert torch.equal(reference[method.form]['mu'], method.tables[0]['mu'])
            assert torch.equal(reference[method.form]['sd'], method.tables[0]['sd'])
            assert torch.equal(before.diagonal(dim1=1,dim2=2), after.diagonal(dim1=1,dim2=2))
            assert torch.allclose(torch.linalg.eigvalsh(before), torch.linalg.eigvalsh(after), rtol=1e-10, atol=1e-10)
            assert not torch.allclose(before, after)
        minimum = torch.linalg.eigvalsh(covariance(method.tables[0])).min()
        assert minimum >= (0. if method.form == 'full' else .5e-4)
        results.append(dict(arm=arm, finite=True, tamper_rejected=True,
            bytes_per_client={d:sum(v[-1] for v in run.bytes if v[1]==0 and v[2]==d) for d in ('up','down')},
            diagnostics=method.diag))
        print('V7 math PASS', arm, results[-1]['bytes_per_client'], flush=True)
    out = ROOT / 'papers/v7'
    out.mkdir(exist_ok=True)
    (out/'math-probe.json').write_text(json.dumps(dict(scope='512uniformtrainrow/client,CPU,noF1', results=results,
        source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'mh').glob('v7*.py')}), indent=2))


if __name__ == '__main__':
    main()
