"""K5 packet/table invariants and inherited model/prediction verification."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_v2 import verify as common
from mh.dagsnet import build_model, CFG
from mh.v6_geometry import Method, decode


def verify(out, complete=True, infer=False):
    out = Path(out)
    cfg = json.loads((out / 'config.json').read_text())['config']
    result = common(out, complete, infer, proxy_count=0, smoke_train_rows=1024)
    model = build_model(dict(CFG, n_features=66, num_classes=16), latent=True)
    method = Method(SimpleNamespace(init=model, dev=torch.device('cpu'),
                                   a=SimpleNamespace(**cfg)), initialize=False)
    previous = None
    for row in sorted((out / 'complete').glob('round_*.json')):
        marker = json.loads(row.read_text())
        state = torch.load(out / f'resume/round_{marker["round"]:03d}.pt', map_location='cpu', weights_only=True)
        algorithm = state['algorithm']
        tables = algorithm['tables']
        assert len(tables) == (20 if method.mode == 'own' else 1)
        assert algorithm['version'] == 0
        for table in tables:
            assert len(table['keys']) == len(table['mu']) == len(table['chol'])
            assert table['keys'][:16] == [(c, -1) for c in range(16)]
            assert (table['chol'].diagonal(dim1=-2, dim2=-1) > 0).all()
            assert torch.isfinite(table['mu']).all() and torch.isfinite(table['chol']).all()
            assert (table['mapping'] >= 0).all() and (table['mapping'] < len(table['keys'])).all()
        for kind, packet in algorithm['packets'].items():
            values = decode(packet, kind, method.context, 'cpu')
            if kind == 'factors':
                table = tables[0]
                d = len(method.columns)
                lower = torch.tril_indices(d, d)
                actual = torch.cat([table['mu'], table['chol'][:, lower[0], lower[1]]], 1)
                assert torch.equal(values.float().reshape(actual.shape), actual)
            if kind == 'mask':
                assert torch.equal(values.float().reshape(tables[0]['parameters'].shape), tables[0]['parameters'])
            if kind == 'eligibility':
                assert ((values == 0) | (values == 1)).all()
                keys = [tuple(v) for v in values.reshape(16,32).bool().nonzero().tolist()]
                assert tables[0]['keys'][16:] == keys
        if previous is not None:
            assert algorithm['packets'] == previous['packets']
            for current, old in zip(tables, previous['tables']):
                for name in ('mu', 'chol', 'mapping', 'parameters'):
                    assert torch.equal(current[name], old[name])
        expected = sum((v['rows'] + cfg['batch']-1)//cfg['batch'] for v in marker['clients']) * cfg['aux_batch']
        assert marker['diagnostics']['sampler']['rows'] == expected
        previous = algorithm
    print('VERIFIED V6', out.name)
    return result
