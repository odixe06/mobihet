"""Static Nyström aggregate/factor packets plus committed model/prediction verification."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from verify_v2 import verify as common
from check_v3_driver import equal
from mh.dagsnet import build_model, CFG
from mh.v7_nystrom import Method, decode
from mh.v7_basis import OMEGA_BYTES


def verify(path, complete=True, infer=False):
    path = Path(path)
    cfg = json.loads((path/'config.json').read_text())['config']
    audit = common(path, complete, infer, proxy_count=0, smoke_train_rows=1024)
    method = Method(SimpleNamespace(init=build_model(dict(CFG,n_features=66,num_classes=16),latent=True),
                                   dev=torch.device('cpu'), a=SimpleNamespace(**cfg)), initialize=False)
    before = None
    for p in sorted((path/'complete').glob('round_*.json')):
        row = json.loads(p.read_text())
        state = torch.load(path/f'resume/round_{row["round"]:03d}.pt',map_location='cpu',weights_only=True)['algorithm']
        assert state['version'] == 0 and len(state['tables']) == (20 if method.mode == 'own' else 1)
        if before is not None:
            assert equal(before, state), 'static bootstrap changed across rounds'
        if method.form.startswith('nys'):
            _,_,basis = decode(state['packets']['basis'],3,528,method.context)
            assert basis.tobytes() == OMEGA_BYTES
        for table in state['tables']:
            assert table['mu'].shape == (16,66) and all(torch.isfinite(v).all() for v in table.values())
            if 'sd' in table:
                assert (table['sd'] > 0).all()
            else:
                assert (table['factor'].diagonal(dim1=1,dim2=2) > 0).all()
        if method.mode != 'own':
            n, contributors = state['aggregate_counts'], state['aggregate_contributors']
            assert n.shape == contributors.shape == (16,) and (n>0).all() and (contributors>=5).all()
            expected, _ = method.reconstruct(n.numpy(),state['aggregate_statistics'].numpy())
            size = expected.size
            _,_,wire = decode(state['packets']['factors'],2,size,method.context)
            wire = wire.reshape(expected.shape)
            assert np.array_equal(wire[:,:66], expected[:,:66].astype('float16').astype('float64'))
            # Eigenvector signs/bases can differ across BLAS runtimes; certify represented covariance.
            reference_cov, actual_cov = method.factor_covariance(expected), method.factor_covariance(wire)
            error = np.linalg.norm(actual_cov-reference_cov,axis=(1,2))/np.maximum(np.linalg.norm(reference_cov,axis=(1,2)),1e-4)
            assert (error <= .010001).all(), 'factor packet inconsistent with aggregate covariance'
            actual = method.table(wire)
            if method.mode == 'shuffle':
                signs = np.random.default_rng(7603).integers(0,2,66)*2-1
                actual['factor'] *= torch.from_numpy(signs.astype('float32'))[None,:,None]
            assert equal(actual,state['tables'][0]), 'training table differs from decoded factor packet'
        before = state
    print('VERIFIED V7',path.name,flush=True)
    return audit


if __name__ == '__main__':
    torch.set_num_threads(1)
    for value in sys.argv[1:]:
        verify(value,infer=True)
