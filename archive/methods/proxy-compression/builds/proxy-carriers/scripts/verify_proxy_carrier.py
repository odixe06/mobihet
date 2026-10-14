"""V2 model/prediction audit plus actual V3 projection, local residual and two-stage bytes."""
import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from verify_v2 import verify as common_verify
from mh.dagsnet import build_model,CFG
from mh.v2_math import Layout,unpack,pack_vector
from mh.v3_math import matrix,unpad,coordinates,decode,context


def verify(out,complete=True,infer=False):
    out=Path(out); cfg=json.loads((out/'config.json').read_text())['config']; arm=cfg['arm']
    if arm=='a_dense32': return common_verify(out,complete,infer)
    audit=common_verify(out,complete,infer,payload_unpack=decode)
    family=arm.split('_')[1]; own='own' in arm; ef=arm.endswith('_ef')
    torch.manual_seed(cfg['seed']); layout=Layout(build_model(dict(CFG,n_features=66,num_classes=16),latent=True))
    torch.manual_seed(cfg['seed']); initial=layout.flatten(build_model(dict(CFG,n_features=66,num_classes=16),latent=True))
    old_global=initial; old_own=[initial for _ in range(20)]; old_res=[torch.zeros_like(initial) for _ in range(20)]
    n=[256 if cfg['smoke'] else len(np.load(ROOT/f'papers/pilot/fixture/train_{k}_y.npy')) for k in range(20)]
    ksize=512 if 'small' in arm else 24576
    for marker in sorted((out/'complete').glob('round_*.json')):
        row=json.loads(marker.read_text()); r=row['round']
        state=torch.load(out/'resume'/f'round_{r:03d}.pt',map_location='cpu',weights_only=True)
        algo=state['algorithm']; mask=algo['mask']; q=algo['q']
        assert len(algo['local_delta'])==len(algo['client_residual'])==len(algo['client_coeff'])==20
        if mask is not None: assert torch.equal(mask,coordinates(layout,ksize,r,family=='t'))
        else: assert torch.linalg.matrix_norm(q.T@q-torch.eye(q.shape[1]))<=1e-5
        def reconstruct(coeff,left=None):
            if mask is not None:
                value=torch.zeros_like(initial); value[mask]=coeff; return value
            return unpad(coeff.reshape(512,-1)@q.T) if left is None else unpad(left@coeff.reshape(772,-1).T)
        for k,(delta,residual,coeff) in enumerate(zip(algo['local_delta'],algo['client_residual'],algo['client_coeff'])):
            assert all(torch.isfinite(v).all() for v in (delta,residual,coeff))
            target=delta+(old_res[k] if ef else 0)
            left=(algo['own_left'][k] if own else algo['left_factor']) if family=='p' else None
            if left is not None:
                assert torch.linalg.matrix_norm(left.T@left-torch.eye(left.shape[1]))<=1e-5
            expected=target[mask] if mask is not None else ((matrix(target)@q if left is None else matrix(target).T@left).reshape(-1))
            decoded=unpack(pack_vector('coeff',expected*n[k],count=n[k]))['values']/n[k]
            assert torch.allclose(coeff,decoded,atol=2e-5,rtol=1e-5)
            restored=reconstruct(coeff,left)
            if ef:
                assert torch.allclose(residual,target-restored,atol=2e-5,rtol=1e-5)
                assert residual.norm()<=target.norm()*(1+1e-5)+2e-5
            else: assert torch.count_nonzero(residual)==0
            if family=='t' and arm.endswith('_train'):
                outside=torch.ones(len(delta),dtype=torch.bool); outside[mask]=False
                assert torch.count_nonzero(delta[outside])==0
            if own: assert torch.allclose(algo['own_vectors'][k],old_own[k]+restored,atol=2e-5,rtol=1e-5)
        if not own:
            token=context(cfg,r,'Q' if family=='p' else 'proxy_update',old_global,mask,q,algo.get('left_factor'))
            decoded=decode((out/'knowledge'/f'round_{r:03d}.bin').read_bytes(),expected_base=r-1,expected_context=token)['values']
            average=sum(c*count for c,count in zip(algo['client_coeff'],n))/sum(n)
            assert torch.allclose(decoded,average,atol=2e-5,rtol=1e-5)
            assert torch.allclose(algo['global_vector'],old_global+reconstruct(decoded,algo.get('left_factor')),atol=2e-5,rtol=1e-5)
            if family=='p':
                first=decode(bytes(algo['left_packet']),expected_base=r-1,
                    expected_context=context(cfg,r,'P_basis',old_global,mask,q))
                assert torch.equal(first['values'].reshape_as(algo['left_factor']),algo['left_factor'])
                records=[v for v in state['bytes'] if v[0]==r]
                assert len(records)==80 and {v[3] for v in records}=={'P','Q','P_basis'}
        old_global=algo['global_vector']; old_own=algo['own_vectors']; old_res=algo['client_residual']
    audit['v3']='actual projected messages, local EF identity/contraction tolerance, rolling mask, PowerSGD two-stage ledger'
    print('VERIFIED V3',out.name,flush=True); return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('paths',nargs='+'); p.add_argument('--partial',action='store_true')
    p.add_argument('--infer',action='store_true'); a=p.parse_args(); torch.set_num_threads(1)
    for out in a.paths: verify(out,not a.partial,a.infer)
