"""Teacher-only full-state model/BN payload audit and run-derived client metrics."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mh.dagsnet import CFG,build_model
from mh.model_packet import Codec
from mh.v2_math import Layout,unpack
from verify_v2 import verify as common
from check_v3_driver import equal


def verify(path,complete=True,infer=False):
    path=Path(path);cfg=json.loads((path/'config.json').read_text())['config']
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
    codec=Codec(model,f'teacher:{cfg["arm"]}:{cfg["seed"]}')
    def decode(packet,expected_base=None):
        _,base,version=codec.decode(packet,expected_base=expected_base)
        return dict(base=base,version=version,kind=254)
    def logs(name,rows,args,train):
        assert name=='proxy_clients' and [v['client'] for v in rows]==list(range(20))
        assert [v['rows'] for v in rows]==train
        for value in rows:
            if args['arm'].startswith('r') and args['arm']!='r_dense_ce':
                assert value['applied']==value['skipped']==0 and value['residual_rows']==value['rows']
            else:
                assert value['applied']>0 and value['skipped']<=16
                assert value['applied']+value['skipped']==(value['rows']+args['batch']-1)//args['batch']
    audit=common(path,complete,infer,payload_unpack=decode,proxy_count=20,log_checker=logs,smoke_train_rows=1024)
    previous_gaussian=None
    for p in sorted((path/'complete').glob('round_*.json')):
        row=json.loads(p.read_text());r=row['round']
        weights=torch.load(path/f'weights/round_{r:03d}.pt',map_location='cpu',weights_only=True)
        state=torch.load(path/f'resume/round_{r:03d}.pt',map_location='cpu',weights_only=True)['algorithm']
        assert state['version']==r
        if 'payload_sha' in row:
            decoded,_,version=codec.decode((path/f'knowledge/round_{r:03d}.bin').read_bytes(),expected_base=r-1)
            assert version==r and equal(decoded,state['global_state'])
            assert all(equal(decoded,s) for s in weights['proxy'])
            model.load_state_dict(decoded,strict=True)
            assert torch.equal(Layout(model).flatten(model),state['global_vector'])
        gaussian=(state['mu'],state['sd'])
        if previous_gaussian is not None: assert equal(gaussian,previous_gaussian)
        previous_gaussian=gaussian
        assert (state['sd']>0).all()
        if 'targets' in state:
            assert row['diagnostics']['teacher_applied']==20*len(state['targets'])
            assert len(state['targets'])==(20 if cfg['arm'].endswith('_own') else 1)
            anchor=unpack(state['anchor_packet'],'cpu')['values'].view(state['M'],66)
            assert torch.equal(anchor,state['anchors'])
            for target in state['targets']:
                assert torch.isfinite(target).all() and (target.mean(1).abs()<1e-5).all()
        if 'chains' in state:
            assert all(torch.isfinite(x).all() and x.shape==(256,66) for x in state['chains'].values())
            assert sorted(state['permutation'].tolist())==list(range(66))
    print('VERIFIED TEACHER',path.name,flush=True)
    return audit
