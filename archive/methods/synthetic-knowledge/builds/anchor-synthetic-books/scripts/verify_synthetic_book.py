"""V5 anchor statistics/book invariants, actual optimizer rows and independent model metrics."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from verify_v2 import verify as common
from mh.v2_math import unpack


def logs(name,records,cfg,train):
    assert [v['client'] for v in records]==list(range(20))
    anchor=cfg['arm'].startswith('b_anchor_')
    if anchor:
        assert name=='proxy_clients'; m=32 if 'small' in cfg['arm'] else 512
        assert [v['rows'] for v in records]==[20*m]*20
        assert [v['real_stat_rows'] for v in records]==train
        assert all(v['anchor_batch']==m and v['applied']+v['skipped']==20 for v in records)
    else:
        assert [v['rows'] for v in records]==train
        assert all(v['applied']+v['skipped']==int(np.ceil(v['rows']/cfg['batch'])) for v in records)
    assert all(v['applied']>0 and v['skipped']<=16 for v in records)


def parameters_hash(sd):
    hash_=hashlib.sha256()
    from mh.dagsnet import build_model,CFG
    names=set(dict(build_model(dict(CFG,n_features=66,num_classes=16),latent=True).named_parameters()))
    for name,value in sorted(sd.items()):
        if name in names: hash_.update(name.encode()); hash_.update(value.contiguous().numpy().tobytes())
    return hash_.hexdigest()


def verify(out,complete=True,infer=False):
    out=Path(out); cfg=json.loads((out/'config.json').read_text())['config']; arm=cfg['arm']
    anchor=arm.startswith('b_anchor_'); book=arm.startswith(('g1_','g4_'))
    result=common(out,complete,infer,proxy_count=0 if book else None,log_checker=logs)
    rows=[json.loads(p.read_text()) for p in sorted((out/'complete').glob('round_*.json'))]
    first=None; previous=None
    for row in rows:
        r=row['round']; state=torch.load(out/f'resume/round_{r:03d}.pt',map_location='cpu',weights_only=True)
        algorithm=state['algorithm']
        if anchor:
            m=32 if 'small' in arm else 512
            assert algorithm['anchors'].shape==(m,66)
            assert torch.equal(unpack(algorithm['anchor_packet'])['values'].reshape(m,66),algorithm['anchors'])
            assert algorithm['h'] in (1.,2.,4.,66**.5) and algorithm['coverage_per_class'].min()>=.8
            assert len(algorithm['coverage_packet'])==100
            selected=unpack(algorithm['coverage_packet'])['values']; assert float(selected[0])==np.float32(algorithm['h'])
            assert torch.equal(selected[1:],algorithm['coverage_per_class'])
            if algorithm['target_packet']:
                raw=unpack(algorithm['target_packet'])['values'].reshape(m,16)
                assert torch.equal(raw/raw.sum(1,keepdim=True).clamp_min(1e-12),algorithm['target'])
            else: assert len(algorithm['own_targets'])==20
            if first is None: first=algorithm
            else:
                assert torch.equal(first['anchors'],algorithm['anchors']) and first['anchor_packet']==algorithm['anchor_packet']
                assert first['target_packet']==algorithm['target_packet']
            assert not row['clients'] and algorithm['version']==0
        elif book:
            assert not row['proxy_clients'] and algorithm['round_no']==r
            mode=arm.split('_',1)[1]; ipc=1 if arm.startswith('g1_') else 4
            if mode=='diag': assert algorithm['phase']==0 and not algorithm['book_packets']
            else:
                period=2 if cfg['smoke'] else 5; phase=(r-1)//period+1
                assert algorithm['phase']==phase and algorithm['version']==phase
                assert algorithm['books'].shape==(20,16,ipc,66) and torch.isfinite(algorithm['books']).all()
                for k,packet in enumerate(algorithm['book_packets']):
                    decoded=unpack(packet,expected_base=phase-1)
                    assert decoded['version']==phase and len(packet)==32+16*ipc*66*2
                    assert torch.equal(decoded['values'].reshape(16,ipc,66),algorithm['books'][k])
                assert len(algorithm['peers'])==20
                for k,peers in enumerate(algorithm['peers']):
                    assert peers==([k] if mode=='own' else [(k+1+((phase-1)*4+j)%19)%20 for j in range(4)])
                if r%period!=1 and previous is not None:
                    assert torch.equal(algorithm['books'],previous['books'])
                if r%period==1 and r>1 and mode in ('mean_trained','grad_trained','own'):
                    weights=torch.load(out/f'weights/round_{r-1:03d}.pt',map_location='cpu',weights_only=True)
                    for k,producer in enumerate(algorithm['diag']['producers']):
                        assert producer['encoder_parameter_sha256']==parameters_hash(weights['private'][k])
        previous=algorithm
    print('VERIFIED V5',out.name)
    return result
