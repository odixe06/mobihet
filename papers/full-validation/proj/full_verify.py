"""Independent confusion/metric and coherent checkpoint verification for full runs."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mh.dagsnet import build_model,CFG
from mh.metrics import METRIC_KEYS,metrics_from_confusion,per_class_from_confusion
from mh.new_fl import digest
from mh.model_packet import Codec
from mh.v2_math import Layout
from check_v3_driver import equal


def verify(path,complete=True,infer=False):
    path=Path(path);identity=json.loads((path/'config.json').read_text());fp=identity.pop('fingerprint')
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==fp
    cfg=identity['config'];rows=[json.loads(p.read_text()) for p in sorted((path/'complete').glob('round_*.json'))]
    result=json.loads((path/'result.json').read_text())
    assert rows==result['history'] and result['fingerprint']==fp
    assert [r['round'] for r in rows]==list(range(1,len(rows)+1))
    if complete:assert result['complete'] and len(rows)==cfg['rounds']
    if not cfg['smoke']:assert cfg['seed']==42 and cfg['rounds']==50 and sum(result['train_rows'])==43045415
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True);previous=None
    proxy_count=20 if 'dml' in cfg['arm'] else 0
    codec=Codec(model,f'full:{cfg["arm"]}:{cfg["seed"]}') if proxy_count else None
    for row in rows:
        r=row['round'];assert row['prev_sha']==previous and row['fingerprint']==fp
        wp=path/f'weights/round_{r:03d}.pt';rp=path/f'resume/round_{r:03d}.pt';pp=path/f'predictions/round_{r:03d}.npz'
        assert digest(wp)==row['weights_sha'] and digest(rp)==row['resume_sha'] and digest(pp)==row['predictions_sha']
        previous=row['weights_sha']
        weights=torch.load(wp,map_location='cpu',weights_only=True)
        state=torch.load(rp,map_location='cpu',weights_only=True)
        assert weights['round']==state['round']==r and weights['fingerprint']==state['fingerprint']==fp
        assert weights['architecture']==dict(CFG,n_features=66,num_classes=16)
        assert len(weights['private'])==20 and len(weights['proxy'])==proxy_count
        for sd in weights['private']+weights['proxy']:
            model.load_state_dict(sd,strict=True)
            assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
        if 'payload_sha' in row:
            packet=path/f'knowledge/round_{r:03d}.bin';assert digest(packet)==row['payload_sha']
            decoded,_,version=codec.decode(packet.read_bytes(),expected_base=r-1)
            assert version==r and equal(decoded,state['algorithm']['global_state'])
            assert all(equal(decoded,sd) for sd in weights['proxy'])
            model.load_state_dict(decoded,strict=True)
            assert torch.equal(Layout(model).flatten(model),state['algorithm']['global_vector'])
        for name in ('clients','proxy_clients'):
            logs=row[name]
            if logs:
                assert [v['client'] for v in logs]==list(range(20))
                assert [v['rows'] for v in logs]==result['train_rows']
                assert all(v['applied']>0 and v['skipped']<=16 and v['applied']+v['skipped']==
                           (v['rows']+cfg['batch']-1)//cfg['batch'] for v in logs)
        expected_lr=cfg['lr_min']+.5*(cfg['lr']-cfg['lr_min'])*(1+np.cos(np.pi*(r-1)/max(cfg['rounds']-1,1)))
        assert np.isclose(row['lr'],expected_lr,atol=1e-12,rtol=0)
        with np.load(pp) as predictions:
            y=predictions['test_y'];assert len(y)==(256 if cfg['smoke'] else 10761343)
            if cfg['smoke']:
                assert np.array_equal(y,np.load(ROOT/'papers/pilot/fixture/group_y.npy')[:256])
            for role,holds in row['eval'].items():
                assert set(holds)=={'test'}
                for rule,ev in holds['test'].items():
                    pred=predictions[f'{role}_test_{rule}'];assert pred.shape==(20,len(y)) and (pred<16).all()
                    cc=np.stack([np.bincount(y.astype('int64')*16+p,minlength=256).reshape(16,16) for p in pred])
                    assert np.array_equal(cc,ev['confusion_clients'])
                    metric=[metrics_from_confusion(c) for c in cc]
                    assert set(ev['mean_metrics'])==set(METRIC_KEYS)
                    for key in METRIC_KEYS:
                        assert np.isclose(np.mean([m[key] for m in metric]),ev['mean_metrics'][key],atol=1e-12,rtol=0)
                    f1=np.mean([[v['f1'] for v in per_class_from_confusion(c,[str(j) for j in range(16)])] for c in cc],axis=0)
                    assert np.allclose(f1,ev['f1_per_class'],atol=1e-12,rtol=0)
                    assert np.isclose(f1[0],ev['benign_f1'])
            if infer and cfg['smoke'] and r in (1,len(rows)):
                x=np.load(ROOT/'papers/pilot/fixture/group_X.npy')[:256]
                model.load_state_dict(weights['private'][0],strict=True);model.cuda().eval()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):z=model(torch.from_numpy(x).cuda().float())[0].float()
                margin=z.topk(2,1).values.diff(dim=1).abs().squeeze(1);decisive=margin>.02
                stored=torch.from_numpy(predictions['private_test_head'][0].astype('int64')).cuda()
                assert torch.equal(z.argmax(1)[decisive],stored[decisive]);model.cpu()
    print('VERIFIED FULL',path.name,len(rows),'rounds',flush=True)
    return dict(run=path.name,rounds=len(rows),complete=result['complete'])
