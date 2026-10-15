import hashlib,json
from pathlib import Path
import numpy as np
import torch
from mh.metrics import METRIC_KEYS,metrics_from_confusion,per_class_from_confusion
from mh.dagsnet import build_model,CFG
from mh.pilot import ROOT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run, expected, fixture_sha, expected_rounds=2, calibration=True):
    identity=json.loads((run/'config.json').read_text())
    fp=identity.pop('fingerprint')
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==fp
    assert identity['fixture']==fixture_sha and identity['config']['rounds']==20
    assert identity['config']['seed'] in (42,43) and not identity['config']['smoke']
    for name,digest in identity['source'].items():assert expected[name.split('/',1)[1]]==digest
    rows=sorted((run/'complete').glob('round_*.json'))
    assert len(rows)==expected_rounds, ('wrong committed round count',len(rows),expected_rounds)
    classes=json.loads((ROOT/'knowledge/meta.json').read_text())['class_names']
    previous=None
    metric_checks=0
    for r,file in enumerate(rows,1):
        row=json.loads(file.read_text());assert row['round']==r and row['fingerprint']==fp and row['prev_sha']==previous
        for folder,key,ext in [('weights','weights_sha','pt'),('resume','resume_sha','pt'),('predictions','predictions_sha','npz')]:
            assert sha(run/folder/f'round_{r:03d}.{ext}')==row[key]
        if 'payload_sha' in row:
            assert sha(run/'knowledge'/f'round_{r:03d}.bin')==row['payload_sha']
        previous=row['weights_sha']
        weights=torch.load(run/'weights'/f'round_{r:03d}.pt',map_location='cpu',weights_only=True)
        resume=torch.load(run/'resume'/f'round_{r:03d}.pt',map_location='cpu',weights_only=True)
        assert weights['round']==resume['round']==r
        assert len(weights['private'])==len(row['clients'])==20
        for sd in weights['private']+weights['proxy']:
            assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
        mass=resume['algorithm']['mass'];logs=resume['algorithm']['log_pi_effective']
        for k,item in enumerate(row['clients']):
            assert item['applied']>0 and item['skipped']<=16
            prior=mass[k]/item['applied']+1./item['rows']
            assert torch.allclose((prior/prior.sum()).log().float(),logs[k],atol=1e-6,rtol=1e-6)
        with np.load(run/'predictions'/f'round_{r:03d}.npz') as pred:
            for role,holds in row['eval'].items():
                for hold,rules in holds.items():
                    y=pred[hold+'_y'];assert np.isin(y,np.arange(16)).all()
                    for rule,ev in rules.items():
                        values=pred[f'{role}_{hold}_{rule}'];assert values.shape==(20,len(y))
                        cms=np.stack([np.bincount(y.astype('int64')*16+p,minlength=256).reshape(16,16) for p in values])
                        assert np.array_equal(cms,np.asarray(ev['confusion_clients']))
                        mm=[metrics_from_confusion(c) for c in cms]
                        for key in METRIC_KEYS:
                            assert np.isclose(np.mean([m[key] for m in mm]),ev['mean_metrics'][key],atol=1e-12,rtol=1e-12)
                            metric_checks+=1
                        pc=np.mean([[p['f1'] for p in per_class_from_confusion(c,classes)] for c in cms],axis=0)
                        assert np.allclose(pc,ev['f1_per_class'],atol=1e-12,rtol=1e-12)
                        assert np.isclose(pc[0],ev['benign_f1'],atol=1e-12,rtol=1e-12)
        model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).eval()
        model.load_state_dict(weights['private'][0],strict=True)
        assert sum(p.numel() for p in model.parameters())==395024
        del model,weights,resume
    cost=None
    if calibration:
        cost=json.loads((run/'cost_gate.json').read_text())
        assert np.allclose(cost['algorithm_seconds'],[json.loads(f.read_text())['algorithm_seconds'] for f in rows])
        assert np.isclose(cost['mean_seconds'],np.mean(cost['algorithm_seconds']))
        assert np.isclose(cost['ratio'],cost['mean_seconds']/cost['baseline_mean_seconds'])
        assert cost['pass_cost']==(cost['ratio']<=2)
    return dict(run=run.name,rounds=len(rows),metric_checks=metric_checks,cost=cost,
                checkpoint_strict_rebuild=True,sampled_saved_prediction_inference=False,utility_r20=False)

