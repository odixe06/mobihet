"""Committed state/wire/fixed-prior and prediction-to-ten-metrics audit, with explicit CE-r2 lineage."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from mh.metrics import METRIC_KEYS,metrics_from_confusion,per_class_from_confusion
from mh.dagsnet import build_model,CFG
from rank_math import C,aggregate,decode_up,decode_down
from rank_method import unframe


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(run,manifest,fixture,fx):
    identity=json.loads((run/'config.json').read_text());fp=identity.pop('fingerprint')
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==fp
    cfg=identity['config'];assert identity['fixture']==fixture and cfg['rounds']==20 and not cfg['smoke']
    assert cfg['arm'] in ('cr_local','cr_own','cr_fl','cr_shuffle','cr_fixed','cr_classweight','cr_gaussian','cr_dml')
    for name,digest in identity['source'].items():assert manifest[name.split('/',1)[1]]==digest
    counts=np.stack([np.bincount(np.load(fx/f'train_{k}_y.npy'),minlength=C) for k in range(20)])
    log_pi=torch.from_numpy(np.log((counts+1)/(counts.sum(1,keepdims=True)+C))).float()
    log_pg=torch.from_numpy(np.log((counts.sum(0)+1)/(counts.sum()+C))).float()
    inherited=identity['inherited_rounds'];assert inherited==(0 if cfg['arm'] in ('cr_gaussian','cr_dml') else 2)
    previous=identity['fork']['sha256'] if inherited else None
    files=sorted((run/'complete').glob('round_*.json'));model=build_model(dict(CFG,n_features=66,num_classes=C),latent=True)
    rows=[];metric_checks=0
    for r,file in enumerate(files,inherited+1):
        row=json.loads(file.read_text());assert row['round']==r and row['fingerprint']==fp and row['prev_sha']==previous
        expected_lr=cfg['lr_min']+.5*(cfg['lr']-cfg['lr_min'])*(1+math.cos(math.pi*(r-1)/19))
        assert abs(row['lr']-expected_lr)<1e-12
        for kind,key,ext in [('weights','weights_sha','pt'),('resume','resume_sha','pt'),('predictions','predictions_sha','npz')]:
            assert sha(run/kind/f'round_{r:03d}.{ext}')==row[key]
        weights=torch.load(run/f'weights/round_{r:03d}.pt',map_location='cpu',weights_only=True)
        state=torch.load(run/f'resume/round_{r:03d}.pt',map_location='cpu',weights_only=True)
        assert weights['round']==state['round']==r and weights['fingerprint']==state['fingerprint']==fp
        assert len(weights['private'])==20 and len(weights['proxy'])==(20 if not inherited else 0)
        assert torch.allclose(state['true_log_pi'],log_pi,rtol=1e-6,atol=1e-6)
        assert torch.allclose(state['log_pg'],log_pg,rtol=1e-6,atol=1e-6)
        for sd in weights['private']+weights['proxy']:
            model.load_state_dict(sd,strict=True)
            assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
        assert sum(p.numel() for p in model.parameters())==395024
        for k,client in enumerate(row['clients']):
            assert client['rows']==int(counts[k].sum()) and client['applied']>0 and client['skipped']<=16
            assert client['applied']+client['skipped']==math.ceil(client['rows']/cfg['batch'])
        if inherited and cfg['arm']!='cr_local':
            algorithm=state['algorithm'];snapshot=2 if cfg['arm']=='cr_fixed' else r;key=bytes.fromhex(fp)[:16]
            assert algorithm['diag']['snapshot_round']==snapshot
            wire=algorithm['wire'];inner=unframe(wire,snapshot,key);expected,cohort=aggregate(algorithm['up'],key)
            assert expected==inner and cohort==algorithm['diag']['cohort']
            means,ng,eligible,N=decode_down(inner,key)
            assert sha(run/f'knowledge/round_{r:03d}.bin')==row['payload_sha']
            assert (run/f'knowledge/round_{r:03d}.bin').read_bytes()==wire
            for k,up in enumerate(algorithm['up']):
                n,s,_=decode_up(up,key);assert np.array_equal(n[0],counts[k]);valid=eligible&(n>=32).all(0)
                used=np.divide(s[:2],n,out=np.zeros((2,C)),where=n>0) if cfg['arm']=='cr_own' else means.copy()
                cn,total=(n,int(n[0].sum())) if cfg['arm']=='cr_own' else (ng,N)
                if cfg['arm']=='cr_shuffle':used=np.roll(used,1,axis=1)
                coef=np.divide(.25*total/C,cn,out=np.zeros((2,C)),where=cn>0)*valid
                assert np.array_equal(algorithm['support'][k].numpy(),valid)
                assert np.allclose(algorithm['means'][k].numpy(),used,rtol=1e-6,atol=1e-7)
                assert np.allclose(algorithm['coefficient'][k].numpy(),coef,rtol=1e-6,atol=1e-7)
            assert torch.isfinite(algorithm['beta']).all() and (algorithm['beta']>=0).all() and (algorithm['beta']<=4).all()
        with np.load(run/f'predictions/round_{r:03d}.npz') as predictions:
            for role,holds in row['eval'].items():
                for hold,rules in holds.items():
                    y=predictions[hold+'_y'];assert np.array_equal(y,np.load(fx/f'{hold}_y.npy'))
                    for rule,ev in rules.items():
                        pred=predictions[f'{role}_{hold}_{rule}'];assert pred.shape==(20,len(y)) and np.isin(pred,np.arange(C)).all()
                        cms=np.stack([np.bincount(y.astype('int64')*C+p,minlength=C*C).reshape(C,C) for p in pred])
                        assert np.array_equal(cms,ev['confusion_clients'])
                        metrics=[metrics_from_confusion(c) for c in cms]
                        for name in METRIC_KEYS:
                            assert np.isclose(np.mean([v[name] for v in metrics]),ev['mean_metrics'][name],rtol=1e-12,atol=1e-12)
                            metric_checks+=1
                        pc=np.mean([[v['f1'] for v in per_class_from_confusion(c,list(range(C)))] for c in cms],axis=0)
                        assert np.allclose(pc,ev['f1_per_class'],rtol=1e-12,atol=1e-12) and np.isclose(pc[0],ev['benign_f1'])
        previous=row['weights_sha'];rows.append(row)
    return dict(run=run.name,committed_rounds=len(rows),inherited_rounds=inherited,
        logical_rounds=inherited+len(rows),complete=inherited+len(rows)==20,metric_checks=metric_checks,
        level='all strict SD/BN/state/wire/fixed priors and saved predictions to ten metrics; no independent saved-weight inference or NNLS refit')
