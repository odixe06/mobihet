"""Audit V2 committed models/state/payloads/predictions, independent metric reconstruction."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG
from mh.metrics import METRIC_KEYS,metrics_from_confusion,per_class_from_confusion
from mh.new_fl import digest
from mh.v2_math import unpack,Layout,CountSketch,KINDS


def verify(out,complete=True,infer=False,payload_unpack=unpack,proxy_count=None,log_checker=None,smoke_train_rows=256):
    out=Path(out); identity=json.loads((out/'config.json').read_text()); fingerprint=identity.pop('fingerprint')
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==fingerprint
    a=identity['config']
    fx=ROOT/'papers/pilot/fixture'; assert digest(fx/'manifest.json')==identity['fixture']
    labels={h:np.load(fx/f'{h}_y.npy')[:256 if a['smoke'] else None] for h in ('old','group')}
    train=[smoke_train_rows if a['smoke'] else len(np.load(fx/f'train_{k}_y.npy')) for k in range(20)]
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
    assert Layout(model).size==395024
    rows=[json.loads(f.read_text()) for f in sorted((out/'complete').glob('round_*.json'))]
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text())
    else:
        assert not complete and rows,'no complete result or committed evidence'
        result=dict(fingerprint=fingerprint,history=rows,complete=False,train_rows=train,
                    role='proxy' if a['arm'].startswith('b_') else 'private')
    assert result['fingerprint']==fingerprint
    assert result['train_rows']==train
    assert rows==result['history'] and [r['round'] for r in rows]==list(range(1,len(rows)+1))
    if complete: assert len(rows)==a['rounds'] and result['complete']
    previous=None; previous_algorithm=None; derived={}; inferences=[]
    expected_eval=[r for r in range(1,len(rows)+1) if r%a['eval_every']==0 or r==a['rounds']]
    assert [r['round'] for r in rows if 'eval' in r]==expected_eval
    for row in rows:
        r=row['round']; wp=out/'weights'/f'round_{r:03d}.pt'; rp=out/'resume'/f'round_{r:03d}.pt'
        assert row['fingerprint']==fingerprint and row['prev_sha']==previous
        assert digest(wp)==row['weights_sha'] and digest(rp)==row['resume_sha']; previous=row['weights_sha']
        state=torch.load(wp,map_location='cpu',weights_only=True)
        resume=torch.load(rp,map_location='cpu',weights_only=True); algorithm=resume['algorithm']
        assert state['round']==resume['round']==r and state['fingerprint']==resume['fingerprint']==fingerprint
        assert state['architecture']==dict(CFG,n_features=66,num_classes=16)
        expected_proxies=(0 if a['arm'] in ('local','diag') else 20) if proxy_count is None else proxy_count
        assert len(state['private'])==20 and len(state['proxy'])==expected_proxies
        for sd in state['private']+state['proxy']:
            model.load_state_dict(sd,strict=True)
            assert all(torch.isfinite(t).all() for t in sd.values() if t.is_floating_point())
        for key in ('global_vector','coeff','residual','last_mean','mu','sd'):
            if key in algorithm: assert torch.isfinite(algorithm[key]).all(),key
        assert algorithm['version']==row['version']
        if 'payload_sha' in row:
            pp=out/'knowledge'/f'round_{r:03d}.bin'; assert digest(pp)==row['payload_sha']
            decoded=payload_unpack(pp.read_bytes(),expected_base=r-1); assert decoded['version']==r
            if decoded['kind']==KINDS['dense']:
                assert torch.equal(decoded['values'],algorithm['global_vector'])
            if decoded['kind']==KINDS['coeff']:
                assert torch.equal(decoded['values'],algorithm['coeff'])
            if decoded['kind']==KINDS['patch']:
                old=previous_algorithm['global_vector'] if previous_algorithm else None
                if old is not None:
                    delta=torch.zeros_like(old); delta[decoded['indices']]=decoded['values']
                    assert torch.equal(old+delta,algorithm['global_vector'])
                    depth,width=(3,256) if 'small' in a['arm'] else (5,2048)
                    sk=CountSketch(395024,depth,width,1701,'cpu')
                    target=algorithm['last_mean']+previous_algorithm['residual']-sk.encode(delta)
                    if a['arm'].endswith('_ef'):
                        assert torch.allclose(target,algorithm['residual'],atol=2e-5,rtol=1e-5)
                    else: assert torch.count_nonzero(algorithm['residual'])==0
        for name in ('clients','proxy_clients'):
            logs=row[name]
            if logs:
                if log_checker is not None: log_checker(name,logs,a,train)
                else:
                    assert [v['client'] for v in logs]==list(range(20)) and [v['rows'] for v in logs]==train
                    assert all(v['applied']>0 and v['skipped']<=16 and v['applied']+v['skipped']==int(np.ceil(v['rows']/a['batch'])) for v in logs)
        lr=a['lr_min']+.5*(a['lr']-a['lr_min'])*(1+np.cos(np.pi*(r-1)/max(a['rounds']-1,1)))
        assert np.isclose(row['lr'],lr,rtol=0,atol=1e-12)
        if 'eval' in row:
            pf=out/'predictions'/f'round_{r:03d}.npz'; assert digest(pf)==row['predictions_sha']
            with np.load(pf) as preds:
                for role,holds in row['eval'].items():
                    for hold,rules in holds.items():
                        y=preds[hold+'_y']; assert np.array_equal(y,labels[hold])
                        for rule,ev in rules.items():
                            pr=preds[f'{role}_{hold}_{rule}']; assert pr.shape==(20,len(y)) and (pr<16).all()
                            cc=np.stack([np.bincount(y.astype('int64')*16+p,minlength=256).reshape(16,16) for p in pr])
                            assert np.array_equal(cc,ev['confusion_clients'])
                            mm=[metrics_from_confusion(c) for c in cc]
                            means={key:float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS}
                            assert set(ev['mean_metrics'])==set(METRIC_KEYS)
                            for key,value in means.items(): assert np.isclose(value,ev['mean_metrics'][key],rtol=0,atol=1e-12)
                            pc=np.mean([[v['f1'] for v in per_class_from_confusion(c,[str(j) for j in range(16)])] for c in cc],axis=0)
                            assert np.allclose(pc,ev['f1_per_class'],atol=1e-12,rtol=0)
                            assert np.isclose(pc[0],ev['benign_f1']) and np.isclose(means['f1_macro'],ev['f1_macro'])
                            derived[(r,role,hold,rule)]=means
                if infer and r in (expected_eval[0],expected_eval[-1]):
                    role=result['role']; hold='group'; rule='head'
                    model.load_state_dict(state['proxy' if role=='proxy' else 'private'][0],strict=True)
                    model.to('cuda').eval()
                    # Same first evaluation batch size; tolerate cross-runtime near ties explicitly.
                    x=np.load(fx/f'{hold}_X.npy')[:min(16384,len(labels[hold]))]
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                        z=model(torch.from_numpy(x).cuda().float())[0].float()
                    margin=z.topk(2,dim=1).values.diff(dim=1).abs().squeeze(1)
                    decisive=margin>.02
                    old=torch.from_numpy(preds[f'{role}_{hold}_{rule}'][0,:len(x)].astype('int64')).cuda()
                    assert torch.equal(z.argmax(1)[decisive],old[decisive]),'saved weights disagree on decisive rows'
                    inferences.append(dict(round=r,role=role,rows=len(x),decisive=int(decisive.sum()),ambiguous=int((~decisive).sum())))
                    model.cpu()
        previous_algorithm=algorithm
    if (out/'metrics.csv').exists():
        with (out/'metrics.csv').open() as f: csvrows=list(csv.DictReader(f))
        assert len(csvrows)==len(derived)
        for row in csvrows:
            mm=derived.pop((int(row['round']),row['role'],row['holdout'],row['rule']))
            for key,value in mm.items(): assert np.isclose(float(row[key]),value,rtol=0,atol=1e-12)
        assert not derived
    else: assert not complete, 'complete run missing metrics CSV'
    audit=dict(run=out.name,rounds=len(rows),complete=result['complete'],inference=inferences,
               level='strict model/state/payload + saved predictions to client metrics; optional sampled decisive inference')
    print('VERIFIED',json.dumps(audit),flush=True)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('paths',nargs='+'); p.add_argument('--partial',action='store_true')
    p.add_argument('--infer',action='store_true'); a=p.parse_args()
    torch.set_num_threads(1)
    for path in a.paths: verify(path,not a.partial,a.infer)
