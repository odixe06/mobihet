"""Terminal pull and audit for BoundaryCells calibration; never launch or stop training."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import nbformat
import numpy as np
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
BASE=ROOT/'.compat/papers/boundary-cells'
HELPER=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS,metrics_from_confusion,per_class_from_confusion
from mh.dagsnet import build_model,CFG


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


def collect(folder, utility=False):
    metadata=json.loads((folder/'kernel-metadata.json').read_text());kernel=metadata['id'];owner=kernel.split('/')[0]
    dest=BASE/'kaggle'/folder.name;marker=dest/'AUDITED.json'
    if marker.exists():return True
    prefix=[sys.executable,str(HELPER/'kaggle_as.py'),owner,'--']
    result=subprocess.run(prefix+['kaggle','kernels','status',kernel],capture_output=True,text=True,check=True)
    print(result.stdout.strip(),flush=True)
    if not any(word in result.stdout for word in ('COMPLETE','ERROR')):return False
    subprocess.run(prefix+[sys.executable,str(HELPER/'pull_kernel_output.py'),kernel,'-p',str(dest),'-j','6',
                   '--pattern',r'^(source/|runs/(?![^/]+/wandb/))'],check=True)
    private=BASE/'private_push'/folder.name
    executed=private/'executed.ipynb'
    subprocess.run(prefix+[sys.executable,str(HELPER/'download_executed_notebook.py'),kernel,'--version','1',
                   '--output',str(executed)],check=True,umask=0o077)
    executed.chmod(0o600)
    reference=nbformat.read(folder/metadata['code_file'],as_version=4)
    pushed=nbformat.read(private/metadata['code_file'],as_version=4)
    key=next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n,ast.Assign) and
             any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
    actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4)
    assert len(actual.cells)==len(reference.cells)
    for i,cell in enumerate(actual.cells):
        if cell.cell_type=='code':assert cell.source==pushed.cells[i].source.replace(key,'[REDACTED]')
    actual.cells[1].source=reference.cells[1].source
    actual.metadata['credential_cell_redacted']=True
    nbformat.write(actual,folder/'executed.redacted.ipynb')
    expected=json.loads((folder/'source_manifest.json').read_text())
    source=dest/'source'
    assert json.loads((source/'source_manifest.json').read_text())==expected
    for name,digest in expected.items():
        assert sha(source/name)==digest
        assert key not in (source/name).read_text(errors='ignore')
    fixture_sha=sha(source/'fixture_manifest.json')
    path=dest/'runs'/('utility_summary.json' if utility else 'calibration_summary.json')
    assert path.exists(), 'terminal output missing summary; inspect failure before retry'
    target_arms=None
    if utility:
        node=next(n for n in ast.parse(reference.cells[3].source).body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='METHODS' for t in n.targets))
        target_arms=ast.literal_eval(node.value)
    summary=json.loads(path.read_text());assert len(summary)==(len(target_arms) if utility else 5)
    audits=[]
    with (ROOT/'.compat/papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,item in summary.items():
            if utility:
                assert item['target_rounds']==20 and item.get('rc',0)==0
                rounds=item['rounds']
                run=dest/'runs'/name
                assert len(list((run/'complete').glob('round_*.json')))==rounds
                if rounds:
                    audits.append(verify(run,expected,fixture_sha,expected_rounds=rounds,calibration=False))
            else:
                assert item['calibration_valid'] and item['rc']==0
                audits.append(verify(dest/'runs'/name,expected,fixture_sha))
    if not utility:
        for seed in {int(name.rsplit('_s',1)[1]) for name in summary}:
            baseline=next(a['cost']['mean_seconds'] for a in audits if a['run']==f'bc_single_s{seed}')
            assert all(np.isclose(a['cost']['baseline_mean_seconds'],baseline) for a in audits if a['run'].endswith(f'_s{seed}'))
    record=dict(kernel=kernel,version=1,source=expected,summary=summary,audits=audits,
                calibration_valid=not utility,
                target_arms=target_arms,
                utility_r20=utility and all(v['complete'] and v['rounds']==20 for v in summary.values()))
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(marker)
    print('AUDITED UTILITY' if utility else 'AUDITED CALIBRATION',kernel,
          'all arms r20',record['utility_r20'],flush=True)
    return True


if __name__=='__main__':
    torch.set_num_threads(1)
    folders=sorted(p.parent for p in (BASE/'notebook').glob('bc-cal-s4?-20261013-initial2-v1/PUSHED'))
    assert len(folders)==2
    while True:
        results=[collect(folder) for folder in folders]
        if all(results):break
        time.sleep(55)
