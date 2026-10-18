"""Terminal exact-source/r2 audit and paired mechanism gate; never infer utility from calibration."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from tail_verify import verify
from mh.dagsnet import build_model,CFG


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(seed,utility=False):
    rid=f'ct-{"u" if utility else "cal"}-s{seed}-20261015-v1'
    folder=BASE/'notebooks'/rid
    dest=BASE/'artifacts/kaggle'/rid
    marker=dest/'AUDITED.json'
    if marker.exists():return json.loads(marker.read_text())
    metadata=json.loads((folder/'kernel-metadata.json').read_text())
    owner=metadata['id'].split('/')[0]
    helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
    prefix=[sys.executable,str(helper/'kaggle_as.py'),owner,'--']
    status=subprocess.run(prefix+['kaggle','kernels','status',metadata['id']],capture_output=True,text=True,check=True)
    print(status.stdout.strip(),flush=True)
    if not any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE','ERROR')):
        return None
    subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),metadata['id'],'-p',str(dest),'-j','6',
        '--pattern',r'^(source/|runs/(?![^/]+/wandb/))'],check=True)
    private=BASE/'private_push'/rid
    executed=private/'executed.ipynb'
    subprocess.run(prefix+[sys.executable,str(helper/'download_executed_notebook.py'),metadata['id'],
        '--version','1','--output',str(executed)],check=True,umask=0o077)
    executed.chmod(0o600)
    reference=nbformat.read(folder/metadata['code_file'],as_version=4)
    pushed=nbformat.read(private/metadata['code_file'],as_version=4)
    key=next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
    actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4)
    assert len(actual.cells)==len(reference.cells)
    for i,c in enumerate(actual.cells):
        if c.cell_type=='code':
            assert c.source==(pushed.cells[i].source.replace(key,'[REDACTED]') if i==1 else reference.cells[i].source)
    actual.cells[1].source=reference.cells[1].source
    actual.metadata['credential_cell_redacted']=True
    nbformat.write(actual,folder/'executed.redacted.ipynb')
    manifest=json.loads((folder/'source_manifest.json').read_text())
    assert manifest==json.loads((dest/'source/source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(dest/'source'/name)==digest
    fixture=sha(dest/'source/fixture_manifest.json')
    assert fixture==sha(ROOT/'.compat/papers/pilot/fixture/manifest.json')
    path=dest/'runs'/('utility_summary.json' if utility else 'calibration_summary.json')
    summary=json.loads(path.read_text()) if path.exists() else {}
    rounds={p.name.rsplit('_s',1)[0]:len(list((p/'complete').glob('round_*.json')))
            for p in (dest/'runs').glob(f'*_s{seed}')}
    calibration_valid=not utility and len(rounds)==8 and all(n==2 for n in rounds.values())
    if not utility and 'KernelWorkerStatus.COMPLETE' in status.stdout:
        assert calibration_valid and len(summary)==8 and all(x['rc']==0 and x['rounds']==2 for x in summary.values())
    audits=[]
    with (ROOT/'.compat/papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
        for arm,n in rounds.items():
            if not n:continue
            run=dest/'runs'/f'{arm}_s{seed}'
            audits.append(verify(run,manifest,fixture,expected_rounds=n,calibration=False))
            for r in range(1,n+1):
                state=torch.load(run/f'weights/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                for sd in state['private']+state['proxy']:model.load_state_dict(sd,strict=True)
                with np.load(run/f'predictions/round_{r:03d}.npz') as pred:
                    for hold in ('group','old'):
                        assert np.array_equal(pred[hold+'_y'],np.load(ROOT/f'.compat/papers/pilot/fixture/{hold}_y.npy'))
    probe=summary.get('ct_local',{}).get('mechanism_probe') if calibration_valid else None
    if probe is not None:
        saved=json.loads((dest/'runs'/f'ct_local_s{seed}/complete/round_002.json').read_text())['diagnostics']
        assert probe==saved
        assert probe['classes_in_tail_range']==sum(v and .1<=f<=.4 for v,f in zip(probe['valid'],probe['tail_fraction']))
        assert np.isclose(probe['macro_membership_difference'],np.mean(probe['membership_difference']),atol=1e-12,rtol=0)
        assert probe['pass_probe']==(probe['classes_in_tail_range']>=12 and probe['macro_membership_difference']>=.05)
    record=dict(kernel=metadata['id'],version=1,source=manifest,summary=summary,audits=audits,
                calibration_valid=calibration_valid,mechanism_probe=probe,
                mechanism_pass=probe['pass_probe'] if probe else None,
                utility_r20=utility and len(rounds)==8 and all(n==20 for n in rounds.values()))
    marker.write_text(json.dumps(record,indent=2)+'\n')
    print('AUDITED ClassTail',seed,'mechanism',record['mechanism_pass'],'all r20',record['utility_r20'],flush=True)
    return record


if __name__=='__main__':
    torch.set_num_threads(1)
    utility='--utility' in sys.argv
    rows=[collect(s,utility=utility) for s in (42,43)]
    if not utility and all(rows) and all(r.get('calibration_valid',bool(r.get('mechanism_probe'))) for r in rows):
        passed=all(r['mechanism_pass'] for r in rows)
        (BASE/'probes/paired-mechanism-gate.json').write_text(json.dumps(dict(
            pass_probe=passed,seeds={str(s):r['mechanism_probe'] for s,r in zip((42,43),rows)},
            decision='eligible for locked utility only' if passed else 'stop; no r20',utility_pass=False),indent=2)+'\n')
        print('PAIRED MECHANISM',passed,'utility gate pending' if passed else 'STOP before r20')
