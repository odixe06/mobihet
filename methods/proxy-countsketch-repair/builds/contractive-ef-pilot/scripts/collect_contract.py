"""Collect terminal repaired-EF trials; audit source, metrics and decoded residual contraction."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
import torch

ROOT=Path(__file__).resolve().parents[5]
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/scripts'))
from verify_sketch_subspace_ratio import verify
from mh.v2_math import CountSketch,unpack


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def contraction(run):
    """Check actual saved patch/state at every committed round, including first round."""
    arm=json.loads((run/'config.json').read_text())['config']['arm']
    if '_contract_' not in arm:return []
    depth,width=(3,256) if 'small' in arm else (5,2048)
    sketch=CountSketch(395024,depth,width,1701,'cpu')
    previous=torch.zeros(depth*width)
    checks=[]
    for path in sorted((run/'complete').glob('round_*.json')):
        row=json.loads(path.read_text());r=row['round']
        state=torch.load(run/'resume'/f'round_{r:03d}.pt',map_location='cpu',weights_only=True)['algorithm']
        packet=unpack((run/'knowledge'/f'round_{r:03d}.bin').read_bytes(),expected_base=r-1)
        delta=torch.zeros(395024);delta[packet['indices']]=packet['values']
        accumulated=state['last_mean']+previous
        residual=state['residual']
        assert torch.allclose(accumulated-sketch.encode(delta),residual,atol=2e-5,rtol=1e-5)
        before=float(accumulated.double().square().sum())
        after=float(residual.double().square().sum())
        # CPU/GPU encode reductions can differ; acceptance on worker used strict float64 <=.
        assert after<=before*(1+1e-6)+1e-12,('saved residual expands',arm,r,before,after)
        assert 0<=row['diagnostics']['gamma']<=1
        assert row['diagnostics']['contraction']<=1+1e-6
        checks.append(dict(round=r,before_squared=before,after_squared=after,gamma=row['diagnostics']['gamma']))
        previous=residual
    return checks


def locked_gates(root,seed):
    def ev(arm,role='private',hold='group',rule='gp'):
        return json.loads((root/f'{arm}_s{seed}/complete/round_020.json').read_text())['eval'][role][hold][rule]
    local=ev('local')
    def common(arm):
        value=ev(arm)
        return value['f1_macro']-local['f1_macro']>=.03 and ev(arm,hold='old')['f1_macro']>ev('local',hold='old')['f1_macro'] and value['benign_f1']>=local['benign_f1']-.03
    own=ev('a_own')['f1_macro'];dense_gain=ev('a_dense')['f1_macro']-own
    teacher=ev('a_dense',role='proxy',rule='head')['f1_macro']-local['f1_macro']>=.05
    gates={}
    for size in ('small','medium'):
        arm=f'a_{size}_contract_ef';gain=ev(arm)['f1_macro']-own
        ef=ev(arm)['f1_macro']-ev(f'a_{size}_noef')['f1_macro']
        gates[size]=dict(pass_quality=bool(teacher and common('a_dense') and common(arm)
            and dense_gain>0 and gain>=.01 and gain/dense_gain>=.8),teacher_pass=bool(teacher),
            gain_vs_own=gain,retention=gain/dense_gain if dense_gain>0 else None,
            ef_contrast=ef,ef_contribution=bool(ef>=.005))
    return gates


def collect(seed):
    rid=f'cef-s{seed}-20261015-v1';folder=BASE/'notebooks'/rid
    dest=BASE/'artifacts/kaggle'/rid;marker=dest/'AUDITED.json'
    if marker.exists():return json.loads(marker.read_text())
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
    prefix=[sys.executable,str(helper/'kaggle_as.py'),meta['id'].split('/')[0],'--']
    status=subprocess.run(prefix+['kaggle','kernels','status',meta['id']],check=True,capture_output=True,text=True)
    print(status.stdout.strip(),flush=True)
    if not any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE','ERROR')):return None
    subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),meta['id'],'-p',str(dest),'-j','4',
        '--pattern',r'^(source/|runs/(?![^/]+/wandb/))'],check=True)
    private=BASE/'private_push'/rid;executed=private/'executed.ipynb'
    subprocess.run(prefix+[sys.executable,str(helper/'download_executed_notebook.py'),meta['id'],
        '--version','1','--output',str(executed)],check=True,umask=0o077)
    executed.chmod(0o600)
    reference=nbformat.read(folder/meta['code_file'],as_version=4)
    pushed=nbformat.read(private/meta['code_file'],as_version=4)
    key=next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
    actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4)
    assert len(actual.cells)==len(reference.cells)
    for i,cell in enumerate(actual.cells):
        if cell.cell_type=='code':
            assert cell.source==(pushed.cells[i].source.replace(key,'[REDACTED]') if i==1 else reference.cells[i].source)
    actual.cells[1].source=reference.cells[1].source
    actual.metadata['credential_cell_redacted']=True
    nbformat.write(actual,folder/'executed.redacted.ipynb')
    manifest=json.loads((folder/'source_manifest.json').read_text())
    assert manifest==json.loads((dest/'source/source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(dest/'source'/name)==digest
    assert sha(dest/'source/fixture_manifest.json')==sha(ROOT/'.compat/papers/pilot/fixture/manifest.json')
    audits=[]
    with (ROOT/'.compat/papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for run in sorted((dest/'runs').glob(f'*_s{seed}')):
            if not list((run/'complete').glob('round_*.json')):continue
            identity=json.loads((run/'config.json').read_text())
            assert identity['source']=={k:v for k,v in manifest.items() if k!='meta.json'}
            assert identity['config']['rounds']==20 and identity['config']['seed']==seed and not identity['config']['smoke']
            audit=verify(run,complete=False)
            audit['decoded_contraction']=contraction(run)
            audits.append(audit)
    all_r20=len(audits)==7 and all(x['rounds']==20 for x in audits)
    record=dict(kernel=meta['id'],version=1,source=manifest,audits=audits,
        all_r20=all_r20,quality_gates=locked_gates(dest/'runs',seed) if all_r20 else None,
        evidence='strict source/state/payload/10metrics and saved decoded contraction; paired seeds/contact/cost still separate')
    marker.write_text(json.dumps(record,indent=2)+'\n')
    print('AUDITED repaired EF',seed,'arms',len(audits),'all r20',record['all_r20'],flush=True)
    return record


if __name__=='__main__':
    torch.set_num_threads(1)
    for seed in (42,43):collect(seed)
