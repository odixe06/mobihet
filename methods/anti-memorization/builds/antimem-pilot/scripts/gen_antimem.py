"""Generate AntiMem/QProxy pilot notebooks (one per seed) and the flat source+fixture dataset stage.

Public notebook keeps an empty W&B key; the private_push copy (git-ignored, 0600) carries the owner's key.
Nothing is uploaded or pushed here.

    python methods/anti-memorization/builds/antimem-pilot/scripts/gen_antimem.py [--hours 7]
"""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import nbformat

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
TAG = '20261015'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
ORDER = ['am_dml', 'am_hybrid', 'am_gaussian', 'am_dml_m4b', 'am_hybrid_m4b', 'am_gaussian_m4b',
         'am_dml_swap', 'am_hybrid_swap', 'am_gaussian_swap', 'am_dml_swappos', 'am_hybrid_swappos',
         'am_gaussian_swappos', 'am_dml_q8', 'am_dml_q4']

WANDB = r'''import time
SESSION_T0=time.monotonic()

import os, subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','--quiet','wandb==0.29.0'],check=True)
import wandb
_INLINE_WANDB_KEY = ""
try:
    from kaggle_secrets import UserSecretsClient
    _wandb_key=_INLINE_WANDB_KEY or UserSecretsClient().get_secret('wandb_key')
except Exception:
    _wandb_key=_INLINE_WANDB_KEY
assert _wandb_key, 'W&B key required before training'
assert wandb.login(key=_wandb_key,anonymous='never',relogin=True,verify=True)
os.environ['WANDB_API_KEY']=_wandb_key
del _wandb_key
assert wandb.Api(timeout=30).default_entity, 'W&B online authentication failed'
print('W&B authenticated online',flush=True)
'''

SETUP = r'''
import hashlib,json,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np, torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('input_manifest.json') if json.loads(p.read_text())==INPUTS]
assert len(hits)==1,('missing/ambiguous inputs',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/antimem');SOURCE=Path('/kaggle/working/source');OUT=Path('/kaggle/working/runs')
for folder in (WORK/'mh',WORK/'knowledge',WORK/'papers/pilot/fixture',WORK/'proj',SOURCE,OUT):folder.mkdir(parents=True,exist_ok=True)
for name,digest in INPUTS.items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest,name
fixture=json.loads((SRC/'manifest.json').read_text())
for name in MH:shutil.copyfile(SRC/name,WORK/'mh'/name)
shutil.copyfile(SRC/'antimem.py',WORK/'proj/antimem.py');shutil.copyfile(SRC/'antimem.py',SOURCE/'antimem.py')
shutil.copyfile(SRC/'meta.json',WORK/'knowledge/meta.json')
for name in list(fixture['files'])+['manifest.json']:shutil.copyfile(SRC/name,WORK/'papers/pilot/fixture'/name)
shutil.copyfile(SRC/'input_manifest.json',SOURCE/'input_manifest.json')
sys.path.insert(0,str(WORK))
from mh.dagsnet import build_model,CFG
torch.set_num_threads(1)
for gpu in (0,1):
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}')
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy',mmap_mode='r')[::100][:32].astype('float32')).to(f'cuda:{gpu}')
    z=model(x)[0];assert torch.isfinite(z).all();z.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del model,x,z;torch.cuda.empty_cache()
print('All source/fixture SHA and T4 forward/backward PASS',flush=True)
'''

RUN = r'''
import re
KEEP=(10,20)

def prune():
    """Keep weights/resume of r10, r20 and the latest committed round only (disk); no cross-session resume."""
    for folder in OUT.glob(f'*_s{SEED}'):
        done=sorted(int(p.stem.split('_')[1]) for p in (folder/'complete').glob('round_*.json'))
        if not done:continue
        for kind in ('weights','resume'):
            for p in (folder/kind).glob('round_*.pt'):
                r=int(p.stem.split('_')[1])
                if r<done[-1] and r not in KEEP:p.unlink()

pending=list(ARMS);active={};summary={}
while active or pending:
    for gpu in (0,1):
        if gpu in active or not pending:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<1.0:continue
        arm=pending.pop(0);folder=OUT/f'{arm}_s{SEED}';folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(WORK/'proj/antimem.py'),'--arm',arm,'--seed',str(SEED),'--rounds','20','--eval_every','1',
                 '--wandb_online','--wandb_project','mobihet-antimem','--max_hours',str(round(remaining-.2,3)),'--out',str(OUT)]
        log=open(folder/'log.txt','a');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(arm,proc,log,time.monotonic())
        print('start',arm,SEED,'gpu',gpu,'remaining_h',round(remaining,2),flush=True)
    prune()
    for gpu,(arm,proc,log,t) in list(active.items()):
        if proc.poll() is None:continue
        log.close();del active[gpu]
        rp=OUT/f'{arm}_s{SEED}/result.json'
        result=json.loads(rp.read_text()) if rp.exists() else dict(history=[])
        summary[arm]=dict(rc=proc.returncode,rounds=len(result['history']),complete=result.get('complete',False),
                          minutes=round((time.monotonic()-t)/60,1))
        (OUT/'antimem_summary.json').write_text(json.dumps(summary,indent=2))
        print('finish',arm,summary[arm],flush=True)
        if proc.returncode:print(open(OUT/f'{arm}_s{SEED}/log.txt').read()[-3000:],flush=True)
    if not active and pending and SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600<1.0:break
    if active or pending:time.sleep(5)
prune()
for arm in pending:summary[arm]=dict(rc=None,rounds=0,complete=False,reason='session budget')
(OUT/'antimem_summary.json').write_text(json.dumps(summary,indent=2))
print('AntiMem session; gates only after collector audit',json.dumps(summary),flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--hours', type=float, default=7.0)
    parser.add_argument('--arms', nargs='+', default=ORDER); parser.add_argument('--suffix', default='v1')
    parser.add_argument('--wandb_project', default='mobihet-antimem')
    # Another kernel owner only escapes the 2-batch-GPU-session limit; the same dataset is shared to it (web UI).
    parser.add_argument('--owner', default='nguyenminhtran060113'); parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--dataset_owner', default='nguyenminhtran060113')
    args = parser.parse_args(); assert set(args.arms) <= set(ORDER)
    OWNER = args.owner; DATASET = f'{args.dataset_owner}/mobihet-antimem-src-{TAG}'
    stage = BASE / 'artifacts/input' / f'antimem-src-{TAG}'
    mh = sorted(p.name for p in (ROOT / '.compat/mh').glob('*.py'))
    assert len(mh) == 40
    fx = ROOT / '.compat/papers/pilot/fixture'; fixture = json.loads((fx / 'manifest.json').read_text())
    if stage.exists():   # reuse the uploaded dataset; its manifest must still match the files
        inputs = json.loads((stage / 'input_manifest.json').read_text())
        assert all(sha(stage / n) == d for n, d in inputs.items())
        assert inputs['antimem.py'] == sha(BASE / 'source/proj/antimem.py')
    else:
        inputs = None; stage.mkdir(parents=True)
    for name in ([] if inputs else mh): shutil.copyfile(ROOT / '.compat/mh' / name, stage / name)
    if inputs is None:
        for name in list(fixture['files']) + ['manifest.json']: shutil.copyfile(fx / name, stage / name)
        shutil.copyfile(ROOT / 'knowledge/meta.json', stage / 'meta.json')
        shutil.copyfile(BASE / 'source/proj/antimem.py', stage / 'antimem.py')
        inputs = {p.name: sha(p) for p in sorted(stage.iterdir())}
        (stage / 'input_manifest.json').write_text(json.dumps(inputs, indent=1) + '\n')
    (stage / 'dataset-metadata.json').write_text(json.dumps(dict(id=DATASET, title=DATASET.split('/')[1],
                                                                 licenses=[{'name': 'CC0-1.0'}]), indent=2) + '\n')
    secret = netrc.netrc().authenticators('api.wandb.ai')[2]
    for seed in args.seeds:
        rid = f'am-s{seed}-{TAG}-{args.suffix}'
        public, private = BASE / 'notebooks' / rid, BASE / 'private_push' / rid
        assert not public.exists() and not private.exists()
        public.mkdir(parents=True); private.mkdir(parents=True, mode=0o700)
        head = f'MH={mh!r}\nINPUTS={inputs!r}\n'
        run = f'SEED={seed}\nSESSION_HOURS={args.hours}\nARMS={args.arms!r}\n' + RUN.replace("'mobihet-antimem'", repr(args.wandb_project))
        cells = [nbformat.v4.new_markdown_cell(f'# AntiMem + QProxy pilot r20 seed{seed}\nContract: docs/methods/anti-memorization/design.md'),
                 nbformat.v4.new_code_cell(WANDB), nbformat.v4.new_code_cell(head + SETUP), nbformat.v4.new_code_cell(run)]
        nb = nbformat.v4.new_notebook(cells=cells)
        nb.metadata = dict(kaggle=dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True,
                                       language='python', sourceType='notebook'),
                           kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                           language_info=dict(name='python', version='3.12'))
        for c in nb.cells:
            if c.cell_type == 'code': compile(c.source, rid, 'exec')
        nbformat.validate(nb)
        meta = dict(id=f'{OWNER}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=f'{rid}.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
                    machine_shape='NvidiaTeslaT4', dataset_sources=[DATASET], kernel_sources=[],
                    competition_sources=[], docker_image=IMAGE)
        nbformat.write(nb, public / meta['code_file'])
        (public / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        (public / 'source_manifest.json').write_text(json.dumps(dict(antimem=inputs['antimem.py'], inputs=sha(stage / 'input_manifest.json'),
                                                                     arms=args.arms, session_hours=args.hours), indent=2) + '\n')
        subprocess.run(['git', 'check-ignore', '-q', str((private / meta['code_file']).resolve())], cwd=ROOT, check=True)
        pushed = nbformat.reads(nbformat.writes(nb), as_version=4)
        pushed.cells[1].source = pushed.cells[1].source.replace('_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = ' + json.dumps(secret))
        assert pushed.cells[1].source != nb.cells[1].source
        for name, value in [(meta['code_file'], nbformat.writes(pushed)), ('kernel-metadata.json', json.dumps(meta, indent=2))]:
            fd = os.open(private / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f: f.write(value)
        print(meta['id'], 'prepared', len(args.arms), 'arms', args.hours, 'h')


if __name__ == '__main__':
    main()
