"""Generate FullDrift probe notebooks (per round plan) and the flat source dataset stage. Nothing is pushed here.

Public notebook keeps an empty W&B key; the private_push copy (git-ignored, 0600) carries the owner's key.

    python methods/full-drift/builds/fd-probe/scripts/gen_fd.py [--round r2] [--suffix v1]
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
FULL = ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj'
TAG = '20261015'
RUNTIME = json.loads((ROOT / 'knowledge/runtime.json').read_text())
# (owner, seed, session, per-GPU queues, hours). Owner 15/10: one shared seed screens methods; seeds come later.
# Queue item 'arm' runs epochs 1-6; 'arm:15' runs epochs 1-15 into <arm>_s42_e15 (round 2 horizon).
PLANS = {'r1': [('annhannh331', 42, 'a', [['fd_dml'], ['fd_dml_sub4']], 7.0),
                ('annhannh331', 42, 'b', [['fd_dml_prox'], ['fd_gaussian', 'fd_gaussian_swappos']], 8.5),
                ('anhtruong1909', 42, 'c', [['fd_fedavg', 'fd_fedavg_sub4'], ['fd_dml_avgm']], 7.5)],
         # round 2 (owner 15/10 ~22Z); measured round 1: DML 31 min/epoch (+24 test), prox 48, FedAvg 16
         'r2': [('anhtruong1909', 42, 'd', [['fd_dml:15'], ['fd_dml_sub4:15']], 9.5),
                ('anhtruong1909', 42, 'e', [['fd_fedavg:15', 'fd_fedavg_sub4_q4', 'fd_fedavg_sub4_prox'],
                                            ['fd_fedavg_sub4:15', 'fd_fedavg_sub16_q4']], 9.0),
                ('annhannh331', 42, 'f', [['fd_dml_sub4_q4', 'fd_dml_sub4_prox'], ['fd_dml_sub16_q4']], 10.5)],
         # round 3 (owner 16/10: client private gp; R1/R2-safe arms); DML epoch 1-6 measured 217 min + setup 5 min
         'r3': [('nguyenminhtran060113', 42, 'g', [['fd_dml_m4b'], ['fd_dml_swappos']], 4.5),
                ('nguyenminhtran060113', 42, 'h', [['fd_dml_ntd'], ['fd_dml_q4']], 4.5)],
         # round 4 (owner 16/10: short-quota accounts); odixe0502 4.0h live, Gaussian epoch 1-6 measured 150 min
         'r4': [('odixe0502', 42, 'i', [['fd_dml_hkd'], ['fd_gaussian_m4b']], 3.95)],
         # round 5 (owner 16/10 ~08:20Z): m4b horizon + combinations; nguyenminhtran060113 9.27h live; DML 32.5 min/epoch (r2)
         'r5': [('nguyenminhtran060113', 42, 'j', [['fd_dml_m4b:15'], ['fd_dml_m4b_q4', 'fd_dml_m4b_sub4_q4']], 8.95)],
         # round 6 (owner 16/10 ~16:45Z): seed 43 confirmation; anhtruong1909 9.36h live; DML epoch 1-6 measured 205-217 min
         'r6': [('anhtruong1909', 43, 'k', [['fd_dml'], ['fd_dml_m4b']], 4.2),
                ('anhtruong1909', 43, 'l', [['fd_dml_sub4_q4'], ['fd_dml_m4b_sub4_q4']], 4.2)],
         # round 7 (owner 16/10 ~17:25Z): seed 43 for swappos and Gaussian m4b; annhannh331 5.29h live
         'r7': [('annhannh331', 43, 'm', [['fd_dml_swappos'], ['fd_gaussian_m4b']], 4.2)]}

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
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('input_manifest.json') if json.loads(p.read_text())==INPUTS]
assert len(hits)==1,('missing/ambiguous inputs',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/fd');SOURCE=Path('/kaggle/working/source');OUT=Path('/kaggle/working/runs')
PROJ=WORK/'papers/full-drift/proj'
for folder in (WORK/'mh',WORK/'knowledge',PROJ,SOURCE,OUT):folder.mkdir(parents=True,exist_ok=True)
for name,digest in INPUTS.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest,name
    target=WORK/'mh'/name if name in MH else PROJ/name if name.endswith('.py') else WORK/'knowledge'/name
    shutil.copyfile(SRC/name,target);shutil.copyfile(SRC/name,SOURCE/name)
shutil.copyfile(SRC/'input_manifest.json',SOURCE/'input_manifest.json')
sys.path.insert(0,str(WORK));sys.path.insert(0,str(PROJ))
t=time.monotonic()
from fd_prepare import prepare_fd
data=prepare_fd(WORK)
(SOURCE/'data_manifest.json').write_text(json.dumps(data,indent=2))
print('verified source; prepared full holdout fixture',data['source_rows'],{k:sum(v) for k,v in data['counts'].items()},
      'test',data['test_rows'],'sec',round(time.monotonic()-t),flush=True)
'''

RUN = r'''
pending={gpu:list(q) for gpu,q in enumerate(QUEUES)};active={};summary={}
while active or any(pending.values()):
    for gpu in (0,1):
        if gpu in active or not pending[gpu]:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<1.0:pending[gpu]=[];continue
        item=pending[gpu].pop(0);arm,_,ep=item.partition(':');ep=int(ep or 6)
        folder=OUT/(f'{arm}_s{SEED}'+('' if ep==6 else f'_e{ep}'));folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{PROJ}:{WORK}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(PROJ/'fd.py'),'--arm',arm,'--seed',str(SEED),'--backend','compiled','--wandb_online',
                 '--wandb_project','mobihet-fulldrift','--max_hours',str(round(remaining-.3,3)),'--out',str(OUT),'--stop_after',str(ep)]
        log=open(folder/'log.txt','a');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(item,folder,proc,log,time.monotonic())
        print('start',item,SEED,'gpu',gpu,'remaining_h',round(remaining,2),flush=True)
    for gpu,(arm,folder,proc,log,t) in list(active.items()):
        if proc.poll() is None and (time.monotonic()-SESSION_T0)/3600>=SESSION_HOURS-.05:proc.terminate()
        if proc.poll() is None:continue
        log.close();del active[gpu]
        rp=folder/'result.json'
        result=json.loads(rp.read_text()) if rp.exists() else dict(history=[])
        summary[arm]=dict(rc=proc.returncode,epochs=len(result['history']),complete=result.get('complete',False),
                          minutes=round((time.monotonic()-t)/60,1))
        (OUT/'fd_summary.json').write_text(json.dumps(summary,indent=2))
        print('finish',arm,summary[arm],flush=True)
        if proc.returncode:print(open(folder/'log.txt').read()[-3000:],flush=True)
    time.sleep(5)
(OUT/'fd_summary.json').write_text(json.dumps(summary,indent=2))
print('FullDrift session; gates only after collector audit',json.dumps(summary),flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--round', choices=sorted(PLANS), default='r2')
    parser.add_argument('--suffix', default='v1'); args = parser.parse_args()
    tag = TAG if args.round == 'r1' else f'{TAG}-{args.round}'   # round 1 stage/dataset kept as pushed
    stage = BASE / 'artifacts/input' / f'fd-src-{tag}'
    mh = sorted(p.name for p in (ROOT / 'mh').glob('*.py')); assert len(mh) == 40
    files = {**{n: ROOT / 'mh' / n for n in mh}, **{n: FULL / n for n in ('full_run.py', 'full_method.py', 'full_driver.py')},
             'fd.py': BASE / 'source/proj/fd.py', 'fd_prepare.py': BASE / 'source/proj/fd_prepare.py',
             'meta.json': ROOT / 'knowledge/meta.json', 'scaler.json': ROOT / 'knowledge/scaler.json'}
    if stage.exists():   # reuse the uploaded dataset; its manifest must still match the sources
        inputs = json.loads((stage / 'input_manifest.json').read_text())
        assert inputs == {n: sha(p) for n, p in files.items()}, 'source changed after staging: new tag'
    else:
        stage.mkdir(parents=True)
        for name, path in files.items(): shutil.copyfile(path, stage / name)
        inputs = {n: sha(stage / n) for n in sorted(files)}
        (stage / 'input_manifest.json').write_text(json.dumps(inputs, indent=1) + '\n')
    secret = netrc.netrc().authenticators('api.wandb.ai')[2]
    for owner, seed, session, queues, hours in PLANS[args.round]:
        dataset = f'{owner}/mobihet-fulldrift-src-{tag}'   # same bytes uploaded under each running account
        rid = f'fd-s{seed}{session}-{tag}-{args.suffix}'
        public, private = BASE / 'notebooks' / rid, BASE / 'private_push' / rid
        assert not public.exists() and not private.exists()
        public.mkdir(parents=True); private.mkdir(parents=True, mode=0o700)
        head = f'MH={mh!r}\nINPUTS={inputs!r}\n'
        run = f'SEED={seed}\nSESSION_HOURS={hours}\nQUEUES={queues!r}\n' + RUN
        cells = [nbformat.v4.new_markdown_cell(
                     f'# FullDrift probe seed{seed} session {session}\n'
                     'Full 100% train80% minus internal holdout (temporal2% + receiver-group5%), epochs 1-6 (or 1-15) of cosine T50. '
                     'Decisions on group holdout; official test once after the last epoch, descriptive. '
                     'Contract: docs/methods/full-drift/design.md'),
                 nbformat.v4.new_code_cell(WANDB), nbformat.v4.new_code_cell(head + SETUP), nbformat.v4.new_code_cell(run)]
        nb = nbformat.v4.new_notebook(cells=cells)
        nb.metadata = dict(kaggle=dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True,
                                       language='python', sourceType='notebook'),
                           kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                           language_info=dict(name='python', version='3.12'))
        for c in nb.cells:
            if c.cell_type == 'code': compile(c.source, rid, 'exec')
        nbformat.validate(nb)
        meta = dict(id=f'{owner}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=f'{rid}.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
                    machine_shape=RUNTIME['machine_shape'], docker_image=RUNTIME['docker_image'],
                    dataset_sources=[dataset, 'trietbackup/veremi-fl-20client', 'trietbackup/veremi-nextgen2026-test'],
                    kernel_sources=[], competition_sources=[])
        assert len(meta['title']) <= 50
        nbformat.write(nb, public / meta['code_file'])
        (public / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        (public / 'source_manifest.json').write_text(json.dumps(dict(inputs=sha(stage / 'input_manifest.json'), fd=inputs['fd.py'],
                                                                     queues=queues, session_hours=hours, seed=seed), indent=2) + '\n')
        (public / 'checkpoint-dataset.json').write_text(json.dumps(dict(
            dataset=dataset, files={n: dict(size=(stage / n).stat().st_size) for n in [*inputs, 'input_manifest.json']},
            uploaded=False), indent=2) + '\n')
        subprocess.run(['git', 'check-ignore', '-q', str((private / meta['code_file']).resolve())], cwd=ROOT, check=True)
        pushed = nbformat.reads(nbformat.writes(nb), as_version=4)
        pushed.cells[1].source = pushed.cells[1].source.replace('_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = ' + json.dumps(secret))
        assert pushed.cells[1].source != nb.cells[1].source
        for name, value in [(meta['code_file'], nbformat.writes(pushed)), ('kernel-metadata.json', json.dumps(meta, indent=2))]:
            fd = os.open(private / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f: f.write(value)
        print(meta['id'], 'prepared', queues, hours, 'h')


if __name__ == '__main__':
    main()
