"""Frozen, online-W&B two-seed BoundaryCells cost calibration; never auto-start utility."""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat as nbf

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
sys.path.insert(0,str(ROOT))
from scripts.kaggle.gen_v2 import AUTH

SETUP = r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
assert torch.cuda.device_count()==2
assert all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('boundary_driver.py') if (p.parent/'train_0_X.npy').exists()]
assert len(hits)==1,hits
SRC=hits[0];WORK=Path('/kaggle/working/work');OUT=Path('/kaggle/working/runs')
SOURCE=Path('/kaggle/working/source');SOURCE.mkdir(exist_ok=True)
for folder in ('mh','knowledge','papers/pilot/fixture','papers/boundary-cells/proj'):
    (WORK/folder).mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'source_manifest.json').read_text())
for name,sha in manifest.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,('source mismatch',name)
    shutil.copyfile(SRC/name,SOURCE/name)
for path in SRC.iterdir():
    if path.name in MH: dst=WORK/'mh'/path.name
    elif path.name in PROJ: dst=WORK/'papers/boundary-cells/proj'/path.name
    elif path.name=='meta.json': dst=WORK/'knowledge'/path.name
    elif path.name=='contract.md': continue
    else: dst=WORK/'papers/pilot/fixture'/path.name
    shutil.copyfile(path,dst)
fixture=json.loads((SRC/'manifest.json').read_text())
for name,sha in fixture['files'].items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,('fixture mismatch',name)
shutil.copyfile(SRC/'manifest.json',SOURCE/'fixture_manifest.json')
shutil.copyfile(SRC/'source_manifest.json',SOURCE/'source_manifest.json')
print('source/fixture verified; CUDA',torch.__version__,flush=True)
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/boundary-cells/proj'))
from mh.dagsnet import build_model,CFG
from cells import packet,decode
import numpy as np
for gpu in (0,1):
    torch.cuda.set_device(gpu);torch.manual_seed(42)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}').eval()
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy',mmap_mode='r')[::100][:32].astype('float32')).to(f'cuda:{gpu}')
    z=model(x)[0];assert torch.isfinite(z).all()
    z.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    identity=bytes(16);encoded=packet(1,2,b'preflight',identity);assert decode(encoded,1,2,identity)==b'preflight'
    print('T4 forward/backward/codec PASS',gpu,flush=True)
    del model,x,z;torch.cuda.empty_cache()
'''

RUN = r'''
T0=time.time();OUT.mkdir(exist_ok=True)
methods=['bc_single','bc_direct2_fl','bc_direct4_fl','bc_teacher2_fl','bc_teacher4_fl']
pending=list(methods);active={};summary={};baseline_cost=None;seen={}
while active or pending:
    for gpu in (0,1):
        if gpu in active or not pending or time.time()-T0>1.5*3600:continue
        if baseline_cost is None and (pending[0]!='bc_single' or gpu!=0):continue
        arm=pending.pop(0);seed=SEED
        folder=OUT/f'{arm}_s{seed}';folder.mkdir(exist_ok=True)
        log=open(folder/'log.txt','w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(WORK/'papers/boundary-cells/proj/boundary_driver.py'),
                 '--arm',arm,'--seed',str(seed),'--out',str(OUT),'--rounds','20',
                 '--stop_after','2','--eval_every','1','--aux_weight','.5','--temperature','1',
                 '--wandb_online','--max_hours','.5']
        active[gpu]=(arm,seed,subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT),log)
        print('start calibration',arm,'seed',seed,'gpu',gpu,flush=True)
    for gpu,(arm,seed,proc,log) in list(active.items()):
        folder=OUT/f'{arm}_s{seed}';label=folder.name
        lines=(folder/'log.txt').read_text().splitlines()
        for line in lines[seen.get(label,0):]:
            if line.startswith(('setup','r','done','WANDB','Traceback','budget')) or 'Error' in line:
                print(label,'|',line,flush=True)
        seen[label]=len(lines)
        if proc.poll() is None:continue
        log.close();del active[gpu]
        file=folder/'result.json';result=json.loads(file.read_text()) if file.exists() else {}
        rows=result.get('history',[])
        if proc.returncode!=0 or len(rows)!=2:
            summary[label]={'rc':proc.returncode,'rounds':len(rows),'calibration_valid':False};continue
        seconds=[row['algorithm_seconds'] for row in rows]
        measured=sum(seconds)/2
        if arm=='bc_single':baseline_cost=measured
        ratio=measured/baseline_cost
        gate={'algorithm_seconds':seconds,'mean_seconds':measured,'baseline_mean_seconds':baseline_cost,
              'ratio':ratio,'pass_cost':ratio<=2,'physical_gpu':gpu,'seed':seed,'rounds':2,
              'round_schedule':20,'utility_measured':False}
        (folder/'cost_gate.json').write_text(json.dumps(gate,indent=2))
        summary[label]={'rc':0,'rounds':2,'calibration_valid':True,'cost':gate}
        print('COST',label,ratio,ratio<=2,flush=True)
    if not active and time.time()-T0>1.5*3600:break
    if not active and baseline_cost is None and pending and pending[0]!='bc_single':break
    if active or pending:time.sleep(2)
for arm in pending:summary[f'{arm}_s{SEED}']={'reason':'missing baseline or parent budget before start','calibration_valid':False}
(OUT/'calibration_summary.json').write_text(json.dumps(summary,indent=2))
print('CALIBRATION ONLY; no utility continuation',summary,flush=True)
assert len(summary)==5 and all(v.get('calibration_valid') for v in summary.values()),'incomplete calibration'
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--owner',required=True);p.add_argument('--seed',type=int,choices=(42,43),required=True);a=p.parse_args()
    base=ROOT/'papers/boundary-cells';rid=f'bc-cal-s{a.seed}-20261013-initial2-v1'
    public=base/'notebook'/rid;private=base/'private_push'/rid;source=base/'input'/a.owner
    assert not public.exists(), 'immutable notebook already exists'
    private.mkdir(parents=True,exist_ok=True);os.chmod(private,0o700)
    subprocess.run(['git','check-ignore','-q',str(private.resolve())],cwd=ROOT,check=True)
    mh=sorted((ROOT/'mh').glob('*.py'));proj=sorted((base/'proj').glob('*.py'))
    files=mh+proj+[ROOT/'knowledge/meta.json',base/'contract.md']
    assert len({f.name for f in files})==len(files)
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    dataset='mobihet-boundary-cal-20261013-initial2-src'
    if source.exists():
        assert json.loads((source/'source_manifest.json').read_text())==manifest
        assert json.loads((source/'dataset-metadata.json').read_text())['id']==f'{a.owner}/{dataset}'
        for f in files:assert hashlib.sha256((source/f.name).read_bytes()).hexdigest()==manifest[f.name]
    else:
        source.mkdir(parents=True)
        for f in files:shutil.copyfile(f,source/f.name)
        for f in (ROOT/'papers/pilot/fixture').iterdir():
            if f.is_file():shutil.copyfile(f,source/f.name)
        (source/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (source/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{a.owner}/{dataset}',title=dataset,
            licenses=[{'name':'CC0-1.0'}]),indent=2))
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        f'# BoundaryCells cost calibration seed{a.seed}\n5 arms ×2 rounds, fixed20roundschedule; no utility claim. '
        'Every learned model DAGSNet; local records stay at their client. Cohort cell counts/targets only. '
        'W&B online before training; mandatory private credential workflow.'),
        nbf.v4.new_code_cell(AUTH),nbf.v4.new_code_cell(f'MH={[f.name for f in mh]!r}\nPROJ={[f.name for f in proj]!r}\n'+SETUP),
        nbf.v4.new_code_cell(f'SEED={a.seed}\n'+RUN)])
    nb.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
                       language_info={'name':'python','version':'3.12'},
                       kaggle={'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,'isInternetEnabled':True,
                               'language':'python','sourceType':'notebook'})
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    public.mkdir(parents=True);filename=rid+'.ipynb'
    nbf.write(nb,public/filename)
    runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    metadata=dict(id=f'{a.owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,language='python',
                  kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
                  machine_shape=runtime['machine_shape'],docker_image=runtime['docker_image'],
                  dataset_sources=[f'{a.owner}/{dataset}'],kernel_sources=[],competition_sources=[])
    assert len(metadata['title'])<=50
    (public/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (public/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    key=netrc.netrc().authenticators('api.wandb.ai')[2];assert key
    private_nb=nbf.reads(nbf.writes(nb),as_version=4)
    private_nb.cells[1].source=private_nb.cells[1].source.replace('_INLINE_WANDB_KEY = ""',
                                                              '_INLINE_WANDB_KEY = '+json.dumps(key))
    for name,body in [(filename,nbf.writes(private_nb)),('kernel-metadata.json',json.dumps(metadata,indent=2))]:
        fd=os.open(private/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(body)
    assert all(key not in (source/f.name).read_text(errors='ignore') for f in files)
    print(metadata['id'],'prepared; no push',flush=True)


if __name__=='__main__':main()
