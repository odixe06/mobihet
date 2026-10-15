"""Paired two-round mechanism calibration on the locked twenty-round schedule."""
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts
from scripts.kaggle.gen_v2 import AUTH

SETUP = r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('tail_driver.py') if (p.parent/'train_0_X.npy').exists()]
assert len(hits)==1,hits
SRC=hits[0]; WORK=Path('/kaggle/temp/class-tail'); OUT=Path('/kaggle/working/runs')
SOURCE=Path('/kaggle/working/source'); SOURCE.mkdir(exist_ok=True)
for folder in ('mh','knowledge','papers/pilot/fixture','papers/class-tail/proj'):
    (WORK/folder).mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'source_manifest.json').read_text())
for name,sha in manifest.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,name
    shutil.copyfile(SRC/name,SOURCE/name)
for path in SRC.iterdir():
    if path.name in MH: dst=WORK/'mh'/path.name
    elif path.name in PROJ: dst=WORK/'papers/class-tail/proj'/path.name
    elif path.name=='meta.json': dst=WORK/'knowledge'/path.name
    else: dst=WORK/'papers/pilot/fixture'/path.name
    shutil.copyfile(path,dst)
fixture=json.loads((SRC/'manifest.json').read_text())
for name,sha in fixture['files'].items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,name
shutil.copyfile(SRC/'manifest.json',SOURCE/'fixture_manifest.json')
shutil.copyfile(SRC/'source_manifest.json',SOURCE/'source_manifest.json')
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/class-tail/proj'))
from mh.dagsnet import build_model,CFG
import numpy as np
for gpu in (0,1):
    torch.cuda.set_device(gpu)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}').eval()
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy',mmap_mode='r')[::100][:32].astype('float32')).to(f'cuda:{gpu}')
    z=model(x)[0];assert torch.isfinite(z).all()
    z.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del model,x,z;torch.cuda.empty_cache()
    print('T4 ClassTail forward/backward PASS',gpu,flush=True)
'''

RUN = r'''
OUT.mkdir(parents=True,exist_ok=True)
pending=list(METHODS);active={};summary={}
while active or pending:
    for gpu in (0,1):
        if gpu in active or not pending:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        assert remaining>.5,'calibration budget exhausted; inspect outputs before continuation'
        arm=pending.pop(0);folder=OUT/f'{arm}_s{SEED}';folder.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(WORK/'papers/class-tail/proj/tail_driver.py'),
            '--arm',arm,'--seed',str(SEED),'--rounds','20','--eval_every','1',
            '--stop_after','2','--wandb_online','--max_hours',str(remaining-.25),'--out',str(OUT)]
        log=open(folder/'log.txt','w');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(arm,proc,log)
        print('start ClassTail calibration',arm,SEED,gpu,flush=True)
    for gpu,(arm,proc,log) in list(active.items()):
        if proc.poll() is None:continue
        log.close();del active[gpu]
        path=OUT/f'{arm}_s{SEED}/result.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        rows=result.get('history',[])
        summary[arm]=dict(rc=proc.returncode,rounds=len(rows),target_rounds=20,
            mechanism_probe=rows[-1].get('diagnostics') if rows and arm=='ct_local' else None)
        assert proc.returncode==0 and len(rows)==2,('failed calibration',arm,summary[arm])
    time.sleep(2)
(OUT/'calibration_summary.json').write_text(json.dumps(summary,indent=2))
print('ClassTail calibration complete; paired mechanism gate must be audited before r20',summary,flush=True)
'''


def main():
    assert (BASE/'probes/local-preflight.json').exists()
    seed, owner = int(sys.argv[1]), sys.argv[2]
    rid = f'ct-cal-s{seed}-20261015-v1'
    stage = BASE/'artifacts/input'/rid
    public, private = BASE/'notebooks'/rid, BASE/'private_push'/rid
    assert not stage.exists() and not public.exists() and not private.exists()
    stage.mkdir(parents=True)
    files = list((ROOT/'mh').glob('*.py'))+list((BASE/'source/proj').glob('*.py'))+[ROOT/'knowledge/meta.json']
    manifest = {}
    for p in files:
        shutil.copyfile(p,stage/p.name)
        manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT/'papers/pilot/fixture').iterdir():
        if p.is_file():shutil.copyfile(p,stage/p.name)
    (stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dataset=f'{owner}/mobihet-{rid}-src'
    (stage/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title=f'mobihet-{rid}-src',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    logging=(ROOT/'methods/boundary-cell-counts/builds/boundary-cell-pilot/scripts/boundary_logging.py').read_text()
    setup=f'MH={[p.name for p in (ROOT/"mh").glob("*.py")]!r}\nPROJ={[p.name for p in (BASE/"source/proj").glob("*.py")]!r}\n'+SETUP
    # W&B adapter also applies to the emitted driver without changing model/source identity.
    setup+=f'\n(WORK/"boundary_logging.py").write_text({logging!r})\n'
    run=RUN.replace("[sys.executable,str(WORK/'papers/class-tail/proj/tail_driver.py'),", "[sys.executable,str(WORK/'boundary_logging.py'),str(WORK/'papers/class-tail/proj/tail_driver.py'),")
    methods=['ct_local','ct_fixed','ct_own','ct_fl','ct_shuffle','ct_classmass','ct_gaussian','ct_dml']
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# ClassTail — locked paired calibration\nTwo CE warm-up rounds for tail arms; reference recipes active. No official test. No r20 before paired probe audit. Private inline W&B key persists in version history.'),
        nbf.v4.new_code_cell('import time\nSESSION_T0=time.monotonic()\n'+AUTH),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(f'SEED={seed}\nSESSION_HOURS=3.0\nMETHODS={methods!r}\n'+run)])
    nb.metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python',version='3.12'),kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True,language='python',sourceType='notebook'))
    filename=rid+'.ipynb'
    meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,machine_shape='NvidiaTeslaT4',enable_internet=True,
        dataset_sources=[dataset],kernel_sources=[],competition_sources=[],docker_image=json.loads((ROOT/'knowledge/runtime.json').read_text())['docker_image'])
    nbf.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,rid,'exec')
    public.mkdir(parents=True);private.mkdir(parents=True,mode=0o700)
    nbf.write(nb,public/filename)
    for name,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest)]:
        (public/name).write_text(json.dumps(value,indent=2)+'\n')
    subprocess.run(['git','check-ignore','-q',str(private/filename)],check=True,cwd=ROOT)
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    credential=nbf.reads(nbf.writes(nb),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for name,text in [(filename,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(text)
    print(meta['id'],'prepared; upload private source dataset before push')


if __name__=='__main__':main()
