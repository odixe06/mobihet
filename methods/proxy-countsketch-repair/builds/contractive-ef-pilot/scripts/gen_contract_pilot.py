"""Try repaired EF with the original seven contrasts, fixture, schedule and gates."""
import hashlib
import json
import netrc
import os
from pathlib import Path
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts
from scripts.kaggle.gen_v2 import AUTH

SETUP = r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('tail_driver.py') if (p.parent/'train_0_X.npy').exists()]
assert len(hits)==1,hits
SRC=hits[0];WORK=Path('/kaggle/temp/contract-ef');OUT=Path('/kaggle/working/runs')
SOURCE=Path('/kaggle/working/source');SOURCE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
for folder in ('mh','knowledge','papers/pilot/fixture','papers/contract-ef/proj'):
    (WORK/folder).mkdir(parents=True,exist_ok=True)
for name,sha in MANIFEST.items():
    body=EMBED[name].encode() if name in EMBED else (SRC/name).read_bytes()
    assert hashlib.sha256(body).hexdigest()==sha,name
    (SOURCE/name).write_bytes(body)
    target=WORK/'papers/contract-ef/proj'/name if name in EMBED else WORK/'knowledge'/name if name=='meta.json' else WORK/'mh'/name
    target.write_bytes(body)
fixture=json.loads((SRC/'manifest.json').read_text())
for name,sha in fixture['files'].items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,name
    shutil.copyfile(SRC/name,WORK/'papers/pilot/fixture'/name)
shutil.copyfile(SRC/'manifest.json',WORK/'papers/pilot/fixture/manifest.json')
shutil.copyfile(SRC/'manifest.json',SOURCE/'fixture_manifest.json')
(SOURCE/'source_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/contract-ef/proj'))
from mh.dagsnet import build_model,CFG
for gpu in (0,1):
    torch.cuda.set_device(gpu)
    m=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}').eval()
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy',mmap_mode='r')[::100][:32].astype('float32')).to(f'cuda:{gpu}')
    z=m(x)[0];assert torch.isfinite(z).all();z.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    del m,x,z;torch.cuda.empty_cache()
    print('T4 repaired EF forward/backward PASS',gpu,flush=True)
'''

RUN = r'''
pending=list(METHODS);active={};summary={}
while active or pending:
    for gpu in (0,1):
        if gpu in active or not pending:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<.5:continue
        arm=pending.pop(0);folder=OUT/f'{arm}_s{SEED}';folder.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(WORK/'boundary_logging.py'),str(WORK/'papers/contract-ef/proj/contract_driver.py'),
            '--arm',arm,'--seed',str(SEED),'--rounds','20','--eval_every','1','--wandb_online',
            '--max_hours',str(remaining-.25),'--out',str(OUT)]
        log=open(folder/'log.txt','w');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(arm,proc,log)
        print('start repaired EF',arm,SEED,gpu,flush=True)
    for gpu,(arm,proc,log) in list(active.items()):
        if proc.poll() is None:continue
        log.close();del active[gpu]
        path=OUT/f'{arm}_s{SEED}/result.json';result=json.loads(path.read_text()) if path.exists() else {}
        summary[arm]=dict(rc=proc.returncode,rounds=len(result.get('history',[])),complete=result.get('complete',False))
        assert proc.returncode==0,('worker failed; preserve output',arm)
    if not active and pending and SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600<.5:break
    if active or pending:time.sleep(2)
for arm in pending:summary[arm]=dict(rounds=0,complete=False,reason='session budget')
(OUT/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
print('Repaired EF batch',summary,flush=True)
'''


def main():
    seed,owner=int(sys.argv[1]),sys.argv[2]
    rid=f'cef-s{seed}-20261015-v1'
    public,private=BASE/'notebooks'/rid,BASE/'private_push'/rid
    assert not public.exists() and not private.exists()
    embed={p.name:p.read_text() for p in (BASE/'source/proj').glob('*.py')}
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'.compat/mh').glob('*.py')}
    manifest.update({name:hashlib.sha256(body.encode()).hexdigest() for name,body in embed.items()})
    manifest['meta.json']=hashlib.sha256((ROOT/'knowledge/meta.json').read_bytes()).hexdigest()
    setup=f'MANIFEST={manifest!r}\nEMBED={embed!r}\n'+SETUP
    logging=(ROOT/'methods/boundary-cell-counts/builds/boundary-cell-pilot/scripts/boundary_logging.py').read_text()
    setup+=f'\n(WORK/"boundary_logging.py").write_text({logging!r})\n'
    methods=['local','a_own','a_dense','a_small_noef','a_small_contract_ef','a_medium_noef','a_medium_contract_ef']
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# CountSketch EF — contraction repair\nOriginal fixed20schedule, seven contrasts, two seeds. Only aggregate patch acceptance changes; no run of unstable original EF. Byte/utility/R1 gates remain distinct. Private inline W&B key persists in version history.'),
        nbf.v4.new_code_cell('import time\nSESSION_T0=time.monotonic()\n'+AUTH),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(f'SEED={seed}\nSESSION_HOURS=3.0\nMETHODS={methods!r}\n'+RUN)])
    nb.metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.12'),
        kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True,language='python',sourceType='notebook'))
    name=rid+'.ipynb'
    meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name,language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,machine_shape='NvidiaTeslaT4',enable_internet=True,
        dataset_sources=[f'{owner}/mobihet-ct-cal-s{seed}-20261015-v1-src'],kernel_sources=[],competition_sources=[],
        docker_image=json.loads((ROOT/'knowledge/runtime.json').read_text())['docker_image'])
    nbf.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,rid,'exec')
    public.mkdir(parents=True);private.mkdir(parents=True,mode=0o700)
    nbf.write(nb,public/name)
    for filename,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest)]:
        (public/filename).write_text(json.dumps(value,indent=2)+'\n')
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    credential=nbf.reads(nbf.writes(nb),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for filename,value in [(name,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value)
    print(meta['id'],'prepared; original scientific gates unchanged')


if __name__=='__main__':main()
