"""Build immutable V2 input + secret-free audit notebooks + ignored private key copies."""
import argparse
import hashlib
import json
import netrc
import os
import shutil
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
FAMILIES={
 'a':['local','a_own','a_dense','a_small_noef','a_small_ef','a_medium_noef','a_medium_ef'],
 'b':['b_dense','b_own512','b_seed512','b_own1024','b_seed1024'],
 'c':['diag','c_own','c_fl','c_shuffle']}
DATASET='mobihet-v2-20261011-src'

AUTH=r'''
import os, subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','--quiet','wandb==0.29.0'],check=True)
import wandb
_INLINE_WANDB_KEY = ""
try:
    from kaggle_secrets import UserSecretsClient
    _wandb_key=UserSecretsClient().get_secret('wandb_key')
except Exception:
    _wandb_key=_INLINE_WANDB_KEY
assert _wandb_key, 'W&B key required before training'
assert wandb.login(key=_wandb_key,anonymous='never',relogin=True,verify=True)
os.environ['WANDB_API_KEY']=_wandb_key
del _wandb_key
assert wandb.Api(timeout=30).default_entity, 'W&B online authentication failed'
print('W&B authenticated online',flush=True)
'''

SETUP=r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
count=torch.cuda.device_count()
assert count==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(count))
assert torch.__version__=='2.10.0+cu128',torch.__version__
print('torch',torch.__version__,'gpus',[torch.cuda.get_device_name(i) for i in range(count)],flush=True)
hits=[p.parent for p in Path('/kaggle/input').rglob('v2_run.py') if (p.parent/'train_0_X.npy').exists()]
assert len(hits)==1,hits
SRC=hits[0]; WORK=Path('/kaggle/working/work'); OUT=Path('/kaggle/working/runs')
SOURCE=Path('/kaggle/working/source'); SOURCE.mkdir(exist_ok=True)
for name in ('mh','knowledge','papers/pilot/fixture'): (WORK/name).mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'source_manifest.json').read_text())
for name,sha in manifest.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,('source mismatch',name)
    shutil.copyfile(SRC/name,SOURCE/name)
for f in SRC.iterdir():
    if f.name in CODE: dst=WORK/'mh'/f.name
    elif f.name=='meta.json': dst=WORK/'knowledge'/f.name
    elif f.name=='pilot-v2.md': continue
    else: dst=WORK/'papers/pilot/fixture'/f.name
    shutil.copyfile(f,dst)
fixture=json.loads((SRC/'manifest.json').read_text())
for name,sha in fixture['files'].items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,('fixture mismatch',name)
shutil.copyfile(SRC/'source_manifest.json',SOURCE/'source_manifest.json')
shutil.copyfile(SRC/'manifest.json',SOURCE/'fixture_manifest.json')
print('verified source and fixture',len(manifest),len(fixture['files']),flush=True)
sys.path.insert(0,str(WORK))
import numpy as np
from mh.dagsnet import build_model,CFG
from mh.v2_math import Layout,CountSketch,SeedBasis,unpack
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
for gpu in range(2):
    device=torch.device(f'cuda:{gpu}'); torch.cuda.set_device(device); torch.manual_seed(42)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(device).eval()
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy')[:32].astype('float32')).to(device)
    y=torch.from_numpy(np.load(SRC/'train_0_y.npy')[:32].astype('int64')).to(device)
    layout=Layout(model); assert layout.size==395024
    with torch.autocast('cuda',dtype=torch.float16): z=model(x)[0].float()
    loss=torch.nn.functional.cross_entropy(z,y); assert torch.isfinite(loss); loss.backward()
    gradient=torch.cat([p.grad.float().reshape(-1) for p in model.parameters()])
    assert torch.isfinite(gradient).all()
    sketch=CountSketch(layout.size,3,256,1701,device)
    packet,delta=sketch.patch(sketch.encode(gradient),128,0); assert torch.isfinite(delta).all()
    assert unpack(packet,device,0)['version']==1
    basis=SeedBasis(model,8,1901,device); coeff=torch.zeros(8,device=device,requires_grad=True)
    torch.nn.functional.cross_entropy(basis.forward(model,coeff,x)[0].float(),y).backward()
    assert torch.isfinite(coeff.grad).all()
    cm=torch.bincount(y*16+z.argmax(1),minlength=256); assert int(cm.sum())==len(y)
    print('T4 real-model deterministic preflight PASS',gpu,flush=True)
    del model,x,y,z,gradient,sketch,basis,coeff,delta; torch.cuda.empty_cache()
torch.cuda.set_device(0)
'''

RUN=r'''
T0=time.time(); OUT.mkdir(exist_ok=True)
queue=list(METHODS); slots=[0,0,1,1]; active={}; done={}; seen={}
while queue or active:
    for slot,gpu in enumerate(slots):
        if slot in active or not queue or time.time()-T0>=3*3600: continue
        arm=queue.pop(0); folder=OUT/f'{arm}_s{SEED}'; folder.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,'-m','mh.v2_run','--arm',arm,'--seed',str(SEED),'--out',str(OUT),
                 '--wandb_online','--max_hours','4']
        log=open(folder/'log.txt','w')
        active[slot]=(arm,subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT),log)
        print('start',arm,'gpu',gpu,flush=True)
    for slot,(arm,proc,log) in list(active.items()):
        lines=(OUT/f'{arm}_s{SEED}'/'log.txt').read_text().splitlines()
        for line in lines[seen.get(arm,0):]:
            if line.startswith(('setup','r','done','WANDB','Traceback','budget')) or 'Error' in line:
                print(arm,'|',line,flush=True)
        seen[arm]=len(lines)
        if proc.poll() is not None:
            done[arm]=proc.returncode; log.close(); del active[slot]
    if queue and not active and time.time()-T0>=3*3600: break
    if queue or active: time.sleep(20)
summary={arm:{'rc':rc,'complete':json.loads((OUT/f'{arm}_s{SEED}'/'result.json').read_text())['complete']
             if (OUT/f'{arm}_s{SEED}'/'result.json').exists() else False} for arm,rc in done.items()}
(OUT/f'summary_s{SEED}.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
shutil.rmtree(WORK)  # raw fixture never published as notebook output
needle=os.environ['WANDB_API_KEY'].encode()
for root in (OUT,SOURCE):
  for path in root.rglob('*'):
    if not path.is_file(): continue
    carry=b''
    with path.open('rb') as handle:
        while True:
            part=handle.read(1024*1024)
            if not part: break
            assert needle not in carry+part,('credential in output',path.name)
            carry=(carry+part)[-len(needle):]
assert set(summary)==set(METHODS) and all(v['rc']==0 and v['complete'] for v in summary.values()),summary
'''


def main():
    p=argparse.ArgumentParser(); p.add_argument('--family',choices=FAMILIES,required=True)
    p.add_argument('--owner42',default='minhtran0601'); p.add_argument('--owner43',default='dinhan25')
    a=p.parse_args(); base=ROOT/'papers/v2'; runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    code=[f.name for f in sorted((ROOT/'mh').glob('*.py'))]
    key=netrc.netrc().authenticators('api.wandb.ai')[2]; assert key
    assert '**/private_push/' in (ROOT/'.gitignore').read_text().splitlines()
    for owner,seed in ((a.owner42,42),(a.owner43,43)):
        rid=f'v2{a.family}-s{seed}-20261011-v1'; d=base/'notebook'/rid; private=base/'private_push'/rid
        assert not (d/'PUSHED').exists() and not (private/'PUSHED').exists(),'never edit pushed notebooks'
        src=base/'input'/owner
        expected={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'mh').glob('*.py')}
        expected['meta.json']=hashlib.sha256((ROOT/'knowledge/meta.json').read_bytes()).hexdigest()
        expected['pilot-v2.md']=hashlib.sha256((ROOT/'docs/research/pilot-v2.md').read_bytes()).hexdigest()
        if (src/'source_manifest.json').exists():
            assert json.loads((src/'source_manifest.json').read_text())==expected,'input source already frozen; new attempt required'
        else:
            src.mkdir(parents=True,exist_ok=True)
            for name in code: shutil.copyfile(ROOT/'mh'/name,src/name)
            shutil.copyfile(ROOT/'knowledge/meta.json',src/'meta.json')
            shutil.copyfile(ROOT/'docs/research/pilot-v2.md',src/'pilot-v2.md')
            for f in (ROOT/'papers/pilot/fixture').iterdir(): shutil.copyfile(f,src/f.name)
            (src/'source_manifest.json').write_text(json.dumps(expected,indent=2))
            (src/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{DATASET}',title=DATASET,
                licenses=[{'name':'CC0-1.0'}])))
        d.mkdir(parents=True,exist_ok=True)
        for name in ('source_manifest.json','manifest.json'): shutil.copyfile(src/name,d/name)
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet V2 '+a.family.upper()+' — feasibility\n'
            'All learned models DAGSNet. No local raw/perturbed record shared. Internal group/old only. '
            'Protocol and gates frozen in input pilot-v2.md. R20 headline; R2 design only. Unlimited contact stage.\n\n'
            'Ten metrics per client confusion: accuracy=TP/N; class precision=TP/(TP+FP), recall=TP/(TP+FN), '
            'F1=2PR/(P+R); macro=class mean, weighted=support-weighted mean, micro=pooled counts. '
            'Single-label accuracy=precision_micro=recall_micro=recall_weighted=f1_micro. '
            'Private headline gp; teacher-only B headline head.\n\n'
            'W&B online before training; credential preferred attached secret, private inline fallback owner-authorized.'),
            nbf.v4.new_code_cell(AUTH),nbf.v4.new_code_cell(f'CODE={code!r}\n'+SETUP),
            nbf.v4.new_code_cell(f'SEED={seed}\nMETHODS={FAMILIES[a.family]!r}\n'+RUN)])
        nb.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
            language_info={'name':'python','version':'3.12'},kaggle={'accelerator':'nvidiaTeslaT4',
            'isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
        name=rid+'.ipynb'; nbf.write(nb,d/name); nbf.validate(nb)
        for c in nb.cells:
            if c.cell_type=='code': compile(c.source,str(d),'exec')
        meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name,language='python',
            kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
            machine_shape=runtime['machine_shape'],docker_image=runtime['docker_image'],
            dataset_sources=[f'{owner}/{DATASET}'],kernel_sources=[],competition_sources=[])
        assert meta['is_private'] is True and len(meta['title'])<=50
        (d/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        private.mkdir(parents=True,exist_ok=True)
        credential=nbf.reads(nbf.writes(nb),as_version=4)
        auth=credential.cells[1].source; assert auth.count('_INLINE_WANDB_KEY = ""')==1
        credential.cells[1].source=auth.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
        assert key not in (d/name).read_text()
        nbf.write(credential,private/name); os.chmod(private/name,0o600)
        (private/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)); os.chmod(private/'kernel-metadata.json',0o600)
        print(meta['id'],'audit',d.relative_to(ROOT),'push',private.relative_to(ROOT),flush=True)


if __name__=='__main__': main()
