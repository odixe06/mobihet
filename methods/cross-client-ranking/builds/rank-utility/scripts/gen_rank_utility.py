"""Owner-locked paired utility; automatic full-T r3 cost gate, then fixed20 schedule."""
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.kaggle.gen_v2 import AUTH

SETUP=r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('input_manifest.json') if json.loads(p.read_text())==INPUTS]
assert len(hits)==1,('missing/ambiguous inputs',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/rank-utility');SOURCE=Path('/kaggle/working/source');OUT=Path('/kaggle/working/runs')
for folder in (WORK/'mh',WORK/'knowledge',WORK/'papers/pilot/fixture',WORK/'papers/rank-utility/proj',SOURCE,OUT):folder.mkdir(parents=True,exist_ok=True)
for name,digest in INPUTS.items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest,name
fixture=json.loads((SRC/'manifest.json').read_text())
for name in MH:shutil.copyfile(SRC/name,WORK/'mh'/name)
for name in PROJ:shutil.copyfile(SRC/name,WORK/'papers/rank-utility/proj'/name)
shutil.copyfile(SRC/'meta.json',WORK/'knowledge/meta.json')
for name in list(fixture['files'])+['manifest.json']:shutil.copyfile(SRC/name,WORK/'papers/pilot/fixture'/name)
for name in MANIFEST:shutil.copyfile(SRC/name,SOURCE/name)
for name in ('source_manifest.json','warm-proof.json','input_manifest.json'):shutil.copyfile(SRC/name,SOURCE/name)
shutil.copyfile(SRC/'manifest.json',SOURCE/'fixture_manifest.json')
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/rank-utility/proj'))
from mh.dagsnet import build_model,CFG
torch.set_num_threads(1)
warm=torch.load(SRC/'warm_weights.pt',map_location='cpu',weights_only=True)
proof=json.loads((SRC/'warm-proof.json').read_text())
assert warm['round']==2 and warm['fingerprint']==proof['fingerprint'] and len(warm['private'])==20 and not warm['proxy']
model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
for sd in warm['private']:
    model.load_state_dict(sd,strict=True)
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
del model,warm
print('All source/data/warm SHA and20 strict warm models PASS',flush=True)
'''

PREFLIGHT=r'''
import numpy as np
for gpu in (0,1):
    torch.cuda.set_device(gpu)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}').eval()
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy',mmap_mode='r')[::100][:32].astype('float32')).to(f'cuda:{gpu}')
    z=model(x)[0];assert torch.isfinite(z).all();z.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del model,x,z;torch.cuda.empty_cache()
    print('T4 rank utility forward/backward PASS',gpu,flush=True)
(SOURCE/'T4_preflight.json').write_text(json.dumps(dict(gpus=2,sm75=True,torch=torch.__version__,forward_backward=True)))
'''

RUN=r'''
def launch(arm,gpu,stop=None):
    remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
    assert remaining>.20,'budget exhausted before worker'
    folder=OUT/f'{arm}_s{SEED}';folder.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(WORK/'papers/rank-utility/proj/boundary_logging.py'),str(WORK/'papers/rank-utility/proj/rank_driver.py'),
        '--arm',arm,'--seed',str(SEED),'--rounds','20','--eval_every','1','--warm',str(SRC/'warm_weights.pt'),
        '--wandb_online','--temperature','1','--weight_cap','4','--max_hours',str(remaining-.15),'--out',str(OUT)]
    if (folder/'result.json').exists():command.append('--resume')
    if stop is not None:command+=['--stop_after',str(stop)]
    log=open(folder/f'worker_{"gate" if stop else "utility"}.log','w')
    proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('START',arm,SEED,gpu,'stop',stop,flush=True);return proc,log

active={gpu:(arm,*launch(arm,gpu,3)) for gpu,arm in enumerate(('cr_local','cr_fl'))}
while active:
    for gpu,(arm,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[gpu];assert p.returncode==0,('cost worker failed',arm,p.returncode)
    assert time.monotonic()-SESSION_T0<SESSION_HOURS*3600,'cost gate exhausted session'
    if active:time.sleep(2)
from rank_verify import verify
for arm in ('cr_local','cr_fl'):
    audited=verify(OUT/f'{arm}_s{SEED}',MANIFEST,hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest(),WORK/'papers/pilot/fixture')
    assert audited['logical_rounds']==3 and audited['committed_rounds']==1
gate_rows={arm:json.loads((OUT/f'{arm}_s{SEED}/complete/round_003.json').read_text()) for arm in ('cr_local','cr_fl')}
ratios={k:gate_rows['cr_fl'][k]/gate_rows['cr_local'][k] for k in ('algorithm_seconds','sec')}
cost=dict(seed=SEED,ratios=ratios,pass_cost=all(v<=2 for v in ratios.values()),round=3,
    basis='actual full-T post-warm epoch+scan and full round; initialization/import/eval charged separately in session budget')
(OUT/'cost_gate.json').write_text(json.dumps(cost,indent=2));print('COST GATE',cost,flush=True)
assert cost['pass_cost'],'cost FAIL; preserve artifacts, do not open r20'
pending=list(METHODS);active={};summary={}
while pending or active:
    for gpu in (0,1):
        if gpu in active or not pending:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<.20+1.15*max(v['sec'] for v in gate_rows.values())/3600:continue
        arm=pending.pop(0);active[gpu]=(arm,*launch(arm,gpu))
    for gpu,(arm,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[gpu]
        path=OUT/f'{arm}_s{SEED}/result.json';r=json.loads(path.read_text()) if path.exists() else {}
        summary[arm]=dict(rc=p.returncode,logical_rounds=r.get('logical_rounds',0),inherited_rounds=r.get('inherited_rounds'),complete=r.get('complete',False))
        (OUT/'utility_summary.json').write_text(json.dumps(summary,indent=2))
        assert p.returncode==0,('utility worker failed',arm,p.returncode)
        print('FINISH',arm,summary[arm],flush=True)
    if not active and pending and SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600<.20+1.15*max(v['sec'] for v in gate_rows.values())/3600:break
    if pending or active:time.sleep(2)
for arm in pending:summary[arm]=dict(complete=False,reason='session boundary; inspect committed rounds before continuation')
(OUT/'utility_summary.json').write_text(json.dumps(summary,indent=2))
assert len(summary)==8
print('RANK UTILITY BATCH',summary,flush=True)
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seed=int(sys.argv[1]);owner='minhtriethihi';rid=f'cr-u-s{seed}-20261015-v1'
    assert all(json.loads((BASE/'probes'/f).read_text())['pass_check'] for f in ('local-preflight.json','driver-preflight.json'))
    public=BASE/'notebooks'/rid;private=BASE/'private_push'/rid;stage=BASE/'artifacts/input'/rid
    assert not any(p.exists() for p in (public,private,stage))
    old=ROOT/'methods/class-tail-risk/builds/class-tail-pilot';parent=old/'artifacts/kaggle'/f'ct-cal-s{seed}-20261015-v1'
    audit=json.loads((parent/'AUDITED.json').read_text())
    assert audit['mechanism_pass'] and len(audit['audits'])==8 and all(a['rounds']==2 for a in audit['audits'])
    warm=parent/f'runs/ct_local_s{seed}/weights/round_002.pt'
    committed=json.loads((parent/f'runs/ct_local_s{seed}/complete/round_002.json').read_text());assert sha(warm)==committed['weights_sha']
    stage.mkdir(parents=True);manifest={}
    sources=list((ROOT/'.compat/mh').glob('*.py'))+list((BASE/'source/proj').glob('*.py'))+[ROOT/'knowledge/meta.json']
    for source in sources:shutil.copyfile(source,stage/source.name);manifest[source.name]=sha(source)
    fx=ROOT/'.compat/papers/pilot/fixture';fixture=json.loads((fx/'manifest.json').read_text())
    assert not any(name.startswith('test') for name in fixture['files'])
    for name in list(fixture['files'])+['manifest.json']:
        if name in fixture['files']:assert sha(fx/name)==fixture['files'][name]
        os.link((fx/name).resolve(),stage/name)
    shutil.copyfile(warm,stage/'warm_weights.pt')
    proof=dict(kernel=audit['kernel'],version=1,arm=f'ct_local_s{seed}',round=2,
        fingerprint=committed['fingerprint'],weights_sha=sha(warm),audited=True,rng='deterministic per-round recipe unchanged')
    for name,value in [('source_manifest.json',manifest),('warm-proof.json',proof)]:
        (stage/name).write_text(json.dumps(value,indent=2)+'\n')
    inputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()};(stage/'input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    dataset=f'{owner}/mobihet-{rid}-src';(stage/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title=f'mobihet-{rid}-src',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    setup=f'MH={[p.name for p in (ROOT/".compat/mh").glob("*.py")]!r}\nPROJ={[p.name for p in (BASE/"source/proj").glob("*.py")]!r}\nMANIFEST={manifest!r}\nINPUTS={inputs!r}\n'+SETUP
    auth='import time\nSESSION_T0=time.monotonic()\n'+AUTH
    auth=auth.replace("_wandb_key=UserSecretsClient().get_secret('wandb_key')","_wandb_key=_INLINE_WANDB_KEY or UserSecretsClient().get_secret('wandb_key')")
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# CrossClientRank locked paired utility\nSix audited CE-r2 forks and fresh Gaussian/DML; fixed true-class gp. Full-T r3 cost gate before r20. Private inline W&B key persists in private version history.'),
        nbf.v4.new_code_cell(auth),nbf.v4.new_code_cell(setup+PREFLIGHT),
        nbf.v4.new_code_cell(f'SEED={seed}\nSESSION_HOURS=2.5\nMETHODS={["cr_local","cr_fl","cr_own","cr_shuffle","cr_fixed","cr_classweight","cr_gaussian","cr_dml"]!r}\n'+RUN)])
    nb.metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.12'),
        kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True,language='python',sourceType='notebook'))
    name=rid+'.ipynb';meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name,language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,machine_shape='NvidiaTeslaT4',enable_internet=True,dataset_sources=[dataset],kernel_sources=[],competition_sources=[],
        docker_image=json.loads((ROOT/'knowledge/runtime.json').read_text())['docker_image'])
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    public.mkdir();private.mkdir(mode=0o700);nbf.write(nb,public/name)
    for filename,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest),('input_manifest.json',inputs),('warm-proof.json',proof)]:
        (public/filename).write_text(json.dumps(value,indent=2)+'\n')
    subprocess.run(['git','check-ignore','-q',str((private/name).resolve())],cwd=ROOT,check=True)
    key=netrc.netrc().authenticators('api.wandb.ai')[2];credential=nbf.reads(nbf.writes(nb),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for filename,value in [(name,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value)
    probe=BASE/'probes/mount-probe'/rid;probe.mkdir(parents=True)
    cpu='import os,sys\n'+setup
    cpu=cpu.replace('assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))','assert torch.cuda.device_count()==0')
    cpu=cpu.replace("assert torch.__version__=='2.10.0+cu128'",'')
    cpu+="\n(SOURCE/'MOUNT_VERIFIED.json').write_text(json.dumps(dict(cpu_only=True,inputs_verified=len(INPUTS),models=20,round=2,weights_sha=proof['weights_sha'])))\n"
    cp=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(cpu)]);cp.metadata=dict(nb.metadata,kaggle=dict(accelerator='none',isGpuEnabled=False,isInternetEnabled=True,language='python',sourceType='notebook'))
    cm=dict(meta,id=meta['id']+'-cpu',title=meta['title']+'-cpu',enable_gpu=False);cm.pop('machine_shape',None)
    nbf.validate(cp);compile(cpu,rid,'exec');nbf.write(cp,probe/name);(probe/'kernel-metadata.json').write_text(json.dumps(cm,indent=2)+'\n')
    print(meta['id'],'prepared; full-T cost gate and fixed20; input files',len(inputs))


if __name__=='__main__':main()
