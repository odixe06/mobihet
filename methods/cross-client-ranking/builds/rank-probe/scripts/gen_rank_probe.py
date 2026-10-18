"""Pair of train-only Kaggle probes using already-audited CE-r2 private checkpoints."""
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
hits=[p.parent for p in Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()==PROTOCOL_SHA]
assert len(hits)==1,('input missing/ambiguous',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/cross-client-rank');SOURCE=Path('/kaggle/working/source');OUT=Path('/kaggle/working/runs')
assert json.loads((SRC/'input_manifest.json').read_text())==INPUTS
for folder in (WORK/'mh',WORK/'proj',WORK/'knowledge',SOURCE,OUT):folder.mkdir(parents=True,exist_ok=True)
for name,sha in INPUTS.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,name
for name in MH:shutil.copyfile(SRC/name,WORK/'mh'/name)
for name in PROJ:shutil.copyfile(SRC/name,WORK/'proj'/name)
shutil.copyfile(SRC/'meta.json',WORK/'knowledge/meta.json')
for name in MANIFEST:shutil.copyfile(SRC/name,SOURCE/name)
for name in ('manifest.json','protocol.json','warm-proof.json','input_manifest.json'):
    shutil.copyfile(SRC/name,SOURCE/name)
(SOURCE/'source_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'proj'))
from mh.dagsnet import build_model,CFG
torch.set_num_threads(1)
warm=json.loads((SRC/'warm-proof.json').read_text())
weights=torch.load(SRC/'warm_weights.pt',map_location='cpu',weights_only=True)
assert weights['round']==2 and weights['fingerprint']==warm['fingerprint'] and len(weights['private'])==20 and not weights['proxy']
model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
for sd in weights['private']:
    model.load_state_dict(sd,strict=True)
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
del model,weights
print('Exact audited CE-r2 weights/20 models/source/train data PASS',flush=True)
'''

RUN=r'''
protocol=json.loads((SRC/'protocol.json').read_text())
identity=hashlib.sha256(json.dumps(dict(protocol=protocol,source=MANIFEST),sort_keys=True).encode()).hexdigest()
wb=wandb.init(project='mobihet-cross-client-rank-probe',id=f'ccr_probe_s{SEED}-{identity[:8]}',
    name=f'ccr_probe_s{SEED}',config=dict(protocol=protocol,source=MANIFEST,identity=identity),mode='online',
    settings=wandb.Settings(save_code=False,disable_code=True))
assert not wb.settings._offline
print('WANDB',wb.url,flush=True)
command=[sys.executable,str(WORK/'proj/rank_probe.py'),'--input',str(SRC),'--out',str(OUT),'--seed',str(SEED)]
for phase in ('scan','aggregate','probe','finish'):
    if phase in ('aggregate','finish'):
        subprocess.run(command+['--phase',phase],cwd=WORK,env=dict(os.environ,PYTHONPATH=str(WORK)),check=True,timeout=120)
        continue
    active=[]
    for shard in (0,1):
        remaining=SESSION_HOURS*3600-(time.monotonic()-SESSION_T0)
        assert remaining>120,'probe preparation exhausted budget'
        log=open(OUT/f'{phase}_gpu{shard}.log','w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        proc=subprocess.Popen(command+['--phase',phase,'--shard',str(shard)],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active.append((proc,log))
    while active:
        assert time.monotonic()-SESSION_T0<SESSION_HOURS*3600,'probe budget exhausted; preserve diagnostic output'
        for proc,log in list(active):
            if proc.poll() is None:continue
            log.close();active.remove((proc,log));assert proc.returncode==0,('probe worker failed',phase,proc.returncode)
        if active:time.sleep(2)
result=json.loads((OUT/'probe_result.json').read_text())
result.update(identity=identity,source=MANIFEST,elapsed_seconds=time.monotonic()-SESSION_T0,
    evidence='Two-T4 train-only CE-r2 snapshot probe; no new SGD, holdout use or utility measurement')
(OUT/'probe_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
wb.log({k:v for k,v in result.items() if isinstance(v,(int,float,bool))})
wb.finish()
print('CrossClientRank seed',SEED,'probe PASS',result['pass_probe'],'utility not measured',flush=True)
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seed,owner=int(sys.argv[1]),sys.argv[2]
    assert json.loads((BASE/'probes/local-preflight.json').read_text())['pass_check']
    rid=f'ccr-probe-s{seed}-20261015-v1';public,private=BASE/'notebooks'/rid,BASE/'private_push'/rid
    stage=BASE/'artifacts/input'/rid
    assert not any(p.exists() for p in (public,private,stage))
    old=ROOT/'methods/class-tail-risk/builds/class-tail-pilot';parent=old/'artifacts/kaggle'/f'ct-cal-s{seed}-20261015-v1'
    audit=json.loads((parent/'AUDITED.json').read_text());assert audit['mechanism_pass']
    assert any(a['run']==f'ct_local_s{seed}' and a['rounds']==2 for a in audit['audits'])
    run=parent/'runs'/f'ct_local_s{seed}';cfg=json.loads((run/'config.json').read_text())
    warm=run/'weights/round_002.pt'
    assert sha(warm)==json.loads((run/'complete/round_002.json').read_text())['weights_sha']
    protocol=dict(seed=seed,score='softmax_T1',margin=1.,weight=.25,warmup_round=2,warm_fingerprint=cfg['fingerprint'],
        primary='private/group/gp fixed true-class prior; head diagnostic; no target-mass claim',
        sample_seed=1729,sample_max_per_class_client=256,fit_fraction=.5,comparison='common-support losses',
        min_global_rows_each_side=128,min_contributors_each_side=5,min_own_rows_each_side=32,
        min_classes=12,min_clients_per_class=5,signal_rms_min=.05,scan_coverage_min=.9,
        residual_macro_min=.2,energy_degenerate_max=1e-20,projection_rcond=1e-10,
        projection_basis='16 class-CE +15 centered-bias +1 centered-logit; fit RMS; held-out T',
        model_training=False,holdout_read=False,client_pooling='equal valid clients per true class')
    stage.mkdir(parents=True)
    manifest={}
    for p in list((ROOT/'.compat/mh').glob('*.py'))+list((BASE/'source/proj').glob('*.py'))+[ROOT/'knowledge/meta.json']:
        shutil.copyfile(p,stage/p.name);manifest[p.name]=sha(p)
    fx=ROOT/'.compat/papers/pilot/fixture'
    for p in list(fx.glob('train_*_X.npy'))+list(fx.glob('train_*_y.npy'))+[fx/'manifest.json']:
        shutil.copyfile(p,stage/p.name)
    shutil.copyfile(warm,stage/'warm_weights.pt')
    proof=dict(kernel=audit['kernel'],version=1,arm=f'ct_local_s{seed}',round=2,
        fingerprint=cfg['fingerprint'],weights_sha=sha(warm),original_fixture_sha=cfg['fixture'],audited=True)
    for name,value in [('warm-proof.json',proof),('protocol.json',protocol),('source_manifest.json',manifest)]:
        (stage/name).write_text(json.dumps(value,indent=2)+'\n')
    inputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
    (stage/'input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    dataset=f'{owner}/mobihet-{rid}-src'
    (stage/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title=f'mobihet-{rid}-src',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    setup=f'MH={[p.name for p in (ROOT/".compat/mh").glob("*.py")]!r}\nPROJ={[p.name for p in (BASE/"source/proj").glob("*.py")]!r}\nINPUTS={inputs!r}\nMANIFEST={manifest!r}\nPROTOCOL_SHA={sha(stage/"protocol.json")!r}\n'+SETUP
    auth='import time\nSESSION_T0=time.monotonic()\n'+AUTH
    auth=auth.replace("_wandb_key=UserSecretsClient().get_secret('wandb_key')","_wandb_key=_INLINE_WANDB_KEY or UserSecretsClient().get_secret('wandb_key')")
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# CrossClientRank — locked train-only probe\nReuses audited CE-r2 private weights. No new SGD, group/old/test input, per-record score/gradient output or utility claim. Private inline W&B key persists in private version history.'),
        nbf.v4.new_code_cell(auth),nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(f'SEED={seed}\nSESSION_HOURS=.5\n'+RUN)])
    nb.metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python',version='3.12'),kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True,language='python',sourceType='notebook'))
    name=rid+'.ipynb';meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name,language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,machine_shape='NvidiaTeslaT4',enable_internet=True,dataset_sources=[dataset],kernel_sources=[],competition_sources=[],
        docker_image=json.loads((ROOT/'knowledge/runtime.json').read_text())['docker_image'])
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    public.mkdir(parents=True);private.mkdir(parents=True,mode=0o700);nbf.write(nb,public/name)
    for filename,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest),('input_manifest.json',inputs),('protocol.json',protocol),('warm-proof.json',proof)]:
        (public/filename).write_text(json.dumps(value,indent=2)+'\n')
    subprocess.run(['git','check-ignore','-q',str((private/name).resolve())],cwd=ROOT,check=True)
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    credential=nbf.reads(nbf.writes(nb),as_version=4);credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for filename,value in [(name,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value)
    probe=BASE/'probes/mount-probe'/rid;probe.mkdir(parents=True)
    cpu='import os,sys\n'+setup
    cpu=cpu.replace('assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))','assert torch.cuda.device_count()==0')
    cpu=cpu.replace("assert torch.__version__=='2.10.0+cu128'",'')
    cpu+="\n(SOURCE/'MOUNT_VERIFIED.json').write_text(json.dumps(dict(cpu_only=True,round=2,models=20,weights_sha=warm['weights_sha'],inputs_verified=len(INPUTS))))\n"
    cpu_nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(cpu)]);cpu_nb.metadata=dict(nb.metadata,kaggle=dict(accelerator='none',isGpuEnabled=False,isInternetEnabled=True,language='python',sourceType='notebook'))
    cpu_meta=dict(meta,id=meta['id']+'-cpu',title=meta['title']+'-cpu',enable_gpu=False);cpu_meta.pop('machine_shape',None)
    nbf.validate(cpu_nb);compile(cpu,rid,'exec');nbf.write(cpu_nb,probe/name);(probe/'kernel-metadata.json').write_text(json.dumps(cpu_meta,indent=2)+'\n')
    print(meta['id'],'prepared; input',len(inputs),'files; no holdout arrays')


if __name__=='__main__':main()
