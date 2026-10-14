"""Immutable V5 notebooks: anchor teachers and cost-capped dynamic books."""
import argparse
import hashlib
import json
import netrc
import os
import shutil
import sys
from pathlib import Path
import nbformat as nbf
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2]); sys.path.insert(0,str(ROOT))
from scripts.kaggle.gen_v2 import AUTH,SETUP,RUN
from mh.v5_run import FAMILIES
DATASET='mobihet-v5-20261012-src'
DOCS=['pilot-v5-anchors.md','pilot-v5-books.md']
ANCHOR_PREFLIGHT=r'''
torch.set_num_threads(1)
for gpu in range(2):
    device=torch.device(f'cuda:{gpu}'); torch.cuda.set_device(device); torch.manual_seed(SEED)
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy')[:32]).to(device).float()
    y=torch.from_numpy(np.load(SRC/'train_0_y.npy')[:32]).to(device).long()
    for m in (32,512):
        anchors=torch.randn(m,66,device=device).half().float()
        distance=(x.square().sum(1)[:,None]+anchors.square().sum(1)[None]-2*x@anchors.T).clamp_min(0)
        kernels=(-distance/32).exp()
        h=torch.stack([kernels[y==c].sum(0)/max(int((y==c).sum()),1) for c in range(16)],dim=1)
        target=h/h.sum(1,keepdim=True).clamp_min(1e-30)
        model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(device).eval()
        with torch.autocast('cuda',dtype=torch.float16):
            loss=-(target*torch.nn.functional.log_softmax(model(anchors)[0].float(),dim=1)).sum(1).mean()
        loss.backward(); assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        print('T4 V5 anchor kernel/teacher preflight PASS',gpu,m,flush=True)
        del model,anchors,distance,kernels,h,target,loss
    del x,y; torch.cuda.empty_cache()
torch.cuda.set_device(0)
'''
PREFLIGHT=r'''
import copy
from types import SimpleNamespace
from mh.v5_books import Method as Books
torch.set_num_threads(1)
_book_times={1:{'mean':[],'grad':[]},4:{'mean':[],'grad':[]}}
for gpu in range(2):
    device=torch.device(f'cuda:{gpu}'); torch.cuda.set_device(device); torch.manual_seed(SEED)
    source_y=np.load(SRC/'train_0_y.npy'); source_x=np.load(SRC/'train_0_X.npy',mmap_mode='r')
    indices=np.concatenate([np.flatnonzero(source_y==c)[:32] for c in range(16)])
    x=torch.from_numpy(np.array(source_x[indices])).to(device).float()
    y=torch.from_numpy(np.array(source_y[indices])).to(device).long()
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(device)
    mu=torch.stack([x[y==c].mean(0) for c in range(16)])
    sd=torch.stack([x[y==c].var(0,correction=0).clamp_min(1e-4).sqrt() for c in range(16)])
    for ipc in (1,4):
      for objective in ('mean','grad'):
        run=SimpleNamespace(init=model,priv=[copy.deepcopy(model)],dev=device,K=1,bytes=[],X=[x],Y=[y],n=[len(y)],
             a=SimpleNamespace(arm=f'g{ipc}_{objective}_trained',smoke=False,seed=SEED))
        run.gen=lambda *keys: torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([SEED,*keys]).generate_state(1)[0]))
        method=Books(run,initialize=False); method.mu=mu.half().float(); method.sd=sd.half().float()
        torch.cuda.synchronize(); started=time.perf_counter(); book,diagnostic=method.condense(0,1,steps=80); torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        assert diagnostic['slots']==diagnostic['applied']==80 and torch.isfinite(book).all()
        _book_times[ipc][objective].append(seconds)
        print('T4 V5 full80 book preflight PASS',gpu,ipc,objective,round(seconds,3),flush=True)
        del method,run,book,diagnostic
    del model,x,y,mu,sd,source_x,source_y,indices
    torch.cuda.empty_cache()
(SOURCE/'book_preflight.json').write_text(json.dumps({'seconds':_book_times,'seed':SEED,
    'evidence':'one producer <=32rows/class, initial DAGSNet; full80 steps both objectives/IPC/devices, no F1/mixing/current-trained guarantee'}))
torch.cuda.set_device(0)
'''

BOOK_RUN=r'''
OUT.mkdir(exist_ok=True); baseline=list(METHODS[:2]); queue=list(METHODS[2:])
assert baseline[0].endswith('_diag') and baseline[1].endswith('_noise')
slots=[0,0,1,1]; active={}; done={}; seen={}; costs={}; budget=None; baseline_done=False
T0=time.monotonic()
def not_run(arm,reason):
    d=OUT/f'{arm}_s{SEED}'; d.mkdir(exist_ok=True)
    identity=json.loads((OUT/f'{METHODS[0]}_s{SEED}'/'config.json').read_text())
    identity.pop('fingerprint'); identity['config']['arm']=arm
    identity['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    (d/'config.json').write_text(json.dumps(identity)); (d/'log.txt').write_text(reason+'\n')
    done[arm]={'rc':-998,'complete':False,'not_started':True,'reason':reason}
while baseline or queue or active:
    if not baseline_done and all(arm in done for arm in METHODS[:2]):
        baseline_done=True
        if not all(done[arm]['complete'] and done[arm]['rc']==0 for arm in METHODS[:2]):
            for arm in queue: not_run(arm,'COST BASELINE MISSING: no valid comparable diagonal timing')
            queue=[]
        else:
            base=json.loads((OUT/f'{METHODS[0]}_s{SEED}'/'result.json').read_text())
            baseline_seconds=base['minutes']*60; budget=2*baseline_seconds
            ipc=1 if METHODS[0].startswith('g1_') else 4
            for arm in list(queue):
                objective='grad' if 'grad' in arm or arm.endswith('_own') else 'mean'
                projected=baseline_seconds+2*1.15*max(_book_times[ipc][objective])*20*4+60
                if projected>budget:
                    not_run(arm,'COST PREFLIGHT REJECT: projected total exceeds 2x measured baseline'); queue.remove(arm)
            (SOURCE/'cost_reference.json').write_text(json.dumps(dict(baseline_seconds=baseline_seconds,
                max_seconds=budget,baseline_arm=METHODS[0],concurrency='baseline diag+noise onGPU0; cases two workers/device',
                prediction_rule='baseline + 2*1.15*single_producer_book_seconds*20*4 +60')))
    for slot,gpu in enumerate(slots):
        if slot in active: continue
        if not baseline_done:
            if slot>1 or not baseline: continue
            arm=baseline.pop(0); limit=None
        else:
            if not queue: continue
            arm=queue.pop(0); limit=budget
        d=OUT/f'{arm}_s{SEED}'; d.mkdir(exist_ok=True); log=(d/'log.txt').open('w')
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
        command=[sys.executable,'-m','mh.v5_run','--arm',arm,'--seed',str(SEED),'--out',str(OUT),
                 '--wandb_online','--max_hours',str(limit/3600 if limit is not None else 4)]
        active[slot]=dict(arm=arm,proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT),
                          log=log,start=time.monotonic(),limit=limit,terminated=False)
    for slot,item in list(active.items()):
        arm=item['arm']; proc=item['proc']; d=OUT/f'{arm}_s{SEED}'
        lines=(d/'log.txt').read_text().splitlines()
        for line in lines[seen.get(arm,0):]:
            if line.startswith(('setup','r','done','WANDB','Traceback','budget')) or 'Error' in line: print(arm,'|',line,flush=True)
        seen[arm]=len(lines)
        elapsed=time.monotonic()-item['start']
        if item['limit'] is not None and elapsed>=item['limit']-30 and proc.poll() is None:
            if not item['terminated']:
                print('COST CAP terminate',arm,round(elapsed),flush=True); proc.terminate(); item['terminated']=True
            elif elapsed>=item['limit']-10: proc.kill()
        if proc.poll() is not None:
            result=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else {}
            active_seconds=result.get('minutes',elapsed/60)*60
            cost_pass=item['limit'] is None or (active_seconds<=item['limit'] and not item['terminated'])
            done[arm]={'rc':proc.returncode,'complete':result.get('complete',False),'cost_stopped':item['terminated'],'cost_pass':cost_pass}
            costs[arm]=dict(worker_wall_seconds=elapsed,active_seconds=active_seconds,
                            max_seconds=item['limit'],cost_stopped=item['terminated'],cost_pass=cost_pass)
            item['log'].close(); del active[slot]
    if baseline or queue or active: time.sleep(5)
(OUT/f'summary_s{SEED}.json').write_text(json.dumps(done,indent=2))
(SOURCE/'cost_observed.json').write_text(json.dumps(costs,indent=2))
print(json.dumps(done),flush=True)
shutil.rmtree(WORK)
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
assert set(done)==set(METHODS) and all(v['rc']==0 and v['complete'] and v.get('cost_pass',True) for v in done.values()),done
'''


def main():
    p=argparse.ArgumentParser(); p.add_argument('--family',choices=FAMILIES,required=True)
    p.add_argument('--owner42',required=True); p.add_argument('--owner43',required=True); args=p.parse_args()
    base=ROOT/'papers/v5'; runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    key=netrc.netrc().authenticators('api.wandb.ai')[2]; assert key
    assert '**/private_push/' in (ROOT/'.gitignore').read_text().splitlines()
    code=[f.name for f in sorted((ROOT/'mh').glob('*.py'))]
    manifest={name:hashlib.sha256((ROOT/'mh'/name).read_bytes()).hexdigest() for name in code}
    manifest['meta.json']=hashlib.sha256((ROOT/'knowledge/meta.json').read_bytes()).hexdigest()
    manifest.update({name:hashlib.sha256((ROOT/'docs/research'/name).read_bytes()).hexdigest() for name in DOCS})
    for owner,seed in ((args.owner42,42),(args.owner43,43)):
        rid=f'v5{args.family}-s{seed}-20261012-v1'; public=base/'notebook'/rid; private=base/'private_push'/rid
        assert not (public/'PUSHED').exists() and not (private/'PUSHED').exists()
        src=base/'input'/owner
        if (src/'source_manifest.json').exists():
            assert json.loads((src/'source_manifest.json').read_text())==manifest,'new source needs new dataset/attempt'
        else:
            src.mkdir(parents=True,exist_ok=True)
            for name in code: shutil.copyfile(ROOT/'mh'/name,src/name)
            shutil.copyfile(ROOT/'knowledge/meta.json',src/'meta.json')
            for name in DOCS: shutil.copyfile(ROOT/'docs/research'/name,src/name)
            for f in (ROOT/'papers/pilot/fixture').iterdir(): shutil.copyfile(f,src/f.name)
            (src/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
            (src/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{DATASET}',title=DATASET,licenses=[{'name':'CC0-1.0'}])))
        public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
        for name in ('source_manifest.json','manifest.json'): shutil.copyfile(src/name,public/name)
        setup=SETUP.replace("elif f.name=='pilot-v2.md'","elif f.name.startswith('pilot-v5-') and f.suffix=='.md'")
        runner=RUN.replace("'mh.v2_run'","'mh.v5_run'") if args.family=='h' else BOOK_RUN
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet V5 '+args.family.upper()+'\nFrozen input contracts; DAGSNet-only. '
            'No raw/perturbed/latent/per-record gradients exported; aggregate stats >=5 or noise-init synthetic exception. '
            'Teacher H first, private G/J; r20 internal holdouts, ten metrics from mean client confusion. '
            'W&B authenticated online before train. Books capped at 2x measured comparable baseline. Not R1/R2/crypto/DP proof.'),
            nbf.v4.new_code_cell(AUTH),nbf.v4.new_code_cell(f'SEED={seed}\nCODE={code!r}\n'+setup+(ANCHOR_PREFLIGHT if args.family=='h' else PREFLIGHT)),
            nbf.v4.new_code_cell(f'SEED={seed}\nMETHODS={FAMILIES[args.family]!r}\n'+runner)])
        nb.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
            language_info={'name':'python','version':'3.12'},kaggle={'accelerator':'nvidiaTeslaT4',
            'isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
        filename=rid+'.ipynb'; nbf.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code': compile(cell.source,rid,'exec')
        nbf.write(nb,public/filename)
        meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,language='python',
            kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
            machine_shape=runtime['machine_shape'],docker_image=runtime['docker_image'],
            dataset_sources=[f'{owner}/{DATASET}'],kernel_sources=[],competition_sources=[])
        assert len(meta['title'])<=50
        (public/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        credential=nbf.reads(nbf.writes(nb),as_version=4)
        credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
        nbf.write(credential,private/filename); os.chmod(private/filename,0o600)
        (private/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)); os.chmod(private/'kernel-metadata.json',0o600)
        print(meta['id'],'prepared; not pushed')


if __name__=='__main__': main()
