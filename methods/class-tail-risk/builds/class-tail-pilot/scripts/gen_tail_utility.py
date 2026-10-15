"""Prepare exact r2→r20 continuation only after both locked mechanism probes pass."""
import hashlib
import json
import netrc
import os
from pathlib import Path
import sys
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[5]
BASE=Path(__file__).resolve().parents[1]
RUN=r'''
pending=list(METHODS);active={};summary={}
while active or pending:
    for gpu in (0,1):
        if gpu in active or not pending:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<.5:continue
        arm=pending.pop(0);folder=OUT/f'{arm}_s{SEED}'
        assert (folder/'config.json').exists(),'require-resume failed'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(WORK/'boundary_logging.py'),str(WORK/'papers/class-tail/proj/tail_driver.py'),
            '--arm',arm,'--seed',str(SEED),'--rounds','20','--eval_every','1','--resume','--wandb_online',
            '--max_hours',str(remaining-.25),'--out',str(OUT)]
        log=open(folder/'log.txt','a');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(arm,proc,log)
        print('start ClassTail utility',arm,SEED,gpu,flush=True)
    for gpu,(arm,proc,log) in list(active.items()):
        if proc.poll() is None:continue
        log.close();del active[gpu]
        result=json.loads((OUT/f'{arm}_s{SEED}/result.json').read_text())
        summary[arm]=dict(rc=proc.returncode,rounds=len(result['history']),complete=result.get('complete',False))
        (OUT/'utility_summary.json').write_text(json.dumps(summary,indent=2))
        assert proc.returncode==0 and summary[arm]['rounds']>=2,('worker failed; preserve output',arm,summary[arm])
    if not active and pending and SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600<.5:break
    if active or pending:time.sleep(2)
for arm in pending:summary[arm]=dict(rc=None,rounds=2,complete=False,reason='session budget; imported warmup only')
assert len(summary)==len(METHODS)
(OUT/'utility_summary.json').write_text(json.dumps(summary,indent=2))
print('ClassTail utility session; evaluate locked gates only after audited r20',summary,flush=True)
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seed,owner,hours=int(sys.argv[1]),sys.argv[2],float(sys.argv[3])
    gate=json.loads((BASE/'probes/paired-mechanism-gate.json').read_text());assert gate['pass_probe']
    parent=f'ct-cal-s{seed}-20261015-v1';source=BASE/'artifacts/kaggle'/parent
    audited=json.loads((source/'AUDITED.json').read_text());assert audited['mechanism_pass']
    rid=f'ct-u-s{seed}-20261015-v1';public,private=BASE/'notebooks'/rid,BASE/'private_push'/rid
    stage=BASE/'artifacts/checkpoint-staging'/rid
    assert not any(p.exists() for p in (public,private,stage))
    oldmeta=json.loads((BASE/'notebooks'/parent/'kernel-metadata.json').read_text())
    reference=nbf.read(BASE/'notebooks'/parent/oldmeta['code_file'],as_version=4)
    manifest=json.loads((BASE/'notebooks'/parent/'source_manifest.json').read_text())
    expected,files={},{}
    for arm in audited['summary']:
        run=source/'runs'/f'{arm}_s{seed}';identity=json.loads((run/'config.json').read_text())
        expected[run.name]={k:identity[k] for k in ('fingerprint','source','fixture')}
        expected[run.name]['last_round']=2
        for p in run.rglob('*'):
            if not p.is_file() or 'wandb' in p.relative_to(run).parts or p.name=='log.txt':continue
            relative=Path('runs')/run.name/p.relative_to(run);target=stage/relative
            target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
            files[str(relative)]=dict(sha256=sha(target),size=target.stat().st_size)
    package=stage/'package';package.mkdir()
    for name,digest in manifest.items():
        original=source/'source'/name;assert sha(original)==digest
        os.link(original,package/name);files['package/'+name]=dict(sha256=digest,size=original.stat().st_size)
    for p in (ROOT/'papers/pilot/fixture').iterdir():
        if p.is_file():
            os.link(p.resolve(),package/p.name)
            files['package/'+p.name]=dict(sha256=sha(p),size=p.stat().st_size)
    (package/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dataset=f'{owner}/mobihet-{rid}-ckpt'
    bridge=dict(dataset=dataset,files=files,parent_kernel=audited['kernel'],parent_version=1,
                verified_full_parent=True,private=True,uploaded=False)
    for name,value in [('dataset-metadata.json',dict(id=dataset,title=f'mobihet-{rid}-ckpt',licenses=[{'name':'CC0-1.0'}])),
                       ('checkpoint-manifest.json',bridge)]:
        (stage/name).write_text(json.dumps(value,indent=2)+'\n')
    setup=reference.cells[2].source
    before,after=setup.split('from mh.dagsnet import build_model,CFG',1)
    imports=f'\nEXPECTED={expected!r}\nIMPORTER={(BASE/"scripts/tail_resume.py").read_text()!r}\n'
    imports+="(WORK/'tail_resume.py').write_text(IMPORTER)\nfrom tail_resume import import_run\nfrom tail_verify import verify\ntorch.set_num_threads(1)\nimports=[import_run('/kaggle/input',OUT,name,item,manifest,verify) for name,item in EXPECTED.items()]\n(SOURCE/'resume-import.json').write_text(json.dumps(imports))\n"
    setup=before+imports+'from mh.dagsnet import build_model,CFG'+after
    run=f'SEED={seed}\nSESSION_HOURS={hours!r}\nMETHODS={list(audited["summary"])!r}\n'+RUN
    auth=reference.cells[1].source.replace("_wandb_key=UserSecretsClient().get_secret('wandb_key')", "_wandb_key=_INLINE_WANDB_KEY or UserSecretsClient().get_secret('wandb_key')")
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# ClassTail utility continuation\nPaired r2 mechanism PASS. Exact warm-up state, fixed20schedule, eight locked arms. Budget boundary is not r20/utility PASS. Private inline W&B key persists in version history.'),
        nbf.v4.new_code_cell(auth),nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(run)])
    nb.metadata=reference.metadata.copy();name=rid+'.ipynb'
    meta=dict(oldmeta,id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name,dataset_sources=[dataset],kernel_sources=[])
    public.mkdir(parents=True);private.mkdir(parents=True,mode=0o700)
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    nbf.write(nb,public/name)
    for filename,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest),('resume-plan.json',expected),('checkpoint-dataset.json',bridge)]:
        (public/filename).write_text(json.dumps(value,indent=2)+'\n')
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    credential=nbf.reads(nbf.writes(nb),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for filename,value in [(name,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value)
    probe=BASE/'probes/mount-probe'/rid;probe.mkdir(parents=True)
    cpu='import os,sys\n'+before+imports+"\n(SOURCE/'MOUNT_VERIFIED.json').write_text(json.dumps(dict(cpu_only=True,imports=imports)))\n"
    cpu=cpu.replace('assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))','assert torch.cuda.device_count()==0')
    cpu=cpu.replace("assert torch.__version__=='2.10.0+cu128'",'')
    cpu_nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(cpu)])
    cpu_nb.metadata=dict(nb.metadata,kaggle=dict(accelerator='none',isGpuEnabled=False,isInternetEnabled=True,language='python',sourceType='notebook'))
    cpu_meta=dict(meta,id=meta['id']+'-cpu',title=meta['title']+'-cpu',enable_gpu=False);cpu_meta.pop('machine_shape',None)
    nbf.validate(cpu_nb);compile(cpu,rid,'exec');nbf.write(cpu_nb,probe/name)
    (probe/'kernel-metadata.json').write_text(json.dumps(cpu_meta,indent=2)+'\n')
    print(meta['id'],'prepared; CPU exact import required before GPU push')


if __name__=='__main__':main()
