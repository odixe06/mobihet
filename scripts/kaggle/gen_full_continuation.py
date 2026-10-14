"""New same-account notebooks resume the immutable H1/H2 source and its fixed fifty-round schedule."""
import hashlib
import json
import netrc
import os
import subprocess
from pathlib import Path
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.kaggle.gen_v2 import AUTH
from scripts.kaggle.gen_full_validation import PLAN, DATASET


def main():
    base = ROOT / 'papers/full-validation'
    runtime = json.loads((ROOT / 'knowledge/runtime.json').read_text())
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    helper = (ROOT / 'scripts/kaggle/full_resume.py').read_text()
    verifier_support = (ROOT / 'scripts/check_v3_driver.py').read_text()
    for index, (owner, methods) in enumerate(PLAN, 1):
        parent = f'full{index}-s42-20261012-cal-v1'
        audited = json.loads((base / 'kaggle' / parent / 'AUDITED.json').read_text())
        assert audited['calibration_complete'] and all(v['rounds'] == 3 for v in audited['audits'])
        original = base / 'notebook' / parent
        metadata = json.loads((original / 'kernel-metadata.json').read_text())
        source_manifest = json.loads((original / 'source_manifest.json').read_text())
        expected = {}
        for arm in methods:
            identity = json.loads((base / 'kaggle' / parent / 'runs' / f'{arm}_s42/config.json').read_text())
            expected[arm] = dict(fingerprint=identity['fingerprint'], source=identity['source'], fixture=identity['fixture'], last_round=3)
        code = [n for n in source_manifest if n in {p.name for p in (ROOT / 'mh').glob('*.py')}]
        project = [p.name for p in (base / 'proj').glob('*.py')]
        rid = f'full{index}-s42-20261013-cont01-v1'
        public = base / 'notebook' / rid
        private = base / 'private_push' / rid
        assert not (public / 'PUSHED').exists() and not (private / 'PUSHED').exists()
        public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
        setup = f'CODE={code!r}\nPROJ={project!r}\nEXPECTED={expected!r}\n' + r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
torch.set_num_threads(1)
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
print('torch',torch.__version__,'gpus',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)
hits=[p.parent for p in Path('/kaggle/input').rglob('full_run.py')
      if (p.parent/'source_manifest.json').exists() and not (p.parent/'data_manifest.json').exists()]
assert len(hits)==1,('source package missing/ambiguous',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/full-context');OUT=Path('/kaggle/working/runs')
SOURCE=Path('/kaggle/working/source');SOURCE.mkdir(exist_ok=True)
for name in ('mh','knowledge','papers/full-validation/proj'):(WORK/name).mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'source_manifest.json').read_text())
for name,sha in manifest.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,('source mismatch',name)
    shutil.copyfile(SRC/name,SOURCE/name)
    target=WORK/'mh'/name if name in CODE else WORK/'papers/full-validation/proj'/name if name in PROJ else WORK/'knowledge'/name
    shutil.copyfile(SRC/name,target)
shutil.copyfile(SRC/'source_manifest.json',SOURCE/'source_manifest.json')
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/full-validation/proj'))
'''
        setup += f'HELPER={helper!r}\n'
        setup += f'VERIFIER_SUPPORT={verifier_support!r}\n'
        setup += r'''
(WORK/'full_resume.py').write_text(HELPER)
(SOURCE/'full_resume.py').write_text(HELPER)
(WORK/'scripts').mkdir(exist_ok=True)
(WORK/'scripts/check_v3_driver.py').write_text(VERIFIER_SUPPORT)
(SOURCE/'check_v3_driver.py').write_text(VERIFIER_SUPPORT)
from full_resume import import_run
from full_verify import verify
imports=[import_run('/kaggle/input',OUT,arm,value,verify) for arm,value in EXPECTED.items()]
(SOURCE/'resume-import.json').write_text(json.dumps(imports,indent=2))
print('REQUIRE RESUME PASS before data decode',imports,flush=True)
from full_prepare import prepare_full
data=prepare_full(WORK)
assert hashlib.sha256((WORK/'papers/pilot/fixture/manifest.json').read_bytes()).hexdigest()==next(iter(EXPECTED.values()))['fixture']
(SOURCE/'data_manifest.json').write_text(json.dumps(data,indent=2))
print('verified full source and data',data['source_rows'],data['test_rows'],flush=True)
'''
        run = f'METHODS={methods!r}\n' + r'''
active={};summary={}
for gpu,arm in enumerate(METHODS):
    folder=OUT/f'{arm}_s42'
    remaining=4.8-(time.monotonic()-SESSION_T0)/3600
    worker_hours=min(4.0,remaining-.25)
    prior=json.loads((folder/'result.json').read_text())
    worst=max(row['sec'] for row in prior['history'])/3600
    assert worker_hours>=1.15*worst+.05,'not enough session time for a complete resumed round'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(WORK/'papers/full-validation/proj/full_run.py'),
        '--arm',arm,'--seed','42','--rounds','50','--eval_every','1','--backend','compiled',
        '--wandb_online','--wandb_project','mobihet-full-validation','--out',str(OUT),'--resume','--max_hours',str(worker_hours)]
    log=open(folder/'log.txt','w');process=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    active[gpu]=(arm,process,log)
while active:
    for gpu,(arm,process,log) in list(active.items()):
        if process.poll() is None and time.monotonic()-SESSION_T0>=4.8*3600:process.terminate()
        if process.poll() is None and time.monotonic()-SESSION_T0>=4.85*3600:process.kill()
        if process.poll() is None:continue
        log.close();del active[gpu]
        path=OUT/f'{arm}_s42/result.json';result=json.loads(path.read_text()) if path.exists() else {}
        rows=result.get('history',[]);new=[r for r in rows if r['round']>EXPECTED[arm]['last_round']]
        summary[arm]=dict(rc=process.returncode,complete=result.get('complete',False),rounds=len(rows),
            previous_round=EXPECTED[arm]['last_round'],new_rounds=len(new),target_rounds=50,
            continuation_verified=bool(new),worst_round_seconds=max([r['sec'] for r in rows] or [0]))
    time.sleep(3)
(OUT/'summary_s42.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
assert all(v['rc']==0 and v['continuation_verified'] for v in summary.values()),'continuation failed; preserve committed rounds'
'''
        notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# Full H1/H2 checkpoint continuation\nResume verified r3; next r4, fixed50round schedule, full train/test. '
            'Require-resume before data decode. W&B online; private inline key persists in private version history. '
            'Session budget stops at a complete boundary; target remains r50. R1/R2 not proved.'),
            nbf.v4.new_code_cell('import time\nSESSION_T0=time.monotonic()\n'+AUTH),
            nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(run)])
        notebook.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
            language_info={'name':'python','version':'3.12'},kaggle={'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,
            'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
        filename=rid+'.ipynb'
        meta=dict(metadata,id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,kernel_sources=[metadata['id']])
        assert len(meta['title'])<=50 and meta['is_private'] and meta['docker_image']==runtime['docker_image']
        nbf.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':compile(cell.source,rid,'exec')
        nbf.write(notebook,public/filename)
        (public/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        (public/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
        (public/'resume-plan.json').write_text(json.dumps(expected,indent=2))
        subprocess.run(['git','check-ignore','-q',str(private/filename)],cwd=ROOT,check=True)
        credential=nbf.reads(nbf.writes(notebook),as_version=4)
        credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
        descriptor=os.open(private/filename,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
        with os.fdopen(descriptor,'w') as file:file.write(nbf.writes(credential))
        (private/'kernel-metadata.json').write_text(json.dumps(meta,indent=2));(private/'kernel-metadata.json').chmod(0o600)
        print(meta['id'],'prepared; not pushed',flush=True)


if __name__=='__main__':main()
