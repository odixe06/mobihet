"""Private full-data calibration notebooks: first3 of the fixed50 rounds, resumable."""
import hashlib
import json
import netrc
import os
import shutil
import subprocess
import sys
from pathlib import Path
import nbformat as nbf

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
sys.path.insert(0,str(ROOT))
from scripts.project_paths import current_path
from scripts.kaggle.gen_v2 import AUTH

PLAN=[('minhtrit06',['h_local','h_dml']),
      ('odixeuit',['h_gaussian','h_hybrid']),
      ('trietbackup',['h_gaussian_own','h_dml_own']),
      ('odixe0502',['h_hybrid_own'])]
DATASET='mobihet-hybrid-full-20261013-src'
SETUP=r'''
import hashlib,json,shutil,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
print('torch',torch.__version__,'gpus',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)
hits=[p.parent for p in Path('/kaggle/input').rglob('full_run.py') if (p.parent/'source_manifest.json').exists()]
assert len(hits)==1,hits
SRC=hits[0];WORK=Path('/kaggle/temp/full-context');OUT=Path('/kaggle/working/runs')
SOURCE=Path('/kaggle/working/source');SOURCE.mkdir(exist_ok=True)
for name in ('mh','knowledge','papers/hybrid-full/proj'):(WORK/name).mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'source_manifest.json').read_text())
for name,sha in manifest.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,('source mismatch',name)
    shutil.copyfile(SRC/name,SOURCE/name)
    target=WORK/'mh'/name if name in CODE else WORK/'papers/hybrid-full/proj'/name if name in PROJ else WORK/'knowledge'/name
    shutil.copyfile(SRC/name,target)
shutil.copyfile(SRC/'source_manifest.json',SOURCE/'source_manifest.json')
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/hybrid-full/proj'))
from full_prepare import prepare_full
data=prepare_full(WORK)
(SOURCE/'data_manifest.json').write_text(json.dumps(data,indent=2))
print('verified full source and data',data['source_rows'],data['test_rows'],flush=True)
'''
RUN=r'''
OUT.mkdir(exist_ok=True);active={};summary={}
for gpu,arm in enumerate(METHODS):
    folder=OUT/f'{arm}_s42';folder.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(WORK/'papers/hybrid-full/proj/full_run.py'),
             '--arm',arm,'--seed','42','--rounds','50','--eval_every','1','--backend','compiled',
             '--wandb_online','--wandb_project','mobihet-hybrid-full','--out',str(OUT),
             '--stop_after','3','--max_hours','4.2']
    log=open(folder/'log.txt','w');process=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    active[gpu]=(arm,process,log)
while active:
    for gpu,(arm,process,log) in list(active.items()):
        if process.poll() is None and time.monotonic()-SESSION_T0>=4.8*3600:process.terminate()
        if process.poll() is None and time.monotonic()-SESSION_T0>=4.85*3600:process.kill()
        if process.poll() is None:continue
        log.close();del active[gpu]
        path=OUT/f'{arm}_s42/result.json';result=json.loads(path.read_text()) if path.exists() else {}
        rows=result.get('history',[])
        summary[arm]=dict(rc=process.returncode,complete=result.get('complete',False),
                         calibration_complete=len(rows)==3,rounds=len(rows),target_rounds=50)
        if rows:
            worst=max(row['sec'] for row in rows)
            setup=max(0,result['minutes']*60-sum(row['sec'] for row in rows))
            estimate=1.15*(setup+worst*(50-len(rows)))/3600
            summary[arm].update(worst_round_seconds=worst,setup_seconds=setup,
                conservative_remaining_gpu_hours=estimate,
                note='one GPU per arm; Kaggle account/session cost must be measured separately')
    time.sleep(3)
(OUT/'summary_s42.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
assert all(v['rc']==0 and v['calibration_complete'] for v in summary.values()),'incomplete first3 full rounds; preserve partial'
'''


def main():
    base=ROOT/'.compat/papers/hybrid-full';runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key and '**/private_push/' in (ROOT/'.gitignore').read_text().splitlines()
    mh=sorted((ROOT/'.compat/mh').glob('*.py'));project=sorted((base/'proj').glob('*.py'))
    files=mh+project+[ROOT/'knowledge/meta.json',ROOT/'knowledge/scaler.json',ROOT/current_path('docs/research/full-hybrid.md')]
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert len(manifest)==len(files)
    for index,(owner,methods) in enumerate(PLAN,1):
        rid=f'hybf{index}-s42-20261013-cal-v1';public=base/'notebook'/rid;private=base/'private_push'/rid;source=base/'input'/owner
        assert not (public/'PUSHED').exists() and not (private/'PUSHED').exists()
        source.mkdir(parents=True,exist_ok=True);public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
        for file in files:shutil.copyfile(file,source/file.name)
        (source/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
        (source/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{DATASET}',title=DATASET,licenses=[{'name':'CC0-1.0'}])))
        code=[p.name for p in mh];proj=[p.name for p in project]
        notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# Hybrid full first3 of50 rounds\n'
            '100%train80%, fulltest20% everyround, descriptive only. AlllearnedmodelsDAGSNet, seed42. '
            'Compiled-vs-eager real-data numerical gate beforetrain; one arm/GPU. '
            'No rawrecords exported; prepareddata only/kaggle/temp. W&Bonline/privateinlinekey owner-authorized. '
            'Checkpoint willresume fixed50round schedule after actualbudget review. R1/R2notproved.'),
            nbf.v4.new_code_cell('import time\nSESSION_T0=time.monotonic()\n'+AUTH),
            nbf.v4.new_code_cell(f'CODE={code!r}\nPROJ={proj!r}\n'+SETUP),
            nbf.v4.new_code_cell(f'METHODS={methods!r}\n'+RUN)])
        notebook.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
            language_info={'name':'python','version':'3.12'},kaggle={'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,
            'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
        filename=rid+'.ipynb';nbf.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':compile(cell.source,rid,'exec')
        meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,language='python',
            kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
            machine_shape=runtime['machine_shape'],docker_image=runtime['docker_image'],
            dataset_sources=[f'{owner}/{DATASET}','trietbackup/veremi-fl-20client','trietbackup/veremi-nextgen2026-test'],
            kernel_sources=[],competition_sources=[])
        assert len(meta['title'])<=50
        nbf.write(notebook,public/filename);(public/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        (public/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
        subprocess.run(['git','check-ignore','-q',str((private/filename).resolve())],cwd=ROOT,check=True)
        credential=nbf.reads(nbf.writes(notebook),as_version=4)
        credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
        descriptor=os.open(private/filename,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
        with os.fdopen(descriptor,'w') as file:file.write(nbf.writes(credential))
        (private/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        os.chmod(private/filename,0o600);os.chmod(private/'kernel-metadata.json',0o600)
        print(meta['id'],'prepared; not pushed')


if __name__=='__main__':main()
