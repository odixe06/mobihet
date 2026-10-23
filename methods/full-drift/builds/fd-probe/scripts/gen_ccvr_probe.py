"""Prepare immutable private 2xT4 notebook: frozen-head teacher/support probes for both audited seeds.

No push here. Reuse the original 2.5% pilot fixture, all old/group rows; never attach the official test.
"""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess

import nbformat

from gen_fd import BASE, ROOT, FULL, RUNTIME, WANDB


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--owner', required=True); a = p.parse_args()
    rid = 'fd-ccvr-probes-20261022-v1'
    stage = BASE / 'artifacts/input' / rid
    public, private = BASE / 'notebooks' / rid, BASE / 'private_push' / rid
    assert not stage.exists() and not public.exists() and not private.exists()
    fixture = ROOT / 'archive/methods/shared-baselines/builds/pilot-baselines/artifacts/fixture'
    weights = {42: ('fd-s42xa-20261015-r22-v1', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_s42_e15'),
               43: ('fd-s43ya-20261015-r23-v2', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_s43_e15')}
    mh = sorted(path.name for path in (ROOT / '.compat/mh').glob('*.py')); assert len(mh) == 40
    files = {name: ROOT / '.compat/mh' / name for name in mh}
    files.update({name: FULL / name for name in ('full_run.py', 'full_method.py', 'full_driver.py')})
    files.update({name: BASE / 'source/proj' / name for name in ('fd.py', 'fd_probe14.py', 'fd_probe23.py', 'fd_probe24.py')})
    files.update({name: ROOT / 'knowledge' / name for name in ('meta.json', 'scaler.json')})
    manifest = json.loads((fixture / 'manifest.json').read_text())
    files.update({name: fixture / name for name in ['manifest.json', *manifest['files']]})
    for seed, (session, run) in weights.items():
        artifact = BASE / 'artifacts/kaggle' / session
        proof = json.loads((artifact / 'AUDITED.json').read_text()); assert proof['audits'][0]['complete']
        directory = artifact / 'runs' / run
        weight = directory / 'weights/round_015.pt'
        marker = json.loads((directory / 'complete/round_015.json').read_text())
        assert sha(weight) == marker['weights_sha']
        files[f'weights_s{seed}.pt'] = weight
    stage.mkdir(); public.mkdir(); private.mkdir(mode=0o700)
    for name, source in files.items(): shutil.copyfile(source, stage / name)
    inputs = {name: sha(stage / name) for name in sorted(files)}
    (stage / 'input_manifest.json').write_text(json.dumps(inputs, indent=1) + '\n')
    dataset = f'{a.owner}/mobihet-ccvr-probes-20261022-v1'
    (stage / 'dataset-metadata.json').write_text(json.dumps(dict(id=dataset, title=dataset.split('/')[1], licenses=[dict(name='CC0-1.0')])))
    setup = f'MH={mh!r}\nINPUTS={inputs!r}\n' + r'''
import hashlib,json,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('input_manifest.json') if json.loads(p.read_text())==INPUTS]
assert len(hits)==1,('missing/ambiguous inputs',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/ccvr');PROJ=WORK/'papers/full-drift/proj';FX=WORK/'fixture'
OUT=Path('/kaggle/working/probes');SOURCE=Path('/kaggle/working/source')
for directory in (WORK/'mh',WORK/'knowledge',PROJ,FX,OUT,SOURCE):directory.mkdir(parents=True,exist_ok=True)
for name,expected in INPUTS.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
    target=WORK/'mh'/name if name in MH else PROJ/name if name.endswith('.py') else WORK/'knowledge'/name if name in ('meta.json','scaler.json') else FX/name if name=='manifest.json' or name.endswith('.npy') else WORK/name
    if name.endswith('.py'):
        shutil.copyfile(SRC/name,target);shutil.copyfile(SRC/name,SOURCE/name)
    else:target.symlink_to(SRC/name)
(SOURCE/'input_manifest.json').write_text(json.dumps(INPUTS,indent=1))
print('Verified',len(INPUTS),'files; cached fixture; no official test',flush=True)
'''
    run = r'''
active={};logs={}
for gpu,seed in enumerate((42,43)):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{PROJ}:{WORK}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=(OUT/f'seed{seed}.log').open('w');logs[seed]=log
    # Sequential per seed; two independent persistent shell jobs, no synchronisation between FL clients.
    code="""
import os,subprocess,sys
from pathlib import Path
proj,fx,out,seed,work=map(str,sys.argv[1:])
for number in (23,24):
    command=[sys.executable,str(Path(proj)/f'fd_probe{number}.py'),'--weights',str(Path(work)/f'weights_s{seed}.pt'),'--fixture',fx,'--out',str(Path(out)/f'probe{number}-s{seed}.json')]
    subprocess.run(command,check=True,env=os.environ)
"""
    active[seed]=subprocess.Popen([sys.executable,'-c',code,str(PROJ),str(FX),str(OUT),str(seed),str(WORK)],env=env,stdout=log,stderr=subprocess.STDOUT)
wb=wandb.init(project='mobihet-fulldrift',name='ccvr-probes-20261022-v1',config=dict(scope='frozen-head diagnostic',seeds=[42,43],official_test=False),settings=wandb.Settings(save_code=False,disable_code=True))
started=time.monotonic()
while any(proc.poll() is None for proc in active.values()):
    state={seed:proc.poll() for seed,proc in active.items()}
    wb.log({'elapsed_seconds':time.monotonic()-started,'finished_seeds':sum(value is not None for value in state.values())})
    print('probe heartbeat',round(time.monotonic()-started),state,flush=True)
    assert time.monotonic()-started<3600,'one-hour mechanism-screen budget'
    time.sleep(30)
for seed,proc in active.items():
    logs[seed].close();print('seed',seed,'rc',proc.returncode,(OUT/f'seed{seed}.log').read_text()[-4000:],flush=True)
assert all(proc.returncode==0 for proc in active.values()),'probe subprocess failed'
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
(OUT/'COMPLETE.json').write_text(json.dumps(dict(inputs=INPUTS,files=files,seconds=time.monotonic()-started),indent=1))
wb.finish()
'''
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# CCVR compression and teacher × support probes\nFrozen audited ep15 backbones; two seeds; old/group only. No full FL utility claim.'),
                                      nbformat.v4.new_code_cell(WANDB), nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell(run)])
    nb.metadata = dict(kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True,language='python',sourceType='notebook'),
                       kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12'))
    for cell in nb.cells:
        if cell.cell_type == 'code': compile(cell.source, rid, 'exec')
    nbformat.validate(nb)
    meta = dict(id=f'{a.owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=f'{rid}.ipynb',language='python',kernel_type='notebook',
                is_private=True,enable_gpu=True,enable_internet=True,machine_shape=RUNTIME['machine_shape'],docker_image=RUNTIME['docker_image'],
                dataset_sources=[dataset],kernel_sources=[],competition_sources=[])
    nbformat.write(nb, public / meta['code_file'])
    (public / 'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (public / 'source_manifest.json').write_text(json.dumps(inputs,indent=1)+'\n')
    secret = netrc.netrc().authenticators('api.wandb.ai')[2]
    assert meta['is_private'] is True
    subprocess.run(['git','check-ignore','-q',str(private / meta['code_file'])],cwd=ROOT,check=True)
    nb.cells[1].source=nb.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(secret))
    for name,body in [(meta['code_file'],nbformat.writes(nb)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        descriptor=os.open(private/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:stream.write(body)
    print(meta['id'],'prepared',len(inputs),'files',sum(path.stat().st_size for path in files.values()),'bytes')


if __name__ == '__main__': main()
