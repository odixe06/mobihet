"""Exercise generated controller failures/budget and verify real notebook metadata/source."""
import hashlib
import json
from pathlib import Path
import sys
import nbformat

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
BASE=ROOT/'papers/boundary-cells'


def controller(code,seed,case):
    folder=BASE/'controller-fixture'/f's{seed}-{case}'
    assert not folder.exists();folder.mkdir(parents=True)
    starts=[]
    class Clock:
        now=0
        def time(self):return self.now
        def sleep(self,seconds):self.now+=6000 if case=='budget' else seconds
    clock=Clock()
    class Process:
        def __init__(self,command,cwd,env,stdout,stderr):
            arm=command[command.index('--arm')+1]
            assert command[command.index('--seed')+1]==str(seed)
            gpu=int(env['CUDA_VISIBLE_DEVICES'])
            starts.append((arm,gpu))
            path=folder/f'{arm}_s{seed}';path.mkdir(exist_ok=True)
            secs=1. if arm=='bc_single' else 2.1 if arm=='bc_teacher2_fl' else 1.5
            n=1 if case=='baseline_fail' and arm=='bc_single' else 2
            (path/'result.json').write_text(json.dumps({'history':[{'algorithm_seconds':secs}]*n}))
            self.returncode=7 if case=='candidate_fail' and arm=='bc_direct4_fl' else 0
            self.polls=0
        def poll(self):
            self.polls+=1
            return None if self.polls==1 else self.returncode
    import types,os
    scope=dict(SEED=seed,OUT=folder,WORK=ROOT,time=clock,os=os,sys=sys,json=json,
               subprocess=types.SimpleNamespace(Popen=Process,STDOUT=-2))
    failed=False
    try:exec(code,scope)
    except AssertionError:failed=True
    summary=json.loads((folder/'calibration_summary.json').read_text())
    assert starts[0]==('bc_single',0) and len(summary)==5
    if case=='normal':
        assert not failed and len(starts)==5
        assert not summary[f'bc_teacher2_fl_s{seed}']['cost']['pass_cost']
        assert summary[f'bc_direct4_fl_s{seed}']['cost']['pass_cost']
    elif case in ('baseline_fail','budget'):
        assert failed and len(starts)==1
    else:
        assert failed and len(starts)==5 and not summary[f'bc_direct4_fl_s{seed}']['calibration_valid']
    return dict(seed=seed,case=case,starts=starts,expected_failure=failed)


def main():
    proof=json.loads((BASE/'local-proof-initial2.json').read_text())
    tests=[];packages=[]
    for seed in (42,43):
        folder=BASE/f'notebook/bc-cal-s{seed}-20261013-initial2-v1'
        metadata=json.loads((folder/'kernel-metadata.json').read_text())
        nb=nbformat.read(folder/metadata['code_file'],as_version=4)
        assert metadata['is_private'] and metadata['enable_gpu'] and metadata['enable_internet']
        assert metadata['machine_shape']=='NvidiaTeslaT4' and '@sha256:' in metadata['docker_image']
        assert len(metadata['title'])<=50
        assert nb.metadata['kaggle']['isGpuEnabled'] and nb.metadata['kaggle']['isInternetEnabled']
        assert nb.metadata['kaggle']['accelerator']=='nvidiaTeslaT4'
        assert '_INLINE_WANDB_KEY = ""' in nb.cells[1].source
        assert nb.cells[3].source.startswith(f'SEED={seed}\n')
        manifest=json.loads((folder/'source_manifest.json').read_text())
        assert all(manifest[name]==sha for name,sha in proof['source_sha'].items())
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,str(folder),'exec')
        assert 'terminate(' not in nb.cells[3].source and 'kill(' not in nb.cells[3].source
        code=nb.cells[3].source.split('\n',1)[1]
        for case in ('normal','baseline_fail','candidate_fail','budget'):
            tests.append(controller(code,seed,case))
        packages.append(dict(seed=seed,kernel=metadata['id'],notebook_sha=hashlib.sha256((folder/metadata['code_file']).read_bytes()).hexdigest()))
    (BASE/'notebook-proof-initial2.json').write_text(json.dumps(dict(packages=packages,controller_tests=tests,
           online_key_boundary=True,dual_GPU_metadata=True,T4_execution_proven=False),indent=2)+'\n')
    print('Boundary notebooks/controller PASS',len(tests),flush=True)


if __name__=='__main__':main()
