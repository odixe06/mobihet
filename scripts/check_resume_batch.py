"""Exercise the emitted continuation cells with unequal rounds and failure boundaries."""
import json
import netrc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def check(folder, case):
    meta = json.loads((folder/'kernel-metadata.json').read_text())
    nb = nbformat.read(folder/meta['code_file'], as_version=4)
    plan = json.loads((folder/'resume-plan.json').read_text())
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    private = folder.parents[1]/'private_push'/folder.name/meta['code_file']
    assert key not in (folder/meta['code_file']).read_text()
    assert key in private.read_text() and private.stat().st_mode & 0o777 == 0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet']
    assert meta['machine_shape']=='NvidiaTeslaT4' and '@sha256:' in meta['docker_image']
    assert nb.metadata['kaggle']['accelerator']=='nvidiaTeslaT4'
    if meta['kernel_sources']:
        assert meta['id'].split('/')[0]==meta['kernel_sources'][0].split('/')[0]
    else:
        bridge=json.loads((folder/'checkpoint-dataset.json').read_text())
        assert bridge['uploaded'] and bridge['verified_full_parent'] and bridge['private']
        assert bridge['dataset'] in meta['dataset_sources']
        assert meta['id'].split('/')[0]==bridge['dataset'].split('/')[0]
    assert nb.cells[2].source.index('import_run(')<nb.cells[2].source.index('data=prepare_full(')
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code': compile(cell.source, str(folder), 'exec')
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)/'runs';out.mkdir();launches=[]
        for arm,value in plan.items():
            run=out/f'{arm}_s42';run.mkdir()
            (run/'result.json').write_text(json.dumps(dict(history=[dict(round=r,sec=100.)
                for r in range(1,value['last_round']+1)])))
        class Clock:
            @staticmethod
            def monotonic(): return 12*3600 if case=='exhausted' else 0
            @staticmethod
            def sleep(_): pass
        class Process:
            def __init__(self,command,**kwargs):
                arm=command[command.index('--arm')+1];launches.append(arm)
                assert '--resume' in command and '--stop_after' not in command
                assert command[command.index('--rounds')+1]=='50' and '--wandb_online' in command
                assert float(command[command.index('--max_hours')+1])>0
                last=plan[arm]['last_round'];new=0 if case=='boundary' else 2
                (out/f'{arm}_s42/result.json').write_text(json.dumps(dict(
                    history=[dict(round=r,sec=100.) for r in range(1,last+new+1)],complete=False)))
                self.returncode=7 if case=='worker-fail' else 0
            def poll(self): return self.returncode
            def terminate(self): self.returncode=-15
            def kill(self): self.returncode=-9
        scope=dict(Path=Path,json=json,os=os,sys=sys,time=Clock,EXPECTED=plan,OUT=out,
                   WORK=Path(directory),SESSION_T0=0,
                   subprocess=SimpleNamespace(Popen=Process,STDOUT=-2))
        try: exec(compile(nb.cells[3].source,'actual generated continuation','exec'),scope)
        except AssertionError: assert case in ('exhausted','worker-fail')
        else: assert case in ('normal','boundary')
        if case=='exhausted': assert not launches
        else:
            assert launches==scope['METHODS']
            summary=json.loads((out/'summary_s42.json').read_text())
            for arm,value in summary.items():
                assert value['previous_round']==plan[arm]['last_round']
                assert value['new_rounds']==(0 if case=='boundary' else 2)
                assert value['target_rounds']==50 and not value['complete']
                assert value['valid_budget_boundary']==(case!='worker-fail')
    return dict(kernel=meta['id'],case=case,pass_check=True)


def main():
    rows=[]
    for build,tag in [('full-validation','20261014-cont02'),('hybrid-full','20261014-cont01')]:
        for folder in sorted((ROOT/'papers'/build/'notebook').glob(f'*{tag}*')):
            rows.extend(check(folder,case) for case in ('normal','boundary','exhausted','worker-fail'))
    assert len(rows)==32
    (ROOT/'docs/operations/resume-batch-preflight-2026-10-14.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Generated resume batch PASS:',len(rows),'controller cases; remote T4/import still checked at startup')


if __name__=='__main__':main()
