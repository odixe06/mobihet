"""Exercise actual emitted controller: completion, partial round boundary, budget, worker failure."""
import contextlib
import io
import json
import netrc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import nbformat

BASE=Path(__file__).resolve().parents[1]


def check(folder,case):
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    nb=nbformat.read(folder/meta['code_file'],as_version=4)
    plan=json.loads((folder/'resume-plan.json').read_text());assert len(plan)==8
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    private=BASE/'private_push'/folder.name/meta['code_file']
    assert key not in (folder/meta['code_file']).read_text() and key in private.read_text()
    assert private.stat().st_mode & 0o777==0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['machine_shape']=='NvidiaTeslaT4'
    assert '@sha256:' in meta['docker_image'] and nb.metadata['kaggle']['isGpuEnabled']
    assert nb.cells[2].source.index('imports=[import_run(')<nb.cells[2].source.index('x=torch.from_numpy(np.load(')
    assert json.loads((folder/'checkpoint-dataset.json').read_text())['uploaded']
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'actual ClassTail notebook','exec')
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);launches=[]
        for name in plan:
            (out/name).mkdir();(out/name/'config.json').write_text('{}')
        class Clock:
            @staticmethod
            def monotonic():return 12*3600 if case=='budget' else 0.
            @staticmethod
            def sleep(_):pass
        class Process:
            def __init__(self,command,**kwargs):
                arm=command[command.index('--arm')+1];seed=command[command.index('--seed')+1]
                name=f'{arm}_s{seed}';assert name in plan;launches.append(name)
                assert '--resume' in command and '--wandb_online' in command and '--stop_after' not in command
                assert command[command.index('--rounds')+1]=='20' and command[command.index('--eval_every')+1]=='1'
                assert float(command[command.index('--max_hours')+1])>0
                n=2 if case=='boundary' else 20
                (out/name/'result.json').write_text(json.dumps(dict(history=[{}]*n,complete=n==20)))
                self.returncode=7 if case=='failure' else 0
            def poll(self):return self.returncode
        scope=dict(Path=Path,json=json,os=os,sys=sys,time=Clock,WORK=out,OUT=out,SESSION_T0=0,
            subprocess=SimpleNamespace(Popen=Process,STDOUT=-2))
        try:exec(compile(nb.cells[3].source,'actual ClassTail controller','exec'),scope)
        except AssertionError:assert case=='failure'
        else:assert case!='failure'
        summary=json.loads((out/'utility_summary.json').read_text())
        if case=='failure':assert any(v.get('rc')==7 for v in summary.values())
        else:
            assert len(summary)==8
            assert len(launches)==(0 if case=='budget' else 8)
            assert all(v['rounds']==(20 if case=='normal' else 2) for v in summary.values())
            assert all(v['complete']==(case=='normal') for v in summary.values())
    return dict(kernel=meta['id'],case=case,pass_check=True)


if __name__=='__main__':
    rows=[]
    for argument in sys.argv[1:]:
        folder=Path(argument)
        with contextlib.redirect_stdout(io.StringIO()):
            checks=[check(folder,case) for case in ('normal','boundary','budget','failure')]
        (folder/'controller-preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
        rows.extend(checks)
    assert rows
    print('ClassTail emitted controllers PASS',len(rows),'cases')
