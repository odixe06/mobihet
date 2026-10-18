"""Real generated continuation controller: next-round, no-progress, exhausted-budget and secret boundaries."""
import json
import netrc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import nbformat

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])


def check(folder,case):
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    nb=nbformat.read(folder/meta['code_file'],as_version=4)
    private=folder.parent.parent/'private_push'/folder.name
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key not in (folder/meta['code_file']).read_text() and key in (private/meta['code_file']).read_text()
    assert (private/meta['code_file']).stat().st_mode&0o777==0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
    assert '@sha256:' in meta['docker_image'] and len(meta['kernel_sources'])==1
    assert meta['id'].split('/')[0]==meta['kernel_sources'][0].split('/')[0]
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,str(folder),'exec')
    expected=json.loads((folder/'resume-plan.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary);work=base/'work';work.mkdir();out=base/'runs';out.mkdir()
        for arm in expected:
            target=out/f'{arm}_s42';target.mkdir()
            (target/'result.json').write_text(json.dumps(dict(history=[dict(round=r,sec=100.) for r in range(1,4)])))
        class Clock:
            @staticmethod
            def monotonic():return 6*3600 if case=='exhausted' else 0.
            @staticmethod
            def sleep(_):pass
        launches=[]
        class Process:
            def __init__(self,command,**kwargs):
                arm=command[command.index('--arm')+1];launches.append(arm)
                assert '--resume' in command and '--stop_after' not in command
                assert command[command.index('--rounds')+1]=='50' and '--wandb_online' in command
                assert 0<float(command[command.index('--max_hours')+1])<=4.0
                self.returncode=0
                history=[dict(round=r,sec=100.) for r in range(1,4 if case=='no-progress' else 7)]
                (out/f'{arm}_s42/result.json').write_text(json.dumps(dict(history=history,complete=False)))
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def kill(self):self.returncode=-9
        scope=dict(Path=Path,sys=sys,os=SimpleNamespace(environ={'WANDB_API_KEY':'TEST_ONLY'}),json=json,time=Clock,
            subprocess=SimpleNamespace(Popen=Process,STDOUT=-2),WORK=work,OUT=out,EXPECTED=expected,SESSION_T0=0.)
        try:exec(compile(nb.cells[-1].source,'actual full continuation','exec'),scope)
        except AssertionError:assert case!='normal'
        else:assert case=='normal'
        if case=='exhausted':assert not launches
        else:
            assert launches==scope['METHODS']
            summary=json.loads((out/'summary_s42.json').read_text())
            assert all(v['previous_round']==3 and v['target_rounds']==50 and not v['complete'] for v in summary.values())
            assert all(v['new_rounds']==(3 if case=='normal' else 0) for v in summary.values())
    print('PASS CONTINUATION NOTEBOOK',meta['id'],case)


if __name__=='__main__':
    for folder in sorted((ROOT/'.compat/papers/full-validation/notebook').glob('*cont01*')):
        for case in ('normal','no-progress','exhausted'):check(folder,case)
