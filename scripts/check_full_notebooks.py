"""Generated full-data controller, credential boundary and partial3-of50 contract."""
import json
import netrc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def check(folder,deadline=False):
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4)
    private=folder.parent.parent/'private_push'/folder.name
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key not in (folder/meta['code_file']).read_text() and key in (private/meta['code_file']).read_text()
    assert (private/meta['code_file']).stat().st_mode&0o777==0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
    assert '@sha256:' in meta['docker_image'] and nb.metadata['kaggle']['isGpuEnabled']
    assert len(meta['dataset_sources'])==3 and not meta['kernel_sources']
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,str(folder),'exec')
    launched=[]
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary);work=base/'work';work.mkdir();out=base/'runs'
        class Clock:
            @staticmethod
            def monotonic():return 6*3600 if deadline else 0.
            @staticmethod
            def sleep(_):pass
        class Process:
            def __init__(self,command,**kwargs):
                arm=command[command.index('--arm')+1];launched.append(arm)
                assert command[command.index('--rounds')+1]=='50' and command[command.index('--stop_after')+1]=='3'
                assert '--wandb_online' in command and command[command.index('--backend')+1]=='compiled'
                assert kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('0','1')
                self.returncode=None if deadline else 0
                history=[dict(sec=100.) for _ in range(1 if deadline else 3)]
                (out/f'{arm}_s42/result.json').write_text(json.dumps(dict(history=history,complete=False,minutes=10.)))
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def kill(self):self.returncode=-9
        scope=dict(Path=Path,sys=sys,os=SimpleNamespace(environ={'WANDB_API_KEY':'TEST_ONLY'}),json=json,time=Clock,
            subprocess=SimpleNamespace(Popen=Process,STDOUT=-2),WORK=work,OUT=out,SESSION_T0=0.)
        try:exec(compile(nb.cells[-1].source,'actual full controller','exec'),scope)
        except AssertionError:assert deadline
        else:assert not deadline
        assert launched==scope['METHODS']
        summary=json.loads((out/'summary_s42.json').read_text())
        assert set(summary)==set(launched) and all(not v['complete'] and v['target_rounds']==50 for v in summary.values())
        assert all(v['calibration_complete']!=deadline for v in summary.values())
    print('PASS FULL NOTEBOOK',meta['id'],'deadline' if deadline else '3-of50',len(launched),'arms')


if __name__=='__main__':
    for folder in sorted((ROOT/'papers/full-validation/notebook').iterdir()):
        for deadline in (False,True):check(folder,deadline)
