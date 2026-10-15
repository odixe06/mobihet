"""Exercise emitted two-GPU probe orchestration; mechanism FAIL remains a successful probe."""
import contextlib
import hashlib
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
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,str(folder),'exec')
    key=netrc.netrc().authenticators('api.wandb.ai')[2];private=BASE/'private_push'/folder.name/meta['code_file']
    assert key not in (folder/meta['code_file']).read_text() and key in private.read_text() and private.stat().st_mode & 0o777==0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['machine_shape']=='NvidiaTeslaT4' and '@sha256:' in meta['docker_image']
    assert nb.metadata['kaggle']['isGpuEnabled']
    inputs=json.loads((folder/'input_manifest.json').read_text())
    assert not any(name.startswith(('old_','group_','test_')) for name in inputs)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);(out/'protocol.json').write_text('{}');calls=[]
        class Clock:
            @staticmethod
            def monotonic():return 1e9 if case=='budget' else 0.
            @staticmethod
            def sleep(_):pass
        class Run:
            settings=SimpleNamespace(_offline=False)
            url='fixture://wandb'
            def log(self,value):pass
            def finish(self):pass
        class Process:
            def __init__(self,command,**kwargs):
                assert '--phase' in command and not any(v in command for v in ('--resume','--rounds','--arm'))
                calls.append((command[command.index('--phase')+1],command[command.index('--shard')+1]))
                assert kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('0','1')
                self.returncode=7 if case=='failure' else 0
            def poll(self):return self.returncode
        def synchronous(command,**kwargs):
            phase=command[command.index('--phase')+1];calls.append((phase,None))
            if phase=='finish':(out/'probe_result.json').write_text(json.dumps(dict(pass_probe=False,seed=42,model_training=False)))
        scope=dict(Path=Path,json=json,os=os,sys=sys,time=Clock,hashlib=hashlib,SRC=out,WORK=out,OUT=out,SESSION_T0=0,
            MANIFEST={},wandb=SimpleNamespace(init=lambda **k:Run(),Settings=lambda **k:SimpleNamespace(**k)),
            subprocess=SimpleNamespace(Popen=Process,run=synchronous,STDOUT=-2))
        try:exec(compile(nb.cells[3].source,'actual CrossClientRank probe controller','exec'),scope)
        except AssertionError:assert case in ('failure','budget')
        else:
            assert case=='normal'
            assert calls==[('scan','0'),('scan','1'),('aggregate',None),('probe','0'),('probe','1'),('finish',None)]
            assert json.loads((out/'probe_result.json').read_text())['pass_probe'] is False
    return dict(kernel=meta['id'],case=case,pass_check=True)


if __name__=='__main__':
    rows=[]
    for arg in sys.argv[1:]:
        folder=Path(arg)
        with contextlib.redirect_stdout(io.StringIO()):checks=[check(folder,c) for c in ('normal','failure','budget')]
        (folder/'controller-preflight.json').write_text(json.dumps(checks,indent=2)+'\n');rows.extend(checks)
    assert rows
    print('CrossClientRank actual controllers PASS',len(rows),'cases')
