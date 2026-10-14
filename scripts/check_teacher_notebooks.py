"""Actual generated two-round cost/resume controller: success, cost stop, deadline."""
import argparse
import ast
import hashlib
import json
import netrc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def check(folder,mode):
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4)
    nbformat.validate(nb);private=folder.parent.parent/'private_push'/folder.name
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key not in (folder/meta['code_file']).read_text()
    assert key in (private/meta['code_file']).read_text() and (private/meta['code_file']).stat().st_mode&0o777==0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and '@sha256:' in meta['docker_image']
    assert nb.metadata['kaggle']['isGpuEnabled'] and meta['machine_shape']=='NvidiaTeslaT4'
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,str(folder),'exec')
    code=ast.literal_eval(ast.parse(nb.cells[2].source).body[0].value)
    manifest=json.loads((folder/'source_manifest.json').read_text())
    methods=ast.literal_eval(ast.parse(nb.cells[-1].source).body[1].value)
    seed=int(folder.name.split('-')[1][1:]);launched=[]
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary);work=base/'work';work.mkdir();out=base/'runs';src=base/'src';src.mkdir()
        (src/'manifest.json').write_bytes((folder/'manifest.json').read_bytes())
        class Clock:
            count=0
            @classmethod
            def time(cls):
                cls.count+=1
                return 0. if mode!='deadline' or cls.count<3 else 6*3600
            @staticmethod
            def sleep(_):pass
        class Process:
            def __init__(self,command,**kwargs):
                arm=command[command.index('--arm')+1];phase='probe' if '--stop_after' in command else 'continue'
                assert '--wandb_online' in command and kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('0','1')
                self.returncode=None if mode=='deadline' else 0;launched.append((arm,phase))
                seconds=3. if mode=='cost' and arm==methods[1] else 1.
                rows=[dict(algorithm_seconds=seconds) for _ in range(2 if phase=='probe' else 20)]
                (out/f'{arm}_s{seed}'/'result.json').write_text(json.dumps(dict(history=rows,complete=phase=='continue')))
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def kill(self):self.returncode=-9
        scope=dict(sys=sys,os=SimpleNamespace(environ={'WANDB_API_KEY':'TEST_ONLY'}),json=json,
            hashlib=hashlib,shutil=__import__('shutil'),time=Clock,Path=Path,subprocess=SimpleNamespace(Popen=Process,STDOUT=-2),
            OUT=out,WORK=work,SOURCE=base/'source',SRC=src,CODE=code,manifest=manifest)
        try:exec(compile(nb.cells[-1].source,'actual teacher controller','exec'),scope)
        except AssertionError:
            assert mode=='deadline'
        else:assert mode!='deadline'
        summary=json.loads((out/f'summary_s{seed}.json').read_text());assert set(summary)==set(methods)
        if mode=='normal':
            assert all(v['rc']==0 and v['complete'] and v['cost_pass'] for v in summary.values())
            assert all((arm,'probe') in launched and (arm,'continue') in launched for arm in methods)
        elif mode=='cost':
            assert summary[methods[1]]['cost_stopped'] and not summary[methods[1]]['complete']
            assert (methods[1],'continue') not in launched
        else:
            assert all(not v['complete'] for v in summary.values())
            for arm in methods[1:]:
                identity=json.loads((out/f'{arm}_s{seed}/config.json').read_text());fp=identity.pop('fingerprint')
                assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==fp
        assert not work.exists()
    print('PASS TEACHER NOTEBOOK',meta['id'],mode,len(methods),'arms')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',choices=['v9','v10'],required=True);args=parser.parse_args()
    for folder in sorted((ROOT/f'papers/{args.build}/notebook').iterdir()):
        for mode in ('normal','cost','deadline'):check(folder,mode)
