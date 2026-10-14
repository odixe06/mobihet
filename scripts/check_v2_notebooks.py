"""Check metadata, credential boundary, and exact generated arm orchestration without GPUs/network."""
import json
import argparse
import netrc
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.kaggle.gen_v2 import FAMILIES


class Clock:
    @staticmethod
    def time(): return 0.
    @staticmethod
    def monotonic(): return 0.
    @staticmethod
    def sleep(_): pass


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--build',choices=['v2','v3','v4','v5','v6','v7','v8'],default='v2'); args=parser.parse_args()
    families=FAMILIES
    if args.build=='v3':
        from mh.v3_math import FAMILIES as families
    elif args.build=='v4':
        from mh.v4_run import FAMILIES as families
    elif args.build=='v5':
        from mh.v5_run import FAMILIES as families
    elif args.build=='v6':
        from mh.v6_run import FAMILIES as families
    elif args.build=='v7':
        from mh.v7_run import FAMILIES as families
    elif args.build=='v8':
        from mh.v8_run import FAMILIES as families
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    for folder in sorted((ROOT/f'papers/{args.build}/notebook').iterdir()):
        meta=json.loads((folder/'kernel-metadata.json').read_text()); nb=nbformat.read(folder/meta['code_file'],as_version=4)
        nbformat.validate(nb)
        assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet']
        assert meta['machine_shape']=='NvidiaTeslaT4' and '@sha256:' in meta['docker_image']
        assert nb.metadata['kaggle']['isGpuEnabled'] and nb.metadata['kaggle']['isInternetEnabled']
        assert key not in (folder/meta['code_file']).read_text()
        private=ROOT/f'papers/{args.build}/private_push'/folder.name
        pnb=nbformat.read(private/meta['code_file'],as_version=4)
        assert json.loads((private/'kernel-metadata.json').read_text())==meta
        assert key in pnb.cells[1].source and 'wandb.login(key=' in pnb.cells[1].source
        assert 'WANDB_API_KEY' in pnb.cells[1].source and 'verify=True' in pnb.cells[1].source
        assert private.joinpath(meta['code_file']).stat().st_mode & 0o777 == 0o600
        for cell in nb.cells:
            if cell.cell_type=='code': compile(cell.source,str(folder),'exec')
        family=folder.name[2]; expected=families[family]; seed=int(folder.name.split('-')[1][1:])
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work'; work.mkdir(); out=Path(tmp)/'runs'
            launched=[]
            class Proc:
                def __init__(self,command,**kwargs):
                    arm=command[command.index('--arm')+1]; launched.append(arm)
                    assert '--wandb_online' in command and kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('0','1')
                    d=out/f'{arm}_s{seed}'; (d/'result.json').write_text(json.dumps({'complete':True,'minutes':1.}))
                    self.returncode=0
                def poll(self): return self.returncode
            class Subprocess:
                Popen=Proc; STDOUT=-2
            runner=nb.cells[-1].source
            source=Path(tmp)/'source'; source.mkdir()
            namespace=dict(sys=sys,os=SimpleNamespace(environ={'WANDB_API_KEY':'TEST_ONLY'}),
                           json=json,shutil=__import__('shutil'),time=Clock,Path=Path,
                           subprocess=Subprocess,OUT=out,WORK=work,SOURCE=source,hashlib=__import__('hashlib'),
                           _book_times={ipc:{kind:[0.,0.] for kind in ('mean','grad')} for ipc in (1,4)})
            exec(compile(runner,'generated runner','exec'),namespace)
            assert launched==expected
            summary=json.loads((out/f'summary_s{seed}.json').read_text())
            assert set(summary)==set(expected) and all(v['complete'] and v['rc']==0 for v in summary.values())
        print('PASS notebook',meta['id'],'exact',len(expected),'arms; private credential boundary')


if __name__=='__main__': main()
