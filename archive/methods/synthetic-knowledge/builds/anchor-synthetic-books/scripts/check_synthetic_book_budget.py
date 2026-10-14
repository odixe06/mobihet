"""Execute actual generated controller with fake workers: reject costly and terminate overtime."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import nbformat
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])


def check(folder,mode):
    meta=json.loads((folder/'kernel-metadata.json').read_text()); nb=nbformat.read(folder/meta['code_file'],as_version=4)
    seed=int(folder.name.split('-')[1][1:]); launched=[]; terminated=[]
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp); work=base/'work'; work.mkdir(); out=base/'runs'; source=base/'source'; source.mkdir()
        class Clock:
            value=0.
            @classmethod
            def monotonic(cls):
                cls.value+=250 if mode=='terminate' else 0
                return cls.value
            @staticmethod
            def sleep(_): pass
        class Proc:
            def __init__(self,command,**kwargs):
                self.arm=command[command.index('--arm')+1]; launched.append(self.arm)
                assert '--wandb_online' in command
                path=out/f'{self.arm}_s{seed}'
                identity=dict(config=dict(arm=self.arm,seed=seed),source={'fake':'0'},fixture='fixture')
                identity['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                (path/'config.json').write_text(json.dumps(identity))
                baseline=self.arm.endswith(('_diag','_noise'))
                self.returncode=0 if baseline else None
                if baseline: (path/'result.json').write_text(json.dumps(dict(complete=True,minutes=1.)))
            def poll(self): return self.returncode
            def terminate(self): terminated.append(self.arm); self.returncode=-15
            def kill(self): self.returncode=-9
        namespace=dict(sys=sys,os=SimpleNamespace(environ={'WANDB_API_KEY':'TEST_ONLY'}),json=json,
            hashlib=hashlib,shutil=__import__('shutil'),time=Clock,Path=Path,subprocess=SimpleNamespace(Popen=Proc,STDOUT=-2),
            OUT=out,WORK=work,SOURCE=source,
            _book_times={ipc:{kind:[10.,10.] if mode=='reject' else [0.,0.] for kind in ('mean','grad')} for ipc in (1,4)})
        try: exec(compile(nb.cells[-1].source,'actual generated controller','exec'),namespace)
        except AssertionError: pass  # parent intentionally ERROR when a locked cost gate fails
        else: raise AssertionError('failed budget unexpectedly accepted')
        summary=json.loads((out/f'summary_s{seed}.json').read_text()); methods=namespace['METHODS']
        assert set(summary)==set(methods)
        if mode=='reject':
            assert launched==methods[:2] and not terminated
            assert all(summary[a]['not_started'] and 'COST PREFLIGHT REJECT' in summary[a]['reason'] for a in methods[2:])
            for arm in methods[2:]:
                cfg=json.loads((out/f'{arm}_s{seed}/config.json').read_text()); fp=cfg.pop('fingerprint')
                assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==fp
        else:
            assert launched==methods and terminated==methods[2:]
            assert all(summary[a]['cost_stopped'] and summary[a]['cost_pass'] is False for a in methods[2:])
        assert not work.exists(),'raw temporary fixture workspace was retained'
    print('PASS actual controller',folder.name,mode)


def main():
    for folder in sorted((ROOT/'papers/v5/notebook').glob('v5[gj]-*')):
        for mode in ('reject','terminate'): check(folder,mode)


if __name__=='__main__': main()
