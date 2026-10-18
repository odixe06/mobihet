"""Actual driver fork/atomic markers/interrupted-round replay on a tiny synthetic runtime; mocked W&B only here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from mh.dagsnet import build_model,CFG
from scripts.check_v3_driver import equal
from rank_verify import verify


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='rank-driver-fixture-') as tmp:
        runtime=Path(tmp);fx=runtime/'papers/pilot/fixture';proj=runtime/'proj'
        for p in (fx,proj,runtime/'mh',runtime/'knowledge',runtime/'stub'):p.mkdir(parents=True)
        for p in (ROOT/'.compat/mh').glob('*.py'):shutil.copyfile(p,runtime/'mh'/p.name)
        for p in (BASE/'source/proj').glob('*.py'):shutil.copyfile(p,proj/p.name)
        shutil.copyfile(ROOT/'knowledge/meta.json',runtime/'knowledge/meta.json')
        rng=np.random.default_rng(1729)
        for k in range(20):
            np.save(fx/f'train_{k}_X.npy',rng.normal(size=(512,66)).astype('float32'))
            np.save(fx/f'train_{k}_y.npy',(np.arange(512)%16).astype('int64'))
        for hold in ('group','old'):
            np.save(fx/f'{hold}_X.npy',rng.normal(size=(64,66)).astype('float32'));np.save(fx/f'{hold}_y.npy',np.arange(64)%16)
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fx.iterdir()}
        (fx/'manifest.json').write_text(json.dumps(dict(files=files,fixture='synthetic fixture; not VeReMi utility')))
        torch.manual_seed(42);model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
        warm=runtime/'warm.pt';torch.save(dict(round=2,private=[model.state_dict()]*20,proxy=[],fingerprint='synthetic-warm-fixture'),warm)
        (runtime/'stub/wandb.py').write_text('''from types import SimpleNamespace
def login(**kwargs):return True
def Settings(**kwargs):return None
class Run:
 settings=SimpleNamespace(_offline=False)
 url='fixture-only'
 summary={}
 def log(self,*args,**kwargs):pass
 def finish(self):pass
def init(**kwargs):return Run()
sdk=SimpleNamespace(wandb_run=SimpleNamespace(Run=Run))
''')
        env=dict(os.environ,PYTHONPATH=str(runtime/'stub')+os.pathsep+str(runtime),CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_API_KEY='synthetic-fixture-only')
        command=[sys.executable,str(proj/'rank_driver.py'),'--arm','cr_fl','--warm',str(warm),'--wandb_online','--eval_every','1','--stop_after','4','--max_hours','.5']
        direct=runtime/'direct';replay=runtime/'replay'
        subprocess.run(command+['--out',str(direct)],check=True,env=env,cwd=runtime)
        first=command.copy();first[first.index('--stop_after')+1]='3'
        subprocess.run(first+['--out',str(replay)],check=True,env=env,cwd=runtime)
        run=replay/'cr_fl_s42';(run/'weights/round_005.pt').write_bytes(b'uncommitted orphan must not be imported')
        subprocess.run(command+['--out',str(replay),'--resume'],check=True,env=env,cwd=runtime)
        for r in (3,4):
            for kind in ('weights','resume'):
                a=torch.load(direct/f'cr_fl_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                b=torch.load(replay/f'cr_fl_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                assert equal(a,b),(r,kind)
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list((runtime/'mh').glob('*.py'))+list(proj.glob('*.py'))}
        audit=verify(run,manifest,hashlib.sha256((fx/'manifest.json').read_bytes()).hexdigest(),fx)
        assert audit['inherited_rounds']==2 and audit['logical_rounds']==4 and audit['committed_rounds']==2 and not audit['complete']
        prediction=run/'predictions/round_004.npz';prediction.write_bytes(prediction.read_bytes()+b'tamper')
        try:verify(run,manifest,hashlib.sha256((fx/'manifest.json').read_bytes()).hexdigest(),fx)
        except AssertionError:pass
        else:raise AssertionError('tampered prediction accepted')
        # CE carries empty bytes, unlike pooled. Exercise the actual recovery launcher
        # and both uninterrupted/interrupted driver paths with weights_only=True.
        ce=[sys.executable,str(BASE/'scripts/rank_bytes_compat.py'),str(proj/'boundary_logging.py'),*command[1:]]
        ce[ce.index('--arm')+1]='cr_local'
        ce_direct=runtime/'ce-direct';ce_replay=runtime/'ce-replay'
        subprocess.run(ce+['--out',str(ce_direct)],check=True,env=env,cwd=runtime)
        ce_first=ce.copy();ce_first[ce_first.index('--stop_after')+1]='3'
        subprocess.run(ce_first+['--out',str(ce_replay)],check=True,env=env,cwd=runtime)
        subprocess.run(ce+['--out',str(ce_replay),'--resume'],check=True,env=env,cwd=runtime)
        with torch.serialization.safe_globals([bytes]):
            for r in (3,4):
                for kind in ('weights','resume'):
                    a=torch.load(ce_direct/f'cr_local_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    b=torch.load(ce_replay/f'cr_local_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    assert equal(a,b),('CE compatible resume',r,kind)
            ce_audit=verify(ce_replay/'cr_local_s42',manifest,hashlib.sha256((fx/'manifest.json').read_bytes()).hexdigest(),fx)
        assert ce_audit['logical_rounds']==4 and ce_audit['committed_rounds']==2
    proof=dict(actual_driver_fork=True,exact_resume=True,orphan_ignored=True,tamper_rejected=True,
        audit=audit,ce_compat_resume=True,ce_audit=ce_audit,pass_check=True,
        scope='Tiny synthetic 20x512-row fixture; mocked W&B; no production data, cost or utility claim')
    (BASE/'probes/driver-preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print('Rank driver fixture PASS',proof)


if __name__=='__main__':main()
