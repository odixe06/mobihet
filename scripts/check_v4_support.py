"""Support fidelity on actual sampled fixture, five driver paths, exact bootstrap resume."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from mh.v4_support import Method
from mh.v4_schema import FIRST,PROFILES,STRUCTURAL
from mh.dagsnet import build_model,CFG
from mh.v4_run import FAMILIES
from check_v3_driver import equal
from verify_v4 import verify


def fidelity():
    torch.set_num_threads(1); g=torch.Generator(device='cuda').manual_seed(4801)
    fx=ROOT/'papers/pilot/fixture'; X=[]; Y=[]
    for k in range(20):
        x=np.load(fx/f'train_{k}_X.npy',mmap_mode='r'); y=np.load(fx/f'train_{k}_y.npy',mmap_mode='r')
        idx=np.random.default_rng(4801+k).choice(len(y),min(2048,len(y)),replace=False)
        X.append(torch.tensor(np.array(x[idx]),device='cuda')); Y.append(torch.tensor(np.array(y[idx]),device='cuda').long())
    run=SimpleNamespace(dev=torch.device('cuda'),K=20,X=X,Y=Y,n=[len(y) for y in Y],bytes=[],
        init=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).cuda(),a=SimpleNamespace(arm='s_structural'))
    method=Method(run); labels=torch.arange(65536,device='cuda')%16
    rows=[]
    for arm in ('s_diag','s_marginal','s_structural','s_shuffle'):
        method.arm=arm; samples=method.draw_support(labels,g,0)
        zero=samples==method.atom0; one=samples==method.atom1
        binary=[FIRST]+sum(PROFILES,[]); binary_invalid=int((~(zero[:,binary]|one[:,binary])).any(1).sum())
        profile_invalid=sum(int((one[:,idx].sum(1)>1).sum()) for idx in PROFILES)
        first=one[:,FIRST]; structural=int((~zero[first][:,STRUCTURAL]).any(1).sum())
        if arm!='s_diag': assert binary_invalid==profile_invalid==0
        if arm in ('s_structural','s_shuffle'): assert structural==0
        rows.append(dict(arm=arm,rows=len(samples),binary_invalid=binary_invalid,
            profile_invalid=profile_invalid,first_rows=int(first.sum()),first_structural_invalid=structural))
    assert len(method.support_packet)==10592
    up=[row[4] for row in run.bytes if row[2]=='up']; assert up==[12776]*20
    (ROOT/'papers/v4/support-math.json').write_text(json.dumps(dict(evidence='2048 real train rows/client aggregate, 65536 synthetic draws per analytic sampler; no F1',checks=rows,
        one_shot_upload=12776,one_shot_download=10592,with_prior_per_client=23568),indent=2))
    del method,run; torch.cuda.empty_cache()


def run(arm,out,extra=()):
    subprocess.run([sys.executable,'-m','mh.v4_run','--arm',arm,'--smoke','--out',str(out),*extra],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',default='support'); args=parser.parse_args()
    fidelity(); base=ROOT/'papers/v4/smoke'/args.tag; resumed=ROOT/'papers/v4/resume_reference'/args.tag
    for arm in FAMILIES['s']:
        assert not (base/f'{arm}_s42').exists(),'fresh tag required'; run(arm,base)
    for arm in ('s_structural','s_own'):
        run(arm,resumed,('--stop_after','1')); run(arm,resumed,('--resume',))
        for r in range(1,4):
            for kind in ('weights','resume'):
                a=torch.load(base/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                b=torch.load(resumed/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                assert equal(a,b),(arm,r,kind)
    for arm in FAMILIES['s']: verify(base/f'{arm}_s42',infer=True)
    print('PASS support schema/fidelity/bytes, five arms x3rounds, exact structural/own resume and inference')


if __name__=='__main__': main()
