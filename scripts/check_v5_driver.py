"""All V5 small driver paths; phase refresh, tensor state resume and decisive inference."""
import argparse
import subprocess
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from mh.v5_run import ARMS
from check_v3_driver import equal
from verify_v5 import verify


def run(arm,out,extra=()):
    subprocess.run([sys.executable,'-m','mh.v5_run','--arm',arm,'--smoke','--out',str(out),*extra],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',default='final')
    parser.add_argument('--arms',nargs='+',choices=ARMS); args=parser.parse_args(); arms=args.arms or ARMS
    base=ROOT/'papers/v5/smoke'/args.tag; resumed=ROOT/'papers/v5/resume_reference'/args.tag
    for arm in arms:
        assert not (base/f'{arm}_s42').exists(),'fresh tag required'; run(arm,base)
    representatives=[arm for arm in ('b_anchor_small_fl','b_anchor_medium_own','g1_grad_trained','g1_own','g4_grad_trained','g4_own') if arm in arms]
    for arm in representatives:
        run(arm,resumed,('--stop_after','1')); run(arm,resumed,('--resume',))
        for r in range(1,4):
            for kind in ('weights','resume'):
                a=torch.load(base/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                b=torch.load(resumed/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                assert equal(a,b),(arm,r,kind)
    torch.set_num_threads(1)
    for arm in arms: verify(base/f'{arm}_s42',infer=True)
    print('PASS V5',len(arms),'arms x3rounds; exact',len(representatives),'resumes across small refresh; inference')


if __name__=='__main__': main()
