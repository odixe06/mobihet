"""Bounded eager arithmetic/resume proof, no full data or local compile training."""
import argparse
import fcntl
import subprocess
import sys
from pathlib import Path
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
PROJECT=ROOT/'papers/hybrid-full/proj'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(PROJECT))
import scripts  # register compatibility imports before loading the frozen verifier
from full_method import ARMS
from full_verify import verify
from scripts.check_v3_driver import equal


def run(arm,out,extra=()):
    subprocess.run([sys.executable,str(PROJECT/'full_run.py'),'--arm',arm,'--smoke',
                    '--backend','eager','--out',str(out),*extra],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    base=ROOT/'papers/hybrid-full/smoke'/args.tag
    resumed=ROOT/'papers/hybrid-full/resume_reference'/args.tag
    with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for arm in ARMS:run(arm,base)
        for arm in ('h_gaussian','h_dml','h_hybrid','h_hybrid_own','h_dml_own'):
            run(arm,resumed,('--stop_after','1'));run(arm,resumed,('--resume',))
            for r in range(1,4):
                for kind in ('weights','resume'):
                    a=torch.load(base/f'{arm}_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    b=torch.load(resumed/f'{arm}_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    assert equal(a,b),(arm,r,kind)
        torch.set_num_threads(1)
        for arm in ARMS:verify(base/f'{arm}_s42',infer=True)
    print('PASS FULL local:7boundedarms/5exactresumes/strictBN/payload/pred/inference; T4compile/fullcoverage pending')


if __name__=='__main__':main()
