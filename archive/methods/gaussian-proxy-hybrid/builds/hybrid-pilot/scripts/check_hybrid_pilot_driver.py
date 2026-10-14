"""All ten bounded paths and exact resume, with artifact-derived model inference."""
import argparse
import fcntl
import subprocess
import sys
from pathlib import Path
import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from mh.v8_hybrid import ARMS
from check_v3_driver import equal
from verify_v8 import verify


def run(arm, out, extra=()):
    subprocess.run([sys.executable,'-m','mh.v8_run','--arm',arm,'--smoke','--out',str(out),*extra],
                   cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    base=ROOT/'papers/v8/smoke'/args.tag
    resumed=ROOT/'papers/v8/resume_reference'/args.tag
    with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for arm in ARMS:
            assert not (base/f'{arm}_s42').exists(),'fresh tag required'
            run(arm,base)
        for arm in ('h_dml','h_hybrid','h_hybrid_own'):
            run(arm,resumed,('--stop_after','1'))
            run(arm,resumed,('--resume',))
            for r in range(1,4):
                for kind in ('weights','resume'):
                    left=torch.load(base/f'{arm}_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    right=torch.load(resumed/f'{arm}_s42/{kind}/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    assert equal(left,right),(arm,r,kind)
        torch.set_num_threads(1)
        for arm in ARMS:
            verify(base/f'{arm}_s42',infer=True)
    print('PASS V8:7armsx3rounds;3exactresumes;strict/payload/inference',flush=True)


if __name__=='__main__':
    main()
