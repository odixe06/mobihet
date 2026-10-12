"""All fifteen bounded driver paths and exact bootstrap resume representatives."""
import argparse
import fcntl
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from mh.v6_run import ARMS
from check_v3_driver import equal
from verify_v6 import verify


def run(arm, path, extra=()):
    subprocess.run([sys.executable, '-m', 'mh.v6_run', '--arm', arm, '--smoke',
                    '--out', str(path), *extra], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    base = ROOT / 'papers/v6/smoke' / args.tag
    resumed = ROOT / 'papers/v6/resume_reference' / args.tag
    # Share local GPU with remote artifact inference, not another trainer.
    with (ROOT / 'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for arm in ARMS:
            assert not (base / f'{arm}_s42').exists(), 'fresh tag required'
            run(arm, base)
        representatives = ['k66_plain_fl', 'k59_joint_fl', 'k61_marginal_shuffle', 'k61_joint_own']
        for arm in representatives:
            run(arm, resumed, ('--stop_after', '1'))
            run(arm, resumed, ('--resume',))
            for r in range(1, 4):
                for kind in ('weights', 'resume'):
                    a = torch.load(base / f'{arm}_s42/{kind}/round_{r:03d}.pt', weights_only=True)
                    b = torch.load(resumed / f'{arm}_s42/{kind}/round_{r:03d}.pt', weights_only=True)
                    assert equal(a, b), (arm, r, kind)
        torch.set_num_threads(1)
        for arm in ARMS:
            verify(base / f'{arm}_s42', infer=True)
    print('PASS V6:15arms x3rounds;4exactresumes;strict/inference')


if __name__ == '__main__':
    main()
