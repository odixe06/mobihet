"""Nyström pilot adapter, keeping the verified private epoch and checkpoint driver."""
import sys
from mh import v2_run as driver
from mh.v6_run import Run
from mh.v7_nystrom import ARMS, FAMILIES, Method


if __name__ == '__main__':
    driver.ARMS, driver.Method, driver.Run = ARMS, Method, Run
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(driver.P.ROOT / 'papers/v7/runs')]
    if '--wandb_project' not in sys.argv:
        sys.argv += ['--wandb_project', 'mobihet-v7']
    driver.main()
