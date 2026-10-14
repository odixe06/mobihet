"""Hybrid pilot: reuse committed-round, online W&B and per-client metric driver."""
import sys
from mh import v2_run as driver
from mh.v6_run import Run as Base
from mh.v8_hybrid import Method, ARMS, FAMILIES, private_epoch


class Run(Base):
    def private_epoch(self, method, k, r):
        return private_epoch(self, method, k, r)


if __name__ == '__main__':
    driver.ARMS, driver.Method, driver.Run = ARMS, Method, Run
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(driver.P.ROOT / 'papers/v8/runs')]
    if '--wandb_project' not in sys.argv:
        sys.argv += ['--wandb_project', 'mobihet-v8']
    driver.main()
