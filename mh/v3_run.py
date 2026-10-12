"""V3 transport adapter over unchanged, verified V2 CE/KD/evaluation/commit driver."""
import sys
from mh import v2_run as driver
from mh.v3_math import ARMS
from mh.v3_methods import Method
from mh.v3_dense import Method as Dense


def method(run,initialize=True):
    return (Dense if run.a.arm=='a_dense32' else Method)(run,initialize=initialize)


if __name__=='__main__':
    driver.ARMS=ARMS; driver.Method=method
    if '--out' not in sys.argv: sys.argv+=['--out',str(driver.P.ROOT/'papers/v3/runs')]
    if '--wandb_project' not in sys.argv: sys.argv+=['--wandb_project','mobihet-v3']
    driver.main()
