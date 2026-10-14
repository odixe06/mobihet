"""V4 local MALA transport over unchanged V2 training/evaluation/commit driver."""
import sys
from mh import v2_run as driver
from mh.v2_methods import Method as Base
from mh.v4_mala import Method as Mala
from mh.v4_support import Method as Support

FAMILIES={'m':['diag','c_fl','c_mala_fl','c_mala_own','c_mala_shuffle']}
FAMILIES['s']=['s_diag','s_marginal','s_structural','s_own','s_shuffle']
ARMS=sum(FAMILIES.values(),[])


def method(run,initialize=True):
    cls=Support if run.a.arm.startswith('s_') else (Mala if 'mala' in run.a.arm else Base)
    return cls(run,initialize)


if __name__=='__main__':
    driver.ARMS=ARMS; driver.Method=method
    if '--out' not in sys.argv: sys.argv+=['--out',str(driver.P.ROOT/'papers/v4/runs')]
    if '--wandb_project' not in sys.argv: sys.argv+=['--wandb_project','mobihet-v4']
    driver.main()
