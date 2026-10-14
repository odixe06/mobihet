"""V5 new knowledge adapters; immutable V3/V4 training sources stay unchanged."""
import sys
from mh import v2_run as driver
from mh.v2_methods import Method as Base
from mh.v5_anchor import Method as Anchor
from mh.v5_books import Method as Books

FAMILIES={'h':['b_dense']+[f'b_anchor_{size}_{mode}' for size in ('small','medium') for mode in ('fl','own','shuffle','gauss')]}
for family,ipc in (('g',1),('j',4)):
    FAMILIES[family]=[f'g{ipc}_{mode}' for mode in ('diag','noise','mean_frozen','mean_trained','grad_frozen','grad_trained','own')]
ARMS=sum(FAMILIES.values(),[])


def method(run,initialize=True):
    cls=Anchor if run.a.arm.startswith('b_anchor_') else (Books if run.a.arm.startswith(('g1_','g4_')) else Base)
    return cls(run,initialize)


if __name__=='__main__':
    driver.ARMS=ARMS; driver.Method=method
    if '--out' not in sys.argv: sys.argv+=['--out',str(driver.P.ROOT/'papers/v5/runs')]
    if '--wandb_project' not in sys.argv: sys.argv+=['--wandb_project','mobihet-v5']
    driver.main()
