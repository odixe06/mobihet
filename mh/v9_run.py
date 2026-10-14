"""Residual-anchor teacher pilot, with the existing atomic audit/W&B driver."""
import sys
from mh import teacher_driver as driver
from mh.v6_run import Run
from mh.v9_anchor import Method, ARMS, FAMILIES

if __name__=='__main__':
    driver.ARMS,driver.Method,driver.Run=ARMS,Method,Run
    if '--out' not in sys.argv: sys.argv+=['--out',str(driver.P.ROOT/'papers/v9/runs')]
    if '--wandb_project' not in sys.argv: sys.argv+=['--wandb_project','mobihet-v9']
    driver.main()
