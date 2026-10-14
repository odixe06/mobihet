"""K5/G59/G61 adapter; prior, private training and audit driver stay immutable."""
import sys
import torch
from mh import v2_run as driver
from mh.v6_geometry import Method

FAMILIES = {'a': ['k66_plain_fl', 'k66_marginal_fl', 'k66_joint_fl'] +
                 [f'k59_{mask}_{mode}' for mask in ('marginal', 'joint')
                  for mode in ('fl', 'own', 'shuffle')],
            'b': [f'k61_{mask}_{mode}' for mask in ('marginal', 'joint')
                  for mode in ('fl', 'own', 'shuffle')]}
ARMS = sum(FAMILIES.values(), [])


class Run(driver.Run):
    def __init__(self, args):
        if not args.smoke:
            super().__init__(args)
            return
        # Covariance/cohort gates need more than the legacy256-row smoke.
        driver.P.Run.__init__(self, args)
        self.bytes = []
        self.X = [x[:1024] for x in self.X]
        self.Y = [y[:1024] for y in self.Y]
        self.H = {h: (x[:256], y[:256]) for h, (x, y) in self.H.items()}
        self.n = [len(y) for y in self.Y]
        self.counts = torch.stack([torch.bincount(y, minlength=16) for y in self.Y]).double()
        self.log_pi = ((self.counts+1)/(self.counts+1).sum(1, keepdim=True)).log().float()

if __name__ == '__main__':
    driver.ARMS = ARMS
    driver.Method = Method
    driver.Run = Run
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(driver.P.ROOT / 'papers/v6/runs')]
    if '--wandb_project' not in sys.argv:
        sys.argv += ['--wandb_project', 'mobihet-v6']
    driver.main()
