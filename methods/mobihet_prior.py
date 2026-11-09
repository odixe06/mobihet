"""Y50 prior: Core with synthetic rows and Q in one prior coordinate.

Real CE trains z + log p_k - log pi_g, so raw z is a posterior under the global prior pi_g. Core's balanced synthetic rows
train raw z directly, a posterior under the uniform prior. Here s = log(1/16) - log pi_g is added on every synthetic row, so
those rows also make raw z a pi_g posterior. Q is collected through the same s: a Q measured on raw z already holds pi_g,
and matching softmax(z + s) to it would count the prior twice. Same messages and byte counts as Core; pi_g is the bootstrap's.
"""
import math

import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_wstudy import WLive, active


class Method(WLive):
    def __init__(self, env):
        super().__init__(env)
        assert self.variant == "stream" and self.w_synth == "full" and not self.no_kd
        total = env.allreduce(self.cnt.sum(0).double())
        self.prior_shift = (-math.log(16) - ((total + 1) / (total.sum() + 16)).log()).float()

    def _loss_stream(self, ctx):
        cfg, s = self.env.cfg, self.prior_shift
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        xp = torch.cat([ctx.x, xg], 1)
        mp = torch.cat([ctx.mask, active(ctx)[:, None].expand(G, a)], 1)
        zp, nBp = grouped.TRAIN(ctx.P, ctx.B, xp, cfg, mask=mp, train=True)
        y = ctx.y
        ce = local.masked_mean(F.cross_entropy((zp[:, :b] + self._shift[:, None, :]).reshape(-1, 16), y.reshape(-1),
                                               reduction="none").view(G, b), ctx.mask)
        with torch.no_grad():
            pr = torch.where(ctx.mask[..., None], F.softmax(zp[:, :b].detach().float() + s, -1), 0.0)
            self._csum.index_put_((torch.arange(G, device=y.device)[:, None].expand_as(y), y), pr, accumulate=True)
        zg = zp[:, b:] + s
        if self.Q is None:
            return ce + F.cross_entropy(zg.reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a).mean(1), nBp
        lt_g = self.Q.clamp_min(1e-6).log()[lab]
        return ce + (lt_g.exp() * (lt_g - F.log_softmax(zg, -1))).sum(-1).mean(1), nBp
