"""CBB reception: KL of class-mean probabilities, with a half-dose row-KL control."""
import math

import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_refine import Method as CBB
from methods.mobihet_wstudy import active


def class_losses(logp, target):
    """Balanced cyclic labels: four rows per class, separately for each client.

    Preserve CBB's target clamp (including its tiny excess mass), and average
    probabilities in log space so extreme logits retain finite gradients.
    """
    assert logp.ndim == 3 and logp.shape[1:] == (64, 16)
    lq = target.detach().clamp_min(1e-6).log()
    lp = logp.reshape(logp.shape[0], 4, 16, 16)
    row = (lq.exp() * (lq - lp)).sum(-1).mean((1, 2))
    mean_lp = torch.logsumexp(lp, dim=1) - math.log(4)
    mean = (lq.exp() * (lq - mean_lp)).sum(-1).mean(1)
    return row, mean


class Method(CBB):
    def __init__(self, env):
        super().__init__(env)
        self.mean_mode = env.hp.get("mean_kd", "row")
        assert self.mean_mode in {"row", "mean", "half"}
        assert self.refine_loss == "forward" and self.real_smoothing == 0 and self.q_agg == "count"
        assert self.aux == 64 and torch.equal(self.labels, torch.arange(64, device=env.dev) % 16)
        assert not self.cf and self.synth == "copula"

    def _loss_stream(self, ctx):
        if self.mean_mode == "row":
            return super()._loss_stream(ctx)
        G, b = ctx.Gfix, ctx.x.shape[1]
        xg, lab = self._synth(ctx)
        enabled = active(ctx)
        mask = torch.cat([ctx.mask, enabled[:, None].expand(G, 64)], 1)
        z, bn = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), self.env.cfg, mask=mask, train=True)
        ce = local.ce_loss(z[:, :b] + self._shift[:, None], ctx.y, ctx.mask)
        lp = F.log_softmax(z, -1)
        with torch.no_grad():
            prob = torch.where(ctx.mask[..., None], lp[:, :b].detach().exp(), 0.)
            self._csum.index_put_((torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y),
                                  prob, accumulate=True)
            self._mean_stats[5].add_(enabled.sum())
            self._mean_stats[6].add_((ctx.mask.sum(1) * enabled).sum())
        if self.Q is None:
            # Isolate reception of soft Q: hard synthetic CE at r1 stays exact.
            synth = F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, 64).mean(1)
        else:
            row, mean = class_losses(lp[:, b:], self.Q)
            synth = mean if self.mean_mode == "mean" else .5 * row
            with torch.no_grad():
                gap = row - mean
                self._mean_stats[:5].add_(torch.stack([
                    enabled.sum(), torch.where(enabled, row, 0.).sum(),
                    torch.where(enabled, mean, 0.).sum(), torch.where(enabled, gap, 0.).sum(),
                    (enabled & (gap < -1e-5)).sum()]).double())
        return ce + synth, bn

    def _round_private(self, r, lr):
        if self.mean_mode == "row":
            return super()._round_private(r, lr)
        self._mean_stats = torch.zeros(7, dtype=torch.float64, device=self.env.dev)
        info = super()._round_private(r, lr)
        stat = self.env.allreduce(self._mean_stats)
        assert int(stat[6]) == int(self.env.allreduce(self.cnt.sum())), "Every real row must contribute once"
        assert int(stat[4]) == 0, "Jensen inequality failed beyond FP32 tolerance"
        n = stat[0].clamp_min(1)
        info.update(mean_kd_steps=int(stat[0]), mean_kd_row_loss=float(stat[1] / n),
                    mean_kd_mean_loss=float(stat[2] / n), mean_kd_jensen_gap=float(stat[3] / n),
                    mean_kd_violations=int(stat[4]), mean_kd_active_steps=int(stat[5]), mean_kd_real_rows=int(stat[6]))
        return info
