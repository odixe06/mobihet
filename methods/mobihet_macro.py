"""Macro-first CBB adaptations: aligned prior coordinate or real-row non-target KD.

Only existing class counts/probabilities are exchanged. Real rows stay local.
Defaults delegate exactly; each non-default arm changes one mechanism.
"""
import math

import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_refine import Method as CBB
from methods.mobihet_wstudy import active, nckd


class Method(CBB):
    def __init__(self, env):
        super().__init__(env)
        self.macro_mode = env.hp.get("macro_mode", "cbb")
        assert self.macro_mode in {"cbb", "prior", "real_nckd"}
        assert self.refine_loss == "forward" and self.real_smoothing == 0 and self.q_agg == "count"
        assert self.aux == 64 and not self.cf and self.synth == "copula"
        self.prior_shift = torch.zeros(16, device=env.dev)
        self._real_keep = torch.zeros(16, dtype=torch.bool, device=env.dev)
        if self.macro_mode == "prior":
            # Counts already belong to the bootstrap schema. This repeats their
            # reduction across simulator ranks, not a new client payload/query.
            total = env.allreduce(self.cnt.sum(0).double())
            self.prior_shift = (-math.log(16) - ((total + 1) / (total.sum() + 16)).log()).float()

    def _loss_stream(self, ctx):
        if self.macro_mode == "cbb":
            return super()._loss_stream(ctx)
        G, b = ctx.Gfix, ctx.x.shape[1]
        enabled = active(ctx)
        with torch.no_grad():
            self._macro_stats[0].add_(enabled.sum())
            self._macro_stats[1].add_((ctx.mask.sum(1) * enabled).sum())
        if self.macro_mode == "real_nckd" and self.Q is None:
            return super()._loss_stream(ctx)
        xg, lab = self._synth(ctx)
        mask = torch.cat([ctx.mask, enabled[:, None].expand(G, 64)], 1)
        z, bn = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), self.env.cfg, mask=mask, train=True)
        ce = local.ce_loss(z[:, :b] + self._shift[:, None], ctx.y, ctx.mask)
        lp = F.log_softmax(z, -1)
        with torch.no_grad():
            prob = F.softmax(z[:, :b].detach().float() + self.prior_shift, -1) if self.macro_mode == "prior" else lp[:, :b].detach().exp()
            prob = torch.where(ctx.mask[..., None], prob, 0.)
            self._csum.index_put_((torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y),
                                  prob, accumulate=True)
        synth_z = z[:, b:] + self.prior_shift if self.macro_mode == "prior" else z[:, b:]
        if self.Q is None:
            synth = F.cross_entropy(synth_z.reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, 64).mean(1)
        else:
            lq = self.Q.clamp_min(1e-6).log()[lab]
            synth_lp = F.log_softmax(synth_z, -1) if self.macro_mode == "prior" else lp[:, b:]
            synth = (lq.exp() * (lq - synth_lp)).sum(-1).mean(1)
        if self.macro_mode == "real_nckd":
            kd = nckd(self.Q.detach()[ctx.y], z[:, :b], ctx.y) * self._real_keep[ctx.y]
            ce = ce + local.masked_mean(kd, ctx.mask)
            with torch.no_grad():
                self._macro_stats[2].add_((ctx.mask.sum(1) * enabled).sum())
                self._macro_stats[3].add_(torch.where(ctx.mask & enabled[:, None], kd, 0.).sum())
                self._macro_stats[4].add_((ctx.mask & enabled[:, None] & self._real_keep[ctx.y]).sum())
        return ce + synth, bn

    def _round_private(self, r, lr):
        if self.macro_mode == "cbb":
            return super()._round_private(r, lr)
        self._macro_stats = torch.zeros(5, dtype=torch.float64, device=self.env.dev)
        self._real_keep.zero_()
        if self.macro_mode == "real_nckd" and self.Q is not None:
            # Preserve W20's binary validity gate; no margin/quality tuning.
            self._real_keep = self.Q.argmax(1) == torch.arange(16, device=self.env.dev)
        info = super()._round_private(r, lr)
        stats = self.env.allreduce(self._macro_stats)
        assert int(stats[1]) == int(self.env.allreduce(self.cnt.sum())), "Every real row must contribute once"
        info.update(macro_active_steps=int(stats[0]), macro_real_rows=int(stats[1]),
                    macro_real_kd_rows=int(stats[2]), macro_real_kd_loss=float(stats[3] / stats[2].clamp_min(1)),
                    macro_real_kd_kept_rows=int(stats[4]), macro_real_kd_keep=self._real_keep.cpu().tolist(),
                    macro_prior_shift=self.prior_shift.detach().cpu().tolist())
        return info
