"""Fixed row budget on Lite2 (rebuild 17/11/2026, docs/research/screening-budget.md); every knob off = mobihet_lite bit-exact.

A vehicle's step keeps its count and its 64-row slot; part of the slot's real rows is not trained on that round, so the
synthetic rows replace real-row work instead of adding to it (client FLOPs ~ rows through the model).

  real_tier_rows list[int] ([] = off)  one entry per synthetic tier (synth_tier_min, needed): a vehicle of tier t trains on at
      most real_tier_rows[t] real rows per step: the first rows of the slot, which core.local's per-round permutation makes
      a fresh random subset every round. The other real rows of the step get no forward, no BN statistics, no CE and no
      class-Q contribution that round. Its class-Q packet is the mean over the rows it trained on (weight n_kc as before;
      a class with no trained row that round sends zeros, which only scales the pooled row before its normalisation).
  synth_off  bool (false)  reference: no synthetic row is trained on (bootstrap still runs; use synth_step_floor=0).
  q_real     bool (false)  reference: + KL(Q_prev[y] || softmax(z)) on the trained real rows (class-Q distillation without inputs).

Round info gains real_frac = trained real rows / real rows (measured) when a knob is on.
"""
import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_lite import Method as Lite
from methods.mobihet_wstudy import active


class Method(Lite):
    def __init__(self, env):
        super().__init__(env)
        hp = env.hp
        rows = list(hp.get("real_tier_rows", []))
        self.synth_off, self.q_real = bool(hp.get("synth_off", False)), bool(hp.get("q_real", False))
        self.budget = bool(rows) or self.synth_off or self.q_real
        if not self.budget:
            return
        assert self.tier_min, "row budget is defined on the synthetic tiers"
        rows = rows or [64] * (len(self.tier_min) + 1)
        assert len(rows) == len(self.tier_min) + 1 and all(isinstance(v, int) and 0 < v <= 64 for v in rows), rows
        assert not self.synth_off or not hp.get("synth_step_floor", 0), "a floor step without synthetic rows has no loss"
        tier = torch.bucketize(torch.as_tensor(env.train.sizes, device=env.dev),
                               torch.tensor(self.tier_min, device=env.dev), right=True)
        self._real_k = torch.tensor(rows, device=env.dev)[tier]          # (K,) real rows per step

    def _round_private(self, r, lr):
        if not self.budget:
            return super()._round_private(r, lr)
        env, epoch = self.env, local.local_epoch
        seen = torch.zeros(2, dtype=torch.float64, device=env.dev)

        def scaled(lay, flat, bufs, data, idx, *a, **k):
            self._seen = torch.zeros_like(self._csum[:, :, 0])
            stat = epoch(lay, flat, bufs, data, idx, *a, **k)
            G = len(idx)
            cnt = self.cnt[idx]
            self._csum[:G] *= (cnt / self._seen[:G].clamp_min(1))[..., None]   # sum over trained rows -> mean * n_kc
            seen.add_(torch.stack([self._seen[:G].sum(), cnt.sum()]).double())
            return stat

        local.local_epoch = scaled                                       # innermost: the Q codec wrappers see the rescaled sums
        try:
            info = super()._round_private(r, lr)
        finally:
            local.local_epoch = epoch
        seen = env.allreduce(seen)
        info["real_frac"] = float(seen[0] / seen[1])
        return info

    def _loss_tier(self, ctx):
        if not self.budget:
            return super()._loss_tier(ctx)
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        rows = self._rows_k[ctx.clients][:, None]
        pos = torch.arange(a, device=lab.device)[None]
        first = (rows == 8) * (8 * (ctx.step % 2)) if bool((self._rows_k == 8).any()) else 0
        keep = active(ctx)[:, None] & (pos >= first) & (pos < first + rows) & (not self.synth_off)
        real = ctx.mask & (torch.arange(b, device=lab.device)[None] < self._real_k[ctx.clients][:, None])
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), self.env.cfg,
                                 mask=torch.cat([real, keep], 1), train=True)
        ce = local.masked_mean(F.cross_entropy((z[:, :b] + self._shift[:, None, :]).reshape(-1, 16), ctx.y.reshape(-1),
                                               reduction="none").view(G, b), real)
        logp = F.log_softmax(z, -1)
        with torch.no_grad():
            at = (torch.arange(G, device=lab.device)[:, None].expand_as(ctx.y), ctx.y)
            self._csum.index_put_(at, torch.where(real[..., None], logp[:, :b].detach().exp(), 0.0), accumulate=True)
            self._seen.index_put_(at, real.float(), accumulate=True)
        if self.Q is None:
            return ce + local.masked_mean(F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1),
                                                          reduction="none").view(G, a), keep), new_b
        lq = self.Q.clamp_min(1e-6).log()
        loss = ce + local.masked_mean((lq[lab].exp() * (lq[lab] - logp[:, b:])).sum(-1), keep)
        if self.q_real:
            loss = loss + local.masked_mean((lq[ctx.y].exp() * (lq[ctx.y] - logp[:, :b])).sum(-1), real)
        return loss, new_b
