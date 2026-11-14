"""Client-compute screen on CBB (B20): cheaper synthetic gradient path or fewer late synthetic rows.

Every knob off delegates to the frozen CBB stream loss (mobihet_refine defaults). Exchange, bytes, Q and the real-row
pass are unchanged; only how the vehicle spends FLOPs on its own synthetic rows changes. Design: docs/research/screening-compute-b20.md.

  synth_grad  "full" (CBB) | "head" | "mix16"
      head : real rows forward+backward through the whole model (BN batch statistics and BN buffer update from the
             real rows only). Synthetic rows: backbone forward under no_grad (BN batch statistics of the synthetic rows,
             buffers discarded), head (LayerNorm -> Linear -> ReLU -> Linear, dropout as in training) with gradient.
             Same KL(Q[y] || p) per row, same mean over the 64 rows. Client FLOPs ~4,16 x F per real row (CBB 6,09).
      mix16: the first 16 synthetic rows (one per class) go through the full model with the real rows exactly like
             CBB (shared BN batch); the other 48 take the head-only path. ~4,6 x F.
  synth_rows_late int (0 = off), synth_late_from int: synthetic rows per step = synth_rows_late once round > late_from
      (balanced labels, same loss weight: mean over rows). SD in the design; ~5,3 x F at r50 with 32 from r26.
  synth_tier_min list[int] ([] = off), synth_tier_rows list[[early, late]] (B21, docs/research/screening-dose-b21.md):
      vehicles split into len(synth_tier_min) + 1 tiers by their own train-row count (ascending thresholds, n >= threshold
      -> next tier); tier t trains on synth_tier_rows[t][0] synthetic rows per step, [1] once round > synth_late_from.
      The step generates the largest row count of the round; a vehicle with fewer rows masks the rest out of its BN batch
      and of its synthetic mean (its first rows stay class-balanced), so the masked rows cost nothing on a real vehicle.

A vehicle with no real row in a step (step floor, padding) normalises its masked real pass with all row slots, so the
masked CE stays finite; its BN buffers are not updated on that step (core.local.local_epoch: has_real only).
"""
import torch
import torch.nn.functional as F

from core import grouped
from methods.mobihet_refine import Method as Refine
from methods.mobihet_wstudy import active
from core import local

_FEAT = None


def _features(P, B, x, cfg, mask):
    """Backbone (Eq. 47) in training mode: BN batch statistics over `mask` rows; returned BN buffers are discarded."""
    return grouped._combined(grouped._Ctx(P, B, mask, True), cfg, x)


def features(P, B, x, cfg, mask):
    global _FEAT
    if _FEAT is None:
        _FEAT = _features if grouped.TRAIN is grouped.forward else torch.compile(_features, dynamic=False)
    return _FEAT(P, B, x, cfg, mask)


def head(P, z, cfg, dtype, train=True):
    """grouped.forward's head on precomputed backbone features z (G, n, comb)."""
    c = grouped._Ctx(P, None, None, False)
    z = F.layer_norm(z, (z.shape[-1],)) * P["head.0.weight"][:, None] + P["head.0.bias"][:, None]
    z = F.dropout(z, cfg["dropout"], train)
    h = torch.relu(grouped._linear(c, "head.2", z.to(dtype)))
    return grouped._f32(grouped._linear(c, "head.5", F.dropout(h, cfg["dropout"], train)))


class Method(Refine):
    def __init__(self, env):
        super().__init__(env)
        hp = env.hp
        self.synth_grad = hp.get("synth_grad", "full")
        assert self.synth_grad in {"full", "head", "mix16"}, self.synth_grad
        self.rows_late = int(hp.get("synth_rows_late", 0))
        self.late_from = int(hp.get("synth_late_from", 0))
        assert self.rows_late == 0 or (self.rows_late % 16 == 0 and 0 < self.rows_late < 64 and self.late_from >= 1)
        assert self.refine_loss == "forward" and self.real_smoothing == 0 and self.w_synth != "nckd"
        assert self.aux == 64 and torch.equal(self.labels, torch.arange(64, device=env.dev) % 16)
        self._rows0 = self.aux
        self.tier_min, rows = list(hp.get("synth_tier_min", [])), hp.get("synth_tier_rows", [])
        if self.tier_min:
            assert self.synth_grad == "full" and self.rows_late == 0 and self.late_from >= 1
            assert self.tier_min == sorted(set(self.tier_min)) and len(rows) == len(self.tier_min) + 1
            assert all(len(x) == 2 and all(v % 16 == 0 and 0 < v <= 64 for v in x) for x in rows)
            tier = torch.bucketize(torch.as_tensor(env.train.sizes, device=env.dev),
                                   torch.tensor(self.tier_min, device=env.dev), right=True)
            self._tier_rows = torch.tensor(rows, device=env.dev)[tier]   # (K, 2) rows per step early / late

    def _rows_for(self, r):
        if self.tier_min:
            self._rows_k = self._tier_rows[:, int(r > self.late_from)]
            return int(self._rows_k.max())
        return self.rows_late if self.rows_late and r > self.late_from else self._rows0

    def _round_private(self, r, lr):
        a = self._rows_for(r)
        if a != self.aux:
            self.aux, self.labels = a, torch.arange(a, device=self.env.dev) % 16
            self.lab_rows = torch.bincount(self.labels, minlength=16).float()
        info = super()._round_private(r, lr)
        info["synth_rows"] = self.aux
        return info

    def _loss_tier(self, ctx):
        """CBB stream loss with a per-vehicle synthetic row count (rows beyond it: no BN statistics, no loss)."""
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        keep = active(ctx)[:, None] & (torch.arange(a, device=lab.device)[None] < self._rows_k[ctx.clients][:, None])
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), self.env.cfg,
                                 mask=torch.cat([ctx.mask, keep], 1), train=True)
        ce = local.masked_mean(F.cross_entropy((z[:, :b] + self._shift[:, None, :]).reshape(-1, 16), ctx.y.reshape(-1),
                                               reduction="none").view(G, b), ctx.mask)
        logp = F.log_softmax(z, -1)
        with torch.no_grad():
            pr = torch.where(ctx.mask[..., None], logp[:, :b].detach().exp(), 0.0)
            self._csum.index_put_((torch.arange(G, device=pr.device)[:, None].expand_as(ctx.y), ctx.y), pr,
                                  accumulate=True)
        if self.Q is None:
            synth = F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a)
        else:
            lt = self.Q.clamp_min(1e-6).log()[lab]
            synth = (lt.exp() * (lt - logp[:, b:])).sum(-1)
        return ce + local.masked_mean(synth, keep), new_b

    def _loss_stream(self, ctx):
        if self.tier_min:
            return self._loss_tier(ctx)
        if self.synth_grad == "full":
            return super()._loss_stream(ctx)
        cfg = self.env.cfg
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        nf = 16 if self.synth_grad == "mix16" else 0
        xg, lab = self._synth(ctx)
        act = active(ctx)
        xf = torch.cat([ctx.x, xg[:, :nf]], 1)
        mf = torch.cat([ctx.mask, act[:, None].expand(G, nf)], 1)
        bn = torch.where(mf.any(1)[:, None], mf, torch.ones_like(mf))   # finite statistics for row-less vehicles
        zf, new_b = grouped.TRAIN(ctx.P, ctx.B, xf, cfg, mask=bn, train=True)
        with torch.no_grad():
            feat = features(ctx.P, ctx.B, xg[:, nf:], cfg, torch.ones(G, a - nf, dtype=torch.bool, device=xg.device))
        zh = head(ctx.P, feat, cfg, ctx.x.dtype)
        zs = torch.cat([zf[:, b:], zh], 1)                               # (G, a, 16) synthetic logits, label order kept
        ce = local.masked_mean(F.cross_entropy((zf[:, :b] + self._shift[:, None, :]).reshape(-1, 16),
                                               ctx.y.reshape(-1), reduction="none").view(G, b), ctx.mask)
        with torch.no_grad():
            pr = torch.where(ctx.mask[..., None], F.log_softmax(zf[:, :b], -1).detach().exp(), 0.0)
            self._csum.index_put_((torch.arange(G, device=pr.device)[:, None].expand_as(ctx.y), ctx.y), pr,
                                  accumulate=True)
        if self.Q is None:
            synth = F.cross_entropy(zs.reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a).mean(1)
        else:
            lt = self.Q.clamp_min(1e-6).log()[lab]
            synth = (lt.exp() * (lt - F.log_softmax(zs, -1))).sum(-1).mean(1)
        return ce + synth, new_b
