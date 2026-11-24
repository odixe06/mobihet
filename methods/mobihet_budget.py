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
  real_select "random" (FB) | "top" | "prop"  (FB-L 18/11/2026, docs/research/screening-fbl.md): which real rows of the step's
      slot a capped vehicle trains on. Every real row keeps its CE from the last round it was trained in (never trained: +inf,
      so a row is first in line until trained once). top: the real_k rows with the highest stored CE; prop: sampled without
      replacement with probability ~ stored CE (Gumbel-top-k, own generator so core.local's permutation stream is the FB one).
      Same row count as FB, so the same client compute; the stored CE costs no extra pass. Stale CE overstates a row's
      current loss (the model improved since), so a row left out rises in line by itself.
  real_select_q int (0)  > 0: the first real_select_q rows of the slot stay FB's random subset and are the only rows that
      feed the class-Q packet; the remaining real_k - real_select_q are chosen by real_select from the rest (CE only).

  compact    bool (false)  (18/11/2026, docs/research/compute-harness.md) execute only the rows the step needs: a group's
      grouped step carries the widest real budget and the widest synthetic dose of its own vehicles instead of the full slot
      plus the round's widest dose, and a step past every vehicle's real rows (synthetic floor) carries no real slot. The
      rows dropped were masked out of BN, loss and Q, so the step computes the same function on fewer rows (FB-L: the chosen
      rows are moved to the front first). Synthetic rows are generated as before and then cut, so they are the same rows;
      only dropout draws change with the shape. Executed client compute ~1,00 x FedAvg instead of ~2,07 x (N4).

  la_mode    "full" | "off" | "absent" | "mask"  (24/11/2026, docs/research/screening-la.md) logit adjustment on the real-row
      CE (and the stored CE that ranks rows): full = L4's log pi_local - log pi_global; off = no shift (ablation A3);
      absent = shift 0 on the classes the vehicle has (n_kc > 0), L4's shift on the ones it lacks; mask = shift 0 on the
      classes it has, -1e4 on the ones it lacks (restricted softmax: real rows never push an absent class down).
      Class counts, bootstrap, Q, synthetic rows and bytes are those of L4 in every mode.

Round info gains real_frac = trained real rows / real rows (measured) when a knob is on.
"""
import math

import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_lite import Method as Lite
from methods.mobihet_wstudy import active


class Method(Lite):
    def __init__(self, env):
        super().__init__(env)
        hp = env.hp
        self.la_mode = hp.get("la_mode", "full")
        assert self.la_mode in {"full", "off", "absent", "mask"}, self.la_mode
        if self.la_mode != "full":                                       # la_shift is rebuilt from counts on every start
            lack = {"off": 0.0, "absent": self.la_shift, "mask": -1e4}[self.la_mode]
            self.la_shift = torch.where(self.cnt > 0, 0.0, lack).to(self.la_shift.dtype)
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
        self.compact = bool(hp.get("compact", False))
        if self.compact:
            assert not self.synth_off and not self.q_real, "compact is defined for the synthetic budget arms"
            self._n_host = [int(v) for v in env.train.sizes]
            self._tier_host = tier.tolist()
            self._real_host = rows
        self.select = hp.get("real_select", "random")
        self.select_q = int(hp.get("real_select_q", 0))
        assert self.select in {"random", "top", "prop"} and (self.select != "random" or not self.select_q), self.select
        assert 0 <= self.select_q < min(rows), (self.select_q, rows)
        if self.select != "random":
            n = env.train.X.shape[0]                                     # + one sink slot: untrained positions write there
            self._score = torch.full((n + 1,), float("inf"), device=env.dev)
            self._sgen = torch.Generator(device=env.dev).manual_seed(hp["seed"] * 1000 + getattr(env, "rank", 0) + 7919)

    def state(self):
        s = super().state()
        if self.budget and self.select != "random":
            s["row_score"], s["row_gen"] = self._score.cpu(), self._sgen.get_state().cpu()
        return s

    def load_state(self, s):
        super().load_state(s)
        if self.budget and self.select != "random":
            self._score = s["row_score"].to(self.env.dev)
            self._sgen.set_state(s["row_gen"])

    def _real_rows(self, ctx):
        """(G, b) real rows the vehicles train on this step, and the subset that feeds the class-Q packet."""
        b = ctx.x.shape[1]
        pos = torch.arange(b, device=ctx.x.device)[None]
        k = self._real_k[ctx.clients][:, None]
        if self.select == "random":
            real = ctx.mask & (pos < k)
            return real, real
        q = ctx.mask & (pos < self.select_q)
        pri = self._score[ctx.rows]
        if self.select == "prop":
            u = torch.rand(pri.shape, device=pri.device, generator=self._sgen).clamp_(1e-12, 1 - 1e-7)
            pri = pri.clamp_min(1e-12).log() - (-u.log()).log()
        pri = torch.where(ctx.mask & ~q, pri, -float("inf"))
        rank = pri.argsort(dim=1, descending=True, stable=True).argsort(dim=1)
        real = q | (ctx.mask & ~q & (rank < k - self.select_q))
        return real, (q if self.select_q else real)

    def _round_private(self, r, lr):
        if not self.budget:
            return super()._round_private(r, lr)
        env, epoch = self.env, local.local_epoch
        seen = torch.zeros(2, dtype=torch.float64, device=env.dev)

        def scaled(lay, flat, bufs, data, idx, *a, **k):
            if self.compact:                                             # host widths of this group, once per group
                ids, synth_host = idx.tolist(), self._rows_k.tolist()      # this round's dose (set by _rows_for)
                b = local.slot(max(self._n_host[c] for c in ids), env.hp["batch"])
                self._real_steps = math.ceil(max(self._n_host[c] for c in ids) / b)
                self._wr = max(min(b, self._real_host[self._tier_host[c]]) for c in ids)
                self._ws = max(16 if synth_host[c] == 8 else synth_host[c] for c in ids)
            self._seen = torch.zeros_like(self._csum[:, :, 0])
            self._trained = torch.zeros((), device=self._csum.device)
            stat = epoch(lay, flat, bufs, data, idx, *a, **k)
            G = len(idx)
            cnt = self.cnt[idx]
            self._csum[:G] *= (cnt / self._seen[:G].clamp_min(1))[..., None]   # sum over trained rows -> mean * n_kc
            seen.add_(torch.stack([self._trained, cnt.sum()]).double())
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
        real, qrow = self._real_rows(ctx)
        x, y, idx = ctx.x, ctx.y, (ctx.rows if self.select != "random" else None)
        if self.compact:
            wr = 0 if ctx.step >= self._real_steps else self._wr
            if self.select != "random":                                  # chosen rows first, slot order kept among them
                order = (~real).to(torch.uint8).argsort(dim=1, stable=True)
                x, y, idx = x.gather(1, order[..., None].expand_as(x)), y.gather(1, order), idx.gather(1, order)
                real, qrow = real.gather(1, order), qrow.gather(1, order)
                idx = idx[:, :wr]
            x, y, real, qrow = x[:, :wr], y[:, :wr], real[:, :wr], qrow[:, :wr]
            xg, lab, keep, a = xg[:, :self._ws], lab[:, :self._ws], keep[:, :self._ws], self._ws
            b = wr
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([x, xg], 1), self.env.cfg,
                                 mask=torch.cat([real, keep], 1), train=True)
        row_ce = F.cross_entropy((z[:, :b] + self._shift[:, None, :]).reshape(-1, 16), y.reshape(-1),
                                 reduction="none").view(G, b)
        ce = local.masked_mean(row_ce, real)
        logp = F.log_softmax(z, -1)
        with torch.no_grad():
            at = (torch.arange(G, device=lab.device)[:, None].expand_as(y), y)
            self._csum.index_put_(at, torch.where(qrow[..., None], logp[:, :b].detach().exp(), 0.0), accumulate=True)
            self._seen.index_put_(at, qrow.float(), accumulate=True)
            if hasattr(self, "_trained"):                                # unset when a test calls the loss alone
                self._trained += real.sum()
            if self.select != "random":
                sink = self._score.numel() - 1                           # no boolean index: no host sync per step
                self._score.index_put_((torch.where(real, idx, sink).reshape(-1),),
                                       row_ce.detach().float().reshape(-1))
        if self.Q is None:
            return ce + local.masked_mean(F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1),
                                                          reduction="none").view(G, a), keep), new_b
        lq = self.Q.clamp_min(1e-6).log()
        loss = ce + local.masked_mean((lq[lab].exp() * (lq[lab] - logp[:, b:])).sum(-1), keep)
        if self.q_real:
            loss = loss + local.masked_mean((lq[y].exp() * (lq[y] - logp[:, :b])).sum(-1), real)
        return loss, new_b
