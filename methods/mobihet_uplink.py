"""CBB b14: same-class synthetic interpolation and model-independent log-score uplink.

GP: log of each client's fp16 mean probability; LG: fp16 mean centered log probability.
Cells with fewer than eight rows keep the existing Q8 mean-probability representation.
"""
import os
from pathlib import Path
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_refine import Method as CBB
from methods.mobihet_wstudy import active


def encode(raw):
    compressed = zlib.compress(raw, 6)
    return b"\1" + compressed if len(compressed) < len(raw) else b"\0" + raw


def packet(counts, values, small):
    present, eligible = counts > 0, counts >= 8
    mask = lambda v: sum(int(x) << j for j, x in enumerate(v)).to_bytes(2, "little")
    raw = values[eligible].astype("<f2").tobytes() + small[present & ~eligible].tobytes()
    return bytes(32) + mask(present) + mask(eligible) + encode(raw)


class Method(CBB):
    def __init__(self, env):
        super().__init__(env)
        self.uplink = env.hp.get("uplink_mode", "prob")
        self.mix = float(env.hp.get("same_class_mix", 0.))
        self.wire_audit = bool(env.hp.get("uplink_audit", False))
        assert self.uplink in {"prob", "gp16", "lg16"} and 0 <= self.mix <= 1
        assert self.refine_loss == "forward" and self.real_smoothing == 0 and self.q_agg == "count"
        self._logsum = None

    def _synth(self, ctx):
        x, lab = super()._synth(ctx)
        if not self.mix:
            return x, lab
        match = (lab[..., None] == ctx.y[:, None]) & ctx.mask[:, None]
        scores = torch.rand(match.shape, device=x.device, generator=self.gen_g).masked_fill(~match, 2.)
        idx = scores.argmin(-1)
        real = ctx.x.gather(1, idx[..., None].expand(-1, -1, ctx.x.shape[-1]))
        blended = (1 - self.mix) * x + self.mix * real
        return torch.where(match.any(-1)[..., None], blended, x), lab

    def _loss_stream(self, ctx):
        if self.uplink != "lg16":
            return super()._loss_stream(ctx)
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        mask = torch.cat([ctx.mask, active(ctx)[:, None].expand(G, a)], 1)
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), self.env.cfg, mask=mask, train=True)
        ce = local.ce_loss(z[:, :b] + self._shift[:, None], ctx.y, ctx.mask)
        lp = F.log_softmax(z, -1)
        with torch.no_grad():
            pr = torch.where(ctx.mask[..., None], lp[:, :b].detach().exp(), 0.)
            ix = (torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y)
            self._csum.index_put_(ix, pr, accumulate=True)
            if self._logsum is None:
                self._logsum = torch.zeros_like(self._csum)
            centered = lp[:, :b].detach().float()
            centered = centered - centered.mean(-1, keepdim=True)
            self._logsum.index_put_(ix, torch.where(ctx.mask[..., None], centered, 0.), accumulate=True)
        if self.Q is None:
            loss = F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a).mean(1)
        else:
            lq = self.Q.clamp_min(1e-6).log()[lab]
            loss = (lq.exp() * (lq - lp[:, b:])).sum(-1).mean(1)
        return ce + loss, new_b

    def _round_private(self, r, lr):
        if self.uplink == "prob":
            return super()._round_private(r, lr)
        env, epoch = self.env, local.local_epoch
        accum = torch.zeros(16, 16, dtype=torch.float64, device=env.dev)
        cost = torch.zeros(2, dtype=torch.float64, device=env.dev)
        records = []

        def collect(lay, flat, bufs, data, idx, *args, **kwargs):
            self._logsum = None
            st = epoch(lay, flat, bufs, data, idx, *args, **kwargs)
            G, count = len(idx), self.cnt[idx]
            mean = self._csum[:G] / count[..., None].clamp_min(1)
            values = (self._logsum[:G] / count[..., None].clamp_min(1) if self.uplink == "lg16" else mean).half()
            eligible, present = count >= 8, count > 0
            values = torch.where(eligible[..., None], values, 0.)
            small = torch.where((present & ~eligible)[..., None], (mean.clamp(0, 1) * 255).round(), 0.).byte()
            fallback = (small.float() / 255).clamp_min(1e-6).log()
            fallback = fallback - fallback.mean(-1, keepdim=True)
            decoded = values.float() if self.uplink == "lg16" else values.float().clamp_min(1e-6).log()
            if self.uplink == "gp16":
                decoded = decoded - decoded.mean(-1, keepdim=True)
            decoded = torch.where(eligible[..., None], decoded, fallback)
            accum.add_((decoded.double() * count[..., None].double()).sum(0))
            nc, vc, sc = count.long().cpu().numpy(), values.cpu().numpy(), small.cpu().numpy()
            frames = [packet(nc[k], vc[k], sc[k]) for k in range(G)]
            cost.add_(torch.tensor([sum(map(len, frames)), G], dtype=torch.float64, device=env.dev))
            if self.wire_audit:
                records.append((env.ids[idx.cpu().numpy()], nc, frames))
            return st

        # Parent still runs its Q8 arithmetic simulator path; its temporary teacher
        # and byte estimates are replaced below by the actual GP/LG packet aggregate.
        local.local_epoch = collect
        try:
            info = super()._round_private(r, lr)
        finally:
            local.local_epoch = epoch
        count = env.allreduce(self.cnt.sum(0).double())
        total = env.allreduce(accum)
        prob = (total / count[:, None].clamp_min(1)).float().softmax(-1)
        codes = (prob * 255).round().byte()
        self.Q = codes.float() / codes.float().sum(-1, keepdim=True).clamp_min(1)
        down = bytes(32) + encode(codes.cpu().numpy().tobytes())
        cost = env.allreduce(cost)
        info.update(bytes_up=float(cost[0] / cost[1].clamp_min(1)), bytes_down=len(down))
        if self.wire_audit:
            folder = Path(os.environ["MOBIHET_RECOVERY_DIR"]).parent / "wire"
            folder.mkdir(exist_ok=True, parents=True)
            frames = [f for _, _, group in records for f in group]
            np.savez_compressed(folder / f"rank{env.rank}_r{r:03d}.npz",
                                ids=np.concatenate([x[0] for x in records]), counts=np.concatenate([x[1] for x in records]),
                                frames=np.frombuffer(b"".join(frames), np.uint8), offsets=np.r_[0, np.cumsum(list(map(len, frames)))],
                                global_sum=total.cpu().numpy(), global_count=count.cpu().numpy(),
                                down=np.frombuffer(down, np.uint8), Q=self.Q.cpu().numpy())
        return info
