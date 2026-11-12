"""CBB module screening: class-Q aggregation, synthetic divergence, real-label smoothing.

All defaults delegate to the frozen CBB path. Only probabilities/counts leave clients;
no parameter, embedding or BN alignment is required between client architectures.
"""
import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_sim import Method as CBB
from methods.mobihet_wstudy import active


def divergence(logp, target, kind):
    """Per-row synthetic loss. js4 has forward-KL curvature near p = target."""
    logq = target.clamp_min(1e-6).log()
    if kind == "forward":
        return (logq.exp() * (logq - logp)).sum(-1)
    q = target.clamp_min(1e-6)
    q = q / q.sum(-1, keepdim=True)
    logq, p = q.log(), logp.exp()
    reverse = (p * (logp - logq)).sum(-1)
    if kind == "reverse":
        return reverse
    if kind == "symmetric":
        return .5 * ((q * (logq - logp)).sum(-1) + reverse)
    assert kind == "js4", kind
    logm = ((p + q) * .5).clamp_min(1e-12).log()
    return 2 * ((q * (logq - logm)).sum(-1) + (p * (logp - logm)).sum(-1))


class Method(CBB):
    def __init__(self, env):
        super().__init__(env)
        self.refine_loss = env.hp.get("refine_loss", "forward")
        self.real_smoothing = float(env.hp.get("real_smoothing", 0.0))
        agg = env.hp.get("refine_agg", "count")
        assert self.refine_loss in {"forward", "reverse", "symmetric", "js4"}
        assert 0 <= self.real_smoothing < 1 and agg in {"count", "sqrt_count"}
        assert self.variant == "stream" and self.q_src == "real" and self.q_bits == 8
        assert not self.qk and not self.need_T and self.sw == 1 and self.q_temp == 1
        # Live's initialization locks the historical recipe to count. Its inherited
        # Base aggregation already implements sqrt weights AND the matching divisor.
        self.q_agg = agg

    def _round_private(self, r, lr):
        # Accumulate on device; verify every real client took an optimizer step.
        env, epoch = self.env, local.local_epoch
        missing = torch.zeros((), device=env.dev)

        def checked(*args, **kwargs):
            stat = epoch(*args, **kwargs)
            missing.add_((stat["applied"] <= 0).sum())
            return stat

        local.local_epoch = checked
        try:
            info = super()._round_private(r, lr)
        finally:
            local.local_epoch = epoch
        info["clients_without_applied"] = int(env.allreduce(missing))
        assert info["clients_without_applied"] == 0, "Client received no optimizer update"
        return info

    def _loss_stream(self, ctx):
        if self.refine_loss == "forward" and self.real_smoothing == 0:
            return super()._loss_stream(ctx)
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        mask = torch.cat([ctx.mask, active(ctx)[:, None].expand(G, a)], 1)
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), self.env.cfg, mask=mask, train=True)
        ce = F.cross_entropy((z[:, :b] + self._shift[:, None]).reshape(-1, 16), ctx.y.reshape(-1),
                             reduction="none", label_smoothing=self.real_smoothing).view(G, b)
        ce = local.masked_mean(ce, ctx.mask)
        logp = F.log_softmax(z, -1)
        with torch.no_grad():
            prob = torch.where(ctx.mask[..., None], logp[:, :b].detach().exp(), 0.)
            self._csum.index_put_((torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y), prob, accumulate=True)
        if self.Q is None:
            synth = F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a)
        else:
            synth = divergence(logp[:, b:], self.Q[lab], self.refine_loss)
        return ce + synth.mean(1), new_b
