"""CBB with client-local first-order L2 SAM; no parameter exchange.

Rho zero delegates exactly. Nonzero reuses rows, synthetic and dropout, commits
only the unperturbed BN/Q pass, and supplies the perturbed gradient to AdamW.
"""
import torch

from core import local
from methods.mobihet_refine import Method as CBB


def perturbation(grad, active, rho):
    finite = torch.isfinite(grad).all(1)
    clean = torch.where((finite & active)[:, None], grad, 0.)
    norm = clean.norm(dim=1, keepdim=True)
    return rho * clean / norm.clamp_min(1e-12), finite


class Method(CBB):
    def __init__(self, env):
        super().__init__(env)
        self.rho = float(env.hp.get("sam_rho", 0.))
        assert self.rho >= 0
        assert self.refine_loss == "forward" and self.real_smoothing == 0 and self.q_agg == "count"
        self._sam_inputs = None

    def _synth(self, ctx):
        if self._sam_inputs is not None:
            return self._sam_inputs
        return super()._synth(ctx)

    def _loss_stream(self, ctx):
        if not self.rho:
            return super()._loss_stream(ctx)
        self._sam_inputs = super()._synth(ctx)
        self._sam_cpu_rng = torch.get_rng_state()
        self._sam_cuda_rng = torch.cuda.get_rng_state(ctx.x.device) if ctx.x.is_cuda else None
        loss, bn = super()._loss_stream(ctx)
        self._sam_loss = loss.detach()
        return loss, bn

    def _sam_gradient(self, ctx, grad):
        epsilon, finite_first = perturbation(grad, ctx.active, self.rho)
        old_p, old_sum = ctx.P, self._csum
        layout = self.env.lay
        shifted = ctx.flat.detach() + epsilon
        # Preserve the parent's parameter strides for the compiled forward.
        perturbed = {k: v.detach().requires_grad_(True) for k, v in layout.params(shifted).items()}
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state(ctx.x.device) if ctx.x.is_cuda else None
        scale = self.env.hp.get("loss_scale", 2. ** 12)
        try:
            torch.set_rng_state(self._sam_cpu_rng)
            if ctx.x.is_cuda:
                torch.cuda.set_rng_state(self._sam_cuda_rng, ctx.x.device)
            ctx.P = perturbed
            # Parent's real-row Q collection has no second-pass side effects.
            self._csum = torch.zeros_like(old_sum)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.env.hp.get("amp", True) and ctx.x.is_cuda):
                loss, _ = super()._loss_stream(ctx)
            masked = torch.where(ctx.active, loss, 0.)
            values = torch.autograd.grad(masked.sum() * scale, tuple(perturbed[k] for k, _ in layout.p))
            second = torch.cat([v.reshape(ctx.Gfix, -1) for v in values], 1) / scale
            # An overflowing first pass must not become an apparently valid update.
            grad.copy_(torch.where(finite_first[:, None], second, float("nan")))
            with torch.no_grad():
                valid = ctx.active & finite_first & torch.isfinite(second).all(1)
                norm = epsilon.norm(dim=1)
                nonzero = ctx.active & finite_first & (epsilon.abs().sum(1) > 0)
                err = torch.where(nonzero, (norm - self.rho).abs(), 0.).max()
                self._sam_error = torch.maximum(self._sam_error, err)
                self._sam_stats += torch.stack([
                    ctx.active.sum(), (ctx.active & ~finite_first).sum(),
                    (ctx.active & ~torch.isfinite(second).all(1)).sum(),
                    torch.where(valid, self._sam_loss, 0.).sum(),
                    torch.where(valid, loss.detach(), 0.).sum(), valid.sum(),
                    (ctx.mask.sum(1) * ctx.active).sum()]).double()
        finally:
            ctx.P, self._csum = old_p, old_sum
            self._sam_inputs = None
            torch.set_rng_state(cpu_after)
            if ctx.x.is_cuda:
                torch.cuda.set_rng_state(cuda_after, ctx.x.device)

    def _round_private(self, r, lr):
        if not self.rho:
            return super()._round_private(r, lr)
        env, epoch = self.env, local.local_epoch
        self._sam_stats = torch.zeros(7, dtype=torch.float64, device=env.dev)
        self._sam_error = torch.zeros((), device=env.dev)

        def sam_epoch(*args, **kwargs):
            assert kwargs.get("grad_fn") is None
            return epoch(*args, **kwargs, grad_fn=self._sam_gradient)

        local.local_epoch = sam_epoch
        try:
            info = super()._round_private(r, lr)
        finally:
            local.local_epoch = epoch
            self._sam_inputs = None
        stats = env.allreduce(self._sam_stats)
        error_bound = env.allreduce(self._sam_error)
        assert int(stats[6]) == int(env.allreduce(self.cnt.sum())), "SAM must cover every real row once"
        info.update(sam_active_steps=int(stats[0]), sam_nonfinite_first=int(stats[1]),
                    sam_nonfinite_second=int(stats[2]), sam_real_rows=int(stats[6]),
                    sam_loss_first=float(stats[3] / stats[5].clamp_min(1)),
                    sam_loss_second=float(stats[4] / stats[5].clamp_min(1)),
                    sam_perturb_norm_error_bound=float(error_bound))
        return info
