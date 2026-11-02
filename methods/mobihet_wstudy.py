"""W20: decoupled (non-target) KD, synthetic step floor, compact bootstrap downlink.

Each W cell changes ONE component of an audited Q/R recipe; with every W knob at its default the losses are the Q/R ones.
  w_real   "full" | "nckd"   real-row KD term (mixed_input real/margin/strength variants): full KL(Q[y] || p), or only its
                              non-target part (1 - Q[y]_y) * KL(Q^[y] || p^) — the true-class probability is left to CE.
  w_synth  "full" | "nckd"   synthetic-row target on streaming rows: soft KL(Q[y] || p), or hard target-vs-rest plus the same
                              non-target part, so a confused teacher row (Q[13] ~ .27 benign) no longer labels class y rows as y'.
  synth_step_floor  int      (core/local.py) clients with fewer real steps take synthetic-only steps up to this count.
  w_boot   "fp32" | "compact" copula bootstrap downlink: fp32 histograms/Cholesky, or uint16 CDF + fp16 mean/Cholesky.
NCKD is the non-target term of Decoupled KD (Zhao et al., CVPR 2022, Eq. 2-3): KD = TCKD + (1 - p_t) NCKD; here
(1 - p_t) is the teacher's, as in that decomposition, and no TCKD/NCKD reweighting (alpha/beta) is introduced.
Messages, teacher schedule and evaluation are those of the parent recipe; only client losses or the bootstrap codec change.
"""
import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet import Method as Base
from methods.mobihet_next import Method as Live
from methods.mobihet_rstudy import Replay

# Uplink of the per-client copula histograms coded as sparse (delta-index, count) varints: lossless, so the pooled sums and
# training are unchanged. Mean over the 5343 clients of v4 train+val, measured by scripts/measure_bootstrap_codec.py.
SPARSE_HIST_UP_MEAN = 73847


def nckd(target, logits, y):
    """(G, n) per-row (1 - t_y) * KL(t^ || p^) over non-target classes; t^/p^ renormalised without class y.
    Masked logits use -1000 like the reference DKD code, never -inf (fp16/NaN safe)."""
    onehot = F.one_hot(y, target.shape[-1]).bool()
    t_y = target.gather(-1, y[..., None]).squeeze(-1)
    t_hat = target.masked_fill(onehot, 0.) / (1 - t_y).clamp_min(1e-6)[..., None]
    log_p_hat = F.log_softmax(logits.float() - 1000. * onehot, -1)
    kl = torch.where(onehot, 0., t_hat * (t_hat.clamp_min(1e-6).log() - log_p_hat)).sum(-1)
    return (1 - t_y) * kl


def hard_plus_nckd(target, logits, y):
    """Synthetic target with the target-vs-rest part made hard: -log p_y + (1 - t_y) NCKD."""
    return -F.log_softmax(logits.float(), -1).gather(-1, y[..., None]).squeeze(-1) + nckd(target, logits, y)


def active(ctx):
    """Clients stepping now; ctx built outside local_epoch (shadow diagnostics) has no step floor."""
    return ctx.active if hasattr(ctx, "active") else ctx.mask.any(1)


class WMixin:
    def _w_knobs(self, env):
        """Before the parent __init__: the copula bootstrap runs inside it."""
        hp = env.hp
        self.w_real, self.w_synth, self.w_boot = hp.get("w_real", "full"), hp.get("w_synth", "full"), hp.get("w_boot", "fp32")
        assert self.w_real in {"full", "nckd"} and self.w_synth in {"full", "nckd"} and self.w_boot in {"fp32", "compact"}

    def _w_check(self):
        if self.variant in {"stream", "query"}:   # _loss_stream reproduces Base only for the S0 hp
            assert not self.real_kd and self.kd_support == "off" and self.synth_target == "soft" and self.kd_bins == 1
        assert self.w_real == "full" or self.variant in {"real", "margin", "strength"}
        assert self.w_synth == "full" or self.variant == "stream"

    def _copula_bootstrap(self, client, NB=256):
        super()._copula_bootstrap(client, NB)
        if self.w_boot == "fp32":
            return
        C, D = self.cop_cdf.shape[:2]
        # Downlink as clients decode it: CDF on a uint16 grid (mass = its differences) and fp16 normal-score mean/Cholesky.
        # The copula covariance was fit on fp32 scores at the server; clients sample through the decoded tables.
        cdf = torch.round(self.cop_cdf.double() * 65535) / 65535
        cdf[..., -1] = torch.where(self.cop_cdf[..., -1] > 0, 1., 0.).double()
        self.cop_cdf = cdf.float().contiguous()
        self.cop_mass = torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])).float()
        self.cop_mu, self.cop_L = self.cop_mu.half().float(), self.cop_L.half().float()
        tri = D * (D + 1) // 2
        # counts for LA/prior both ways; no Gaussian moments (unused on the copula path)
        self.bootstrap_bytes = {"up": 4 * C + SPARSE_HIST_UP_MEAN + 4 * C * (D + tri),
                                "down": 4 * C + 2 * C * D * NB + 2 * C * (D + tri)}

    def _loss_private(self, ctx):
        if self.variant in {"stream", "query"}:
            return self._loss_stream(ctx)
        G, b = ctx.Gfix, ctx.x.shape[1]
        x, labels = self._synth(ctx)
        mask = torch.cat([ctx.mask, active(ctx)[:, None].expand(G, 64)], 1)
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, x], 1), self.env.cfg, mask=mask, train=True)
        ce = local.ce_loss(z[:, :b] + self._shift[:, None], ctx.y, ctx.mask)
        lsp = F.log_softmax(z, -1)
        with torch.no_grad():
            probability = torch.where(ctx.mask[..., None], lsp[:, :b].detach().exp(), 0.)
            self._csum.index_put_((torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y), probability, accumulate=True)
        if self.Q is None:
            return ce + F.cross_entropy(z[:, b:].reshape(-1, 16), labels.reshape(-1), reduction="none").view(G, 64).mean(1), new_b
        target = self.Q[labels]
        if self.variant != "mixed_class":
            target = torch.cat([target[:, :32], self.bank_target[self.bank_index]], 1)
        lt = target.clamp_min(1e-6).log()
        synthetic = (lt.exp() * (lt - lsp[:, b:])).sum(-1).mean(1)
        real = torch.zeros_like(ce)
        if self.variant in {"real", "margin", "strength"}:
            if self.w_real == "nckd":
                kr = nckd(self.Q[ctx.y], z[:, :b], ctx.y)
            else:
                qr = self.Q[ctx.y].clamp_min(1e-6).log()
                kr = (qr.exp() * (qr - lsp[:, :b])).sum(-1)
            coefficient = self.receive_weight[ctx.clients].gather(1, ctx.y)
            real = local.masked_mean(coefficient * kr, ctx.mask)
        return ce + synthetic + real, new_b

    def _loss_stream(self, ctx):
        """Base._loss_private for the S0 hp (arm none, online Q, soft synthetic target, no real KD, fixed weight 1)."""
        cfg = self.env.cfg
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        xp = torch.cat([ctx.x, xg], 1)
        mp = torch.cat([ctx.mask, active(ctx)[:, None].expand(G, a)], 1)
        zp, nBp = grouped.TRAIN(ctx.P, ctx.B, xp, cfg, mask=mp, train=True)
        y = ctx.y
        ce = local.masked_mean(F.cross_entropy((zp[:, :b] + self._shift[:, None, :]).reshape(-1, 16), y.reshape(-1),
                                               reduction="none").view(G, b), ctx.mask)
        lsp = F.log_softmax(zp, -1)
        with torch.no_grad():
            pr = torch.where(ctx.mask[..., None], lsp[:, :b].detach().exp(), 0.0)
            self._csum.index_put_((torch.arange(G, device=y.device)[:, None].expand_as(y), y), pr, accumulate=True)
        if self.Q is None:
            aux_ce = F.cross_entropy(zp[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a).mean(1)
            return ce + aux_ce, nBp
        lt_g = self.Q.clamp_min(1e-6).log()[lab]
        if self.w_synth == "nckd":
            gkd = hard_plus_nckd(lt_g.exp(), zp[:, b:], lab).mean(1)
        else:
            gkd = (lt_g.exp() * (lt_g - lsp[:, b:])).sum(-1).mean(1)
        return ce + gkd, nBp


class WLive(WMixin, Live):
    def __init__(self, env):
        self._w_knobs(env)
        super().__init__(env)
        self._w_check()


class WReplay(WMixin, Replay):
    def __init__(self, env):
        self._w_knobs(env)
        super().__init__(env)
        self._w_check()


def Method(env):
    assert env.hp["r_kind"] in {"live", "replay"}
    return WReplay(env) if env.hp["r_kind"] == "replay" else WLive(env)
