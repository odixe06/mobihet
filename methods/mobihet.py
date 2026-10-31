"""MobiHet — owner's method (v1 arm fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl, history/v1/docs/methods/full-drift/design.md).

Every vehicle keeps a PRIVATE model that never leaves it (any architecture in principle; DAGSNet here) and trains a shared
PROXY that is the only thing exchanged, as a 3-bit stochastic-rounded delta + LZMA2 (q3_zl). Per local step on real rows
(x, y) and Gaussian rows (x_g, y_g) drawn from pooled class moments (gpool):

    private:  CE(z_p + log π_k − log π_g, y)                              la: logit-adjusted CE toward the global prior
            + w · KL(T(x) ‖ z_p(x))                                       frz: T = proxy snapshot received this round (frozen)
            + w · KL(T(x_g) ‖ z_p(x_g))       (round 1: CE(z_p(x_g), y_g))  gkd
    proxy:    CE(z_r, y) + min(w, 1) · KL(z_p ‖ z_r)                       kdpx: proxy <- private weight capped at 1
    w = 1 / (CE_private + CE_proxy)   (detached; DML adaptive weight)

KD on real rows, no KD in round 1 (no snapshot yet). Server: exact proxy kept, θ ← θ + mean_k Q3(Δ_k) (unweighted, as in
the original); downlink Q3(exact − replica), nearest rounding; BN buffers fp16, averaged. m4b (zeroing simulator noise and
absolute receiver/sender positions) is the identity on v4: the dataset has none of those columns.

One-time bootstrap (gpool + la): each client sends per-class counts, Σx, Σxxᵀ (77-d); the server returns the pooled class
means, Cholesky factors and the global prior. Deviations from the original: shared protocol (batch 64, wd 1e-4, cosine
1e-3 → 1e-5; original wd 0, batch 512), Gaussian rows per step = batch / 2 (original 256 per 512 real rows, same ratio),
LZMA2 byte count measured on a fixed sample of clients per rank (lossless stage only changes bytes, not training).

Arms (hp "arm"; docs/research/improvement-plan.md, owner 30/10: remove proxy training on the vehicle):
  dml    the v1 arm above (control S0): private + proxy trained together every step.
  srv    (A) only the private trains. After its epoch each vehicle scores a shared transfer set X_s (M Gaussian rows drawn
         from the pooled moments with a per-round seed, so X_s is never sent) and uploads its private's class
         probabilities (uint8). The server averages them per row (weights n_{k,c} of the row's class: a vehicle is
         trusted on classes it has seen; hp agg_soft = "client" uses n_k) and distils the proxy on X_s (S Adam steps,
         BN frozen), then sends the Q3 delta as before. FedDF (Lin et al. 2020) / FedMD (Li & Wang 2019) style.
  cheap  (B) only the private trains on real rows; after the epoch the proxy (from the replica) is distilled from the
         private on X_s on the vehicle (M / 256 steps, BN frozen), and the Q3 delta is uploaded as in dml.
  none   (C) no proxy. Vehicles upload per-class mean private probabilities on their real rows (16 x 16 + counts, as
         FedDistill, Jeong et al. 2018); private KD targets are the pooled class rows Q[y] (real and Gaussian rows).
Synthetic rows (hp "synth", private-only arms; docs/research/screening-r10.md, owner 30/10): a = batch / 2 rows per step,
  gauss   pooled class Gaussian (default, as above), balanced labels
  off     no synthetic rows (KD on real rows only)
  oracle  real train rows of other vehicles (this rank's pool) with balanced labels: upper bound, not deployable
  short   Gaussian, labels drawn per vehicle with p_{k,c} ∝ exp(−16 n_{k,c} / n_k) (classes it lacks first)
  mix     gauss, then each row is mixed λ x + (1 − λ) x_g, λ ~ U(0, 1), with a random real row x of the same class in the
          batch when there is one (FedMix-like, Yoon et al. 2021; the real row never leaves the vehicle)
  copula  Gaussian copula: pooled per-class marginal histograms (256 bins on [−4, 4], summable) and the covariance of
          normal scores; x = F_c^{-1}(Φ(z)), z ~ N(0, R_c), linear inside a bin (v1 P-copula, reopened as an input source)
  copula_cluster  copula per class x scenario/run; sample a cluster by its pooled class count (H1)
  copula_mix      copula rows mixed with same-class local real rows, as in mix (H2)
hp synth_rows (default batch/2) controls the number of synthetic rows (H3: 64 at batch 64).
hp synth_target = onehot replaces synthetic Q[y] KD by CE in arm none; real-row KD stays unchanged (H4).
hp anchor = class/ensemble uses a fixed seeded support (m_transfer rows); ensemble sends uint8 private probabilities
on that support and distils sample-indexed targets directly, without a server proxy. Class is the same-support Q[y]
control. hp real_kd=false disables real-row Q[y] KD only in arm none (diagnostic ablation).
In srv/cheap the proxy never sees real rows, so its BN statistics come from the vehicles: the frozen teacher's forward on
real rows uses batch statistics (dropout off) and its running statistics are uploaded (fp16) and n_k-averaged. The KD
weight w = 1/(CE_private + CE_teacher) replaces CE_proxy by the teacher's CE (no live proxy).
"""
import lzma
import math

import torch
import torch.nn.functional as F

from core import grouped, local
from core.dagsnet import DAGSNet
from core.optim import GroupAdamW

LZMA = [dict(id=lzma.FILTER_LZMA2, preset=9 | lzma.PRESET_EXTREME)]
BITS = 3
KD_CAP = 1.0
BYTES_SAMPLE = 64                                                   # clients per rank whose packets are LZMA-coded each round


def quantize(delta, gen=None):
    """antimem.quantize on (G, rows, -1): symmetric per-row scale (fp16), 3-bit codes in [-3, 3]; stochastic if gen."""
    levels = 2 ** (BITS - 1) - 1
    scale = (delta.abs().amax(2, keepdim=True) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    v = delta / safe
    q = torch.round(v) if gen is None else torch.floor(v + torch.rand(v.shape, device=v.device, generator=gen))
    q = q.clamp(-levels, levels) * (scale > 0)
    return q * scale, q.to(torch.int8), scale.half()


def packet_bytes(codes, scales, bufs_row):
    """Wire bytes of one client's packet: 3-bit codes packed (2 per byte, like the original 8//bits), fp16 scales, fp16
    BN buffers, LZMA2-coded."""
    body = []
    for q, s in zip(codes, scales):
        c = (q.reshape(-1).to(torch.int16) + 4).to(torch.uint8).cpu()
        if len(c) % 2:
            c = torch.cat([c, c.new_zeros(1)])
        body += [s.reshape(-1).cpu().numpy().tobytes(), (c[0::2] | (c[1::2] << 3)).numpy().tobytes()]
    body.append(bufs_row.half().cpu().numpy().tobytes())
    return 32 + len(lzma.compress(b"".join(body), format=lzma.FORMAT_RAW, filters=LZMA))   # 32: header as QHEADER


class Method:
    name = "mobihet"
    per_client = True

    def __init__(self, env):
        self.env = env
        lay, dev = env.lay, env.dev
        p0, b0 = lay.init(env.hp["seed"], 1, dev)
        self.priv_p = p0.repeat(env.K, 1)                            # private bank (this rank's clients)
        self.priv_b = b0.repeat(env.K, 1)
        self.g_p, self.g_b = p0[0].clone(), b0[0].clone()            # proxy replica every client holds
        self.exact = self.g_p.clone()                               # server's exact proxy (error feedback)
        self.version = 0
        self.aux = env.hp.get("synth_rows", env.hp["batch"] // 2)
        assert isinstance(self.aux, int) and self.aux > 0, self.aux
        self.weighted = env.hp.get("agg", "weighted") == "weighted"      # owner 29/10: n_k weights (original: 1/K)
        self.gen_q = torch.Generator(device=dev).manual_seed(env.hp["seed"] * 7 + env.rank)
        self.gen_g = torch.Generator(device=dev).manual_seed(env.hp["seed"] * 11 + env.rank)
        self.arm = env.hp.get("arm", "dml")
        assert self.arm in ("dml", "srv", "cheap", "none"), self.arm
        self.M = env.hp.get("m_transfer", 4096)
        self.S = env.hp.get("server_steps", 200)
        self.Q = None                                                 # arm none: pooled class probabilities (16, 16)
        self.synth = env.hp.get("synth", "gauss")
        assert self.synth in ("gauss", "off", "oracle", "short", "mix", "copula", "copula_cluster", "copula_mix"), self.synth
        assert self.synth == "gauss" or self.arm != "dml", "synth arms are for the private-only arms"
        self.synth_target = env.hp.get("synth_target", "soft")
        assert self.synth_target in ("soft", "onehot", "fallback"), self.synth_target
        assert self.synth_target == "soft" or self.arm == "none", "onehot target is for arm none"
        self.no_kd = env.hp.get("no_kd", False)
        self.kd_weight = env.hp.get("kd_weight", "adaptive")
        self.q_codec = env.hp.get("q_codec", "legacy")
        self.q_collect = env.hp.get("q_collect", "online")
        self.kd_support = env.hp.get("kd_support", "off")
        self.q_agg = env.hp.get("q_agg", "count")
        assert self.q_collect in ("online", "postepoch")
        assert self.kd_support in ("off", "uniform", "local")
        assert self.q_agg in ("count", "sqrt_count")
        assert self.q_agg == "count" or (self.arm == "none" and not self.no_kd and self.q_codec == "mean_fp16"
                                         and env.hp.get("kd_bins", 1) == 1 and env.hp.get("anchor", "off") == "off")
        assert self.q_collect == "online" or (self.arm == "none" and not self.no_kd and self.q_codec == "mean_fp16"
                                              and env.hp.get("kd_bins", 1) == 1 and env.hp.get("anchor", "off") == "off")
        assert self.kd_support == "off" or (self.arm == "none" and self.synth_target == "fallback")
        assert self.kd_weight in ("adaptive", "fixed")
        assert self.q_codec in ("legacy", "mean_fp16")
        assert self.kd_weight == "adaptive" or self.arm == "none"
        self.kd_bins = env.hp.get("kd_bins", 1)
        assert isinstance(self.no_kd, bool) and self.kd_bins in (1, 8)
        assert not (self.no_kd or self.kd_bins > 1) or self.arm == "none"
        assert not self.no_kd or self.kd_bins == 1
        assert self.q_codec == "legacy" or (self.arm == "none" and self.kd_bins == 1)
        self._bootstrap()
        self.bucket_Q = None
        if self.kd_bins > 1:
            self._bucket_bootstrap()
        self.anchor = env.hp.get("anchor", "off")
        self.real_kd = env.hp.get("real_kd", True)
        self.real_kd_gate = env.hp.get("real_kd_gate", False)
        assert isinstance(self.real_kd, bool)
        assert isinstance(self.real_kd_gate, bool)
        assert not self.real_kd_gate or (self.arm == "none" and self.real_kd)
        assert self.real_kd or self.arm == "none"
        assert self.anchor in ("off", "class", "ensemble"), self.anchor
        assert self.anchor == "off" or (not self.no_kd and self.kd_bins == 1)
        self.anchor_Q = None
        if self.anchor != "off":
            assert self.arm == "none" and self.synth in ("gauss", "copula")
            assert self.M % 16 == 0
            # Same finite support on every vehicle/rank; keep the streaming sampler RNG intact.
            from types import SimpleNamespace
            gen, aux, labels = self.gen_g, self.aux, self.labels
            self.gen_g = torch.Generator(device=dev).manual_seed(env.hp["seed"] * 1009)
            self.aux, self.labels = self.M, torch.arange(self.M, device=dev) % 16
            ctx = SimpleNamespace(Gfix=1, x=env.train.X[:1], clients=torch.zeros(1, dtype=torch.long, device=dev))
            self.anchor_X, self.anchor_y = self._synth(ctx)
            self.anchor_X, self.anchor_y = self.anchor_X[0], self.anchor_y[0]
            self.gen_g, self.aux, self.labels = gen, aux, labels

    def _bucket(self, x):
        bits = (x.float() @ self.router > 0).long()
        return (bits * bits.new_tensor([1, 2, 4])).sum(-1)

    def _bucket_bootstrap(self):
        """Fixed input partition, fitted with train counts only; no learned representation is shared."""
        env = self.env
        gen = torch.Generator().manual_seed(20261031)
        self.router = (2 * torch.randint(2, (env.train.X.shape[1], 3), generator=gen) - 1).float().to(env.dev)
        self.bucket_cnt = torch.zeros(env.K * 16 * 8, device=env.dev)
        client = torch.repeat_interleave(torch.arange(env.K, device=env.dev), env.train.off[1:] - env.train.off[:-1])
        for i in range(0, len(client), 1 << 20):
            y = env.train.Y[i:i + (1 << 20)].long()
            b = self._bucket(env.train.X[i:i + (1 << 20)])
            key = (client[i:i + (1 << 20)] * 16 + y) * 8 + b
            self.bucket_cnt.index_add_(0, key, torch.ones_like(key, dtype=torch.float32))
        self.bucket_cnt = self.bucket_cnt.view(env.K, 16, 8)
        self.bucket_tot = env.allreduce(self.bucket_cnt.sum(0).double())
        for direction in ("up", "down"):
            self.bootstrap_bytes[direction] += 16 * 8 * 4

    def _bootstrap(self):
        env, X, Y = self.env, self.env.train.X, self.env.train.Y
        C, D = 16, X.shape[1]
        cnt = torch.zeros(env.K, C, device=env.dev)
        s1 = torch.zeros(C, D, dtype=torch.float64, device=env.dev)
        s2 = torch.zeros(C, D, D, dtype=torch.float64, device=env.dev)
        client = torch.repeat_interleave(torch.arange(env.K, device=env.dev), env.train.off[1:] - env.train.off[:-1])
        cnt.index_put_((client, Y.long()), torch.ones(len(Y), device=env.dev), accumulate=True)
        self.cnt = cnt                                               # n_{k,c} (srv class-aware weights)
        for c in range(C):
            for i in range(0, len(Y), 1 << 21):
                x = X[i:i + (1 << 21)][Y[i:i + (1 << 21)] == c].double()
                s1[c] += x.sum(0); s2[c] += x.T @ x
        tot = env.allreduce(cnt.sum(0).double())
        s1, s2 = env.allreduce(s1), env.allreduce(s2)
        mu = s1 / tot[:, None]
        cov = s2 / tot[:, None, None] - mu[:, :, None] * mu[:, None, :]
        cov = cov + 1e-4 * torch.eye(D, device=env.dev, dtype=torch.float64)
        self.mu, self.L = mu.float(), torch.linalg.cholesky(cov).float()
        log_pk = ((cnt + 1) / (cnt.sum(1, keepdim=True) + C)).log()
        log_pg = ((tot + 1) / (tot.sum() + C)).log().float()
        self.la_shift = log_pk - log_pg                              # (K, 16)
        self.labels = torch.arange(self.aux, device=env.dev) % C
        tri = D * (D + 1) // 2
        self.bootstrap_bytes = {"up": 4 * C * (1 + D + tri), "down": 4 * C * (D + tri) + 4 * C}
        if self.synth == "short":
            self.p_short = torch.exp(-C * cnt / cnt.sum(1, keepdim=True).clamp_min(1))
        elif self.synth == "oracle":
            self.cls_idx = torch.argsort(Y.long())
            n_c = cnt.sum(0).long()
            self.cls_n, self.cls_off = n_c, torch.cumsum(n_c, 0) - n_c
        elif self.synth.startswith("copula"):
            self._copula_bootstrap(client)

    def _client_clusters(self):
        """Public scenario/run metadata; no labels or test rows enter the cluster assignment."""
        import os
        import pandas as pd
        from core import data as D

        c = pd.read_parquet(os.path.join(D.find_root(), "clients.parquet"))
        keys = sorted(set(zip(c.scenario, c.run)))
        assert len(keys) == 8, f"expected 8 scenario/run groups, got {len(keys)}"
        key = {k: i for i, k in enumerate(keys)}
        by_id = dict(zip(c.client_id, [key[k] for k in zip(c.scenario, c.run)]))
        return torch.tensor([by_id[int(i)] for i in self.env.ids], device=self.env.dev), len(keys)

    def _copula_bootstrap(self, client, NB=256):
        """Pooled per-class marginal histograms (C, D, NB) on [−4, 4] and normal-score covariance (both sums)."""
        env, X, Y = self.env, self.env.train.X, self.env.train.Y
        C, D = 16, X.shape[1]
        if self.synth == "copula_cluster":
            cluster, self.n_cop_clusters = self._client_clusters()
            Y = Y.long() * self.n_cop_clusters + cluster[client]
            C *= self.n_cop_clusters
        hist = torch.zeros(C * D * NB, dtype=torch.float64, device=env.dev)
        f = torch.arange(D, device=env.dev)
        bins = lambda x: ((x.float() + 4) * (NB / 8)).long().clamp(0, NB - 1)
        for i in range(0, len(Y), 1 << 20):
            x, y = X[i:i + (1 << 20)], Y[i:i + (1 << 20)].long()
            hist.index_add_(0, ((y[:, None] * D + f) * NB + bins(x)).reshape(-1),
                            torch.ones(x.numel(), dtype=torch.float64, device=env.dev))
        hist = env.allreduce(hist).view(C, D, NB)
        if self.synth == "copula_cluster":
            n = hist[:, 0].sum(-1).view(16, self.n_cop_clusters)
            self.cop_cluster_prob = (n / n.sum(-1, keepdim=True).clamp_min(1)).float()
        mass = hist / hist.sum(-1, keepdim=True).clamp_min(1)
        cdf = mass.cumsum(-1)
        nd = torch.distributions.Normal(0.0, 1.0)
        score = nd.icdf((cdf - mass / 2).clamp(1e-4, 1 - 1e-4)).float()      # mid-rank normal score of each bin
        s1 = torch.zeros(C, D, dtype=torch.float64, device=env.dev)
        s2 = torch.zeros(C, D, D, dtype=torch.float64, device=env.dev)
        for i in range(0, len(Y), 1 << 20):
            x, y = X[i:i + (1 << 20)], Y[i:i + (1 << 20)].long()
            z = score[y[:, None], f, bins(x)].double()
            for c in range(C):
                zc = z[y == c]
                s1[c] += zc.sum(0); s2[c] += zc.T @ zc
        s1, s2 = env.allreduce(s1), env.allreduce(s2)
        tot = hist[:, 0].sum(-1).clamp_min(1)
        mu = s1 / tot[:, None]
        cov = s2 / tot[:, None, None] - mu[:, :, None] * mu[:, None, :] + 1e-4 * torch.eye(D, device=env.dev, dtype=torch.float64)
        self.cop_mu, self.cop_L = mu.float(), torch.linalg.cholesky(cov).float()
        self.cop_cdf, self.cop_mass = cdf.float().contiguous(), mass.float()
        self.bootstrap_bytes["up"] += 4 * C * D * NB + 4 * C * (D + D * (D + 1) // 2)
        self.bootstrap_bytes["down"] += 4 * C * D * NB + 4 * C * (D + D * (D + 1) // 2)

    def _synth(self, ctx):
        """Synthetic rows (G, a, D) and labels (G, a) for one step, per hp synth (see module docstring)."""
        G, a, dev = ctx.Gfix, self.aux, ctx.x.device
        if hasattr(self, "anchor_X"):
            # Balanced classes as in the streaming control; random instance within each class.
            j = torch.randint(self.M // 16, (G, a), device=dev, generator=self.gen_g)
            self._anchor_idx = j * 16 + self.labels[None]
            return self.anchor_X[self._anchor_idx].to(ctx.x.dtype), self.anchor_y[self._anchor_idx]
        if self.synth == "short":
            lab = torch.multinomial(self.p_short[ctx.clients], a, replacement=True, generator=self.gen_g)
        else:
            lab = self.labels[None].expand(G, a)
        if self.synth == "oracle":
            u = torch.rand(G, a, device=dev, generator=self.gen_g)
            pos = self.cls_off[lab] + (u * self.cls_n[lab]).long().minimum((self.cls_n[lab] - 1).clamp_min(0))
            return self.env.train.X[self.cls_idx[pos]].to(ctx.x.dtype), lab
        if self.synth.startswith("copula"):
            cat = lab
            if self.synth == "copula_cluster":
                cluster = torch.multinomial(self.cop_cluster_prob[lab].reshape(-1, self.n_cop_clusters), 1,
                                            generator=self.gen_g).view(G, a)
                cat = lab * self.n_cop_clusters + cluster
            z = torch.randn(G, a, self.cop_mu.shape[1], device=dev, generator=self.gen_g)
            z = self.cop_mu[cat] + (self.cop_L[cat] @ z[..., None]).squeeze(-1)
            u = torch.special.ndtr(z).clamp(1e-6, 1 - 1e-6)
            cdf = self.cop_cdf[cat]                                          # (G, a, D, NB)
            j = torch.searchsorted(cdf, u[..., None]).clamp(max=cdf.shape[-1] - 1)
            hi = cdf.gather(-1, j).squeeze(-1)
            m = self.cop_mass[cat].gather(-1, j).squeeze(-1)
            frac = ((u - (hi - m)) / m.clamp_min(1e-12)).clamp(0, 1)
            xg = -4 + (j.squeeze(-1) + frac) * (8 / cdf.shape[-1])
            if self.synth != "copula_mix":
                return xg.to(ctx.x.dtype), lab
        else:
            z = torch.randn(G, a, self.mu.shape[1], device=dev, generator=self.gen_g)
            xg = (self.mu[lab] + (self.L[lab] @ z[..., None]).squeeze(-1)).clamp(-4, 4)
        if self.synth in ("mix", "copula_mix"):
            same = (ctx.y[:, None, :] == lab[..., None]) & ctx.mask[:, None, :]     # (G, a, b) real rows of the class
            pick = (torch.rand(same.shape, device=dev, generator=self.gen_g) * same).argmax(-1)
            xr = ctx.x.gather(1, pick[..., None].expand(-1, -1, xg.shape[-1])).float()
            lam = torch.rand(G, a, 1, device=dev, generator=self.gen_g)
            xg = torch.where(same.any(-1, keepdim=True), lam * xr + (1 - lam) * xg, xg)
        return xg.to(ctx.x.dtype), lab

    def payload_bytes(self):
        n = 4 * (self.env.lay.P + self.env.lay.B)
        return {"arm": self.arm, "dense_reference": n, "bootstrap": self.bootstrap_bytes,
                "measured": "rounds.jsonl bytes_up/bytes_down"}

    def transfer_set(self, r):
        """X_s for round r: M Gaussian rows, balanced labels, same on every rank and on the server (seeded, never sent)."""
        g = torch.Generator(device=self.env.dev).manual_seed(self.env.hp["seed"] * 1000 + r)
        ys = torch.arange(self.M, device=self.env.dev) % 16
        z = torch.randn(self.M, self.mu.shape[1], device=self.env.dev, generator=g)
        xs = (self.mu[ys] + (self.L[ys] @ z[..., None]).squeeze(-1)).clamp(-4, 4)
        return xs, ys

    def _loss_private(self, ctx):
        """srv / cheap / none: only the private trains (models = 1)."""
        cfg = self.env.cfg
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        act = ctx.mask.any(1)
        if self.synth == "off":
            a, xp, mp = 0, ctx.x, ctx.mask
        else:
            xg, lab = self._synth(ctx)
            xp = torch.cat([ctx.x, xg], 1)
            mp = torch.cat([ctx.mask, act[:, None].expand(G, a)], 1)
        zp, nBp = grouped.TRAIN(ctx.P, ctx.B, xp, cfg, mask=mp, train=True)
        y = ctx.y
        ce = local.masked_mean(F.cross_entropy((zp[:, :b] + self._shift[:, None, :]).reshape(-1, 16), y.reshape(-1),
                                               reduction="none").view(G, b), ctx.mask)
        lsp = F.log_softmax(zp, -1)
        if self.arm == "none":
            if self.no_kd:
                aux_ce = -lsp[:, b:].gather(-1, lab[..., None]).squeeze(-1).mean(1) if a else 0.0
                return ce + aux_ce, nBp
            if self.q_collect == "online":
                with torch.no_grad():                                # class sums of private probabilities, real rows
                    pr = torch.where(ctx.mask[..., None], lsp[:, :b].detach().exp(), 0.0)   # padded rows may be inf/NaN
                    key = y * 8 + self._bucket(ctx.x) if self.kd_bins > 1 else y
                    self._csum.index_put_((torch.arange(G, device=y.device)[:, None].expand_as(y), key), pr, accumulate=True)
            if self.Q is None:
                if a == 0:
                    return ce, nBp
                aux_ce = F.cross_entropy(zp[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a).mean(1)
                return ce + aux_ce, nBp
            lt = self.Q.clamp_min(1e-6).log()
            lt_r = lt[y]
            lt_g = lt[lab] if a else None
            if a and self.bucket_Q is not None:
                lt_g = self.bucket_Q[lab, self._bucket(xg)].clamp_min(1e-6).log()
            if a and self.anchor == "ensemble" and self.anchor_Q is not None:
                lt_g = self.anchor_Q[self._anchor_idx].clamp_min(1e-6).log()
            ce_t = local.masked_mean(-lt_r.gather(-1, y[..., None]).squeeze(-1), ctx.mask)
        else:
            if self.version == 0:
                if a == 0:
                    return ce, nBp
                aux_ce = F.cross_entropy(zp[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a).mean(1)
                return ce + aux_ce, nBp
            with torch.no_grad():
                zr, nBt = grouped.TRAIN(self._teacher_p, self._tb, ctx.x, self._cfg0, mask=ctx.mask, train=True)
                self._tb = {k: torch.where(act.view(-1, *([1] * (v.dim() - 1))), nBt[k], v) for k, v in self._tb.items()}
                lt_r = F.log_softmax(zr.float(), -1)
                lt_g = F.log_softmax(grouped.TRAIN(self._teacher_p, self._teacher_b, xg, cfg, train=False)[0].float(), -1) if a else None
            ce_t = local.masked_mean(F.nll_loss(lt_r.reshape(-1, 16), y.reshape(-1), reduction="none").view(G, b), ctx.mask)
        raw = 1 / (ce + ce_t).detach().clamp_min(1e-6)
        if self.kd_weight == "fixed":
            raw = torch.ones_like(raw)
        kl_real = (lt_r.exp() * (lt_r - lsp[:, :b])).sum(-1)
        if self.kd_support != "off":
            kl_real = kl_real * self._support_weight(y, ctx.clients)
        if self.real_kd_gate:
            kl_real = torch.where(lt_r.argmax(-1) == y, kl_real, 0.0)
        kd = local.masked_mean(kl_real, ctx.mask)
        if not self.real_kd:
            kd = torch.zeros_like(kd)
        if a and self.synth_target == "onehot":
            gkd = -lsp[:, b:].gather(-1, lab[..., None]).squeeze(-1).mean(1)
        elif a and self.synth_target == "fallback":
            rho = self._support_weight(lab, ctx.clients)
            rho = torch.where(lt_g.argmax(-1) == lab, rho, 0.0)
            target = rho[..., None] * lt_g.exp() + (1 - rho[..., None]) * F.one_hot(lab, 16)
            gkd = (target * (target.clamp_min(1e-6).log() - lsp[:, b:])).sum(-1).mean(1)
        else:
            gkd = (lt_g.exp() * (lt_g - lsp[:, b:])).sum(-1).mean(1) if a else 0.0
        return ce + raw * (kd + gkd), nBp

    def _support_weight(self, labels, clients):
        if self.kd_support == "off":
            return torch.ones_like(labels, dtype=torch.float32)
        if self.kd_support == "uniform":
            return torch.full_like(labels, 0.5, dtype=torch.float32)
        count = local.pad_rows(self.cnt[clients], labels.shape[0]).gather(1, labels)
        return self.env.hp["batch"] / (self.env.hp["batch"] + count)

    def _loss(self, ctx):
        env, cfg = self.env, self.env.cfg
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        Pp, Pr = ctx.P
        Bp, Br = ctx.B
        act = ctx.mask.any(1)
        z = torch.randn(G, a, self.mu.shape[1], device=ctx.x.device, generator=self.gen_g)
        xg = (self.mu[self.labels] + (self.L[self.labels] @ z[..., None]).squeeze(-1)).clamp(-4, 4).to(ctx.x.dtype)
        xp = torch.cat([ctx.x, xg], 1)
        mp = torch.cat([ctx.mask, act[:, None].expand(G, a)], 1)
        zp, nBp = grouped.TRAIN(Pp, Bp, xp, cfg, mask=mp, train=True)
        zr, nBr = grouped.TRAIN(Pr, Br, ctx.x, cfg, mask=ctx.mask, train=True)
        y = ctx.y
        shift = self._shift[:, None, :]
        ce = local.masked_mean(F.cross_entropy((zp[:, :b] + shift).reshape(-1, 16), y.reshape(-1), reduction="none").view(G, b), ctx.mask)
        pce = local.masked_mean(F.cross_entropy(zr.reshape(-1, 16), y.reshape(-1), reduction="none").view(G, b), ctx.mask)
        raw = 1 / (ce + pce).detach().clamp_min(1e-6)
        lsp, lsr = F.log_softmax(zp, -1), F.log_softmax(zr, -1)
        kl_proxy = local.masked_mean((lsp[:, :b].detach().exp() * (lsp[:, :b].detach() - lsr)).sum(-1), ctx.mask)
        loss = ce + pce + raw.clamp(max=KD_CAP) * kl_proxy
        if self.version > 0:
            with torch.no_grad():
                zt, _ = grouped.TRAIN(self._teacher_p, self._teacher_b, xp, cfg, train=False)
                lt = F.log_softmax(zt.float(), -1)
            kl = (lt.exp() * (lt - lsp)).sum(-1)                         # KL(T ‖ private) per row
            kd = local.masked_mean(kl[:, :b], ctx.mask)
            gkd = kl[:, b:].mean(1)
            loss = loss + raw * (kd + gkd)
        else:
            aux_ce = F.cross_entropy(zp[:, b:].reshape(-1, 16), self.labels.repeat(G), reduction="none").view(G, a).mean(1)
            loss = loss + aux_ce
        return loss, [nBp, nBr]

    def round(self, r, lr):
        if self.arm != "dml":
            return self._round_private(r, lr)
        env, lay = self.env, self.env.lay
        P = lay.P
        dsum = torch.zeros_like(self.g_p, dtype=torch.float64)
        bsum = torch.zeros_like(self.g_b, dtype=torch.float64)
        loss = torch.zeros(env.K, device=env.dev)
        nbytes, sampled = 0, 0
        for gi, idx in enumerate(env.groups_idx()):
            G = len(idx)
            flat = torch.cat([self.priv_p[idx], self.g_p.repeat(G, 1)], 1)
            bufs = torch.cat([self.priv_b[idx], self.g_b.repeat(G, 1)], 1)
            self._shift = local.pad_rows(self.la_shift[idx], local.group_size(env.train.off, idx, env.hp))
            if self.version > 0:
                Gf = self._shift.shape[0]
                self._teacher_p = lay.params(self.g_p[None].repeat(Gf, 1))
                self._teacher_b = lay.bufs(self.g_b[None].repeat(Gf, 1))
            st = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss,
                                   models=2)
            self.priv_p[idx], self.priv_b[idx] = flat[:, :P], bufs[:, :lay.B]
            # uplink: Q3(proxy − replica) per tensor row, stochastic rounding
            delta = lay.params(flat[:, P:] - self.g_p[None])
            codes, scales = [], []
            for k, shape in lay.p:
                d = delta[k].reshape(G, shape[0] if len(shape) > 1 else 1, -1)
                sent, q, s = quantize(d, self.gen_q)
                wk = env.n[idx][:, None] if self.weighted else 1.0
                dsum[_slice(lay, k)] += (wk * sent.reshape(G, -1)).sum(0).double()
                codes.append(q); scales.append(s)
            wb = env.n[idx][:, None].double() if self.weighted else 1.0
            bsum += (wb * bufs[:, lay.B:].half().double()).sum(0)
            loss[idx] = st["loss"]
            if sampled < BYTES_SAMPLE:
                for i in range(min(G, BYTES_SAMPLE - sampled)):
                    nbytes += packet_bytes([c[i] for c in codes], [s[i] for s in scales], bufs[i, lay.B:])
                sampled += min(G, BYTES_SAMPLE - sampled)
        K = env.allreduce(env.n.sum().clone() if self.weighted else torch.tensor(float(env.K), device=env.dev))
        self.exact = self.exact + (env.allreduce(dsum) / K).float()
        self.g_b = (env.allreduce(bsum) / K).float()
        # downlink: nearest Q3 of (exact − replica); every client rebuilds the same replica
        down = self.exact - self.g_p
        dv = lay.params(down[None])
        dq, codes, scales = torch.zeros_like(down), [], []
        for k, shape in lay.p:
            d = dv[k].reshape(1, shape[0] if len(shape) > 1 else 1, -1)
            sent, q, s = quantize(d)
            dq[_slice(lay, k)] = sent.reshape(-1); codes.append(q[0]); scales.append(s[0])
        self.g_p = self.g_p + dq
        down_bytes = packet_bytes(codes, scales, self.g_b)
        up = env.allreduce(torch.tensor([float(nbytes), float(sampled)], device=env.dev))
        self.version = r
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / env.allreduce(env.n.sum().clone())),
                "bytes_up": float(up[0] / up[1]), "bytes_down": down_bytes,
                "q_residual_rel": float((self.exact - self.g_p).norm() / self.exact.norm())}

    def _round_private(self, r, lr):
        env, lay, dev = self.env, self.env.lay, self.env.dev
        P, Bn = lay.P, lay.B
        loss = torch.zeros(env.K, device=dev)
        steps = torch.zeros(2, device=dev)                              # applied, skipped (non-finite fp16 gradient)
        bsum = torch.zeros_like(self.g_b, dtype=torch.float64)          # teacher BN statistics on real rows
        dsum = torch.zeros_like(self.g_p, dtype=torch.float64)          # cheap: Σ w Q3(Δ proxy)
        xs, ys = self.transfer_set(r) if self.arm in ("srv", "cheap") else (None, None)
        soft = torch.zeros(self.M, 16, device=dev, dtype=torch.float64) if self.arm == "srv" else None
        swt = torch.zeros(self.M, device=dev, dtype=torch.float64) if self.arm == "srv" else None
        csum = torch.zeros(16, 16, device=dev, dtype=torch.float64)
        csum_bucket = torch.zeros(16, 8, 16, device=dev, dtype=torch.float64) if self.kd_bins > 1 else None
        anchor_sum = torch.zeros(self.M, 16, device=dev, dtype=torch.float64) if self.anchor == "ensemble" else None
        anchor_weight = torch.zeros(self.M, device=dev, dtype=torch.float64) if self.anchor == "ensemble" else None
        nbytes, sampled = 0, 0
        self._cfg0 = dict(env.cfg, dropout=0.0)
        for idx in env.groups_idx():
            G = len(idx)
            Gf = local.group_size(env.train.off, idx, env.hp)
            self._shift = local.pad_rows(self.la_shift[idx], Gf)
            if self.version > 0 and self.arm != "none":
                self._teacher_p = lay.params(self.g_p[None].repeat(Gf, 1))
                self._teacher_b = lay.bufs(self.g_b[None].repeat(Gf, 1))
                self._tb = {k: v.clone() for k, v in self._teacher_b.items()}   # running stats, updated every step
            if self.arm == "none":
                self._csum = torch.zeros(Gf, 16 * self.kd_bins, 16, device=dev)
            flat, bufs = self.priv_p[idx], self.priv_b[idx]
            st = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss_private)
            self.priv_p[idx], self.priv_b[idx] = flat, bufs
            loss[idx] = st["loss"]
            steps += torch.stack([st["applied"].sum(), st["skipped"].sum()])
            w = env.n[idx].double()
            body = []
            if self.arm == "none":
                if not self.no_kd and self.kd_bins == 1 and self.q_collect == "online":
                    if self.q_codec == "mean_fp16":
                        packet = (self._csum[:G] / self.cnt[idx, :, None].clamp_min(1)).half()
                        decoded = packet.float()
                        weight = self.cnt[idx, :, None].double()
                        if self.q_agg == "sqrt_count":
                            weight = weight.sqrt()
                        csum += (decoded.double() * weight).sum(0)
                    else:
                        csum += self._csum[:G].sum(0).double()           # historical, unquantized aggregation
                        body.append(self._csum[:G].half().cpu().numpy().tobytes())
                elif self.kd_bins > 1:
                    count = self.bucket_cnt[idx]
                    mean = self._csum[:G].view(G, 16, 8, 16) / count[..., None].clamp_min(1)
                    decoded = mean.half().float()                       # fp16 means on wire; sums could overflow fp16
                    csum_bucket += (decoded.double() * count[..., None].double()).sum(0)
                if self.anchor == "ensemble":
                    prob = self._probs(flat, bufs, self.anchor_X, G)
                    q8 = (prob * 255).round()
                    prob = q8 / q8.sum(-1, keepdim=True).clamp_min(1)
                    wk = self.cnt[idx][:, self.anchor_y].double()
                    anchor_sum += (wk[..., None] * prob.double()).sum(0)
                    anchor_weight += wk.sum(0)
            if self.arm != "none" and self.version > 0:
                tb = lay.flat_bufs(self._tb)[:G]
                bsum += (w[:, None] * tb.half().double()).sum(0)
            if self.arm in ("srv", "cheap"):
                prob = self._probs(flat, bufs, xs, G)
            if self.arm == "srv":
                q8 = (prob * 255).round()                                  # uint8 upload
                pq = q8 / q8.sum(-1, keepdim=True).clamp_min(1)
                wk = self.cnt[idx][:, ys].double() if env.hp.get("agg_soft", "class") == "class" else w[:, None].expand(G, self.M)
                soft += (wk[..., None] * pq.double()).sum(0)
                swt += wk.sum(0)
                if sampled < BYTES_SAMPLE:
                    for i in range(min(G, BYTES_SAMPLE - sampled)):
                        nbytes += 32 + len(lzma.compress(q8[i].to(torch.uint8).cpu().numpy().tobytes() +
                                                         (lay.flat_bufs(self._tb)[i].half().cpu().numpy().tobytes()
                                                          if self.version > 0 else b""),
                                                         format=lzma.FORMAT_RAW, filters=LZMA))
                    sampled += min(G, BYTES_SAMPLE - sampled)
            elif self.arm == "cheap":
                pr = self._distil_on_vehicle(prob, xs, G)
                delta = lay.params(pr - self.g_p[None])
                codes, scales = [], []
                for k, shape in lay.p:
                    d = delta[k].reshape(G, shape[0] if len(shape) > 1 else 1, -1)
                    sent, q, sc = quantize(d, self.gen_q)
                    dsum[_slice(lay, k)] += (w[:, None].float() * sent.reshape(G, -1)).sum(0).double()
                    codes.append(q); scales.append(sc)
                if sampled < BYTES_SAMPLE:
                    tbf = lay.flat_bufs(self._tb) if self.version > 0 else self.g_b[None].expand(G, -1)
                    for i in range(min(G, BYTES_SAMPLE - sampled)):
                        nbytes += packet_bytes([c[i] for c in codes], [sc[i] for sc in scales], tbf[i])
                    sampled += min(G, BYTES_SAMPLE - sampled)
            elif self.arm == "none" and sampled < BYTES_SAMPLE:
                nbytes += 0 if self.no_kd else 32 + 16 * self.kd_bins * 16 * 2
                sampled += 1
        N = env.allreduce(env.n.sum().clone())
        steps = env.allreduce(steps)
        info = {"train_loss": float(env.allreduce((loss * env.n).sum()) / N),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1))}
        if self.arm == "none":
            previous_q = self.Q
            if self.q_collect == "postepoch":
                import time
                from methods.mobihet_teacher import class_probability_sums
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t_collect = time.monotonic()
                sums, counts = class_probability_sums(env, self.priv_p, self.priv_b, env.hp.get("eval_rows", 1 << 15))
                assert torch.equal(counts, self.cnt), "postepoch must cover every local train row once"
                packet = (sums / counts[..., None].clamp_min(1)).half()
                weight = counts[..., None].double()
                if self.q_agg == "sqrt_count":
                    weight = weight.sqrt()
                csum = (packet.float().double() * weight).sum(0)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                info["q_collect_seconds"] = time.monotonic() - t_collect
            down_bytes = 0 if self.no_kd else 32 + 16 * self.kd_bins * 16 * 2
            if self.kd_bins > 1:
                self._aggregate_buckets(env.allreduce(csum_bucket))
                info["bucket_nonempty_frac"] = float((self.bucket_tot > 0).float().mean())
                info["bucket_target_variation"] = float(((self.bucket_Q - self.Q[:, None]) ** 2).sum(-1).mean())
            elif not self.no_kd:
                csum = env.allreduce(csum)
                local_tot = self.cnt.sum(0).double() if self.q_agg == "count" else self.cnt.double().sqrt().sum(0)
                tot = env.allreduce(local_tot)
                self.Q = (csum / tot[:, None].clamp_min(1)).float()
                if self.q_codec == "mean_fp16":
                    self.Q = self.Q.half().float()
                    self.Q = self.Q / self.Q.sum(-1, keepdim=True).clamp_min(1e-12)
                if env.hp.get("teacher_diagnostics", False):
                    info["q_true_probability_mean"] = float(self.Q.diag().mean())
                    info["q_rejected_classes"] = (self.Q.argmax(-1) != torch.arange(16, device=dev)).nonzero().flatten().tolist()
                    info["q_entropy_mean"] = float(-(self.Q * self.Q.clamp_min(1e-6).log()).sum(-1).mean())
                    info["q_change_l1"] = None if previous_q is None else float((self.Q - previous_q).abs().sum(-1).mean())
            if self.anchor == "ensemble":
                target = env.allreduce(anchor_sum) / env.allreduce(anchor_weight)[:, None].clamp_min(1)
                q8 = (target * 255).round()
                self.anchor_Q = (q8 / q8.sum(-1, keepdim=True).clamp_min(1)).float()
                info["anchor_label_agreement"] = float((self.anchor_Q.argmax(-1) == self.anchor_y).float().mean())
                info["anchor_true_probability"] = float(self.anchor_Q.gather(1, self.anchor_y[:, None]).mean())
                info["anchor_entropy"] = float(-(self.anchor_Q * self.anchor_Q.clamp_min(1e-6).log()).sum(-1).mean())
                # Additional uint8 probability packet in each direction; inputs regenerated by seed.
                extra = 32 + self.M * 16
                nbytes += sampled * extra
                down_bytes += extra
        else:
            if self.version > 0:
                self.g_b = (env.allreduce(bsum) / N).float()
            if self.arm == "srv":
                soft, swt = env.allreduce(soft), env.allreduce(swt)
                target = (soft / swt[:, None].clamp_min(1e-12)).float()
                self.exact = self._distil_on_server(target, xs)
                info["server_kl_end"] = self._kl_end
            else:
                self.exact = self.exact + (env.allreduce(dsum) / N).float()
            down = self.exact - self.g_p
            dv = lay.params(down[None])
            dq, codes, scales = torch.zeros_like(down), [], []
            for k, shape in lay.p:
                d = dv[k].reshape(1, shape[0] if len(shape) > 1 else 1, -1)
                sent, q, sc = quantize(d)
                dq[_slice(lay, k)] = sent.reshape(-1); codes.append(q[0]); scales.append(sc[0])
            self.g_p = self.g_p + dq
            down_bytes = packet_bytes(codes, scales, self.g_b)
            info["q_residual_rel"] = float((self.exact - self.g_p).norm() / self.exact.norm())
        up = env.allreduce(torch.tensor([float(nbytes), float(max(sampled, 1))], device=dev))
        self.version = r
        info.update(bytes_up=float(up[0] / up[1]), bytes_down=down_bytes)
        return info

    def _aggregate_buckets(self, sums):
        """Decode downlink, then derive the marginal teacher from the same received conditional table."""
        count = self.bucket_tot
        marginal = sums.sum(1) / count.sum(1)[:, None].clamp_min(1)
        table = sums / count[..., None].clamp_min(1)
        table = torch.where(count[..., None] > 0, table, marginal[:, None])
        decoded = table.half().float()
        self.bucket_Q = decoded / decoded.sum(-1, keepdim=True).clamp_min(1e-12)
        self.Q = (self.bucket_Q.double() * count[..., None]).sum(1).float() / count.sum(1)[:, None].float().clamp_min(1)

    @torch.no_grad()
    def _probs(self, flat, bufs, xs, G, chunk=512):
        """Private class probabilities of a group on X_s -> (G, M, 16), eval mode, in row chunks."""
        env, lay = self.env, self.env.lay
        P, B = lay.params(flat), lay.bufs(bufs)
        out = []
        for s0 in range(0, len(xs), chunk):
            with torch.autocast("cuda", dtype=torch.float16, enabled=env.dev.type == "cuda"):
                z, _ = grouped.EVAL(P, B, xs[s0:s0 + chunk][None].expand(G, -1, -1).to(env.train.X.dtype), env.cfg,
                                    train=False)
            out.append(z.float().softmax(-1))
        return torch.cat(out, 1)

    def _distil_on_server(self, target, xs):
        """Proxy ← pooled soft labels on X_s: S Adam steps (lr 1e-3, batch 512), BN frozen (eval mode, no dropout), from the
        exact proxy. Rank 0 computes, every rank receives the same parameters."""
        env, lay = self.env, self.env.lay
        m = DAGSNet(env.cfg).to(env.dev)
        m.load_state_dict(lay.state_dict(self.exact, self.g_b), strict=False)
        m.eval()
        if env.rank == 0:
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            g = torch.Generator(device=env.dev).manual_seed(env.hp["seed"] + self.version)
            for _ in range(self.S):
                i = torch.randint(0, len(xs), (512,), device=env.dev, generator=g)
                l = F.kl_div(F.log_softmax(m(xs[i]), -1), target[i], reduction="batchmean")
                opt.zero_grad(); l.backward(); opt.step()
            self._kl_end = float(l.detach())
        flat = torch.cat([m.state_dict()[k].reshape(-1).float() for k, _ in lay.p])
        if env.world > 1:
            import torch.distributed as dist
            dist.broadcast(flat, 0)
            kl = torch.tensor([getattr(self, "_kl_end", 0.0)], device=env.dev); dist.broadcast(kl, 0); self._kl_end = float(kl)
        return flat

    def _distil_on_vehicle(self, prob, xs, G):
        """cheap: proxies (from the replica) ← this group's privates on X_s, batch 256, one pass, BN frozen. -> (G, P)."""
        env, lay = self.env, self.env.lay
        P = lay.P
        flat = self.g_p[None].repeat(G, 1)
        Bv = lay.bufs(self.g_b[None].repeat(G, 1))
        opt = GroupAdamW(flat, wd=env.hp["weight_decay"], clip=env.hp["clip"])
        act = torch.ones(G, dtype=torch.bool, device=env.dev)
        bs = 256
        for s0 in range(0, len(xs), bs):
            leaves = {k: v.detach().requires_grad_(True) for k, v in lay.params(flat).items()}
            x = xs[s0:s0 + bs][None].expand(G, -1, -1).to(env.train.X.dtype)
            with torch.autocast("cuda", dtype=torch.float16, enabled=env.dev.type == "cuda"):
                zr, _ = grouped.TRAIN(leaves, Bv, x, self._cfg0, train=False)
            t = prob[:, s0:s0 + bs]
            l = (t * (t.clamp_min(1e-8).log() - F.log_softmax(zr.float(), -1))).sum(-1).mean(1)
            (l.sum() * 1024).backward()
            grad = torch.cat([leaves[k].grad.reshape(G, -1) for k, _ in lay.p], dim=1) / 1024
            opt.step(flat, grad, 1e-3, act & torch.isfinite(grad).all(1))
        return flat

    def evaluate(self):
        out = {"private": self.env.eval_clients(self.priv_p, self.priv_b)}
        if self.arm != "none":
            out["proxy"] = self.env.eval_global(self.g_p, self.g_b)
        return out

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("priv_p", "priv_b", "g_p", "g_b", "exact")} | {
            "version": self.version, "Q": None if self.Q is None else self.Q.cpu(),
            "anchor_Q": None if self.anchor_Q is None else self.anchor_Q.cpu(),
            "bucket_Q": None if self.bucket_Q is None else self.bucket_Q.cpu()}

    def load_state(self, s):
        for k in ("priv_p", "priv_b", "g_p", "g_b", "exact"):
            setattr(self, k, s[k].to(self.env.dev))
        self.version = s["version"]
        self.Q = None if s.get("Q") is None else s["Q"].to(self.env.dev)
        self.anchor_Q = None if s.get("anchor_Q") is None else s["anchor_Q"].to(self.env.dev)
        self.bucket_Q = None if s.get("bucket_Q") is None else s["bucket_Q"].to(self.env.dev)


def _slice(lay, key):
    o = 0
    for k, s in lay.p:
        n = math.prod(s)
        if k == key:
            return slice(o, o + n)
        o += n
    raise KeyError(key)
