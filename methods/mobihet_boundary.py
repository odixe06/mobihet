"""B19 (A2'): generator boundary components chosen by the generator's own confusion on real rows.

Design and gates: docs/research/screening-boundary-b19.md. `km_boundary=off` delegates exactly to methods.mobihet_sim
(KM16 w0 .5, no quantisation per owner 12/11). `on` adds one bootstrap pass: every vehicle scores its real train rows with
the generator's Bayes classifier (mid-rank scores, the tables vehicles hold), and uploads count / sum z / sum zz^T of class-y
rows predicted c' != y per (y, c') cell with >= MIN_CELL rows; the server turns each pair with >= MIN_PAIR network rows
into one extra mixture component of class y, weighted by its real count. Only data-space statistics leave a vehicle:
no parameter, embedding or BN, so every vehicle may run another architecture.
"""
import math

import torch

from core import local
from methods.mobihet_sim import Method as Sim

MIN_CELL, MIN_PAIR, NB, LO, W = 8, 20, 256, -4.0, 8.0 / 256


class GenScore:
    """Per-class log density of x under the decoded generator (fp32 on device, chunked by caller)."""

    def __init__(self, cdf, mass, p, mu, L, floor=1e-7):
        self.score = torch.special.ndtri((cdf - mass / 2).clamp(1e-4, 1 - 1e-4)).float()      # (C, D, NB) mid-rank
        logm = (mass.double().clamp_min(floor) / W).log()
        self.jac = (logm + 0.5 * self.score.double() ** 2 + 0.5 * math.log(2 * math.pi)).float()  # log(m/W) - log phi(z)
        self.mu = mu.float()                                                       # (C, K, D)
        Ld = L.double()
        self.Linv = torch.linalg.inv(Ld).float()                                   # (C, K, D, D)
        self.logdet = torch.diagonal(Ld, dim1=-2, dim2=-1).log().sum(-1).float()   # (C, K)
        self.logp = torch.where(p > 0, p.double().log(), torch.full_like(p.double(), -math.inf)).float()

    def __call__(self, x):
        C, Dm = self.score.shape[0], self.score.shape[1]
        b = ((x.float() - LO) / W).floor().long().clamp(0, NB - 1)                 # (R, D)
        f = torch.arange(Dm, device=x.device)
        out = torch.empty(x.shape[0], C, device=x.device)
        for c in range(C):
            z = self.score[c][f, b]
            d = z[:, None, :] - self.mu[c][None]
            y = torch.einsum("kde,rke->rkd", self.Linv[c], d)
            lg = -0.5 * (y * y).sum(-1) - self.logdet[c] - 0.5 * Dm * math.log(2 * math.pi) + self.logp[c]
            out[:, c] = torch.logsumexp(lg, -1) + self.jac[c][f, b].sum(-1)
        return out

    def zscore(self, x, c):
        f = torch.arange(self.score.shape[1], device=x.device)
        return self.score[c][f, ((x.float() - LO) / W).floor().long().clamp(0, NB - 1)]


def mixture_cov(p, mu, L):
    """Total covariance of each class mixture (C, D, D): shrinkage target known to server and vehicles."""
    p, mu, L = p.double(), mu.double(), L.double()
    m = (p[..., None] * mu).sum(1)
    second = (p[..., None, None] * (L @ L.transpose(-1, -2) + mu[..., :, None] * mu[..., None, :])).sum(1)
    return second - m[:, :, None] * m[:, None, :]


def build_components(n, s1, s2, total, base_p, base_mu, base_L, w0):
    """Server: pairs (y, c') with n >= MIN_PAIR -> extra components of class y. Returns padded p, mu, L (fp16 decoded) + info."""
    C, D = s1.shape[0], s1.shape[-1]
    dev = s1.device
    keep = (n >= MIN_PAIR) & ~torch.eye(C, dtype=torch.bool, device=dev)
    E = C - 1
    p_new = torch.zeros(C, E, dtype=torch.float64, device=dev)
    mu_new = base_mu.double().mean(1, keepdim=True).expand(C, E, D).clone()
    L_new = torch.eye(D, dtype=torch.float64, device=dev).expand(C, E, D, D).clone()
    tgt = mixture_cov(base_p, base_mu, base_L)
    share = torch.zeros(C, dtype=torch.float64, device=dev)
    for y in range(C):
        cols = keep[y].nonzero().squeeze(1)
        for e, c in enumerate(cols.tolist()):
            nn = n[y, c]
            m = s1[y, c] / nn
            S = s2[y, c] / nn - m[:, None] * m[None]
            ev, U = torch.linalg.eigh((S + S.T) / 2)
            S = U @ torch.diag(ev.clamp_min(0)) @ U.T
            S = (nn * S + w0 * tgt[y]) / (nn + w0) + 1e-4 * torch.eye(D, dtype=S.dtype, device=dev)
            mu_new[y, e] = m
            L_new[y, e] = torch.linalg.cholesky(S)
            p_new[y, e] = nn / total[y]
        share[y] = p_new[y].sum()
    assert bool((share < 1).all()), share
    p = torch.cat([base_p.double() * (1 - share[:, None]), p_new], 1)
    p = p / p.sum(1, keepdim=True)
    mu = torch.cat([base_mu.double(), mu_new], 1).half().float()                   # downlink fp16, as KM centres
    L = torch.cat([base_L.double(), L_new], 1).half().float()
    info = {"pairs_kept": int(keep.sum()), "components_per_class": keep.sum(1).tolist(),
            "boundary_share": [round(float(v), 5) for v in share], "pair_rows": n[keep].sum().item()}
    return p.float(), mu, L, info


class Method(Sim):
    def __init__(self, env):
        super().__init__(env)
        assert self.km_boundary in {"off", "on"}
        assert self.q_bits == 16 and self.q_src == "real" and not self.qk and not self.need_T and self.sw == 1
        assert self.variant == "stream" and self.aux == 64 and not self.cf and self.synth == "copula"
        assert self.env.hp.get("cop_kmeans_q", "none") == "none" and not int(self.env.hp.get("cop_kmeans_lite", 0))

    def _copula_bootstrap(self, client, NB_=256):
        super()._copula_bootstrap(client, NB_)
        self.km_boundary = str(self.env.hp.get("km_boundary", "off"))
        self.bd_info = {}
        if self.km_boundary == "on":
            assert self.km == 16 and float(self.env.hp.get("cop_kmeans_w0")) == 0.5
            self._km_boundary()

    @torch.no_grad()
    def _km_boundary(self):
        env, X, Y = self.env, self.env.train.X, self.env.train.Y
        C, K, D = self.km_mu.shape
        dev = env.dev
        self.bd_base = (self.km_p.clone(), self.km_mu.clone(), self.km_L.clone())     # diagnostics: generator before B19
        gs = GenScore(self.cop_cdf, self.cop_mass, self.km_p, self.km_mu, self.km_L)
        total = env.allreduce(self.cnt.sum(0).double())                             # class counts: bootstrap schema
        logpi = ((total + 1) / (total.sum() + C)).log().float()
        n = torch.zeros(C, C, dtype=torch.float64, device=dev)
        s1 = torch.zeros(C, C, D, dtype=torch.float64, device=dev)
        s2 = torch.zeros(C, C, D, D, dtype=torch.float64, device=dev)
        off = env.train.off.tolist()
        cells_up = torch.zeros(2, dtype=torch.float64, device=dev)                 # cells sent, rows scored
        mis_all = torch.zeros(C, C, dtype=torch.float64, device=dev)                # every misclassified row (incl. unsent)
        for v in range(len(off) - 1):
            x, y = X[off[v]:off[v + 1]], Y[off[v]:off[v + 1]].long()
            pred = torch.cat([(gs(x[i:i + 2048]) + logpi).argmax(-1) for i in range(0, len(y), 2048)])
            bad = pred != y
            cell = y[bad] * C + pred[bad]
            cnt = torch.bincount(cell, minlength=C * C).view(C, C).double()
            mis_all += cnt
            cells_up[1] += len(y)
            for yc in (cnt >= MIN_CELL).nonzero().tolist():
                yy, cc = yc
                sel = bad & (y == yy) & (pred == cc)
                z = gs.zscore(x[sel], yy).double()
                n[yy, cc] += z.shape[0]
                s1[yy, cc] += z.sum(0)
                s2[yy, cc] += z.T @ z
                cells_up[0] += 1
        n, s1, s2 = env.allreduce(n), env.allreduce(s1), env.allreduce(s2)
        mis_all, cells_up = env.allreduce(mis_all), env.allreduce(cells_up)
        w0 = float(self.env.hp["cop_kmeans_w0"]) * D
        self.km_p, self.km_mu, self.km_L, info = build_components(n, s1, s2, total, self.km_p, self.km_mu, self.km_L, w0)
        self.km = self.km_p.shape[1]
        nv = env.allreduce(torch.tensor([float(len(off) - 1)], dtype=torch.float64, device=dev))
        tri = D * (D + 1) // 2
        up = float(cells_up[0] / nv) * (2 + 4 + 4 * D + 4 * tri)                # index, count, fp32 sum z, fp32 tri sum zz^T
        kept = info["pairs_kept"]
        down = 4 * C * (C - 1) + kept * 2 * (D + tri)                              # p fp32 for every slot, mean+Cholesky fp16
        self.bootstrap_bytes["up"] += up
        self.bootstrap_bytes["down"] += down
        info.update(rows_scored=int(cells_up[1]), cells_sent=int(cells_up[0]), up_mean_bytes=up, down_bytes=down,
                    misclassified_rows=int(mis_all.sum()), misclassified_sent_rows=int(n.sum()),
                    generator_train_acc=float(1 - mis_all.sum() / cells_up[1]), K_total=self.km)
        self.bd_info = info
        import json
        print("KM_BOUNDARY " + json.dumps(info), flush=True)

    def _round_private(self, r, lr):
        env, epoch = self.env, local.local_epoch                                    # same applied-step check as B13-B17
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
        if self.bd_info and r == 1:
            info["km_boundary"] = self.bd_info
        return info
