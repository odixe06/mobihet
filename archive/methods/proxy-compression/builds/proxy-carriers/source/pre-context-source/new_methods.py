"""New FL screening: Fourier mean embeddings and noise-initialized dataset condensation.

Implementation choices/gates: docs/research/pilot-new-fl.md. These are adaptations, not
reproductions of DP-MERF or FedDM. Only additive statistics or optimized synthetic points
cross clients. The common frozen condensation encoder is DAGSNet initialized from a seed.
"""
import copy, math
import torch
import torch.nn.functional as F
from mh import pilot as P


def moments(x, y):
    n = torch.bincount(y, minlength=P.C).double()
    s = torch.zeros(P.C, P.D, device=x.device, dtype=torch.float64)
    q = torch.zeros_like(s)
    for c in range(P.C):
        z = x[y == c].double()
        s[c], q[c] = z.sum(0), z.square().sum(0)
    return n, s, q


def gaussian(n, s, q):
    mu = s / n.clamp_min(1)[:, None]
    sd = (q / n.clamp_min(1)[:, None] - mu.square()).clamp_min(1e-8).sqrt()
    return mu.float(), sd.float()


def fp16(x):
    out = x.half()
    assert torch.isfinite(out).all(), 'fp16 knowledge overflow'
    return out.float()


def frequencies(device):
    """128 real features: 64 cosine/sine pairs, four fixed bandwidths; no data fitting."""
    g = torch.Generator(device=device).manual_seed(1701)
    scales = torch.tensor([.25, .5, 1., 2.], device=device).repeat_interleave(16) * math.sqrt(P.D)
    return torch.randn(P.D, 64, device=device, generator=g) / scales


def phi(x, w):
    z = x @ w
    return torch.cat([z.cos(), z.sin()], -1) / math.sqrt(w.shape[1])


def kernel_sums(x, y, w):
    out = torch.zeros(P.C, w.shape[1] * 2, device=x.device, dtype=torch.float64)
    for i in range(0, len(y), 8192):
        out.index_add_(0, y[i:i + 8192], phi(x[i:i + 8192].float(), w).double())
    return out


def gaussian_embedding(mu, sd, w):
    phase, damp = mu @ w, torch.exp(-.5 * (sd.square() @ w.square()))
    return torch.cat([damp * phase.cos(), damp * phase.sin()], -1) / math.sqrt(w.shape[1])


def fit_kernel_bank(mu, sd, target, w, g, steps, size=128):
    u = torch.randn(P.C, size, P.D, device=mu.device, generator=g, requires_grad=True)
    opt = torch.optim.Adam([u], lr=.05)
    diagnostics = []
    for step in range(steps + 1):
        x = mu[:, None] + sd[:, None] * u
        fit = (phi(x, w).mean(1) - target).square().sum(1).mean()
        reg = .1 * (u.mean(1).square().mean() + (u.square().mean(1) - 1).square().mean())
        loss = fit + reg
        assert torch.isfinite(loss)
        if step in (0, steps):
            diagnostics.append(dict(step=step, fit=float(fit.detach()), moment_penalty=float(reg.detach())))
        if step < steps:
            opt.zero_grad(); loss.backward(); opt.step()
    return x.detach(), diagnostics


class KernelFL(P.Local):
    """Secure-aggregated input moments + class Fourier sums; client-side synthesis."""
    def setup(self):
        R = self.R; self.mode = R.a.method; self.w = frequencies(R.dev)
        totals = None
        for k in range(R.K):
            # Cast BEFORE aggregation: the simulated payload really contains fp32 sums.
            vals = [v.float().double() for v in moments(R.X[k], R.Y[k])]
            if self.mode != 'diag':
                vals.append(kernel_sums(R.X[k], R.Y[k], self.w).float().double())
            R.ship(0, k, 'up', 'class_sums', 32 + sum(v.numel() * 4 for v in vals))
            totals = vals if totals is None else [a + b for a, b in zip(totals, vals)]
        assert R.K >= 5
        n, s, q = totals[:3]
        self.mu, self.sd = [fp16(v) for v in gaussian(n, s, q)]
        self.diag = {}
        down = 32 + P.C * 4 + (self.mu.numel() + self.sd.numel()) * 2
        if self.mode != 'diag':
            target = fp16((totals[3] / n.clamp_min(1)[:, None]).float())
            if self.mode == 'kme_gauss':
                target = fp16(gaussian_embedding(self.mu, self.sd, self.w))
            self.bank, self.diag['synthesis'] = fit_kernel_bank(
                self.mu, self.sd, target, self.w, R.gen(991), R.a.synth_steps)
            down += target.numel() * 2
        for k in range(R.K):
            R.ship(0, k, 'down', 'class_sums', down)

    def sample(self, k, m, g):
        y = torch.randint(P.C, (m,), device=self.R.dev, generator=g)
        if self.mode == 'diag':
            x = self.mu[y] + self.sd[y] * torch.randn(m, P.D, device=self.R.dev, generator=g)
        else:
            j = torch.randint(self.bank.shape[1], (m,), device=self.R.dev, generator=g)
            x = self.bank[y, j]
        return x, y

    def loss(self, k, x, y, g, r):
        xa, ya = self.sample(k, self.R.a.aux_batch, g)
        z = self.R.priv[k](torch.cat([x, xa]))[0].float()
        return F.cross_entropy(z[:len(y)], y) + self.R.a.aux_weight * F.cross_entropy(z[len(y):], ya)


class CondensedFL(KernelFL):
    """Static FedDM-inspired synthetic codebooks, shared without private model weights.

RSU can read synthetic books (owner exception 2026-10-11). Every book starts from
noise transformed by OWN moments. No real row is selected or used as initialization.
"""
    def setup(self):
        R = self.R; books = []; self.diag = {'synthesis': []}
        encoder = copy.deepcopy(R.init).eval().float()
        for p in encoder.parameters():
            p.requires_grad_(False)
        for k in range(R.K):
            n, s, q = moments(R.X[k], R.Y[k]); mu, sd = gaussian(n, s, q)
            u = torch.randn(P.C, R.a.ipc, P.D, device=R.dev, generator=R.gen(992, k), requires_grad=True)
            if R.a.method == 'dm':
                hsum = torch.zeros(P.C, P.latent_dim(encoder), device=R.dev, dtype=torch.float64)
                hsq = torch.zeros_like(hsum)
                with torch.no_grad():
                    for i in range(0, R.n[k], 4096):
                        h = encoder(R.X[k][i:i + 4096].float())[1].double()
                        hsum.index_add_(0, R.Y[k][i:i + 4096], h)
                        hsq.index_add_(0, R.Y[k][i:i + 4096], h.square())
                target = (hsum / n.clamp_min(1)[:, None]).float()
                scale = (hsq / n.clamp_min(1)[:, None] - target.double().square()).clamp_min(1e-4).sqrt().float()
                opt = torch.optim.Adam([u], lr=.03)
                trace = []
                for step in range(R.a.synth_steps + 1):
                    x = mu[:, None] + sd[:, None] * u
                    h = encoder(x.reshape(-1, P.D))[1].view(P.C, R.a.ipc, -1).mean(1)
                    fit = ((h - target) / scale).square().mean()
                    loss = fit + .01 * u.mean(1).square().mean()
                    assert torch.isfinite(loss)
                    if step in (0, R.a.synth_steps):
                        trace.append(dict(step=step, fit=float(fit.detach())))
                    if step < R.a.synth_steps:
                        opt.zero_grad(); loss.backward(); opt.step()
                self.diag['synthesis'].append(dict(client=k, trace=trace))
            book = fp16((mu[:, None] + sd[:, None] * u).detach())
            books.append(book)
            R.ship(0, k, 'up', 'synthetic_codebook', 32 + P.C * 4 + book.numel() * 2)
        self.books = torch.stack(books)
        for k in range(R.K):
            R.ship(0, k, 'down', 'synthetic_codebook', (R.K - 1) * (32 + P.C * 4 + books[k].numel() * 2))

    def sample(self, k, m, g):
        R = self.R
        donor = torch.randint(R.K - 1, (m,), device=R.dev, generator=g)
        donor = donor + (donor >= k).long()  # a client never replays its own transmitted book
        y = torch.randint(P.C, (m,), device=R.dev, generator=g)
        j = torch.randint(R.a.ipc, (m,), device=R.dev, generator=g)
        return self.books[donor, y, j], y


METHODS = dict(local=P.Local, diag=KernelFL, kme_gauss=KernelFL, kme=KernelFL,
               dm_gauss=CondensedFL, dm=CondensedFL)
