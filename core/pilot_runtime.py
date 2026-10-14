"""MobiHet pilot driver: screen every proposed knowledge channel (H1-H5) under one FL loop, no channel limit yet.

Plan, arms and locked gates: docs/research/pilot-plan.md. Fixture: scripts/build_pilot_fixture.py (20 clients,
alpha=0.5, 2.5% of each client's rows, holdouts `old` + `group`). Test20% is never read here.

Common recipe for every arm (sibling repo fl_dagsnet_veremi, scaled down): R rounds x 1 local epoch, batch 512,
AdamW wd 0 re-created per client per round, per-round cosine LR 1e-3 -> 1e-5, grad-norm clip 1.0 per model,
fp16 autocast with fp32 loss. Every client trains every round. Private models start from one seeded init.
Decision rules scored on every private model: `head` = argmax logits; `gp` = argmax(logits - log pi_i + log pi_g),
pi_g from the secure-aggregated label histogram (64 B per client, charged to every arm).

FL rule (owner 2026-10-10): no client ever sees another client's rows. What crosses a client boundary is listed
per method in WIRE and checked by `ship()`: parameters, additive class statistics, prototypes, logits on
synthetic points, a generator. Sums of several clients only (secure aggregation is simulated: the server only
uses totals; nothing per client is ever handed to another client).

    python -m mh.pilot --arm local --seed 42 [--smoke]
"""
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mh.dagsnet import build_model, CFG
from mh import archs
from mh.metrics import metrics_from_confusion, per_class_from_confusion

ROOT = Path(__file__).resolve().parents[1]
C, D = 16, 66
CLASSES = json.loads((ROOT / 'knowledge/meta.json').read_text())['class_names']
WIRE = {'params', 'class_counts', 'class_sums', 'prototypes', 'soft_labels', 'generator'}  # never rows (x, y)


# ------------------------------------------------------------------ models: every model returns (logits, latent)
class MLP(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(D, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.out = nn.Linear(w, C)

    def forward(self, x):
        h = self.body(x)
        return self.out(h), h


def make(spec):
    if spec == 'dagsnet':
        return build_model(dict(CFG, n_features=D, num_classes=C), latent=True)
    if spec.startswith('mlp'):
        return MLP(int(spec[3:]))
    if spec == 'cnn1d':
        return archs.CNN1d()
    if spec == 'gru':
        return archs.GRUNet()
    raise ValueError(spec)


def last_linear(m):
    return m.head[-1] if hasattr(m, 'head') else m.out


def latent_dim(m):
    return last_linear(m).in_features


def wire_bytes(sd, bpv=2):
    """fp16 on the wire for every float entry of a state_dict (BN running stats included)."""
    return sum(v.numel() * bpv for v in sd.values() if v.is_floating_point())


def average(acc, sd, w):
    if acc is None:
        return {k: (v.float() * w if v.is_floating_point() else v.clone()) for k, v in sd.items()}
    for k, v in sd.items():
        acc[k] = acc[k] + v.float() * w if v.is_floating_point() else torch.maximum(acc[k], v)
    return acc


def finish(acc, total):
    return {k: (v / total if v.is_floating_point() else v) for k, v in acc.items()}


# ------------------------------------------------------------------ the run
class Run:
    def __init__(self, a):
        self.a, self.dev = a, torch.device('cuda')
        fx = ROOT / 'papers/pilot/fixture'
        self.K = 20
        self.X, self.Y = [], []
        for k in range(self.K):
            x, y = np.load(fx / f'train_{k}_X.npy'), np.load(fx / f'train_{k}_y.npy')
            if a.smoke:
                i = np.random.default_rng(k).choice(len(y), 3000, replace=False); x, y = x[i], y[i]
            self.X.append(torch.from_numpy(x).to(self.dev)); self.Y.append(torch.from_numpy(y.astype(np.int64)).to(self.dev))
        self.H = {}
        for h in ('old', 'group'):
            x, y = np.load(fx / f'{h}_X.npy'), np.load(fx / f'{h}_y.npy')
            if a.smoke:
                x, y = x[:8192], y[:8192]
            self.H[h] = (torch.from_numpy(x).to(self.dev), torch.from_numpy(y.astype(np.int64)).to(self.dev))
        self.n = [len(y) for y in self.Y]
        self.counts = torch.stack([torch.bincount(y, minlength=C) for y in self.Y]).double()
        tot = self.counts.sum(0)                                 # secure-aggregated histogram
        self.log_pg = torch.log((tot + 1) / (tot + 1).sum()).float()
        self.log_pi = torch.log((self.counts + 1) / (self.counts + 1).sum(1, keepdim=True)).float()
        self.bytes = []                                          # (round, client, dir, kind, n)
        self.ship(0, None, 'up', 'class_counts', self.K * C * 4)
        self.ship(0, None, 'down', 'class_counts', self.K * C * 4)
        specs = a.private.split(',')                            # P3: client k gets specs[k % len(specs)]
        self.arch = [specs[k % len(specs)] for k in range(self.K)]
        inits = {}
        for s in specs:
            torch.manual_seed(a.seed); inits[s] = make(s).to(self.dev)
        self.init = inits[specs[0]]
        self.priv = [copy.deepcopy(inits[self.arch[k]]) for k in range(self.K)]

    def ship(self, r, k, d, kind, n):
        assert kind in WIRE, kind                                # R3: only knowledge kinds, never rows
        self.bytes.append((r, k, d, kind, int(n)))

    def lr(self, r):
        a = self.a
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * (r - 1) / max(a.rounds - 1, 1)))

    def gen(self, *key):
        return torch.Generator(device=self.dev).manual_seed(int(np.random.SeedSequence([self.a.seed, *key]).generate_state(1)[0]))

    # -------------------------------------------------------------- one client, one local epoch
    def train(self, m, k, r):
        mods = m.trainable(k)
        opt = torch.optim.AdamW([{'params': list(x.parameters())} for x in mods], lr=self.lr(r), weight_decay=0.0)
        scaler = torch.amp.GradScaler()
        for x in mods:
            x.train()
        g = self.gen(r, k)
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed, r, k, 7]).generate_state(1)[0]))  # dropout
        X, Y, B = self.X[k], self.Y[k], self.a.batch
        perm = torch.randperm(len(Y), device=self.dev, generator=g)
        tot, steps = 0.0, 0
        for i in range(0, len(Y), B):
            idx = perm[i:i + B]
            with torch.autocast('cuda', dtype=torch.float16):
                loss = m.loss(k, X[idx].float(), Y[idx], g, r)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            for x in mods:
                nn.utils.clip_grad_norm_(x.parameters(), self.a.clip)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss.detach()); steps += 1
        return tot / max(steps, 1)

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def logits(self, model, X):
        model.eval(); out = []
        for i in range(0, len(X), 16384):
            with torch.autocast('cuda', dtype=torch.float16):
                out.append(model(X[i:i + 16384].float())[0].float())
        return torch.cat(out)

    def score(self, z, y, bias):
        pred = (z + bias).argmax(1)
        return torch.bincount(y * C + pred, minlength=C * C).view(C, C).cpu().numpy()

    def evaluate(self, m):
        res = {}
        for name, models, local_prior in m.eval_models():
            res[name] = {}
            for h, (X, y) in self.H.items():
                cms = {'head': [], 'gp': []}
                for k, model in enumerate(models):
                    z = self.logits(model, X)
                    cms['head'].append(self.score(z, y, 0.0))
                    bias = (self.log_pg - self.log_pi[k]) if local_prior else 0.0
                    cms['gp'].append(self.score(z, y, bias))
                res[name][h] = {}
                for rule, cl in cms.items():
                    f1 = [metrics_from_confusion(c)['f1_macro'] for c in cl]
                    mets = [metrics_from_confusion(c) for c in cl]
                    pc = np.mean([[p['f1'] for p in per_class_from_confusion(c, CLASSES)] for c in cl], 0)
                    res[name][h][rule] = dict(f1_macro=float(np.mean(f1)), f1_macro_clients=f1,
                                              mean_metrics={kk: float(np.mean([mm[kk] for mm in mets])) for kk in mets[0]},
                                              f1_per_class=pc.tolist(), benign_f1=float(pc[0]),
                                              confusion_sum=np.sum(cl, 0).tolist())
        return res


# ------------------------------------------------------------------ methods
class Local:
    """Lower bound: private model, own rows only. Only the label histogram crosses (for gp)."""
    def __init__(self, run):
        self.R = run

    def setup(self):
        pass

    def trainable(self, k):
        return [self.R.priv[k]]

    def ce(self, k, x, y):
        return F.cross_entropy(self.R.priv[k](x)[0].float(), y)

    def loss(self, k, x, y, g, r):
        return self.ce(k, x, y)

    def begin(self, k, r):
        pass

    def end(self, k, r):
        pass

    def aggregate(self, r):
        pass

    def eval_models(self):
        return [('private', self.R.priv, True)]


class FedAvg(Local):
    """Reference for parameter sharing (not feasible at B0): one global DAGSNet, n_k-weighted mean of full state."""
    def setup(self):
        self.glob = copy.deepcopy(self.R.init); self.acc = None

    def begin(self, k, r):
        self.R.priv[k].load_state_dict(self.glob.state_dict())
        self.R.ship(r, k, 'down', 'params', wire_bytes(self.glob.state_dict()))

    def end(self, k, r):
        sd = self.R.priv[k].state_dict()
        self.acc = average(self.acc, sd, self.R.n[k]); self.R.ship(r, k, 'up', 'params', wire_bytes(sd))

    def aggregate(self, r):
        self.glob.load_state_dict(finish(self.acc, sum(self.R.n))); self.acc = None

    def eval_models(self):
        return [('global', [self.glob], False)]


class DML(Local):
    """H1 / NILM-FL: private model + shared proxy, mutual learning (CE + KL weighted 1/sg(ce_s+ce_r)), proxy averaged
    unweighted (as the sibling rebuild). Only proxy parameters leave the vehicle."""
    def setup(self):
        torch.manual_seed(self.R.a.seed + 1000)
        self.glob = make(self.R.a.proxy).to(self.R.dev); self.work = copy.deepcopy(self.glob); self.acc = None

    def trainable(self, k):
        return [self.R.priv[k], self.work]

    def begin(self, k, r):
        self.work.load_state_dict(self.glob.state_dict())
        self.R.ship(r, k, 'down', 'params', wire_bytes(self.glob.state_dict()))

    def loss(self, k, x, y, g, r):
        zs, zr = self.R.priv[k](x)[0].float(), self.work(x)[0].float()
        ce_s, ce_r = F.cross_entropy(zs, y), F.cross_entropy(zr, y)
        ls, lr_ = F.log_softmax(zs, 1), F.log_softmax(zr, 1)
        kl_s = F.kl_div(ls, lr_.detach(), log_target=True, reduction='batchmean')
        kl_r = F.kl_div(lr_, ls.detach(), log_target=True, reduction='batchmean')
        w = 1.0 / (ce_s + ce_r).detach().clamp_min(1e-6)
        return ce_s + w * kl_s + ce_r + w * kl_r

    def end(self, k, r):
        sd = self.work.state_dict()
        self.acc = average(self.acc, sd, 1.0); self.R.ship(r, k, 'up', 'params', wire_bytes(sd))

    def aggregate(self, r):
        self.glob.load_state_dict(finish(self.acc, self.R.K)); self.acc = None

    def eval_models(self):
        return [('private', self.R.priv, True), ('proxy', [self.glob], False)]


def class_sums(X, Y):
    """Additive per-class statistics of one client: n_c, sum x, sum x x^T (float64)."""
    n, s1, s2 = torch.zeros(C, dtype=torch.float64, device=X.device), torch.zeros(C, D, dtype=torch.float64, device=X.device), \
        torch.zeros(C, D, D, dtype=torch.float64, device=X.device)
    for c in range(C):
        xc = X[Y == c].double()
        n[c] = len(xc); s1[c] = xc.sum(0); s2[c] = xc.T @ xc
    return n, s1, s2


class Synth(Local):
    """H2: class statistics in the 66-feature input space, summed over all clients (secure aggregation), broadcast as
    a per-class Gaussian; every local step adds an aux batch of class-uniform synthetic points (CE, weight 1).
    form: full | diag | lr<r> (rank-r + diagonal residual) | mean (point mass; prototype-like control)
          | bal (own class-balanced replay, no communication; prior control)
          | own (full Gaussian fitted on the client's OWN rows, no communication; smoothing/regularisation control)."""
    def setup(self):
        R, form = self.R, self.R.a.form
        self.form = form
        if form == 'own':
            self.own = []
            for k in range(R.K):
                n, s1, s2 = class_sums(R.X[k], R.Y[k]); nn_ = n.clamp_min(1)
                mu = s1 / nn_[:, None]
                cov = s2 / nn_[:, None, None] - torch.einsum('ci,cj->cij', mu, mu); cov = 0.5 * (cov + cov.transpose(1, 2))
                L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(D, dtype=cov.dtype, device=cov.device)).float()
                self.own.append((mu.float(), L, torch.nonzero(n).flatten()))
            return
        if form == 'bal':
            self.order = [torch.argsort(y) for y in R.Y]
            self.cnt = [torch.bincount(y, minlength=C) for y in R.Y]
            return
        n = s1 = s2 = 0
        for k in range(R.K):
            a, b, c = class_sums(R.X[k], R.Y[k]); n, s1, s2 = n + a, s1 + b, s2 + c
            second = 0 if form == 'mean' else (D if form == 'diag' else D * (D + 1) // 2)
            R.ship(0, k, 'up', 'class_sums', C * (4 + 4 * D + 4 * second))
        mu = s1 / n[:, None]
        cov = s2 / n[:, None, None] - torch.einsum('ci,cj->cij', mu, mu)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        self.mu = mu.float()
        var = torch.diagonal(cov, dim1=1, dim2=2).clamp_min(1e-8)
        if form == 'full':
            self.L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(D, dtype=cov.dtype, device=cov.device)).float()
            down = C * (D + D * (D + 1) // 2) * 2
        elif form == 'diag':
            self.sd = var.sqrt().float(); down = C * 2 * D * 2
        elif form.startswith('lr'):
            r = int(form[2:])
            ev, U = torch.linalg.eigh(cov)                       # ascending
            ev, U = ev[:, -r:].clamp_min(0), U[:, :, -r:]
            self.U = (U * ev.sqrt()[:, None, :]).float()
            self.sd = (var - (U ** 2 * ev[:, None, :]).sum(2)).clamp_min(1e-8).sqrt().float()
            down = C * (D + D * r + D) * 2
        elif form == 'mean':
            down = C * D * 2
        else:
            raise ValueError(form)
        for k in range(R.K):
            R.ship(0, k, 'down', 'class_sums', down)

    def sample(self, k, m, g):
        dev = self.R.dev
        if self.form == 'bal':
            own = torch.nonzero(self.cnt[k]).flatten()
            y = own[torch.randint(len(own), (m,), device=dev, generator=g)]
            starts = torch.cumsum(self.cnt[k], 0) - self.cnt[k]
            u = torch.rand(m, device=dev, generator=g)
            idx = self.order[k][starts[y] + (u * self.cnt[k][y]).long().clamp_max(self.cnt[k][y] - 1)]
            return self.R.X[k][idx].float(), y
        if self.form == 'own':
            mu, L, own = self.own[k]
            y = own[torch.randint(len(own), (m,), device=dev, generator=g)]
            z = torch.randn(m, D, device=dev, generator=g)
            return mu[y] + (L[y] @ z[:, :, None]).squeeze(-1), y
        y = torch.randint(C, (m,), device=dev, generator=g)
        if self.form == 'mean':
            return self.mu[y], y
        z = torch.randn(m, D, device=dev, generator=g)
        if self.form == 'full':
            return self.mu[y] + (self.L[y] @ z[:, :, None]).squeeze(-1), y
        if self.form == 'diag':
            return self.mu[y] + self.sd[y] * z, y
        zr = torch.randn(m, self.U.shape[2], 1, device=dev, generator=g)
        return self.mu[y] + (self.U[y] @ zr).squeeze(-1) + self.sd[y] * z, y

    def loss(self, k, x, y, g, r):
        xa, ya = self.sample(k, self.R.a.aux_batch, g)
        z = self.R.priv[k](torch.cat([x, xa.float()]))[0].float()  # one forward: BN sees the mixed batch
        return F.cross_entropy(z[:len(y)], y) + self.R.a.aux_weight * F.cross_entropy(z[len(y):], ya)


class LG(Local):
    """H3a / LG-FedAvg: only the last linear layer (latent -> 16) is shared and n_k-averaged; backbone stays private."""
    def setup(self):
        self.glob = copy.deepcopy(last_linear(self.R.init).state_dict()); self.acc = None

    def begin(self, k, r):
        last_linear(self.R.priv[k]).load_state_dict(self.glob)
        self.R.ship(r, k, 'down', 'params', wire_bytes(self.glob))

    def end(self, k, r):
        sd = last_linear(self.R.priv[k]).state_dict()
        self.acc = average(self.acc, sd, self.R.n[k]); self.R.ship(r, k, 'up', 'params', wire_bytes(sd))

    def aggregate(self, r):
        self.glob = finish(self.acc, sum(self.R.n)); self.acc = None

    def eval_models(self):
        return [('private', self.R.priv, True), ('globalhead', with_head(self.R.priv, self.glob), True)]


def with_head(models, sd):
    """Secondary view for shared-head methods: each private backbone with the shared head (what the vehicle holds right
    after its next download). The primary score is always the private model after its local epoch."""
    out = [copy.deepcopy(m) for m in models]
    for m in out:
        last_linear(m).load_state_dict(sd)
    return out


@torch.no_grad()
def local_protos(run, k):
    """Class means of the private latent over the client's own rows (eval mode)."""
    m = run.priv[k]; m.eval()
    X, Y = run.X[k], run.Y[k]
    s = torch.zeros(C, latent_dim(m), device=run.dev); n = torch.zeros(C, device=run.dev)
    for i in range(0, len(Y), 16384):
        with torch.autocast('cuda', dtype=torch.float16):
            h = m(X[i:i + 16384].float())[1].float()
        s.index_add_(0, Y[i:i + 16384], h); n += torch.bincount(Y[i:i + 16384], minlength=C)
    return s / n.clamp_min(1)[:, None], n


class FedGH(Local):
    """H3b / FedGH: clients upload class prototypes; the server trains one global head (Linear latent->16) on them with
    CE and broadcasts it; the client replaces its head and trains the whole model locally."""
    def setup(self):
        self.glob = None; self.P = []

    def begin(self, k, r):
        if self.glob is not None:
            last_linear(self.R.priv[k]).load_state_dict(self.glob)
            self.R.ship(r, k, 'down', 'params', wire_bytes(self.glob))

    def end(self, k, r):
        p, n = local_protos(self.R, k)
        self.P.append((p, n)); self.R.ship(r, k, 'up', 'prototypes', C * (2 * p.shape[1] + 4))

    def aggregate(self, r):
        P = torch.cat([p for p, n in self.P]); N = torch.cat([n for p, n in self.P])
        Y = torch.arange(C, device=self.R.dev).repeat(len(self.P)); keep = N > 0
        head = copy.deepcopy(last_linear(self.R.init)) if self.glob is None else nn.Linear(P.shape[1], C).to(self.R.dev)
        if self.glob is not None:
            head.load_state_dict(self.glob)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(100):
            loss = F.cross_entropy(head(P[keep]), Y[keep]); opt.zero_grad(); loss.backward(); opt.step()
        self.glob = copy.deepcopy(head.state_dict()); self.P = []

    def eval_models(self):
        return [('private', self.R.priv, True), ('globalhead', with_head(self.R.priv, self.glob), True)]


class FedProto(Local):
    """H4 / FedProto: CE + lambda * MSE(latent, global prototype of the label); prototypes count-weighted at the server."""
    def setup(self):
        self.glob = None; self.acc = None; self.nacc = None

    def begin(self, k, r):
        if self.glob is not None:
            self.R.ship(r, k, 'down', 'prototypes', C * 2 * self.glob.shape[1])

    def loss(self, k, x, y, g, r):
        z, h = self.R.priv[k](x)
        l = F.cross_entropy(z.float(), y)
        if self.glob is not None:
            l = l + self.R.a.aux_weight * F.mse_loss(h.float(), self.glob[y])
        return l

    def end(self, k, r):
        p, n = local_protos(self.R, k)
        self.acc = p * n[:, None] if self.acc is None else self.acc + p * n[:, None]
        self.nacc = n if self.nacc is None else self.nacc + n
        self.R.ship(r, k, 'up', 'prototypes', C * (2 * p.shape[1] + 4))

    def aggregate(self, r):
        self.glob = self.acc / self.nacc.clamp_min(1)[:, None]; self.acc = self.nacc = None


class DSFL(Synth):
    """H5a / DS-FL: the open set is synthetic -- drawn from the pooled full Gaussian (H2 statistics, exchanged once)
    with a seed every vehicle shares, so no input crosses the channel and its Gaussian label is NOT used. Each round
    the client uploads int8 probabilities on a shared subset; the server averages and sharpens them (ERA, T=0.1);
    next round the client distils on that subset (soft-label CE) next to its own CE."""
    def setup(self):
        self.R.a.form = 'full'; super().setup()
        g = self.R.gen(99, 99)
        self.open = self.sample(0, self.R.a.open_size, g)[0]
        self.soft = None; self.acc = None

    def subset(self, r):
        return torch.randperm(len(self.open), device=self.R.dev, generator=self.R.gen(98, r))[:self.R.a.open_round]

    def begin(self, k, r):
        if self.soft is not None:
            self.R.ship(r, k, 'down', 'soft_labels', self.soft[1].numel())

    def loss(self, k, x, y, g, r):
        if self.soft is None:
            return self.ce(k, x, y)
        idx, tgt = self.soft
        j = torch.randint(len(idx), (self.R.a.aux_batch,), device=self.R.dev, generator=g)
        z = self.R.priv[k](torch.cat([x, self.open[idx[j]]]))[0].float()
        soft_ce = -(tgt[j] * F.log_softmax(z[len(y):], 1)).sum(1).mean()
        return F.cross_entropy(z[:len(y)], y) + self.R.a.aux_weight * soft_ce

    def end(self, k, r):
        idx = self.subset(r)
        p = F.softmax(self.R.logits(self.R.priv[k], self.open[idx]), 1)
        p = torch.round(p * 255) / 255                           # int8 on the wire
        self.acc = p if self.acc is None else self.acc + p
        self.R.ship(r, k, 'up', 'soft_labels', p.numel())

    def aggregate(self, r):
        avg = (self.acc / self.R.K).clamp_min(1e-8)
        self.soft = (self.subset(r), F.softmax(torch.log(avg) / 0.1, 1)); self.acc = None


class Generator(nn.Module):
    def __init__(self, dz, out):
        super().__init__()
        self.dz = dz
        self.net = nn.Sequential(nn.Linear(dz + C, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, out), nn.ReLU())

    def forward(self, y, g):
        z = torch.randn(len(y), self.dz, device=y.device, generator=g)
        return self.net(torch.cat([z, F.one_hot(y, C).float()], 1))


class FedGen(LG):
    """H5b / FedGen on the LG interface (shared head => one latent meaning): clients upload their head after local
    training; the server fits a generator G(z, y) -> latent so the count-weighted ensemble of client heads labels it y,
    broadcasts G (+ the averaged head); clients add CE(head(G(z, y~pi_g)), y) to their own CE."""
    def setup(self):
        super().setup()
        torch.manual_seed(self.R.a.seed + 2000)
        self.G = Generator(32, latent_dim(self.R.init)).to(self.R.dev); self.ready = False; self.heads = []
        self.pg = torch.exp(self.R.log_pg)

    def begin(self, k, r):
        super().begin(k, r)
        if self.ready:
            self.R.ship(r, k, 'down', 'generator', wire_bytes(self.G.state_dict()))

    def loss(self, k, x, y, g, r):
        l = self.ce(k, x, y)
        if self.ready:
            yg = torch.multinomial(self.pg, self.R.a.aux_batch, replacement=True, generator=g)
            with torch.no_grad():
                hg = self.G(yg, g)
            l = l + self.R.a.aux_weight * F.cross_entropy(last_linear(self.R.priv[k])(hg).float(), yg)
        return l

    def end(self, k, r):
        super().end(k, r)
        self.heads.append(copy.deepcopy(last_linear(self.R.priv[k])).requires_grad_(False))

    def aggregate(self, r):
        super().aggregate(r)
        self.G.train(); opt = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        w = (self.R.counts / self.R.counts.sum(0, keepdim=True).clamp_min(1)).float().to(self.R.dev)   # (K, C)
        g = self.R.gen(97, r)
        for _ in range(100):
            y = torch.multinomial(self.pg, 512, replacement=True, generator=g)
            h = self.G(y, g)
            loss = sum(w[k, y] * F.cross_entropy(head(h), y, reduction='none') for k, head in enumerate(self.heads)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        self.G.eval(); self.heads = []; self.ready = True


METHODS = dict(local=Local, fedavg=FedAvg, dml=DML, synth=Synth, lg=LG, fedgh=FedGH, fedproto=FedProto,
               dsfl=DSFL, fedgen=FedGen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True); p.add_argument('--method', required=True, choices=METHODS)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--private', default='dagsnet'); p.add_argument('--proxy', default='mlp64')
    p.add_argument('--form', default='full'); p.add_argument('--aux_batch', type=int, default=256)
    p.add_argument('--aux_weight', type=float, default=1.0)
    p.add_argument('--open_size', type=int, default=4096); p.add_argument('--open_round', type=int, default=1024)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-5); p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=5); p.add_argument('--smoke', action='store_true')
    p.add_argument('--out', default=str(ROOT / 'papers/pilot/runs'))
    a = p.parse_args()
    if a.smoke:
        a.rounds, a.eval_every = 2, 1
    out = Path(a.out) / f'{a.arm}_s{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = Run(a); m = METHODS[a.method](R); m.setup()
    hist = []
    for r in range(1, a.rounds + 1):
        t1, losses = time.time(), []
        for k in range(R.K):
            m.begin(k, r); losses.append(R.train(m, k, r)); m.end(k, r)
        m.aggregate(r)
        row = dict(round=r, lr=R.lr(r), loss=float(np.mean(losses)), sec=time.time() - t1)
        if r % a.eval_every == 0 or r == a.rounds:
            row['eval'] = R.evaluate(m)
            e = row['eval'][next(iter(row['eval']))]
            print(f"r{r} {a.arm} s{a.seed} loss {row['loss']:.4f} group gp {e['group']['gp']['f1_macro']:.4f} "
                  f"old gp {e['old']['gp']['f1_macro']:.4f} ({row['sec']:.0f}s)", flush=True)
        hist.append(row)
    b = np.array([(r, -1 if k is None else k, d == 'up', n) for r, k, d, kind, n in R.bytes], dtype=np.int64)
    kinds = sorted({kind for *_, kind, n in R.bytes})
    per = {d: {kind: int(sum(n for _, _, dd, kk, n in R.bytes if dd == d and kk == kind)) for kind in kinds} for d in ('up', 'down')}
    res = dict(config=vars(a), history=hist, bytes_total=per,
               bytes_per_client_round={d: float(b[(b[:, 2] == (d == 'up')) & (b[:, 0] > 0), 3].sum() / R.K / a.rounds) for d in ('up', 'down')},
               bytes_once_per_client={d: float(b[(b[:, 2] == (d == 'up')) & (b[:, 0] == 0), 3].sum() / R.K) for d in ('up', 'down')},
               train_rows=R.n, arch=R.arch, minutes=(time.time() - t0) / 60)
    (out / 'result.json').write_text(json.dumps(res) + '\n')
    print('done', a.arm, a.seed, f'{res["minutes"]:.1f} min', flush=True)


if __name__ == '__main__':
    main()
