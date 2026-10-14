"""P2: knowledge channels that passed P1, under the contact channel (mh/channel.py). Plan: docs/research/pilot-plan.md §5.

Per round, every vehicle: one local epoch with the knowledge it has ACTIVATED so far, then one contact whose true
byte budget comes from the simulator tier. What crosses the contact is an object version split in fragments.

Policies (the H6 question: does contact-aware scheduling keep the P1 gain under a budget?):
  naive  every contact restarts each transfer from byte 0 (no cursor); download first, then upload. An object only
         arrives if it fits in one contact.
  h6     resumable cursor per object version; a version once started is finished before a newer one is fetched/sent
         (coherent snapshot, no restart churn); in-progress transfers go first; the server down-weights stale uploads
         by 1 / (1 + staleness).
Content (form) is a property of the method, the same under both policies.

Methods:
  synth  H2: the vehicle's additive class sums go up once (object `sums`); the RSU publishes the pooled Gaussian as a
         new version each time the contributor count reaches 5 / 10 / 15 / 20 (secure aggregation: never fewer than 5
         clients in a sum); vehicles download the newest version and replay synthetic points from the active one.
  dml    H1: each vehicle keeps a persistent proxy; it downloads the newest global proxy version (replaces its proxy on
         arrival) and uploads a snapshot of its proxy; the RSU averages what arrived in the round into the global.

    python -m mh.p2 --arm h2_full_B0_h6 --method synth --form full --tier B0 --policy h6 --seed 42
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from mh import pilot as P
from mh.channel import budgets, contact, Transfer

SECAGG_MIN = 5


class ChanSynth(P.Synth):
    """H2 over the channel. Server state = sums of the contributors so far; published versions at 5/10/15/20."""
    def setup(self):
        R = self.R; self.form = R.a.form
        self.mine = [P.class_sums(R.X[k], R.Y[k]) for k in range(R.K)]
        full = self.form not in ('diag', 'mean')
        self.up_size = P.C * (4 + 4 * P.D + 4 * (P.D * (P.D + 1) // 2 if full else P.D))
        self.contrib, self.sums = set(), None
        self.versions = {}                   # version -> (contributors, sampler state)
        self.active = [None] * R.K           # version activated by each vehicle
        self.state = {}

    def publish(self):
        n, s1, s2 = self.sums
        mu = s1 / n[:, None]
        cov = s2 / n[:, None, None] - torch.einsum('ci,cj->cij', mu, mu); cov = 0.5 * (cov + cov.transpose(1, 2))
        var = torch.diagonal(cov, dim1=1, dim2=2).clamp_min(1e-8)
        st = dict(mu=mu.float())
        D, C = P.D, P.C
        if self.form == 'full':
            st['L'] = torch.linalg.cholesky(cov + 1e-4 * torch.eye(D, dtype=cov.dtype, device=cov.device)).float()
            size = C * (D + D * (D + 1) // 2) * 2
        elif self.form == 'diag':
            st['sd'] = var.sqrt().float(); size = C * 2 * D * 2
        elif self.form.startswith('lr'):
            r = int(self.form[2:]); ev, U = torch.linalg.eigh(cov); ev, U = ev[:, -r:].clamp_min(0), U[:, :, -r:]
            st['U'] = (U * ev.sqrt()[:, None, :]).float()
            st['sd'] = (var - (U ** 2 * ev[:, None, :]).sum(2)).clamp_min(1e-8).sqrt().float()
            size = C * (D + D * r + D) * 2
        v = len(self.versions) + 1
        self.versions[v] = (len(self.contrib), st, size)

    # ---- channel interface
    def down_object(self, k):
        if not self.versions:
            return None
        v = max(self.versions)
        return None if self.active[k] == v else (('stats', v), self.versions[v][2])

    def up_object(self, k, r):
        return None if k in self.contrib else (('sums', k), self.up_size, k)

    def on_down(self, k, key, data, r):
        self.active[k] = key[1]

    def on_up(self, k, key, data, r, base):
        a, b, c = self.mine[k]
        self.sums = (a, b, c) if self.sums is None else tuple(x + y for x, y in zip(self.sums, (a, b, c)))
        self.contrib.add(k)
        if len(self.contrib) in (5, 10, 15, 20):
            self.publish()

    def end_round(self, r):
        pass

    def base(self, k):
        return 0

    # ---- training uses the vehicle's active version
    def loss(self, k, x, y, g, r):
        v = self.active[k]
        if v is None:
            return self.ce(k, x, y)
        st = self.versions[v][1]
        m, dev = self.R.a.aux_batch, self.R.dev
        ya = torch.randint(P.C, (m,), device=dev, generator=g)
        if self.form == 'mean':
            xa = st['mu'][ya]
        else:
            z = torch.randn(m, P.D, device=dev, generator=g)
            if self.form == 'full':
                xa = st['mu'][ya] + (st['L'][ya] @ z[:, :, None]).squeeze(-1)
            elif self.form == 'diag':
                xa = st['mu'][ya] + st['sd'][ya] * z
            else:
                zr = torch.randn(m, st['U'].shape[2], 1, device=dev, generator=g)
                xa = st['mu'][ya] + (st['U'][ya] @ zr).squeeze(-1) + st['sd'][ya] * z
        zz = self.R.priv[k](torch.cat([x, xa.float()]))[0].float()
        return F.cross_entropy(zz[:len(y)], y) + self.R.a.aux_weight * F.cross_entropy(zz[len(y):], ya)


class ChanDML(P.DML):
    """H1 over the channel: persistent per-vehicle proxy, versioned global proxy."""
    def setup(self):
        super().setup()
        self.prox = [copy.deepcopy(self.glob) for _ in range(self.R.K)]
        self.version, self.vstore = 1, {1: copy.deepcopy(self.glob.state_dict())}
        self.base_of = [1] * self.R.K
        self.bpv = 1 if self.R.a.int8 else 2
        self.size = P.wire_bytes(self.glob.state_dict(), self.bpv)
        self.inbox = []

    def trainable(self, k):
        return [self.R.priv[k], self.prox[k]]

    def begin(self, k, r):
        self.work = self.prox[k]

    def end(self, k, r):
        pass

    def aggregate(self, r):
        pass

    def quant(self, sd):
        if not self.R.a.int8:
            return {k: v.half().float() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        out = {}
        for k, v in sd.items():
            if v.is_floating_point() and v.numel() > 1:
                s = v.abs().max().clamp_min(1e-12) / 127
                out[k] = torch.round(v / s).clamp(-127, 127) * s
            else:
                out[k] = v.clone()
        return out

    def down_object(self, k):
        return None if self.base_of[k] == self.version else (('proxy', self.version), self.size)

    def up_object(self, k, r):
        return (('upd', k, r), self.size, (self.quant(self.prox[k].state_dict()), self.base_of[k]))

    def on_down(self, k, key, data, r):
        v = key[1]
        if v in self.vstore and v > self.base_of[k]:
            self.prox[k].load_state_dict(self.quant(self.vstore[v])); self.base_of[k] = v

    def on_up(self, k, key, data, r, base):
        self.inbox.append(data)

    def end_round(self, r):
        if not self.inbox:
            return
        acc, tot = None, 0.0
        for sd, base in self.inbox:
            w = 1.0 / (1 + self.version - base) if self.R.a.policy == 'h6' else 1.0
            acc = P.average(acc, sd, w); tot += w
        self.glob.load_state_dict(P.finish(acc, tot)); self.inbox = []
        self.version += 1; self.vstore[self.version] = copy.deepcopy(self.glob.state_dict())
        for v in [v for v in self.vstore if v < self.version - 30]:
            del self.vstore[v]


METHODS = dict(synth=ChanSynth, dml=ChanDML)


def run_contact(R, m, k, r, budget, pend, policy):
    """One contact of vehicle k. pend[k] = {'down': Transfer|None, 'up': Transfer|None} carried across contacts."""
    want_down, want_up = m.down_object(k), m.up_object(k, r)
    for d, want in (('down', want_down), ('up', want_up)):
        cur = pend[k][d]
        if policy == 'naive' or cur is None:
            pend[k][d] = None if want is None else Transfer(want[0], want[1], want[2] if len(want) > 2 else None)
            if policy == 'naive' and pend[k][d] is not None:
                pend[k][d].sent = 0
    order = [pend[k]['down'], pend[k]['up']]
    if policy == 'h6':
        order.sort(key=lambda t: 0 if t is not None and t.sent > 0 else 1)   # finish what was started first
    order = [t for t in order if t is not None]
    done, used = contact(order, budget)
    for t in done:
        if t is pend[k]['down']:
            m.on_down(k, t.key, t.data, r); pend[k]['down'] = None
        else:
            m.on_up(k, t.key, t.data, r, None); pend[k]['up'] = None
    return used, len(done)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True); p.add_argument('--method', required=True, choices=METHODS)
    p.add_argument('--tier', required=True, choices=('B0', 'B1', 'B2', 'B3')); p.add_argument('--policy', required=True, choices=('naive', 'h6'))
    p.add_argument('--seed', type=int, default=42); p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--private', default='dagsnet'); p.add_argument('--proxy', default='mlp64'); p.add_argument('--int8', action='store_true')
    p.add_argument('--form', default='full'); p.add_argument('--aux_batch', type=int, default=256)
    p.add_argument('--aux_weight', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-5); p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=5); p.add_argument('--smoke', action='store_true')
    p.add_argument('--out', default=str(P.ROOT / 'papers/p2/runs'))
    a = p.parse_args()
    if a.smoke:
        a.rounds, a.eval_every = 3, 1
    out = Path(a.out) / f'{a.arm}_s{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = P.Run(a); m = METHODS[a.method](R); m.setup()
    btrue, bhat = budgets(a.tier, a.rounds, R.K)
    pend = [{'down': None, 'up': None} for _ in range(R.K)]
    hist = []
    for r in range(1, a.rounds + 1):
        t1, losses, used, done = time.time(), [], [], 0
        for k in range(R.K):
            m.begin(k, r); losses.append(R.train(m, k, r))
            u, n = run_contact(R, m, k, r, btrue[r, k], pend, a.policy); used.append(u); done += n
        m.end_round(r)
        row = dict(round=r, lr=R.lr(r), loss=float(np.mean(losses)), sec=time.time() - t1, bytes_used=float(np.sum(used)),
                   budget=float(btrue[r].sum()), transfers_done=done,
                   active=(sum(v is not None for v in m.active) if hasattr(m, 'active') else sum(b > 1 for b in m.base_of)))
        if r % a.eval_every == 0 or r == a.rounds:
            row['eval'] = R.evaluate(m)
            e = row['eval']['private']
            print(f"r{r} {a.arm} s{a.seed} active {row['active']} group gp {e['group']['gp']['f1_macro']:.4f} "
                  f"old gp {e['old']['gp']['f1_macro']:.4f} bytes {row['bytes_used']/1e3:.0f}/{row['budget']/1e3:.0f} kB ({row['sec']:.0f}s)", flush=True)
        hist.append(row)
    res = dict(config=vars(a), history=hist, minutes=(time.time() - t0) / 60, train_rows=R.n)
    (out / 'result.json').write_text(json.dumps(res) + '\n')
    print('done', a.arm, a.seed, f'{res["minutes"]:.1f} min', flush=True)


if __name__ == '__main__':
    main()
