"""A1 diagnostic: posterior of the CBB generator p_gen(c|x) on synthetic rows and on real fold-val rows (sim panel).

    python scripts/run_local_guarded.py python scripts/diag_gen_posterior.py [--n 200] [--per-class 4096]

Builds the generator exactly as the sim bootstrap (CBB hp: KM16 w0 .5, lite6, int8) on the sim panel's fold-train rows,
no training, no official test. Gate locked in docs/research/method-building-brief.md §5 (A1).
Output: runs/a1-gen-posterior/diag.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import data as D  # noqa: E402
from scripts import sim_fold_val as S  # noqa: E402

CBB = ("cop_kmeans=16 cop_kmeans_w0=0.5 cop_kmeans_q=int8 cop_kmeans_lite=6 q_bits=8 km_bytes=true").split()
NB, LO, W = 256, -4.0, 8.0 / 256


class GenPosterior:
    """log f_c(x) for every class c under the decoded generator tables the vehicles hold."""

    def __init__(self, M, floor=1e-7):
        self.cdf = M.cop_cdf.double()                                    # (C, D, NB) cumulative up to bin end
        self.mass = M.cop_mass.double()
        self.logm = (self.mass.clamp_min(floor) / W).log()
        self.p = M.km_p.double()                                         # (C, K)
        self.mu = M.km_mu.double()                                       # (C, K, D)
        L = M.km_L.double()
        self.Linv = torch.linalg.inv(L)                                  # (C, K, D, D) lower-triangular inverse
        self.logdet = torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)  # (C, K)
        self.logp = torch.where(self.p > 0, self.p.log(), torch.full_like(self.p, -math.inf))

    def logf(self, x, mid=False):
        """mid=True: real rows scored at the bin mid-rank, as the generator statistics were fitted (atoms of real data
        would otherwise sit at a bin edge, far outside the fitted latent spread); synthetic rows: exact PIT."""
        x = x.double()
        R, Dm = x.shape
        C = self.cdf.shape[0]
        j = ((x - LO) / W).floor().long().clamp(0, NB - 1)                # (R, D)
        frac = torch.full_like(x, .5) if mid else ((x - LO) / W - j).clamp(0, 1)
        out = torch.empty(R, C, dtype=torch.float64, device=x.device)
        fidx = torch.arange(Dm, device=x.device)
        for c in range(C):
            hi = self.cdf[c][fidx, j]                                    # (R, D)
            m = self.mass[c][fidx, j]
            u = (hi - m + frac * m).clamp(1e-6, 1 - 1e-6)
            z = torch.special.ndtri(u)
            jac = (self.logm[c][fidx, j] + 0.5 * z * z + 0.5 * math.log(2 * math.pi)).sum(-1)   # log(m/W) - log phi(z)
            d = z[:, None, :] - self.mu[c][None]                         # (R, K, D)
            y = torch.einsum("kde,rke->rkd", self.Linv[c], d)
            lg = -0.5 * (y * y).sum(-1) - self.logdet[c] - 0.5 * Dm * math.log(2 * math.pi) + self.logp[c]
            out[:, c] = torch.logsumexp(lg, -1) + jac
        return out


def sample(M, per_class, gen):
    C, K, Dm = M.km_mu.shape
    lab = torch.arange(C, device=M.km_mu.device).repeat_interleave(per_class)
    k = torch.multinomial(M.km_p[lab], 1, replacement=True, generator=gen).squeeze(1)
    z = M.km_mu[lab, k] + (M.km_L[lab, k] @ torch.randn(len(lab), Dm, 1, device=lab.device, generator=gen)).squeeze(-1)
    x = torch.cat([M._inv(z[None, i:i + 1024], lab[None, i:i + 1024])[0] for i in range(0, len(lab), 1024)])
    return x, lab


def chunked(P, x, bs=1024, mid=False):
    return torch.cat([P.logf(x[i:i + bs], mid) for i in range(0, len(x), bs)])


def report(y, pred, C=16):
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
    present = cm.sum(1) > 0
    return {"accuracy": float(tp.sum() / cm.sum()), "f1_macro": float(f1[present].mean()),
            "recall": np.round(rec, 4).tolist(), "precision": np.round(prec, 4).tolist(), "f1": np.round(f1, 4).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-class", type=int, default=4096)
    ap.add_argument("--out", default="runs/a1-gen-posterior")
    a = ap.parse_args()
    cnt = S.fold_counts()
    ids = S.pick(a.n, cnt=cnt)
    orig = D.load
    D.load = lambda root, split, clients=None, cache=None, folds=None, with_fold=False: orig(
        root, "train", clients, cache, folds=["train"] if split == "train" else ["val"])
    from core import fl
    from core.dagsnet import width
    from scripts.run_fl import PROTOCOL
    fl.Env._sizes = staticmethod(lambda root, cache: cnt[:, 0].sum(1))
    hp = dict(PROTOCOL, rounds=20)
    for s in open("runs/sim/core_hp.txt").read().split() + CBB:
        k, v = s.split("=", 1)
        try:
            hp[k] = json.loads(v)
        except json.JSONDecodeError:
            hp[k] = v
    env = fl.Env(width(1.0), hp, clients=ids.tolist(), cache=os.environ.get("V4_CACHE", "cache"))
    from methods.mobihet_sim import Method
    M = Method(env)
    P = GenPosterior(M)
    gen = torch.Generator(device=env.dev).manual_seed(123)
    out = {"ids_sha": __import__("hashlib").sha256(ids.astype(np.int64).tobytes()).hexdigest()[:16],
           "hp": {k: hp[k] for k in sorted(hp)}, "km_kept": int(M.km_kept)}
    with torch.no_grad():
        xs, ls = sample(M, a.per_class, gen)
        lf = chunked(P, xs.float())
        post = torch.softmax(lf, -1)                                     # uniform prior: synthetic rows are class-balanced
        py = post.gather(1, ls[:, None]).squeeze(1)
        ent = -(post * post.clamp_min(1e-30).log()).sum(-1)
        y = ls.cpu().numpy()
        syn = {"frac_py_lt_0.9": float((py < .9).float().mean()), "frac_py_lt_0.5": float((py < .5).float().mean()),
               "per_class_frac_py_lt_0.9": [float((py[ls == c] < .9).float().mean()) for c in range(16)],
               "per_class_mean_py": [float(py[ls == c].mean()) for c in range(16)],
               "mean_entropy_nats": float(ent.mean()), "argmax": report(y, post.argmax(-1).cpu().numpy())}
        # where the posterior mass of each synthetic class goes (mean posterior row): the row-level soft target content
        syn["mean_posterior"] = [np.round(post[ls == c].mean(0).cpu().numpy(), 4).tolist() for c in range(16)]
        out["synthetic"] = syn
        Xv, Yv = env.test.X, env.test.Y.long()
        lv = chunked(P, Xv.float())
        prior = torch.bincount(env.train.Y.long(), minlength=16).double()
        prior = (prior + 1) / (prior.sum() + 16)
        yv = Yv.cpu().numpy()
        out["real_val_uniform_prior"] = report(yv, lv.argmax(-1).cpu().numpy())
        out["real_val_global_prior"] = report(yv, (lv + prior.log().to(lv.device)).argmax(-1).cpu().numpy())
        out["real_val_finite_frac"] = float(torch.isfinite(lv).all(-1).double().mean())
        lm = chunked(P, Xv.float(), mid=True)
        out["real_val_midrank_uniform_prior"] = report(yv, lm.argmax(-1).cpu().numpy())
        out["real_val_midrank_global_prior"] = report(yv, (lm + prior.log().to(lm.device)).argmax(-1).cpu().numpy())
        out["rows"] = {"synthetic": int(len(ls)), "real_val": int(len(Yv))}
    td = out["real_val_midrank_uniform_prior"]["recall"][13]
    out["gate"] = {"synthetic_soft_ge_2pct": out["synthetic"]["frac_py_lt_0.9"] >= .02, "real_td_recall_ge_0.42": td >= .42}
    out["gate"]["pass"] = all(out["gate"].values())
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "diag.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("gate", "rows", "km_kept", "real_val_finite_frac")}))
    print("synthetic frac p<.9", round(syn["frac_py_lt_0.9"], 4), "per class", np.round(syn["per_class_frac_py_lt_0.9"], 3).tolist())
    for k in ("real_val_uniform_prior", "real_val_midrank_uniform_prior", "real_val_midrank_global_prior"):
        r = out[k]
        print(k, "acc", round(r["accuracy"], 4), "macroF1", round(r["f1_macro"], 4), "recall", r["recall"])
    print("synthetic argmax acc", round(syn["argmax"]["accuracy"], 4), "recall", syn["argmax"]["recall"])


if __name__ == "__main__":
    main()
