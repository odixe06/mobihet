"""Generator-structure probe on the sim panel (fold train -> fold val; no training): does the KM16 mixture improve with
  em1/em2   1-2 full EM rounds from the KM16 mixture (responsibilities under each cluster Gaussian; needs the mixture downlinked)
  soft      final pass with soft k-means responsibilities exp(-|z-c|^2 / 2T), T = mean within-cluster variance per class
            from the hard pass (only centres needed: same bytes as KM16)
  white     k-means in the class-whitened space L_c^-1 (z - mu_c) (class Cholesky downlinked once)
Yardstick: held-out LL per fold-val row (probe_km_codec.py); pass mark for a sim arm = mean16 dLL >= +1 nat (K8->K16 = +4).
    python scripts/probe_km_em.py [--n 200] [--out runs/km-codec-probe]  -> <out>/probe_em.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.probe_km_codec import (C, chol, class_cov, heldout_ll, init_centres, load, mixture, moments,  # noqa: E402
                                    tables, to_z)
from scripts.sim_fold_val import fold_counts, pick  # noqa: E402

K = 16


def lloyd_c(z, Y, cen, iters, metric=None):
    for _ in range(iters):
        n = torch.zeros(C, K, dtype=torch.float64)
        s1 = torch.zeros(C, K, z.shape[1], dtype=torch.float64)
        for c in range(C):
            zc = z[Y == c].double()
            u = zc if metric is None else metric(c, zc)
            a = torch.cdist(u, metric(c, cen[c]) if metric else cen[c]).argmin(1)
            n[c].index_add_(0, a, torch.ones(len(a), dtype=torch.float64))
            s1[c].index_add_(0, a, zc)
        cen = torch.where(n[..., None] > 0, s1 / n.clamp_min(1)[..., None], cen)
    return cen


def soft_moments(z, Y, resp):
    """resp(c, zc) -> (n_c, K) responsibilities; returns n, s1, s2 like moments()."""
    Dm = z.shape[1]
    n = torch.zeros(C, K, dtype=torch.float64)
    s1 = torch.zeros(C, K, Dm, dtype=torch.float64)
    s2 = torch.zeros(C, K, Dm, Dm, dtype=torch.float64)
    for c in range(C):
        zc = z[Y == c].double()
        if not len(zc):
            continue
        for i in range(0, len(zc), 1 << 16):
            zb = zc[i:i + (1 << 16)]
            r = resp(c, zb)
            n[c] += r.sum(0)
            s1[c] += r.T @ zb
            for k in range(K):
                s2[c, k] += (zb * r[:, k:k + 1]).T @ zb
    return n, s1, s2


def gauss_resp(p, m, L):
    def f(c, zb):
        lp = []
        for k in range(K):
            if p[c, k] <= 0:
                lp.append(torch.full((len(zb),), -1e30, dtype=torch.float64))
                continue
            sol = torch.linalg.solve_triangular(L[c, k], (zb - m[c, k]).T, upper=False)
            lp.append(torch.log(p[c, k]) - 0.5 * (sol ** 2).sum(0) - torch.log(torch.diagonal(L[c, k])).sum())
        return torch.softmax(torch.stack(lp, 1), 1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="runs/km-codec-probe")
    a = ap.parse_args()
    ids = pick(a.n, cnt=fold_counts())
    X, Y, off = load(ids, "train")
    Xv, Yv, _ = load(ids, "val")
    score, _ = tables(X, Y)
    z, zv = to_z(X, Y, score), to_z(Xv, Yv, score)
    del X, Xv
    mu, cov = class_cov(z, Y)
    res = {}

    def ev(p, m, S, tag):
        ll = heldout_ll(zv, Yv, p, m, chol(S))
        res[tag] = ll.tolist()
        print(f"{tag:10s} LL mean16 {np.nanmean(ll):.4f}  c7 {ll[7]:.3f} c1 {ll[1]:.3f} c6 {ll[6]:.3f} c0 {ll[0]:.3f}", flush=True)

    cen = lloyd_c(z, Y, init_centres(mu, cov, K), 4)
    n, s1, s2, A = moments(z, Y, cen)
    p, m, S = mixture(n, s1, s2, cov, mu)
    ev(p, m, S, "km16")
    # soft k-means final pass, temperature from the hard pass (mean within-cluster variance per class)
    T = torch.stack([torch.einsum("kdd->k", S[c]).mul(p[c]).sum() / z.shape[1] for c in range(C)])
    for tf in (0.5, 1.0):
        def soft(c, zb, tf=tf):
            return torch.softmax(-torch.cdist(zb, cen[c]) ** 2 / (2 * tf * T[c]), 1)
        ev(*mixture(*soft_moments(z, Y, soft), cov, mu), f"soft{tf}")
    # full EM from the KM16 mixture
    pe, me, Se = p, m, S
    for it in (1, 2):
        pe, me, Se = mixture(*soft_moments(z, Y, gauss_resp(pe, me, chol(Se))), cov, mu)
        ev(pe, me, Se, f"em{it}")
    # whitened k-means
    Lc = torch.linalg.cholesky(cov)
    Lci = torch.linalg.inv(Lc)
    metric = lambda c, u: (u - mu[c]) @ Lci[c].T
    cw = lloyd_c(z, Y, init_centres(mu, cov, K), 4, metric)
    Aw = torch.empty(len(Y), dtype=torch.long)
    for c in range(C):
        idx = (Y == c).nonzero().squeeze(1)
        Aw[idx] = torch.cdist(metric(c, z[idx].double()), metric(c, cw[c])).argmin(1)
    nw, s1w, s2w, _ = moments(z, Y, cw, Aw)
    ev(*mixture(nw, s1w, s2w, cov, mu), "white")
    json.dump(res, open(os.path.join(a.out, "probe_em.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
