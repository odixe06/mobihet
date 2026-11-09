"""KM-copula codec/byte probe on the simulator panel (fold train -> fold val, never the official test). No training.

    python scripts/probe_km_codec.py [--n 200] [--out runs/km-codec-probe]

Rebuilds Core copula tables + KM16 (w0 .5D, 4 Lloyd) exactly as methods/mobihet_sim.py on the sim vehicles' fold train,
then measures, against the uncompressed mixture:
  occupancy   per-vehicle non-empty (class, cluster) cells -> sparse uplink bytes; row mass in small cells
  downlink    codecs of per-cluster (mean, Cholesky): fp16 (current), int8 per-row scale, PPCA-R in own eigenbasis
  lloyd       2 vs 4 Lloyd rounds (bytes of the extra rounds)
  subsample   second moments from a random fraction p of vehicles only (counts/means/CDF from all)
  classK      per-class K in {1,4,8,16,32,48}
Fidelity yardstick = held-out mean log-likelihood per fold-val row in the normal-score space, per class (offline only,
not part of the method; LL/TV is not FL utility, LESSONS). Writes <out>/probe.json.
"""
import argparse
import json
import lzma
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import data as D  # noqa: E402
from scripts.sim_fold_val import fold_counts, pick  # noqa: E402

C, NB = 16, 256
torch.set_num_threads(min(8, os.cpu_count() or 1))


def load(ids, fold):
    s = D.load(D.find_root(), "train", ids, cache="cache", folds=[fold])
    return torch.from_numpy(np.asarray(s.X, np.float32)), torch.from_numpy(np.asarray(s.Y)).long(), np.asarray(s.off)


def tables(X, Y):
    Dm = X.shape[1]
    f = torch.arange(Dm)
    bins = ((X + 4) * (NB / 8)).long().clamp(0, NB - 1)
    hist = torch.zeros(C * Dm * NB, dtype=torch.float64)
    hist.index_add_(0, ((Y[:, None] * Dm + f) * NB + bins).reshape(-1), torch.ones(bins.numel(), dtype=torch.float64))
    hist = hist.view(C, Dm, NB)
    mass = hist / hist.sum(-1, keepdim=True).clamp_min(1)
    cdf = mass.cumsum(-1)
    # Core downlink (compact): CDF on a uint16 grid, scores from decoded tables (as clients and _kmeans use them)
    cdf = torch.round(cdf * 65535) / 65535
    cdf[..., -1] = torch.where(hist.sum(-1) > 0, 1., 0.)
    mass = torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1]))
    score = torch.special.ndtri((cdf - mass / 2).clamp(1e-4, 1 - 1e-4)).float()
    return score, hist


def to_z(X, Y, score):
    f = torch.arange(X.shape[1])
    bins = ((X + 4) * (NB / 8)).long().clamp(0, NB - 1)
    return score[Y[:, None], f, bins]


def moments(z, Y, cen, a=None):
    """n (C,K), s1 (C,K,D), s2 (C,K,D,D) of hard assignment to nearest centre; a = precomputed assignment."""
    K, Dm = cen.shape[1], z.shape[1]
    n = torch.zeros(C, K, dtype=torch.float64)
    s1 = torch.zeros(C, K, Dm, dtype=torch.float64)
    s2 = torch.zeros(C, K, Dm, Dm, dtype=torch.float64)
    A = torch.empty(len(Y), dtype=torch.long) if a is None else a
    for c in range(C):
        idx = (Y == c).nonzero().squeeze(1)
        if not len(idx):
            continue
        zc = z[idx].double()
        if a is None:
            A[idx] = torch.cdist(zc, cen[c]).argmin(1)
        ac = A[idx]
        n[c].index_add_(0, ac, torch.ones(len(ac), dtype=torch.float64))
        s1[c].index_add_(0, ac, zc)
        for k in ac.unique().tolist():
            zk = zc[ac == k]
            s2[c, k] += zk.T @ zk
    return n, s1, s2, A


def class_cov(z, Y):
    Dm = z.shape[1]
    mu = torch.zeros(C, Dm, dtype=torch.float64)
    cov = torch.zeros(C, Dm, Dm, dtype=torch.float64)
    for c in range(C):
        zc = z[Y == c].double()
        mu[c] = zc.mean(0)
        cov[c] = (zc - mu[c]).T @ (zc - mu[c]) / len(zc) + 1e-4 * torch.eye(Dm, dtype=torch.float64)
    return mu, cov


def init_centres(mu, cov, K):
    ev, V = torch.linalg.eigh(cov)
    h = K // 2
    dirs = (V[..., -h:] * ev[:, None, -h:].clamp_min(0).sqrt()).transpose(-1, -2)
    cen = torch.cat([mu[:, None] + dirs, mu[:, None] - dirs], 1)
    if K % 2:
        cen = torch.cat([mu[:, None], cen], 1)
    return cen


def lloyd(z, Y, mu, cov, K, iters):
    cen = init_centres(mu, cov, K)
    for _ in range(iters):
        n = torch.zeros(C, K, dtype=torch.float64)
        s1 = torch.zeros(C, K, z.shape[1], dtype=torch.float64)
        for c in range(C):
            zc = z[Y == c].double()
            a = torch.cdist(zc, cen[c]).argmin(1)
            n[c].index_add_(0, a, torch.ones(len(a), dtype=torch.float64))
            s1[c].index_add_(0, a, zc)
        cen = torch.where(n[..., None] > 0, s1 / n.clamp_min(1)[..., None], cen)
    return cen


def mixture(n, s1, s2, cov, mu, w0f=0.5):
    """Same estimator as _kmeans (mode full, no quantisation): returns p (C,K), m (C,K,D), S (C,K,D,D)."""
    Dm = s1.shape[-1]
    keep = n >= 20
    m = s1 / n.clamp_min(1)[..., None]
    S = s2 / n.clamp_min(1)[..., None, None] - m[..., :, None] * m[..., None, :]
    w0 = w0f * Dm
    S = (n[..., None, None] * S + w0 * cov[:, None]) / (n[..., None, None] + w0) + 1e-4 * torch.eye(Dm, dtype=S.dtype)
    p = torch.where(keep, n, 0.)
    p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
    p = p / p.sum(1, keepdim=True)
    m = m.where(keep[..., None], mu[:, None].expand_as(m))
    return p, m, S


def heldout_ll(zv, Yv, p, m, L):
    """Mean log-likelihood per val row by class under the mixture (L = Cholesky factors as the client samples)."""
    out = np.zeros(C)
    for c in range(C):
        zc = zv[Yv == c].double()
        if not len(zc):
            out[c] = np.nan
            continue
        ks = (p[c] > 0).nonzero().squeeze(1)
        lp = []
        for k in ks.tolist():
            d = (zc - m[c, k]).T
            sol = torch.linalg.solve_triangular(L[c, k], d, upper=False)
            ld = 2 * torch.log(torch.diagonal(L[c, k])).sum()
            lp.append(torch.log(p[c, k]) - 0.5 * (sol ** 2).sum(0) - 0.5 * ld - 0.5 * zc.shape[1] * np.log(2 * np.pi))
        out[c] = torch.logsumexp(torch.stack(lp), 0).mean().item()
    return out


def chol(S):
    return torch.linalg.cholesky((S + S.transpose(-1, -2)) / 2)


def q_rows_int8(L):
    """int8 lower-triangular Cholesky, one fp16 scale per row."""
    sc = L.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    sc = sc.half().double()
    q = (L / sc).round().clamp(-127, 127)
    return q * sc, q


def ppca(S, R):
    e, U = torch.linalg.eigh(S)
    Dm = S.shape[-1]
    lam, Ur = e[..., -R:], U[..., -R:]
    sig = e[..., :Dm - R].mean(-1, keepdim=True).clamp_min(1e-4)
    return Ur @ torch.diag_embed(lam - sig) @ Ur.transpose(-1, -2) + sig[..., None] * torch.eye(Dm, dtype=S.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="runs/km-codec-probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cnt = fold_counts()
    ids = pick(a.n, cnt=cnt)
    X, Y, off = load(ids, "train")
    Xv, Yv, _ = load(ids, "val")
    print("rows train/val", len(Y), len(Yv), flush=True)
    score, hist = tables(X, Y)
    z, zv = to_z(X, Y, score), to_z(Xv, Yv, score)
    del X, Xv
    mu, cov = class_cov(z, Y)
    Dm, tri = z.shape[1], z.shape[1] * (z.shape[1] + 1) // 2
    res = {"ids_n": len(ids), "rows_train": int(len(Y)), "rows_val": int(len(Yv)), "val_rows_per_class": np.bincount(Yv.numpy(), minlength=C).tolist()}

    def evaluate(p, m, S, tag):
        L = chol(S)
        ll = heldout_ll(zv, Yv, p, m, L)
        res.setdefault("ll", {})[tag] = ll.tolist()
        print(f"{tag:28s} LL mean16 {np.nanmean(ll):.4f}  c7 {ll[7]:.3f} c2 {ll[2]:.3f} c1 {ll[1]:.3f}", flush=True)
        return ll

    # Core single Gaussian (K=1) and the reference KM16 w0 .5, 4 Lloyd
    one = torch.ones(C, 1, dtype=torch.float64)
    evaluate(one, mu[:, None], cov[:, None], "core_K1")
    K = 16
    cen4 = lloyd(z, Y, mu, cov, K, 4)
    n, s1, s2, A = moments(z, Y, cen4)
    p, m, S = mixture(n, s1, s2, cov, mu)
    ref = evaluate(p, m, S, "km16_ref")

    # occupancy per vehicle (uplink sparsity) and row mass in small cells
    cells, small = [], {1: 0, 8: 0, 32: 0, 78: 0}
    for v in range(len(off) - 1):
        key = (Y[off[v]:off[v + 1]] * K + A[off[v]:off[v + 1]]).numpy()
        u, c = np.unique(key, return_counts=True)
        cells.append(len(u))
        for t in small:
            small[t] += int(c[c <= t].sum())
    cells = np.array(cells)
    per_cell_int8 = 4 + Dm + 4 + tri + 4                               # count, mean+scale, cov+scale (b8 codec)
    res["occupancy"] = {"cells_mean": float(cells.mean()), "cells_median": float(np.median(cells)),
                        "cells_p10": float(np.percentile(cells, 10)), "cells_p90": float(np.percentile(cells, 90)),
                        "dense_cells": C * K, "dense_up_B": C * K * per_cell_int8,
                        "sparse_up_B_mean": float((cells * (per_cell_int8 + 1)).mean()),        # +1 B cell index
                        "rows_in_cells_le": {str(t): v / len(Y) for t, v in small.items()}}
    print("occupancy", res["occupancy"], flush=True)

    # downlink codecs (what every vehicle receives): current fp16 L, int8 L per-row scale, PPCA-R
    L = chol(S)
    Lh = L.half().double()
    res.setdefault("down", {})["fp16"] = {"B_per_cluster": 2 * (Dm + tri)}
    evaluate(p, m.half().double(), Lh @ Lh.transpose(-1, -2), "down_fp16")
    Lq, q = q_rows_int8(L)
    mq = m.half().double()
    codes = q[..., torch.tril_indices(Dm, Dm)[0], torch.tril_indices(Dm, Dm)[1]].to(torch.int8).numpy().tobytes()
    res["down"]["int8_rows"] = {"B_per_cluster": tri + 2 * Dm + 2 * Dm, "zlib9_ratio": len(zlib.compress(codes, 9)) / len(codes),
                                "lzma_ratio": len(lzma.compress(codes)) / len(codes)}
    evaluate(p, mq, Lq @ Lq.transpose(-1, -2), "down_int8_rows")
    for R in (8, 16, 24, 32, 48):
        Sp = ppca(S, R)
        res["down"][f"ppca{R}"] = {"B_per_cluster_fp16": 2 * (Dm + R * Dm + R + 1), "B_per_cluster_int8": Dm + R * Dm + 2 * R + 2 * R + 2 + 2 * Dm}
        evaluate(p, mq, Sp, f"down_ppca{R}")
    e = torch.linalg.eigvalsh(S)
    frac = (e.flip(-1).cumsum(-1) / e.sum(-1, keepdim=True))
    res["spectrum_R_for_var"] = {str(t): float((frac < t).sum(-1).double().add(1).mean()) for t in (0.9, 0.95, 0.99)}
    print("spectrum", res["spectrum_R_for_var"], flush=True)

    # Lloyd rounds: 1 and 2 instead of 4
    for it in (1, 2):
        c_ = lloyd(z, Y, mu, cov, K, it)
        n_, s1_, s2_, _ = moments(z, Y, c_)
        evaluate(*mixture(n_, s1_, s2_, cov, mu), f"lloyd{it}")

    # second moments from a random fraction of vehicles (counts/means/centres from all)
    g = np.random.default_rng(0)
    veh = np.repeat(np.arange(len(off) - 1), np.diff(off))
    for frac_v in (0.1, 0.25, 0.5):
        sel = torch.from_numpy(np.isin(veh, g.choice(len(off) - 1, int(round(frac_v * (len(off) - 1))), replace=False)))
        ns, s1s, s2s, _ = moments(z[sel], Y[sel], cen4, A[sel])
        Ss = s2s / ns.clamp_min(1)[..., None, None] - (s1s / ns.clamp_min(1)[..., None])[..., :, None] * (s1s / ns.clamp_min(1)[..., None])[..., None, :]
        # mixture with full counts/means, second moment from the subset (cells absent in the subset fall back to class cov)
        m_full = s1 / n.clamp_min(1)[..., None]
        Ss = torch.where((ns >= 2)[..., None, None], Ss, cov[:, None].expand_as(Ss))
        w0 = 0.5 * Dm
        Sf = (ns[..., None, None] * Ss + w0 * cov[:, None]) / (ns[..., None, None] + w0) + 1e-4 * torch.eye(Dm, dtype=Ss.dtype)
        evaluate(p, m_full.where((n >= 20)[..., None], mu[:, None].expand_as(m_full)), Sf, f"subsample{frac_v}")
        res.setdefault("subsample_rows", {})[str(frac_v)] = int(sel.sum())

    # per-class K (independent fits; the column of class c is what K would give that class)
    for k_ in (4, 8, 32, 48):
        c_ = lloyd(z, Y, mu, cov, k_, 4)
        n_, s1_, s2_, _ = moments(z, Y, c_)
        evaluate(*mixture(n_, s1_, s2_, cov, mu), f"K{k_}")
    res["ll_ref_minus_core"] = (np.array(res["ll"]["km16_ref"]) - np.array(res["ll"]["core_K1"])).tolist()
    json.dump(res, open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, "probe.json"))


if __name__ == "__main__":
    main()
