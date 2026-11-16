"""KM-copula codec probe, part 2 (sim panel fold train -> fold val; no training; see probe_km_codec.py for the yardstick).

    python scripts/probe_km_codec2.py [--n 200] [--out runs/km-codec-probe]

  up_int{8,6,4}   per-vehicle sparse upload of the final pass (b8 codec: count, mean and centred covariance quantised with
                  one scale per vector/matrix), server sum n*(q(cov)+q(m)q(m)^T), PSD projection; zlib ratio of codes
  down_int{8,6,4} per-row-scaled Cholesky codes of the final mixture; zlib ratio of codes
  cls_int8        class moments (Lloyd init + shrinkage target) uploaded int8 per vehicle instead of fp32
Writes <out>/probe2.json.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.probe_km_codec import (C, chol, class_cov, heldout_ll, lloyd, load, mixture, moments, tables,  # noqa: E402
                                    to_z)
from scripts.sim_fold_val import fold_counts, pick  # noqa: E402


def qz(t, bits, dims):
    lv = 2 ** (bits - 1) - 1
    sc = t.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / lv
    q = (t / sc).round().clamp(-lv, lv)
    return q * sc, q


def zratio(chunks):
    raw = np.concatenate([c.astype(np.int8).ravel() for c in chunks]).tobytes()
    return len(zlib.compress(raw, 9)) / len(raw)


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
    Dm = z.shape[1]
    tri = Dm * (Dm + 1) // 2
    il = torch.tril_indices(Dm, Dm)
    mu, cov = class_cov(z, Y)
    K = 16
    cen = lloyd(z, Y, mu, cov, K, 4)
    n, s1, s2, A = moments(z, Y, cen)
    res = {"ll": {}, "zlib": {}, "B": {}}

    def ev(p, m, S, tag):
        ll = heldout_ll(zv, Yv, p, m, chol(S))
        res["ll"][tag] = ll.tolist()
        print(f"{tag:22s} LL mean16 {np.nanmean(ll):.4f}  c7 {ll[7]:.3f} c1 {ll[1]:.3f} c11 {ll[11]:.3f}", flush=True)

    ref = mixture(n, s1, s2, cov, mu)
    ev(*ref, "ref")

    # uplink codecs, per vehicle and non-empty cell
    for bits in (8, 6, 4):
        S2 = torch.zeros_like(s2)
        S1 = torch.zeros_like(s1)
        codes, cells = [], 0
        for v in range(len(off) - 1):
            zv_, yv_, av_ = z[off[v]:off[v + 1]].double(), Y[off[v]:off[v + 1]], A[off[v]:off[v + 1]]
            for c in yv_.unique().tolist():
                for k in av_[yv_ == c].unique().tolist():
                    zk = zv_[(yv_ == c) & (av_ == k)]
                    nk = zk.shape[0]
                    mk = zk.mean(0)
                    ck = zk.T @ zk / nk - mk[:, None] * mk[None]
                    qm, cm_ = qz(mk, bits, (-1,))
                    qc, cc_ = qz(ck, bits, (-2, -1))
                    S1[c, k] += nk * qm
                    S2[c, k] += nk * (qc + qm[:, None] * qm[None])
                    cells += 1
                    if len(codes) < 4000:
                        codes += [cm_.numpy(), cc_[il[0], il[1]].numpy()]
        Sx = S2 / n.clamp_min(1)[..., None, None] - (S1 / n.clamp_min(1)[..., None])[..., :, None] * (S1 / n.clamp_min(1)[..., None])[..., None, :]
        e, U = torch.linalg.eigh((Sx + Sx.transpose(-1, -2)) / 2)
        Sx = U @ torch.diag_embed(e.clamp_min(0)) @ U.transpose(-1, -2)
        s2x = n[..., None, None] * (Sx + (S1 / n.clamp_min(1)[..., None])[..., :, None] * (S1 / n.clamp_min(1)[..., None])[..., None, :])
        ev(*mixture(n, S1, s2x, cov, mu), f"up_int{bits}")
        res["zlib"][f"up_int{bits}"] = zratio(codes)
        res["B"][f"up_int{bits}_sparse_mean"] = cells / (len(off) - 1) * (1 + 4 + (Dm * bits + 7) // 8 + 4 + (tri * bits + 7) // 8 + 4)

    # downlink codecs on the reference mixture
    p, m, S = ref
    L = chol(S)
    for bits in (8, 6, 4):
        Lq, q = qz(L, bits, (-1,))                                   # per-row scale (strict lower + diagonal)
        mq = m.half().double()
        ev(p, mq, Lq @ Lq.transpose(-1, -2) + 1e-6 * torch.eye(Dm, dtype=torch.float64), f"down_int{bits}")
        res["zlib"][f"down_int{bits}"] = zratio([q[..., il[0], il[1]].numpy()])
        res["B"][f"down_int{bits}_per_cluster_raw"] = (tri * bits + 7) // 8 + 2 * Dm + 2 * Dm + 4

    # class moments uploaded int8 per vehicle (init + shrinkage target)
    s1c = torch.zeros(C, Dm, dtype=torch.float64)
    s2c = torch.zeros(C, Dm, Dm, dtype=torch.float64)
    nc = torch.zeros(C, dtype=torch.float64)
    for v in range(len(off) - 1):
        zv_, yv_ = z[off[v]:off[v + 1]].double(), Y[off[v]:off[v + 1]]
        for c in yv_.unique().tolist():
            zc = zv_[yv_ == c]
            mk = zc.mean(0)
            ck = zc.T @ zc / len(zc) - mk[:, None] * mk[None]
            qm, _ = qz(mk, 8, (-1,))
            qc, _ = qz(ck, 8, (-2, -1))
            nc[c] += len(zc); s1c[c] += len(zc) * qm; s2c[c] += len(zc) * (qc + qm[:, None] * qm[None])
    mu8 = s1c / nc[:, None]
    cov8 = s2c / nc[:, None, None] - mu8[:, :, None] * mu8[:, None] + 1e-4 * torch.eye(Dm, dtype=torch.float64)
    e, U = torch.linalg.eigh(cov8)
    cov8 = U @ torch.diag_embed(e.clamp_min(1e-4)) @ U.transpose(-1, -2)
    cen8 = lloyd(z, Y, mu8, cov8, K, 4)
    n8, s18, s28, _ = moments(z, Y, cen8)
    ev(*mixture(n8, s18, s28, cov8, mu8), "cls_int8")
    json.dump(res, open(os.path.join(a.out, "probe2.json"), "w"), indent=1)
    print(json.dumps({"zlib": res["zlib"], "B": res["B"]}, indent=1))


if __name__ == "__main__":
    main()
