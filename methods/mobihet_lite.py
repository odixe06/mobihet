"""Lite bootstrap v2 on T3 (rebuild 16/11/2026, docs/research/screening-lite2.md); km2 off = mobihet_compute bit-exact.

  cop_bins   int (256)     bins of the per-class marginal tables on [-4, 4] (histogram uplink, CDF downlink); != 256 needs km2.
  km2        bool (false)  KM mixture through the codec below (needs cop_kmeans > 0, cop_kmeans_q=int8, cop_kmeans_lite=0):
    km2_frac  float (1.0)  class moments (init + shrinkage target) and each Lloyd pass come from a fresh fixed-hash subset of
                           this fraction of the vehicles; the final pass uses every vehicle.
    km2_tau   int (1)      final pass: a (vehicle, class, cluster) cell with < km2_tau rows uploads count + int8 mean only; the
                           component covariance is estimated from the cells that sent a second moment (centred at their mean,
                           shrunk by their rows). Counts/means of every cell still set the mixture weights and means.
    km2_bits  int (6)      downlink Cholesky strict lower part at 8 / 6 / 5 bits, one fp16 scale per row, fp16 diagonal.
    km2_mean8 bool (false) downlink component means int8 + one fp32 scale (default fp16).
  Sampling, marginal decoding, Q, loss and schedule are T3's. With cop_bins=256, km2_frac=1, km2_tau=1, km2_bits=6 the
  mixture equals T3's lite6 (tests/test_mobihet_lite.py); only the byte coders differ.

Bytes are measured per vehicle, printed as KM2_BYTES and written to bootstrap_bytes (means over vehicles):
  up   class counts 4C + histogram as sparse (delta-index, count) varints + class moments (sampled vehicles: per present
       class 13 B header + zlib-6 of its int8 codes) + Lloyd passes it was sampled for and the final pass (per cell uint16
       index + int32 count + fp32 scale, + fp32 scale when a second moment follows; zlib-6 of the vehicle's int8 codes).
  down class counts 4C + CDF (uint16 grid as adjacent differences, best of lzma/bz2/zlib, lossless) + int8 centres of its
       own classes for each pass it takes part in + kept clusters' mixture (weight fp32, mean, fp16 diagonal and row
       scale, Cholesky codes best of lzma/bz2/zlib).
"""
import bz2
import json
import lzma
import zlib

import numpy as np
import torch

from methods.mobihet_compute import Method as Compute


def pack(raw):
    """Smallest of three standard lossless coders + 1 byte naming the coder."""
    return 1 + min(len(lzma.compress(raw, preset=9)), len(bz2.compress(raw, 9)), len(zlib.compress(raw, 6)))


def varint_len(v):
    return 1 + sum((v >= (1 << (7 * k))).long() for k in range(1, 5))


class Method(Compute):
    def _copula_bootstrap(self, client, NB=256):
        hp = self.env.hp
        nb = int(hp.get("cop_bins", 256))
        assert nb in (64, 128, 256) and (nb == 256 or hp.get("km2", False)), nb
        super()._copula_bootstrap(client, nb)

    def _kmeans(self, K, iters):
        hp = self.env.hp
        if not hp.get("km2", False):
            return super()._kmeans(K, iters)
        from core import data as D_
        env, X, Y, dev = self.env, self.env.train.X, self.env.train.Y, self.env.dev
        C, D = self.cop_mu.shape
        NB = self.cop_cdf.shape[-1]
        frac, tau = float(hp.get("km2_frac", 1.0)), int(hp.get("km2_tau", 1))
        bits, mean8 = int(hp.get("km2_bits", 6)), bool(hp.get("km2_mean8", False))
        assert hp.get("cop_kmeans_q") == "int8" and not hp.get("cop_kmeans_lite", 0) and self.w_boot == "compact"
        assert 0 < frac <= 1 and tau >= 1 and bits in (5, 6, 8)
        off_knobs = dict(cop_kmeans_part=1.0, cop_kmeans_cov="full", cop_kmeans_white=False, cop_kmeans_em=0,
                         cop_kmeans_temp=1.0, synth_gapfill=False, cop_pop=0, q_cluster=False, km_chol_bits=0,
                         km_cls_q="fp32", km_cen_q="exact", km_bytes=False)
        assert all(hp.get(k, v) == v for k, v in off_knobs.items()), "km2 replaces the older KM codec knobs"
        score = torch.special.ndtri((self.cop_cdf - self.cop_mass / 2).clamp(1e-4, 1 - 1e-4))   # decoded tables, as clients have
        f = torch.arange(D, device=dev)
        bins = lambda x: ((x.float() + 4) * (NB / 8)).long().clamp(0, NB - 1)
        tri = D * (D + 1) // 2
        il = torch.tril_indices(D, D, device=dev)
        off = env.train.off.tolist()
        nv = len(off) - 1
        gid = env.train.ids.long().cpu()
        f64 = dict(dtype=torch.float64, device=dev)

        def sampled(p):                                                  # pass p: the same subset on every rank
            if frac == 1:
                return [True] * nv
            return torch.from_numpy(np.random.default_rng(20261116 + p).random(D_.N_CLIENTS) < frac)[gid].tolist()

        def quant(t, codes):                                             # int8, one fp32 scale per vector / matrix
            dims = (-1,) if t.dim() == 1 else (-2, -1)
            sc = t.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / 127
            q = (t / sc).round().clamp(-127, 127)
            codes.append((q if t.dim() == 1 else q[il[0], il[1]]).to(torch.int8))
            return q * sc

        def zlen(codes):
            return len(zlib.compress(torch.cat(codes).cpu().numpy().tobytes(), 6)) if codes else 0

        B = {k: torch.zeros(nv, dtype=torch.float64) for k in ("hist", "cls", "lloyd", "final", "cen")}   # per-vehicle bytes
        ncls = torch.zeros(nv, dtype=torch.float64)
        cells = torch.zeros(3, **f64)                                    # final pass: cells, cells with a second moment, rows in them

        def rows(v):
            x, y = X[off[v]:off[v + 1]], Y[off[v]:off[v + 1]].long()
            return x, y, score[y[:, None], f, bins(x)].double()

        # ---- class moments (init directions + shrinkage target) from the sampled vehicles ----
        n = torch.zeros(C, **f64)
        s1 = torch.zeros(C, D, **f64)
        s2 = torch.zeros(C, D, D, **f64)
        sel = sampled(0)
        for v in range(nv):
            x, y, z = rows(v)
            idx = (((y[:, None] * D + f) * NB + bins(x)).reshape(-1)).unique(return_counts=True)
            B["hist"][v] = float((varint_len(torch.diff(idx[0], prepend=idx[0].new_zeros(1))) + varint_len(idx[1])).sum())
            cls = y.unique().tolist()
            ncls[v] = len(cls)
            if not sel[v]:
                continue
            codes = []
            for c in cls:
                zc = z[y == c]
                nk = zc.shape[0]
                mk = quant(zc.mean(0), codes)
                ck = quant(zc.T @ zc / nk - zc.mean(0)[:, None] * zc.mean(0)[None], codes)
                n[c] += nk; s1[c] += nk * mk; s2[c] += nk * (ck + mk[:, None] * mk[None])
            B["cls"][v] = len(cls) * (1 + 4 + 8) + zlen(codes)
        n, s1, s2 = env.allreduce(n), env.allreduce(s1), env.allreduce(s2)
        present = self.cop_cdf[:, 0, -1] > 0
        assert bool((n[present] >= 2 * D).all()), f"km2_frac={frac}: a class has < {2 * D} rows in the sampled vehicles: {n.tolist()}"
        mu0 = s1 / n.clamp_min(1)[:, None]
        cov = s2 / n.clamp_min(1)[:, None, None] - mu0[:, :, None] * mu0[:, None, :]
        e, U = torch.linalg.eigh((cov + cov.transpose(-1, -2)) / 2)
        cov = U @ torch.diag_embed(e.clamp_min(0)) @ U.transpose(-1, -2) + 1e-4 * torch.eye(D, **f64)
        ev, V = torch.linalg.eigh(cov)
        h = K // 2
        dirs = (V[..., -h:] * ev[:, None, -h:].clamp_min(0).sqrt()).transpose(-1, -2)
        cen = torch.cat([mu0[:, None] + dirs, mu0[:, None] - dirs], 1)
        if K % 2:
            cen = torch.cat([mu0[:, None], cen], 1)

        def bcast(cen):                                                  # int8 centres as the vehicles decode them
            sc = cen.abs().amax(-1, keepdim=True).clamp_min(1e-12).float().double() / 127
            return (cen / sc).round().clamp(-127, 127) * sc

        def cell_pass(cen, sel, final):
            n = torch.zeros(C, K, **f64)
            s1 = torch.zeros(C, K, D, **f64)
            s2 = torch.zeros(C, K, D, D, **f64) if final else None
            n_p, s1_p = torch.zeros_like(n), torch.zeros_like(s1)        # cells that sent a second moment
            key = "final" if final else "lloyd"
            for v in range(nv):
                if not sel[v]:
                    continue
                x, y, z = rows(v)
                codes, nc, nc2 = [], 0, 0
                for c in y.unique().tolist():
                    zc = z[y == c]
                    a = torch.cdist(zc, cen[c]).argmin(1)
                    for k in a.unique().tolist():
                        zk = zc[a == k]
                        nk = zk.shape[0]
                        mk = quant(zk.mean(0), codes)
                        n[c, k] += nk; s1[c, k] += nk * mk
                        nc += 1
                        if final and nk >= tau:
                            ck = quant(zk.T @ zk / nk - zk.mean(0)[:, None] * zk.mean(0)[None], codes)
                            s2[c, k] += nk * (ck + mk[:, None] * mk[None])
                            n_p[c, k] += nk; s1_p[c, k] += nk * mk
                            nc2 += 1
                            cells[2] += nk
                B[key][v] += nc * (2 + 4 + 4) + nc2 * 4 + zlen(codes)
                B["cen"][v] += ncls[v] * K * (D + 4)
                if final:
                    cells[0] += nc; cells[1] += nc2
            n, s1 = env.allreduce(n), env.allreduce(s1)
            if final:
                return n, s1, env.allreduce(s2), env.allreduce(n_p), env.allreduce(s1_p)
            return n, s1

        for i in range(iters):
            n, s1 = cell_pass(bcast(cen), sampled(1 + i), False)
            cen = torch.where(n[..., None] > 0, s1 / n.clamp_min(1)[..., None], cen)
        n, s1, s2, nS, s1p = cell_pass(bcast(cen), [True] * nv, True)
        keep = n >= 20
        m = s1 / n.clamp_min(1)[..., None]
        mp = s1p / nS.clamp_min(1)[..., None]
        S = s2 / nS.clamp_min(1)[..., None, None] - mp[..., :, None] * mp[..., None, :]
        e, U = torch.linalg.eigh((S + S.transpose(-1, -2)) / 2)         # quantised moments: server projects to PSD
        S = U @ torch.diag_embed(e.clamp_min(0)) @ U.transpose(-1, -2)
        w0 = float(hp.get("cop_kmeans_w0", 2.0)) * D
        S = (nS[..., None, None] * S + w0 * cov[:, None]) / (nS[..., None, None] + w0) + 1e-4 * torch.eye(D, **f64)
        p = torch.where(keep, n, 0.)
        p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
        self.km_p = (p / p.sum(1, keepdim=True)).float()
        mu = m.where(keep[..., None], mu0[:, None].expand_as(m))
        if mean8:
            sc = mu.abs().amax(-1, keepdim=True).clamp_min(1e-12).float().double() / 127
            self.km_mu = ((mu / sc).round().clamp(-127, 127) * sc).float()
        else:
            self.km_mu = mu.half().float()
        L = torch.linalg.cholesky(S)
        lv = 2 ** (bits - 1) - 1
        off_d = L.masked_fill(torch.eye(D, dtype=torch.bool, device=dev), 0)
        sc = (off_d.abs().amax(-1, keepdim=True) / lv).clamp_min(1e-4).half().double()   # fp16-safe (row 0 has no off-diagonal)
        q = (off_d / sc).round().clamp(-lv, lv)
        diag = torch.diagonal(L, dim1=-2, dim2=-1).half().double()
        assert bool((diag > 0).all()), "fp16 Cholesky diagonal not positive"
        self.km_L = (q * sc + torch.diag_embed(diag)).float()
        self.km_kept = kept = int(keep.sum())

        # ---- bytes ----
        sl = torch.tril_indices(D, D, -1, device=dev)
        chol = pack(q[keep][:, sl[0], sl[1]].to(torch.int8).cpu().numpy().tobytes())
        mix = kept * (4 + (D + 4 if mean8 else 2 * D) + 2 * 2 * D) + chol
        codes = torch.round(self.cop_cdf.double() * 65535).cpu().numpy().astype(np.int64)
        cdf = 28 + pack(np.diff(codes, axis=-1, prepend=0).astype("<u2").tobytes())
        up_v = 4 * C + B["hist"] + B["cls"] + B["lloyd"] + B["final"]
        down_v = 4 * C + cdf + B["cen"] + mix
        red = env.allreduce(torch.stack([up_v.sum(), down_v.sum(), torch.tensor(float(nv), dtype=torch.float64)]
                                        + [B[k].sum() for k in ("hist", "cls", "lloyd", "final", "cen")]).to(dev)).cpu()
        cells = env.allreduce(cells).cpu()
        N = float(red[2])
        rec = {"bins": NB, "K": K, "iters": iters, "frac": frac, "tau": tau, "bits": bits, "mean8": mean8, "vehicles": int(N),
               "kept": kept, "up_mean": float(red[0] / N), "down_mean": float(red[1] / N),
               "up": {"counts": 4 * C, **{k: float(red[3 + i] / N) for i, k in enumerate(("hist", "cls", "lloyd", "final"))}},
               "down": {"counts": 4 * C, "cdf": cdf, "centres": float(red[7] / N), "mixture": mix, "chol_codes": chol},
               "broadcast_copy": 4 * C + cdf + (iters + 1) * C * K * (D + 4) + mix,
               "final_cells_mean": float(cells[0] / N), "final_cells_s2_mean": float(cells[1] / N),
               "s2_row_cover": float(cells[2] / n.sum())}
        rec["total_mean"] = rec["up_mean"] + rec["down_mean"]
        self.bootstrap_bytes = {"up": rec["up_mean"], "down": rec["down_mean"]}
        self.codec_bytes = rec
        print(f"KM-copula K={K}: kept clusters {kept}/{C * K}", flush=True)
        print("KM2_BYTES " + json.dumps(rec), flush=True)
