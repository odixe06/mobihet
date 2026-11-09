"""Simulator candidates on top of Core (WLive stream); every knob off reproduces Core.

  synth_need_T  int  (0 = off)  A: a synthetic row of class c gets zero weight at a vehicle holding >= T real rows of c
                                (the vehicle would not generate it: compute <= Core). Per-row weight of the kept rows unchanged.
  synth_cf      bool             counterfactual twins from the vehicle's own real rows of the step, same copula statistics:
                                class c in CF (attacks that modify one field family S_c, benign twin 100 % in train fold):
                                  attack row = a real benign row of the batch with S_c resampled from the class-c copula
                                  conditional on the other features (latent Gaussian, randomized PIT of the kept values);
                                benign synthetic row = a real CF-attack row of the batch with its S_c resampled from the
                                benign conditional. No suitable real row in the batch -> plain copula row (Core).
  synth_weight  float (1.0)      fixed weight of the synthetic KL term (Core: 1) at vehicles with >= synth_weight_min_rows
  synth_weight_min_rows int (0)  fold-train rows (0 = every vehicle); compute and bytes = Core, 1 local epoch kept.
  oracle_shuffle "none"|"feature"|"block"  diagnostic only (synth=oracle, pooled real rows of the rank, not deployable):
                                each feature / each physical block (BLOCKS) of a synthetic row comes from an independent real
                                row of the same class -> what dependence a generator must keep to reach the oracle.
  cop_kmeans    int (0 = off)    KM-copula: per class a K-component Gaussian mixture in the copula normal-score space, centres by
                                federated k-means (init mu_c +- sqrt(eig) v from the pooled class covariance: no rows leave a vehicle;
                                cop_kmeans_iters Lloyd rounds of per-cluster sums/counts), component covariance shrunk to the class
                                covariance (prior weight 2D rows), clusters < 20 rows dropped. Marginals and sampling = Core
                                (one cluster draw per row); extra bootstrap bytes counted in bootstrap_bytes.
  cop_kmeans_cov "full"|"within"|"diag"  component covariance: full per-cluster second moment (default); "within" = pooled
                                within-cluster covariance cov_c - between-centre scatter, from counts/sums only (no second-moment
                                pass bytes); "diag" = per-cluster variances along the class eigenvectors (D floats per cluster);
                                "pcaR" = full per-cluster covariance inside the top-R class eigenvectors (R(R+1)/2 floats), the
                                pooled within covariance elsewhere.
  cop_kmeans_q  "none"|"fp16"|"int8"  per-vehicle upload codec (full mode): each vehicle sends per-cluster count (int32), mean
                                and centred covariance quantised (fp16, or int8 with one fp32 scale per vector/matrix); the
                                server sums n*(q(cov) + q(mean) q(mean)^T). Simulated per vehicle, not per rank.
  cop_kmeans_lite int (0 = off) KM-lite codec (needs cop_kmeans_q=int8): class moments for init/shrinkage uploaded int8 per
                                vehicle (b8 codec) instead of fp32; centres broadcast int8 (one fp32 scale per centre); downlink
                                Cholesky factors with `lite` bits (8 or 6) and one fp16 scale per row, means fp16; class
                                mean/Cholesky not downlinked (clients sample only the mixture); CDF downlink lossless zlib (L50).
                                Uplink sent sparse (non-empty cells, uint16 index) and zlib-compressed. Bytes measured, printed
                                as KM_LITE_BYTES and written into bootstrap_bytes (whole bootstrap, Core part included).
  cop_kmeans_part float (1.0)   fraction of vehicles (fixed hash of the global id) that upload the final second moments;
                                every vehicle still uploads counts/means. Needs cop_kmeans_q=int8.
  cop_pop       int (0 = off)   simulator only: generator statistics (CDF, class moments, KM) from the fold-train rows of `cop_pop`
                                vehicles (-1 = all 5343; else a fixed random subset, always including the sim vehicles), split
                                over ranks; training, LA prior, Q and evaluation stay on the sim vehicles. Mimics the row volume of
                                the full-scale network behind each statistic; never reads fold val or the official test.
b11 (09/11, docs/research/improvement-proposals.md; all off = the LT6 trajectory, no RNG drawn by any of them):
  ph_grid       bool (false)    evaluation only: extra views ph025/ph050/ph075/ph100 = own-test CM of argmax(z + a * la_shift), a in
                                {.25,.5,.75,1}: the posterior under the vehicle's own (Laplace) prior instead of the network prior the
                                logit-adjusted CE trains toward. 0 bytes, no training change; "private" stays a = 0.
  w_ema         float (0 = off) per-vehicle weight average over rounds w_ema <- b w_ema + (1 - b) w (BN buffers too), started at r1;
                                "private" = own-test CM of the average, "raw" = of the trained weights; training continues from w.
  q_cluster     bool (false)    KM cluster-conditioned teacher: real rows are assigned (deterministic mid-rank score, nearest kept
                                centre of their class) and private probabilities summed per (class, cluster); a vehicle uploads the
                                int8 mean + count of each cell with >= q_cluster_min rows (smaller cells only enter the class Q:
                                no per-row statistic leaves the vehicle); synthetic rows of cluster k take Q[y, k], empty cells Q[y].
  q_cluster_min int (8)
  q_temp        float (1.0)     synthetic target Q^(1/T) renormalised (T > 1 softer), applied at the vehicle; 0 bytes.
b12 (09/11): synth_rows_sim int (0 = off) synthetic rows per step (multiple of 16, balanced; loss weight unchanged: mean over rows);
                                set after init because the Live parent asserts 64 for its bank variants (unused on the stream path).
  fuse_sessions str ("" = off)  evaluation only: session sidecar (scripts/build_sessions.py, keyed (vehicle, world, pseudonym)); extra
                                views fuse004/fuse016/fuseall = causal sum of the session's last 4 / 16 / all log-probs up to each message.
                                Per-message "private" view unchanged; same rows, same labels; no training change, 0 bytes.
codec study (09/11, docs/research/screening-codec.md; every knob off = KM50 bit-exact, needs cop_kmeans_q=int8):
  km_chol_bits  int (0 = fp16)  downlink cluster Cholesky: strict lower part at 8 / 6 bits with one fp16 scale per row, fp16 diagonal.
  km_cls_q      "fp32"|"int8"   class moments (k-means init + shrinkage target) from per-vehicle int8 uploads instead of Core fp32.
  km_cen_q      "exact"|"int8"  centres the vehicles assign with: KM50 simulated exact centres (accounted fp16) or int8 per centre.
  km_bytes      bool (false)    exact per-vehicle bootstrap accounting (KM_CODEC_BYTES): sparse cells + zlib, lossless CDF, no class
                                downlink, unicast and broadcast copy, fp16-centre assignment flips. Accounting only.
  q_bits        int (16)        Q packets both ways: fp16 (16) or p*(2^b-1) integers (8, 4); uplink only the present classes.
  (cop_kmeans_iters, existing: Lloyd passes; cop_kmeans_lite = the bundle cls int8 + centres int8 + Cholesky bits.)
S_c: fixed public table of field families (features each attack type changes), measured on fold-train benign twins of one
scenario (FAMILIES below); 0 extra bytes, same 64 synthetic rows per step (compute = Core + a
77x77 matvec per synthetic row).
"""
import json
import os

import torch

from methods.mobihet_wstudy import WLive

# S_c (input index, knowledge/input_order.json) per class with benign twin rate 1.0; frozen from
# runs/transport-probe/twins-highway_2.json (fold train, changed_frac > .5) by scripts/probe_attack_twins.py.
FAMILIES = {1: [1, 2, 19, 34, 48, 63],
            2: [3, 4, 5, 9, 10, 23, 24, 41, 43, 44, 45, 46, 49, 50, 51, 55, 56, 67, 68, 70],
            3: [0, 5, 6, 7, 18, 22, 23, 32, 33, 36, 37, 40, 51, 52, 66, 67],
            6: [1, 2, 19, 34, 48, 63],
            7: [3, 4, 5, 9, 10, 18, 20, 22, 23, 24, 32, 33, 35, 36, 37, 40, 41, 43, 44, 45, 46, 49, 50, 51, 55, 56, 61, 62,
                64, 66, 67, 68, 70],
            10: [5, 6, 8, 20, 22, 23, 35, 36, 37, 44, 45, 46, 51, 52, 66, 67],
            11: [0, 5, 6, 7, 18, 19, 22, 23, 32, 33, 34, 36, 37, 40, 47, 51, 52, 53, 62, 63, 66, 67],
            12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43,
                 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66, 67, 68, 70, 72, 73, 75],
            15: [0, 5, 6, 7, 18, 19, 22, 23, 32, 33, 34, 36, 37, 40, 47, 51, 52, 53, 62, 63, 66, 67]}


# physical-quantity blocks of the 77 inputs (prefixes p4_std_/win_/ses_ stripped): acl, spd, hed, pos+map, time/session, nbr/traffic
BLOCKS = [[1, 2, 19, 34, 48, 63],
          [0, 5, 6, 7, 17, 18, 32, 33, 40, 47, 51, 52, 53, 61, 62],
          [8, 20, 21, 35, 38, 54, 64, 65, 72, 73, 75],
          [3, 4, 9, 10, 22, 23, 24, 36, 37, 43, 44, 45, 46, 49, 50, 55, 56, 66, 67, 68],
          [11, 12, 13, 14, 15, 16, 25, 39, 57, 58, 59, 60, 71],
          [26, 27, 28, 29, 30, 31, 41, 42, 69, 70, 74, 76]]
assert sorted(i for b in BLOCKS for i in b) == list(range(77))


def family_sets(path=None):
    """Recompute FAMILIES from a twins json (check only); the method uses the frozen table."""
    if path is None:
        return FAMILIES
    here = os.path.dirname(os.path.abspath(__file__))
    lab = json.load(open(os.path.join(here, "..", "knowledge", "label_mapping.json")))["label_to_id"]
    return {lab[n]: [i for i, v in enumerate(o["changed_frac"]) if v > .5]
            for n, o in json.load(open(path)).items() if o.get("changed_frac") and o["twin_rate"] >= 0.999}


class Method(WLive):
    def __init__(self, env):
        super().__init__(env)
        hp = env.hp
        sr = int(hp.get("synth_rows_sim", 0))                           # b12: synthetic rows per step (Live asserts 64)
        if sr:
            assert sr % 16 == 0 and not hp.get("synth_benign", 0), sr
            self.aux, self.labels = sr, torch.arange(sr, device=env.dev) % 16
        self.need_T = int(hp.get("synth_need_T", 0))
        self.cf = bool(hp.get("synth_cf", False))
        self.sw = float(hp.get("synth_weight", 1.0))
        self.sw_min = int(hp.get("synth_weight_min_rows", 0))
        self.shuffle = hp.get("oracle_shuffle", "none")
        assert self.shuffle in {"none", "feature", "block"} and (self.shuffle == "none" or self.synth == "oracle")
        if self.cf:
            self._cf_setup()
        big = (env.n >= self.sw_min).float()
        self.sw_k = 1 + (self.sw - 1) * big                             # (K,) synthetic weight per local client
        sb = int(hp.get("synth_benign", 0))
        if sb:                                                           # S1: sb benign rows of the 64, the rest equal per attack
            assert self.aux == 64 and 0 < sb < 64 and (64 - sb) % 15 == 0, sb
            rest = torch.arange(1, 16, device=env.dev).repeat((64 - sb) // 15)
            self.labels = torch.cat([torch.zeros(sb, dtype=torch.long, device=env.dev), rest])
        self.q_src = hp.get("q_source", "real")
        assert self.q_src in {"real", "synth"}
        self.lab_rows = torch.bincount(self.labels, minlength=16).float()
        if self.shuffle == "block":
            blk = torch.zeros(77, dtype=torch.long)
            for j, b in enumerate(BLOCKS):
                blk[b] = j
            self.blk = blk.to(env.dev)
        self.ph = bool(hp.get("ph_grid", False))
        self.ema_beta = float(hp.get("w_ema", 0.0))
        self.qk = bool(hp.get("q_cluster", False))
        self.qk_min = int(hp.get("q_cluster_min", 8))
        self.q_temp = float(hp.get("q_temp", 1.0))
        assert 0 <= self.ema_beta < 1 and self.q_temp > 0 and self.qk_min >= 1
        self.ema_p = self.ema_b = self.Qk = None
        self.q_bits = int(hp.get("q_bits", 16))                          # codec study: Q packets fp16 (16) or uint8/uint4 p*(2^b-1)
        assert self.q_bits in (16, 8, 4) and (self.q_bits == 16 or self.q_src == "real"), self.q_bits
        self.fuse_path = str(hp.get("fuse_sessions", ""))
        if self.fuse_path:
            self._fuse_setup()
        if self.qk:
            assert getattr(self, "km", 0) and self.q_src == "real" and not self.need_T, "q_cluster: KM generator, real-row Q"
            self._qk_score = torch.special.ndtri((self.cop_cdf - self.cop_mass / 2).clamp(1e-4, 1 - 1e-4))
            self._qk_f = torch.arange(self._qk_score.shape[1], device=env.dev)
            self._qk_dead = self.km_p <= 0                              # dropped clusters: never assigned, never sampled

    def _copula_bootstrap(self, client, NB=256):
        pop = int(self.env.hp.get("cop_pop", 0))
        own = self.env.train
        if pop:
            self.env.train = self._pop_split(pop)
        try:
            super()._copula_bootstrap(client, NB)
            K = int(self.env.hp.get("cop_kmeans", 0))
            self.km = K
            if K:
                self._kmeans(K, int(self.env.hp.get("cop_kmeans_iters", 4)))
        finally:
            if pop:
                self.env.train = own
                torch.cuda.empty_cache()

    def _pop_split(self, pop):
        """This rank's share of the population vehicles' fold-train rows (sim driver maps split train -> fold train)."""
        import numpy as np
        from core import data as D
        env = self.env
        ids = np.arange(D.N_CLIENTS)
        if pop > 0 and pop < D.N_CLIENTS:
            sim = env.all_ids.cpu().numpy() if torch.is_tensor(env.all_ids) else np.asarray(env.all_ids)
            rest = np.setdiff1d(ids, sim)
            extra = np.random.default_rng(11).choice(rest, max(0, pop - len(sim)), replace=False)
            ids = np.sort(np.concatenate([sim, extra]))
        mine = ids[env.rank::env.world]
        s = D.load(D.find_root(), "train", mine.tolist(), None).to(env.dev)
        print(f"cop_pop rank{env.rank}: {len(s.ids)} vehicles, {int(s.off[-1])} fold-train rows for generator statistics", flush=True)
        return s

    def _kmeans(self, K, iters):
        env, X, Y = self.env, self.env.train.X, self.env.train.Y
        C, D = self.cop_mu.shape
        dev = env.dev
        score = torch.special.ndtri((self.cop_cdf - self.cop_mass / 2).clamp(1e-4, 1 - 1e-4))   # decoded tables, as clients have
        f = torch.arange(D, device=dev)
        bins = lambda x: ((x.float() + 4) * (256 / 8)).long().clamp(0, 255)
        tri = D * (D + 1) // 2
        il = torch.tril_indices(D, D, device=dev)
        qm = str(self.env.hp.get("cop_kmeans_q", "none"))
        lite = int(self.env.hp.get("cop_kmeans_lite", 0))
        part = float(self.env.hp.get("cop_kmeans_part", 1.0))
        assert lite in (0, 6, 8) and 0 < part <= 1 and (qm == "int8" or not (lite or part < 1)), (lite, part, qm)
        hp = self.env.hp                                                 # codec study 09/11: lite's pieces one by one
        cls8 = bool(lite) or hp.get("km_cls_q", "fp32") == "int8"
        cen8 = bool(lite) or hp.get("km_cen_q", "exact") == "int8"
        bits = lite or int(hp.get("km_chol_bits", 0))
        acct = bool(hp.get("km_bytes", False))
        assert hp.get("km_cls_q", "fp32") in ("fp32", "int8") and hp.get("km_cen_q", "exact") in ("exact", "int8")
        assert bits in (0, 6, 8) and (qm == "int8" or not (cls8 or cen8 or bits or acct)) and (lite or not bits or acct), bits
        off = env.train.off.tolist()
        nv = len(off) - 1
        SAMPLE = 64                                                      # vehicles per rank whose codes are zlib-measured
        meas = {k: torch.zeros(2, dtype=torch.float64, device=dev) for k in ("cls", "lloyd", "final")}   # summed (hdr, code) B
        zs = {"cls": [], "final": []}                                    # sampled int8 code bytes
        acc = {k: [0.0] * nv for k in ("cls", "lloyd", "final", "ncls")} if acct else None   # exact per-vehicle B
        flips = torch.zeros(2, dtype=torch.float64, device=dev)          # rows whose centre changes under fp16 centres, rows
        if part < 1:
            import numpy as np
            from core import data as D_
            pick = torch.from_numpy(np.random.default_rng(20261109).random(D_.N_CLIENTS) < part)
            is_part = pick[env.train.ids.long().cpu()].tolist()
        else:
            is_part = [True] * nv

        def quant(t, codes=None):                                        # last dim (vector) or last two dims (matrix)
            if qm == "fp16":
                return t.half().double()
            dims = (-1,) if t.dim() == 1 else (-2, -1)
            sc = t.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / 127
            q = (t / sc).round().clamp(-127, 127)
            if codes is not None:
                codes.append((q if t.dim() == 1 else q[il[0], il[1]]).to(torch.int8))
            return q * sc

        if cls8:                                                         # class moments int8 per vehicle (init + shrinkage)
            mu0, cov = self._cls_int8(X, Y, off, score, f, bins, quant, meas, zs, SAMPLE, acc)
        else:
            mu0 = self.cop_mu.double()
            cov = (self.cop_L.double() @ self.cop_L.double().transpose(-1, -2))
        ev, V = torch.linalg.eigh(cov)                                   # ascending
        h = K // 2
        dirs = (V[..., -h:] * ev[:, None, -h:].clamp_min(0).sqrt()).transpose(-1, -2)          # (C, h, D)
        cen = torch.cat([mu0[:, None] + dirs, mu0[:, None] - dirs], 1)  # (C, K, D)
        if K % 2:
            cen = torch.cat([mu0[:, None], cen], 1)
        up = down = 0

        def bcast(cen):                                                  # centres as the vehicles decode them
            if not cen8:
                return cen
            sc = cen.abs().amax(-1, keepdim=True).clamp_min(1e-12).float().double() / 127
            return (cen / sc).round().clamp(-127, 127) * sc

        mode = str(self.env.hp.get("cop_kmeans_cov", "full"))
        R = int(mode[3:]) if mode.startswith("pca") else 0
        assert mode in ("full", "within", "diag") or 0 < R < D, mode
        Vr = V[..., -R:] if R else None                                  # (C, D, R) top-R class eigenvectors
        assert qm in ("none", "fp16", "int8") and (qm == "none" or mode == "full"), (qm, mode)
        pstat = {}
        white = bool(self.env.hp.get("cop_kmeans_white", False))
        assert not white or (not lite and qm == "int8"), "whitened assignment: int8 per-vehicle path, class Cholesky downlink"
        Wi = torch.linalg.inv(torch.linalg.cholesky(cov)) if white else None

        def nearest(c, zc, cc):                                          # assignment metric (Euclid or class-whitened)
            if white:
                return torch.cdist((zc - mu0[c]) @ Wi[c].T, (cc - mu0[c]) @ Wi[c].T).argmin(1)
            return torch.cdist(zc, cc).argmin(1)

        def stats_q(cen, second):
            n = torch.zeros(C, K, dtype=torch.float64, device=dev)
            s1 = torch.zeros(C, K, D, dtype=torch.float64, device=dev)
            s2 = torch.zeros(C, K, D, D, dtype=torch.float64, device=dev) if second else None
            n_p, s1_p = torch.zeros_like(n), torch.zeros_like(s1)        # participants only (second-moment estimate)
            key = "final" if second else "lloyd"
            if second:
                self._km_nloc = torch.zeros(nv, C, K, device=dev)          # own rows per (class, cluster): gap-fill
            cen16 = cen.half().double() if (acct and not cen8) else None    # KM50 accounted fp16 centres, simulated exact ones
            for v in range(nv):
                x, y = X[off[v]:off[v + 1]], Y[off[v]:off[v + 1]].long()
                z = score[y[:, None], f, bins(x)].double()
                cells = 0
                samp = bool(second and bits and v < SAMPLE and is_part[v])
                codes = [] if (samp or acct) else None
                ys = y.unique().tolist()
                for c in ys:
                    zc = z[y == c]
                    a = nearest(c, zc, cen[c])
                    if cen16 is not None:
                        flips.add_(torch.tensor([float((nearest(c, zc, cen16[c]) != a).sum()), float(len(a))],
                                                dtype=torch.float64, device=dev))
                    for k in a.unique().tolist():
                        zk = zc[a == k]
                        nk = zk.shape[0]
                        if second:
                            self._km_nloc[v, c, k] = nk
                        mk = quant(zk.mean(0), codes)
                        n[c, k] += nk
                        s1[c, k] += nk * mk
                        cells += 1
                        if second and is_part[v]:
                            ck = quant(zk.T @ zk / nk - zk.mean(0)[:, None] * zk.mean(0)[None], codes)
                            s2[c, k] += nk * (ck + mk[:, None] * mk[None])
                            n_p[c, k] += nk
                            s1_p[c, k] += nk * mk
                # sparse cell: uint16 index + int32 count + fp32 scale + D int8 (+ fp32 scale + tri int8 if second moment)
                sec = bool(second and is_part[v])
                meas[key] += torch.tensor([cells * (2 + 4 + 4 + (4 if sec else 0)), cells * (D + (tri if sec else 0))],
                                          dtype=torch.float64, device=dev)
                if samp:
                    zs["final"].extend(codes)
                if acct:                                                 # sparse cells: header + zlib-6 of this vehicle's codes
                    acc[key][v] += cells * (2 + 4 + 4 + (4 if sec else 0)) + self._zlen(codes)
                    acc["ncls"][v] = float(len(ys))
            n, s1 = env.allreduce(n), env.allreduce(s1)
            if second:
                s2 = env.allreduce(s2)
                pstat["n"], pstat["s1"] = env.allreduce(n_p), env.allreduce(s1_p)
            return n, s1, s2

        def stats(cen, second=False):
            if qm != "none":
                return stats_q(cen, bool(second))
            n = torch.zeros(C, K, dtype=torch.float64, device=dev)
            s1 = torch.zeros(C, K, D, dtype=torch.float64, device=dev)
            s2 = None
            if second == "full":
                s2 = torch.zeros(C, K, D, D, dtype=torch.float64, device=dev)
            elif second == "diag":
                s2 = torch.zeros(C, K, D, dtype=torch.float64, device=dev)
            elif second == "pca":
                s2 = torch.zeros(C, K, R, R, dtype=torch.float64, device=dev)
            for i in range(0, len(Y), 1 << 18):
                x, y = X[i:i + (1 << 18)], Y[i:i + (1 << 18)].long()
                z = score[y[:, None], f, bins(x)].double()
                for c in y.unique().tolist():
                    zc = z[y == c]
                    a = torch.cdist(zc, cen[c]).argmin(1)
                    n[c].index_add_(0, a, torch.ones_like(a, dtype=torch.float64))
                    s1[c].index_add_(0, a, zc)
                    if second == "full":
                        for k in a.unique().tolist():
                            zk = zc[a == k]
                            s2[c, k] += zk.T @ zk
                    elif second == "diag":
                        s2[c].index_add_(0, a, (zc @ V[c]) ** 2)
                    elif second == "pca":
                        u = zc @ Vr[c]
                        for k in a.unique().tolist():
                            uk = u[a == k]
                            s2[c, k] += uk.T @ uk
            n, s1 = env.allreduce(n), env.allreduce(s1)
            if s2 is not None:
                s2 = env.allreduce(s2)
            return n, s1, s2

        for _ in range(iters):
            n, s1, _ = stats(bcast(cen))
            cen = torch.where(n[..., None] > 0, s1 / n.clamp_min(1)[..., None], cen)
            up += C * K * (4 + {"none": 4 * D, "fp16": 2 * D, "int8": D + 4}[qm]); down += 2 * C * K * D
        n, s1, s2 = stats(bcast(cen), second=("pca" if R else mode) if mode != "within" else False)
        keep = n >= 20
        m = s1 / n.clamp_min(1)[..., None]
        nS = n                                                           # rows behind each second-moment estimate
        if mode == "full":
            if part < 1:                                                 # centred at the participants' mean
                nS = pstat["n"]
                mp = pstat["s1"] / nS.clamp_min(1)[..., None]
                S = s2 / nS.clamp_min(1)[..., None, None] - mp[..., :, None] * mp[..., None, :]
            else:
                S = s2 / n.clamp_min(1)[..., None, None] - m[..., :, None] * m[..., None, :]
            extra = tri
            if qm != "none":                                             # quantised moments: server projects to PSD
                e, U = torch.linalg.eigh((S + S.transpose(-1, -2)) / 2)
                S = U @ torch.diag_embed(e.clamp_min(0)) @ U.transpose(-1, -2)
        elif mode == "diag":                                             # variances along class eigenvectors V[c]
            var = (s2 / n.clamp_min(1)[..., None] - (m @ V) ** 2).clamp_min(1e-4)
            S = V[:, None] @ torch.diag_embed(var) @ V[:, None].transpose(-1, -2)
            extra = D
        else:                                                            # pooled within = class cov - between-centre scatter
            pk = n / n.sum(1, keepdim=True).clamp_min(1)
            dm = m - mu0[:, None]
            Wc = cov - torch.einsum("ck,ckd,cke->cde", pk, dm, dm)
            e, U = torch.linalg.eigh((Wc + Wc.transpose(-1, -2)) / 2)
            Wc = U @ torch.diag_embed(e.clamp_min(1e-3)) @ U.transpose(-1, -2)
            S = Wc[:, None].expand(C, K, D, D).clone()
            extra = 0
            if R:                                                        # cluster covariance inside the top-R subspace
                mr = m @ Vr                                              # (C, K, R)
                Sr = s2 / n.clamp_min(1)[..., None, None] - mr[..., :, None] * mr[..., None, :]
                Vk = Vr[:, None]                                         # (C, 1, D, R)
                Wr = Vk.transpose(-1, -2) @ Wc[:, None] @ Vk
                S = S + Vk @ (Sr - Wr) @ Vk.transpose(-1, -2)
                S = (S + S.transpose(-1, -2)) / 2
                e, U = torch.linalg.eigh(S)
                S = U @ torch.diag_embed(e.clamp_min(1e-3)) @ U.transpose(-1, -2)
                extra = R * (R + 1) // 2
        w0 = float(self.env.hp.get("cop_kmeans_w0", 2.0)) * D
        S = (nS[..., None, None] * S + w0 * cov[:, None]) / (nS[..., None, None] + w0) + 1e-4 * torch.eye(D, device=dev, dtype=S.dtype)
        p = torch.where(keep, n, 0.)
        p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))          # class with no kept cluster: uniform (never empty)
        self.km_p = (p / p.sum(1, keepdim=True)).float()
        self.km_mu = m.where(keep[..., None], mu0[:, None].expand_as(m)).half().float()
        hp = self.env.hp
        em = int(hp.get("cop_kmeans_em", 0))
        assert em == 0 or (qm == "int8" and not lite and part == 1 and mode == "full"), "EM: int8 per-vehicle path only"
        for _ in range(em):        # vehicles receive the mixture (fp16), upload responsibility-weighted int8 moments
            n, s1, s2 = self._em_stats(X, Y, off, score, f, bins, quant, self.km_p.double(), self.km_mu.double(),
                                       torch.linalg.cholesky(S).half().double())
            up += C * K * (4 + D + 4 + tri + 4); down += 4 * C * K + 2 * C * K * (D + tri)
            keep = n >= 20
            m = s1 / n.clamp_min(1)[..., None]
            S = s2 / n.clamp_min(1)[..., None, None] - m[..., :, None] * m[..., None, :]
            e, U = torch.linalg.eigh((S + S.transpose(-1, -2)) / 2)
            S = U @ torch.diag_embed(e.clamp_min(0)) @ U.transpose(-1, -2)
            S = (n[..., None, None] * S + w0 * cov[:, None]) / (n[..., None, None] + w0) + 1e-4 * torch.eye(D, device=dev, dtype=S.dtype)
            p = torch.where(keep, n, 0.)
            p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
            self.km_p = (p / p.sum(1, keepdim=True)).float()
            self.km_mu = m.where(keep[..., None], mu0[:, None].expand_as(m)).half().float()
        tau = float(hp.get("cop_kmeans_temp", 1.0))
        if tau != 1.0:                                                   # cluster draw prob ~ n^tau (rare modes more)
            p = torch.where(keep, n, 0.).pow(tau)
            p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
            self.km_p = (p / p.sum(1, keepdim=True)).float()
        if bool(hp.get("synth_gapfill", False)):                         # per vehicle: favour clusters its own rows lack
            assert qm == "int8" and not int(hp.get("cop_pop", 0)) and hasattr(self, "_km_nloc")
            g = self.km_p[None]
            tot = self._km_nloc.sum(-1, keepdim=True)
            loc = self._km_nloc / tot.clamp_min(1)
            w = torch.where(tot > 0, torch.maximum(g - loc, 0.1 * g), g.expand_as(loc))
            self.km_pv = (w / w.sum(-1, keepdim=True).clamp_min(1e-12)).float()
        if not bits:
            self.km_L = torch.linalg.cholesky(S).half().float()
            up += (4 * C * K * (1 + D + extra) if qm == "none" else
                   C * K * (4 + (2 * (D + extra) if qm == "fp16" else D + 4 + extra + 4))); down += 4 * C * K + 2 * C * K * (D + extra)
            self.bootstrap_bytes["up"] += up; self.bootstrap_bytes["down"] += down
        else:
            self._lite_down(torch.linalg.cholesky(S), bits, iters, meas, zs)
        self.km_kept = int(keep.sum())
        print(f"KM-copula K={K}: kept clusters {self.km_kept}/{C * K}, extra bootstrap up {up} B down {down} B", flush=True)
        if acct:
            self._codec_acct(torch.linalg.cholesky(S), bits, iters, cls8, cen8, acc, flips, keep)

    @staticmethod
    def _zlen(codes):
        import zlib
        return len(zlib.compress(torch.cat(codes).cpu().numpy().tobytes(), 6)) if codes else 0

    def _codec_acct(self, L, bits, iters, cls8, cen8, acc, flips, keep):
        """km_bytes: exact per-vehicle bootstrap bytes of the codec this run uses (accounting only: no tensor, no RNG touched).
        Uplink per vehicle: counts + sparse histogram (Core, lossless) + class moments (int8 codes + zlib, or fp32 sparse by
        present class) + every Lloyd pass and the final pass as sparse cells (header + zlib-6 of that vehicle's int8 codes).
        Downlink unicast: counts + lossless CDF + the centres of the vehicle's own classes for each of the iters + 1 assignment
        passes (fp16, or int8 + fp32 scale) + kept clusters' mixture (p fp32, mean fp16, factor fp16 + zlib or `bits` codes +
        zlib + fp16 diagonal/row scale). No class mean/Cholesky downlink (KM clients sample only the mixture).
        Broadcast copy = the downlink every vehicle receives identically, all classes' centres, sent once."""
        import zlib
        from methods.mobihet_lossless import encode
        from methods.mobihet_wstudy import SPARSE_HIST_UP_MEAN
        env, dev = self.env, self.env.dev
        C, K, D = L.shape[0], L.shape[1], L.shape[-1]
        tri = D * (D + 1) // 2
        km = keep.reshape(-1).cpu()
        kept = int(km.sum())
        Lc = L.reshape(C * K, D, D)[km].double().cpu()
        il = torch.tril_indices(D, D)
        l16 = len(zlib.compress(Lc.half()[:, il[0], il[1]].numpy().tobytes(), 6))     # KM50 factor, lossless coded
        if bits:
            sl = torch.tril_indices(D, D, -1)
            lv = 2 ** (bits - 1) - 1
            offd = Lc.masked_fill(torch.eye(D, dtype=torch.bool), 0)
            sc = (offd.abs().amax(-1, keepdim=True) / lv).clamp_min(1e-4).half().double()
            q = (offd / sc).round().clamp(-lv, lv)[:, sl[0], sl[1]].to(torch.int8)
            chol = len(zlib.compress(q.numpy().tobytes(), 6)) + 2 * kept * D * 2      # + fp16 diagonal, fp16 row scale
        else:
            chol = l16
        mix = 4 * kept + 2 * kept * D + chol
        cdf = len(encode(torch.round(self.cop_cdf.double() * 65535).cpu().numpy().astype("<u2")))
        cen_b = (D + 4) if cen8 else 2 * D
        ncls = torch.tensor(acc["ncls"], dtype=torch.float64)
        cls_v = torch.tensor(acc["cls"], dtype=torch.float64) if cls8 else ncls * (1 + 4 + 4 * (D + tri))
        up_v = 4 * C + SPARSE_HIST_UP_MEAN + cls_v + torch.tensor(acc["lloyd"], dtype=torch.float64) + \
            torch.tensor(acc["final"], dtype=torch.float64)
        cen_v = (iters + 1) * ncls * K * cen_b
        down_v = 4 * C + cdf + cen_v + mix
        tot = up_v + down_v
        BIN = 4096
        hist = lambda t: torch.bincount((t / BIN).long().clamp(0, 2047), minlength=2048).double()
        red = env.allreduce(torch.cat([torch.stack([up_v.sum(), down_v.sum(), cls_v.sum(), cen_v.sum(),
                                                    torch.tensor(float(len(up_v)), dtype=torch.float64)]),
                                       hist(up_v), hist(tot)]).to(dev)).cpu()
        flips = env.allreduce(flips.clone()).cpu()
        n = float(red[4])

        def pct(h, p):
            c = h.cumsum(0)
            return float(((c >= p * c[-1]).nonzero()[0, 0] + 1) * BIN)    # upper edge of the bin

        hu, ht = red[5:5 + 2048], red[5 + 2048:]
        rec = {"bits": bits, "cls_int8": cls8, "cen_int8": cen8, "iters": iters, "vehicles": int(n), "kept": kept,
               "up_mean": float(red[0] / n), "down_unicast_mean": float(red[1] / n), "cls_up_mean": float(red[2] / n),
               "centres_down_mean": float(red[3] / n), "mixture_down": mix, "factor_down": chol, "factor_fp16_lossless": l16,
               "cdf_down": cdf, "hist_up": SPARSE_HIST_UP_MEAN,
               "broadcast_copy": 4 * C + cdf + (iters + 1) * C * K * cen_b + mix,
               "up_p50": pct(hu, .5), "up_p90": pct(hu, .9), "up_max_bin": pct(hu, 1.0),
               "total_p50": pct(ht, .5), "total_p90": pct(ht, .9), "total_max_bin": pct(ht, 1.0),
               "fp16_centre_flip_frac": float(flips[0] / flips[1]) if flips[1] > 0 else None}
        rec["total_unicast_mean"] = rec["up_mean"] + rec["down_unicast_mean"]
        self.bootstrap_bytes = {"up": rec["up_mean"], "down": rec["down_unicast_mean"]}
        self.codec_bytes = rec
        print("KM_CODEC_BYTES " + json.dumps(rec), flush=True)

    def _em_stats(self, X, Y, off, score, f, bins, quant, p, m, L):
        """One federated EM pass: per vehicle and class, responsibilities under the received mixture (p, m, L), then the
        responsibility-weighted count, mean and centred covariance per cluster, int8 (b8 codec); server sums."""
        env, dev = self.env, self.env.dev
        C, K, D = m.shape
        n = torch.zeros(C, K, dtype=torch.float64, device=dev)
        s1 = torch.zeros(C, K, D, dtype=torch.float64, device=dev)
        s2 = torch.zeros(C, K, D, D, dtype=torch.float64, device=dev)
        lpk = torch.where(p > 0, p.clamp_min(1e-300).log(), torch.full_like(p, -1e30))
        ldet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)        # (C, K)
        for v in range(len(off) - 1):
            x, y = X[off[v]:off[v + 1]], Y[off[v]:off[v + 1]].long()
            z = score[y[:, None], f, bins(x)].double()
            for c in y.unique().tolist():
                zc = z[y == c]
                r = []
                for i in range(0, len(zc), 4096):
                    zb = zc[i:i + 4096]
                    sol = torch.linalg.solve_triangular(L[c], (zb[None] - m[c][:, None]).transpose(-1, -2), upper=False)
                    r.append(torch.softmax(lpk[c][:, None] - 0.5 * (sol ** 2).sum(1) - ldet[c][:, None], 0).T)
                r = torch.cat(r)                                             # (n_vc, K)
                for k in (r.sum(0) > 1e-3).nonzero().squeeze(1).tolist():
                    w = r[:, k]
                    nk = w.sum()
                    mk = (w @ zc) / nk
                    dz = zc - mk
                    ck = (dz * w[:, None]).T @ dz / nk
                    mq = quant(mk)
                    n[c, k] += nk; s1[c, k] += nk * mq; s2[c, k] += nk * (quant(ck) + mq[:, None] * mq[None])
        return env.allreduce(n), env.allreduce(s1), env.allreduce(s2)

    def _cls_int8(self, X, Y, off, score, f, bins, quant, meas, zs, sample, acc=None):
        """Class mean/covariance in normal-score space from per-vehicle int8 moments (b8 codec), as the server sums them."""
        env, dev = self.env, self.env.dev
        C, D = self.cop_mu.shape
        tri = D * (D + 1) // 2
        n = torch.zeros(C, dtype=torch.float64, device=dev)
        s1 = torch.zeros(C, D, dtype=torch.float64, device=dev)
        s2 = torch.zeros(C, D, D, dtype=torch.float64, device=dev)
        for v in range(len(off) - 1):
            x, y = X[off[v]:off[v + 1]], Y[off[v]:off[v + 1]].long()
            z = score[y[:, None], f, bins(x)].double()
            codes = [] if (v < sample or acc is not None) else None
            cls = y.unique().tolist()
            for c in cls:
                zc = z[y == c]
                nk = zc.shape[0]
                mk = quant(zc.mean(0), codes)
                ck = quant(zc.T @ zc / nk - zc.mean(0)[:, None] * zc.mean(0)[None], codes)
                n[c] += nk; s1[c] += nk * mk; s2[c] += nk * (ck + mk[:, None] * mk[None])
            # class index + int32 count + 2 fp32 scales + D + tri int8 per present class
            meas["cls"] += torch.tensor([len(cls) * (1 + 4 + 8), len(cls) * (D + tri)], dtype=torch.float64, device=dev)
            if v < sample:
                zs["cls"].extend(codes)
            if acc is not None:
                acc["cls"][v] = len(cls) * (1 + 4 + 8) + self._zlen(codes)
        n, s1, s2 = env.allreduce(n), env.allreduce(s1), env.allreduce(s2)
        mu = s1 / n.clamp_min(1)[:, None]
        cov = s2 / n.clamp_min(1)[:, None, None] - mu[:, :, None] * mu[:, None, :]
        e, U = torch.linalg.eigh((cov + cov.transpose(-1, -2)) / 2)
        cov = U @ torch.diag_embed(e.clamp_min(0)) @ U.transpose(-1, -2) + 1e-4 * torch.eye(D, device=dev, dtype=torch.float64)
        return mu, cov

    def _lite_down(self, L, bits, iters, meas, zs):
        """Downlink Cholesky factors: diagonal fp16, strict lower part `bits` with one fp16 scale per row; account the whole
        lite bootstrap (Core part included)."""
        import zlib
        from methods.mobihet_lossless import encode
        from methods.mobihet_wstudy import SPARSE_HIST_UP_MEAN
        env, dev = self.env, self.env.dev
        C, K, D = L.shape[0], L.shape[1], L.shape[-1]
        lv = 2 ** (bits - 1) - 1
        eye = torch.eye(D, dtype=torch.bool, device=dev)
        off_d = L.masked_fill(eye, 0)
        sc = (off_d.abs().amax(-1, keepdim=True) / lv).clamp_min(1e-4).half().double()   # fp16-safe (row 0 has no off-diagonal)
        q = (off_d / sc).round().clamp(-lv, lv)
        diag = torch.diagonal(L, dim1=-2, dim2=-1).half().double()
        self.km_L = (q * sc + torch.diag_embed(diag)).float()
        assert bool((diag > 0).all()), "fp16 Cholesky diagonal not positive"
        sl = torch.tril_indices(D, D, -1, device=dev)
        down_chol = len(zlib.compress(q[..., sl[0], sl[1]].to(torch.int8).cpu().numpy().tobytes(), 6))
        cdf = len(encode(torch.round(self.cop_cdf.double() * 65535).cpu().numpy().astype("<u2")))
        nv = env.allreduce(torch.tensor([float(len(self.env.train.off) - 1)], dtype=torch.float64, device=dev))
        ratio = {}
        for k in ("cls", "final"):                                       # zlib-6 ratio of the sampled int8 code streams
            raw = torch.cat(zs[k]).cpu().numpy().tobytes() if zs[k] else b""
            t = env.allreduce(torch.tensor([float(len(zlib.compress(raw, 6)) if raw else 0), float(len(raw))],
                                           dtype=torch.float64, device=dev))
            ratio[k] = float(t[0] / t[1]) if t[1] > 0 else 1.0
        per = {k: env.allreduce(meas[k].clone()) / nv for k in meas}     # mean (hdr, code) B per vehicle
        up = {"counts": 4 * C, "hist_sparse": SPARSE_HIST_UP_MEAN,
              "class_moments": float(per["cls"][0] + ratio["cls"] * per["cls"][1]),
              "lloyd": float(per["lloyd"][0] + per["lloyd"][1]),
              "final": float(per["final"][0] + ratio["final"] * per["final"][1])}
        down = {"counts": 4 * C, "cdf_lossless": cdf, "centres": (iters + 1) * C * K * (D + 4),
                "final": 4 * C * K + 3 * 2 * C * K * D + down_chol}          # p fp32; mean, diagonal, row scale fp16
        self.bootstrap_bytes = {"up": sum(up.values()), "down": sum(down.values())}
        rec = {"bits": bits, "up": up, "down": down, "zlib_ratio": ratio, "final_raw_mean": float(per["final"].sum()),
               "total_up": self.bootstrap_bytes["up"], "total_down": self.bootstrap_bytes["down"],
               "cop_kmeans_part": float(env.hp.get("cop_kmeans_part", 1.0)), "vehicles": int(nv)}
        print("KM_LITE_BYTES " + json.dumps(rec), flush=True)

    def _km_synth(self, ctx):
        G, a, dev = ctx.Gfix, self.aux, ctx.x.device
        lab = self.labels[None].expand(G, a)
        probs = self.km_pv[ctx.clients[:, None].expand(G, a), lab] if hasattr(self, "km_pv") else self.km_p[lab]
        k = torch.multinomial(probs.reshape(-1, self.km), 1, replacement=True, generator=self.gen_g).view(G, a)
        self._last_k = k                                                 # q_cluster: target of each synthetic row
        mu, L = self.km_mu[lab, k], self.km_L[lab, k]
        z = mu + (L @ torch.randn(G, a, mu.shape[-1], 1, device=dev, generator=self.gen_g)).squeeze(-1)
        return self._inv(z, lab).to(ctx.x.dtype), lab

    def _cf_setup(self):
        C, D = self.cop_mu.shape
        dev = self.cop_mu.device
        S = torch.zeros(C, D, dtype=torch.bool, device=dev)
        for c, idx in family_sets().items():
            S[c, idx] = True
        self.cf_cls = S.any(1)                                          # classes with a family
        cov = (self.cop_L.double() @ self.cop_L.double().transpose(-1, -2))
        # conditional of S given K = ~S under class c, written in full-D form: z_S = mu_S + A (z_K - mu_K) + Lc eps
        A = torch.zeros(C, C, D, D, dtype=torch.float64, device=dev)    # [target class t, family of class f]
        Lc = torch.zeros_like(A)
        for f in range(C):
            if not self.cf_cls[f]:
                continue
            s, k = S[f].nonzero().squeeze(1), (~S[f]).nonzero().squeeze(1)
            for t in range(C):
                cv = cov[t]
                W = torch.linalg.solve(cv[k][:, k], cv[k][:, s])        # (|K|, |S|)
                cc = cv[s][:, s] - cv[s][:, k] @ W
                A[t, f][s[:, None], k[None]] = W.T
                Lc[t, f][s[:, None], s[None]] = torch.linalg.cholesky(cc + 1e-6 * torch.eye(len(s), device=dev, dtype=cc.dtype))
        self.cf_S, self.cf_A, self.cf_Lc = S, A.float(), Lc.float()
        self.cf_list = self.cf_cls.nonzero().squeeze(1).tolist()

    def _pit(self, x, cat):
        """Randomized PIT of real values x (G, a, D) under the marginals of class cat (G, a) -> latent z."""
        NB = self.cop_cdf.shape[-1]
        j = ((x.float() + 4) * (NB / 8)).floor().long().clamp(0, NB - 1)
        hi = self.cop_cdf[cat].gather(-1, j[..., None]).squeeze(-1)
        m = self.cop_mass[cat].gather(-1, j[..., None]).squeeze(-1)
        u = (hi - m + torch.rand(x.shape, device=x.device, generator=self.gen_g) * m).clamp(1e-6, 1 - 1e-6)
        return torch.special.ndtri(u)

    def _inv(self, z, cat):
        u = torch.special.ndtr(z).clamp(1e-6, 1 - 1e-6)
        cdf = self.cop_cdf[cat]
        j = torch.searchsorted(cdf, u[..., None]).clamp(max=cdf.shape[-1] - 1)
        hi = cdf.gather(-1, j).squeeze(-1)
        m = self.cop_mass[cat].gather(-1, j).squeeze(-1)
        frac = ((u - (hi - m)) / m.clamp_min(1e-12)).clamp(0, 1)
        return -4 + (j.squeeze(-1) + frac) * (8 / cdf.shape[-1])

    def _oracle_shuffled(self, ctx):
        G, a, dev = ctx.Gfix, self.aux, ctx.x.device
        lab = self.labels[None].expand(G, a)
        nb = 77 if self.shuffle == "feature" else len(BLOCKS)
        u = torch.rand(G, a, nb, device=dev, generator=self.gen_g)
        n = self.cls_n[lab][..., None]
        pos = self.cls_off[lab][..., None] + (u * n).long().minimum((n - 1).clamp_min(0))
        if self.shuffle == "block":
            pos = pos[..., self.blk]                                     # (G, a, 77): one source row per block
        rows = self.cls_idx[pos]
        x = self.env.train.X[rows, torch.arange(77, device=dev)]
        return x.to(ctx.x.dtype), lab

    def _synth(self, ctx):
        if self.shuffle != "none":
            return self._oracle_shuffled(ctx)
        if getattr(self, "km", 0):
            return self._km_synth(ctx)
        xg, lab = super()._synth(ctx)
        if not self.cf:
            return xg, lab
        G, a, dev = lab.shape[0], lab.shape[1], lab.device
        real = ctx.mask
        benign = real & (ctx.y == 0)
        attack = real & self.cf_cls[ctx.y]
        # base row per synthetic row: a benign real row for CF attack labels, a CF-attack real row for benign labels
        want_b = self.cf_cls[lab]                                       # (G, a)
        want_a = lab == 0
        rb = torch.rand(G, a, ctx.y.shape[1], device=dev, generator=self.gen_g)
        pick_b = (rb * benign[:, None]).argmax(-1)
        pick_a = (rb * attack[:, None]).argmax(-1)
        ok = (want_b & benign.any(1)[:, None]) | (want_a & attack.any(1)[:, None])
        pick = torch.where(want_b, pick_b, pick_a)
        base = ctx.x.gather(1, pick[..., None].expand(-1, -1, ctx.x.shape[-1])).float()
        fam = torch.where(want_b, lab, ctx.y.gather(1, pick))           # family to resample
        fam = torch.where(ok, fam, torch.zeros_like(fam))
        z = self._pit(base, lab)
        mu = self.cop_mu[lab]
        eps = torch.randn(G, a, mu.shape[-1], device=dev, generator=self.gen_g)
        d = z - mu
        zs = mu.clone()
        for f in self.cf_list:             # pairs (t, f) = (f, f) attack, (0, f) benign
            for t in (f, 0):
                sel = (ok & (fam == f) & (lab == t))[..., None]
                zs = torch.where(sel, mu + d @ self.cf_A[t, f].T + eps @ self.cf_Lc[t, f].T, zs)
        Sm = self.cf_S[fam] & ok[..., None]
        xs = self._inv(torch.where(Sm, zs, z), lab)
        out = torch.where(Sm, xs, base)
        return torch.where(ok[..., None], out.to(xg.dtype), xg), lab

    def _loss_stream(self, ctx):
        if not self.need_T and self.sw == 1.0 and self.q_src == "real" and not self.qk and self.q_temp == 1.0:
            return super()._loss_stream(ctx)
        # identical to Core's stream loss except per-row synthetic weights: 0 for locally abundant classes (A), lambda_k (synth_weight)
        import torch.nn.functional as F
        from core import grouped, local
        from methods.mobihet_wstudy import active
        cfg = self.env.cfg
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        xp = torch.cat([ctx.x, xg], 1)
        mp = torch.cat([ctx.mask, active(ctx)[:, None].expand(G, a)], 1)
        zp, nBp = grouped.TRAIN(ctx.P, ctx.B, xp, cfg, mask=mp, train=True)
        y = ctx.y
        ce = local.masked_mean(F.cross_entropy((zp[:, :b] + self._shift[:, None, :]).reshape(-1, 16), y.reshape(-1),
                                               reduction="none").view(G, b), ctx.mask)
        lsp = F.log_softmax(zp, -1)
        with torch.no_grad():
            if self.q_src == "real":
                pr = torch.where(ctx.mask[..., None], lsp[:, :b].detach().exp(), 0.0)
                self._csum.index_put_((torch.arange(G, device=y.device)[:, None].expand_as(y), y), pr, accumulate=True)
                if self.qk:                                              # per (class, cluster) sums of the same probabilities
                    xb = ((ctx.x.float() + 4) * (256 / 8)).long().clamp(0, 255)
                    zr = self._qk_score[y[..., None], self._qk_f, xb]       # (G, b, D) mid-rank normal scores
                    dist = ((zr[..., None, :] - self.km_mu[y]) ** 2).sum(-1).masked_fill(self._qk_dead[y], float("inf"))
                    cell = y * self.km + dist.argmin(-1)
                    cl = ctx.clients[:, None].expand_as(y)
                    self._qk_sum.index_put_((cl, cell), pr, accumulate=True)
                    self._qk_cnt.index_put_((cl, cell), ctx.mask.float(), accumulate=True)
            else:   # Q1: class sums from the synthetic rows' (pre-update) predictions, scaled so that sum / n_kc = synthetic mean
                nk = self.env.n[ctx.clients]
                steps = torch.clamp(torch.ceil(nk / b), min=float(self.env.hp.get("synth_step_floor", 0)))
                w = self.cnt[ctx.clients].gather(1, lab) / (steps[:, None] * self.lab_rows[lab])
                ps = torch.where(active(ctx)[:, None, None], lsp[:, b:].detach().exp() * w[..., None], 0.0)   # masked rows: no NaN
                self._csum.index_put_((torch.arange(G, device=y.device)[:, None].expand_as(lab), lab), ps, accumulate=True)
        keep = (self.cnt[ctx.clients].gather(1, lab) < self.need_T).float() if self.need_T else torch.ones_like(lab, dtype=torch.float32)
        keep = keep * self.sw_k[ctx.clients][:, None]
        if self.Q is None:
            row = F.cross_entropy(zp[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, a)
        else:
            Q = self.Q
            if self.qk and self.Qk is not None:
                tq = self.Qk[lab, self._last_k]                           # (G, a, 16) cluster-conditioned target
            else:
                tq = Q[lab]
            if self.q_temp != 1.0:
                tq = tq.clamp_min(1e-6).pow(1.0 / self.q_temp)
                tq = tq / tq.sum(-1, keepdim=True)
            lt_g = tq.clamp_min(1e-6).log()
            row = (lt_g.exp() * (lt_g - lsp[:, b:])).sum(-1)
        return ce + (row * keep).mean(1), nBp

    # ---- b11: round hook (cluster Q aggregation, weight average), evaluation views, state ----
    def round(self, r, lr):
        env = self.env
        if self.qk:
            self._qk_sum = torch.zeros(env.K, 16 * self.km, 16, device=env.dev)
            self._qk_cnt = torch.zeros(env.K, 16 * self.km, device=env.dev)
        info = super().round(r, lr)
        if self.qk:
            info.update(self._qk_aggregate())
        if self.ema_beta:
            if self.ema_p is None:
                self.ema_p, self.ema_b = self.priv_p.clone(), self.priv_b.clone()
            else:
                self.ema_p.mul_(self.ema_beta).add_(self.priv_p, alpha=1 - self.ema_beta)
                self.ema_b.mul_(self.ema_beta).add_(self.priv_b, alpha=1 - self.ema_beta)
        return info

    def _round_private(self, r, lr):
        """q_bits < 16: Base's Q packets (fp16 vehicle class means up, fp16 Q down) become p*(2^b-1) integers. The vehicle's
        means are quantised right after its local epoch (Base then reads them through its fp16 cast, a fixed decode table of
        the integer), Q after the server aggregate, before the round's teacher signature. Exact bytes per vehicle: 32-byte
        header + 2-byte class bitmask + present classes only (absent rows are 0: lossless to drop), body raw-packed or zlib-6
        (+1 flag byte), whichever is smaller; downlink likewise over the 16x16 table."""
        if self.q_bits == 16:
            return super()._round_private(r, lr)
        import zlib
        import numpy as np
        from core import local
        env, lv = self.env, 2 ** self.q_bits - 1
        acc = torch.zeros(2, dtype=torch.float64, device=env.dev)

        def body(c):
            raw = c.tobytes() if self.q_bits == 8 else (c.reshape(-1, 2)[:, 0] | (c.reshape(-1, 2)[:, 1] << 4)).astype(np.uint8).tobytes()
            return min(len(raw), len(zlib.compress(raw, 6)) + 1)

        orig = local.local_epoch

        def epoch(lay, flat, bufs, data, idx, *a, **k):
            st = orig(lay, flat, bufs, data, idx, *a, **k)
            G = len(idx)
            cnt = self.cnt[idx, :, None].clamp_min(1)
            q = ((self._csum[:G] / cnt).clamp(0, 1) * lv).round()
            self._csum[:G] = q / lv * cnt
            codes = q.to(torch.uint8).cpu().numpy()
            present = (self.cnt[idx] > 0).cpu().numpy()
            for v in range(G):
                acc.add_(torch.tensor([34.0 + body(codes[v][present[v]]) if present[v].any() else 34.0, 1.0],
                                      dtype=torch.float64, device=env.dev))
            return st

        local.local_epoch = epoch
        try:
            info = super()._round_private(r, lr)
        finally:
            local.local_epoch = orig
        if self.Q is not None:
            q = (self.Q.clamp(0, 1) * lv).round()
            self.Q = (q / q.sum(-1, keepdim=True).clamp_min(1)).float()
            info["bytes_down"] = 32 + body(q.to(torch.uint8).cpu().numpy())
        acc = env.allreduce(acc)
        info["bytes_up"] = float(acc[0] / acc[1].clamp_min(1))
        return info

    def _qk_aggregate(self):
        """Server side of q_cluster from per-vehicle int8 cell means (cells >= q_cluster_min rows); Q[y] fills empty cells."""
        env, C, K = self.env, 16, self.km
        cnt = self._qk_cnt
        ok = cnt >= self.qk_min
        mean = self._qk_sum / cnt.clamp_min(1)[..., None]
        q8 = (mean * 255).round()
        dec = q8 / q8.sum(-1, keepdim=True).clamp_min(1)
        w = torch.where(ok, cnt, 0.).double()
        S = env.allreduce((dec.double() * w[..., None]).sum(0))
        N = env.allreduce(w.sum(0))
        cells = env.allreduce(ok.sum(1).double().sum()[None])           # uploaded cells summed over vehicles
        veh = env.allreduce(torch.tensor([float(env.K)], device=env.dev, dtype=torch.float64))
        Qk = (S / N.clamp_min(1)[:, None]).float().view(C, K, 16)
        fill = (N.view(C, K) == 0)[..., None]
        Qk = torch.where(fill, self.Q[:, None, :].expand_as(Qk), Qk)
        Qk = (Qk * 255).round()                                          # int8 downlink table
        self.Qk = Qk / Qk.sum(-1, keepdim=True).clamp_min(1)
        up = float(cells[0] / veh[0]) * (2 + 2 + 16)                     # cell index, count, int8 mean per uploaded cell
        down = 32 + C * K * 16
        return {"qk_bytes_up": up, "qk_bytes_down": down, "qk_cells_per_vehicle": float(cells[0] / veh[0]),
                "qk_filled_frac": float(fill.float().mean()),
                "qk_tv_vs_q": float(0.5 * (self.Qk - self.Q[:, None]).abs().sum(-1)[~fill[..., 0]].mean()) if (~fill).any() else 0.0}

    def evaluate(self):
        env = self.env
        if self.ema_beta and self.ema_p is not None:
            out = {"private": env.eval_clients(self.ema_p, self.ema_b), "raw": env.eval_clients(self.priv_p, self.priv_b)}
        else:
            out = super().evaluate()
        if self.ph:
            for a, cm in self._own_cm_ph(self.priv_p, self.priv_b, (.25, .5, .75, 1.0)).items():
                out[f"ph{int(round(a * 100)):03d}"] = env.gather_cms(cm)
        if self.fuse_path:
            wp, wb = (self.ema_p, self.ema_b) if (self.ema_beta and self.ema_p is not None) else (self.priv_p, self.priv_b)
            for name, cm in self._fused_cms(self._eval_lp(wp, wb)).items():
                out[name] = env.gather_cms(cm)
        return out

    def _fuse_setup(self):
        """Align the session sidecar (scripts/build_sessions.py) with env.test rows; fail fast on any mismatch."""
        import numpy as np
        env = self.env
        z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", self.fuse_path))
        ids = env.test.ids.cpu().numpy() if torch.is_tensor(env.test.ids) else np.asarray(env.test.ids)
        pos = np.searchsorted(z["ids"], ids)
        assert (pos < len(z["ids"])).all() and (z["ids"][pos] == ids).all(), "fuse: vehicles missing from sidecar"
        blk = [np.arange(z["off"][p], z["off"][p + 1]) for p in pos]
        rows = np.concatenate(blk)
        off = env.test.off.cpu().numpy() if torch.is_tensor(env.test.off) else np.asarray(env.test.off)
        Y = env.test.Y.cpu().numpy() if torch.is_tensor(env.test.Y) else np.asarray(env.test.Y)
        assert len(rows) == off[-1] and np.array_equal(np.diff(off), [len(b) for b in blk]), "fuse: row counts differ"
        assert np.array_equal(z["y"][rows], Y), "fuse: label order differs (sidecar not aligned)"
        sess, rank = z["sess"][rows].astype(np.int64), z["rank"][rows].astype(np.int64)
        perm = np.lexsort((rank, sess))
        ss = sess[perm]
        start = np.where(np.concatenate([[True], ss[1:] != ss[:-1]]), np.arange(len(ss)), 0)
        start = np.maximum.accumulate(start)
        dev = env.dev
        self._fz_perm = torch.from_numpy(perm).to(dev)
        self._fz_start = torch.from_numpy(start).to(dev)
        self._fz_client = torch.from_numpy(np.repeat(np.arange(len(ids)), np.diff(off))).to(dev)
        print(f"fuse: {len(rows)} test rows, {int((ss[1:] == ss[:-1]).sum())} with session history", flush=True)

    @torch.no_grad()
    def _eval_lp(self, flat, bufs):
        """Per-row log-softmax of each vehicle's own model on its own test rows, in env.test row order (fp32)."""
        from core import grouped
        env, lay, test, cfg = self.env, self.env.lay, self.env.test, self.env.cfg
        dev = test.X.device
        clients = torch.arange(env.K, device=dev)
        max_rows = env.hp.get("eval_rows", 1 << 15)
        off = test.off
        n = off[clients + 1] - off[clients]
        out = torch.zeros(int(off[-1]), 16, device=dev)
        order = torch.argsort(n, descending=True)
        i = 0
        while i < len(clients):
            b = min(int(n[order[i]].item()), max_rows)
            g = max(1, min(len(clients) - i, max_rows // max(b, 1)))
            idx = order[i:i + g]
            P, B = lay.params(flat[idx]), lay.bufs(bufs[idx])
            start, cnt = off[clients[idx]], n[idx]
            for s in range(0, int(cnt.max().item()), b):
                j = s + torch.arange(b, device=dev)[None]
                valid = j < cnt[:, None]
                rows = start[:, None] + torch.where(valid, j, 0)
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits, _, _ = grouped.EVAL(P, B, test.X[rows], cfg, train=False, hidden=True)
                out[rows[valid]] = torch.log_softmax(logits.float(), -1)[valid]
            i += g
        return out

    @torch.no_grad()
    def _fused_cms(self, lp):
        """Causal decision fusion per (vehicle, world, pseudonym) session: message t takes argmax of the summed log-probs of
        the last w messages of its session up to and including t (time order). Views fuse004 / fuse016 / fuseall."""
        dev, K = lp.device, self.env.K
        L = lp[self._fz_perm].double()
        cs = torch.cat([torch.zeros(1, 16, dtype=L.dtype, device=dev), L.cumsum(0)])
        i = torch.arange(len(L), device=dev)
        y = self.env.test.Y.long()
        out = {}
        for name, w in (("fuse004", 4), ("fuse016", 16), ("fuseall", 0)):
            lo = self._fz_start if w == 0 else torch.maximum(i - w + 1, self._fz_start)
            pred_s = (cs[i + 1] - cs[lo]).argmax(-1)
            pred = torch.empty_like(pred_s)
            pred[self._fz_perm] = pred_s
            code = self._fz_client * 256 + y * 16 + pred
            out[name] = torch.bincount(code, minlength=K * 256).view(K, 16, 16).cpu().numpy()
        return out

    @torch.no_grad()
    def _own_cm_ph(self, flat, bufs, alphas):
        """core.evaluate.own_cm with argmax(z + a * la_shift[client]) for several a in one forward."""
        from core import grouped
        env, lay, test, cfg = self.env, self.env.lay, self.env.test, self.env.cfg
        dev, N = test.X.device, 16
        clients = torch.arange(env.K, device=dev)
        max_rows = env.hp.get("eval_rows", 1 << 15)
        off = test.off
        n = off[clients + 1] - off[clients]
        out = {a: torch.zeros(len(clients), N * N, dtype=torch.int64, device=dev) for a in alphas}
        order = torch.argsort(n, descending=True)
        i = 0
        while i < len(clients):
            b = min(int(n[order[i]].item()), max_rows)
            g = max(1, min(len(clients) - i, max_rows // max(b, 1)))
            idx = order[i:i + g]
            P, B = lay.params(flat[idx]), lay.bufs(bufs[idx])
            start, cnt = off[clients[idx]], n[idx]
            sh = self.la_shift[idx].float()[:, None, :]
            for s in range(0, int(cnt.max().item()), b):
                j = s + torch.arange(b, device=dev)[None]
                valid = j < cnt[:, None]
                rows = start[:, None] + torch.where(valid, j, 0)
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits, _, _ = grouped.EVAL(P, B, test.X[rows], cfg, train=False, hidden=True)
                for a in alphas:
                    code = test.Y[rows].long() * N + (logits.float() + a * sh).argmax(-1)
                    code = torch.arange(g, device=dev)[:, None] * N * N + code
                    out[a][idx] += torch.bincount(code[valid], minlength=g * N * N).view(g, -1)
            i += g
        return {a: v.view(len(clients), N, N).cpu().numpy() for a, v in out.items()}

    def state(self):
        s = super().state()
        if self.ema_beta:
            s |= {"ema_p": None if self.ema_p is None else self.ema_p.cpu(), "ema_b": None if self.ema_b is None else self.ema_b.cpu()}
        if self.qk:
            s |= {"Qk": None if self.Qk is None else self.Qk.cpu()}
        return s

    def load_state(self, s):
        super().load_state(s)
        dev = self.env.dev
        if self.ema_beta:
            self.ema_p = None if s.get("ema_p") is None else s["ema_p"].to(dev)
            self.ema_b = None if s.get("ema_b") is None else s["ema_b"].to(dev)
        if self.qk:
            self.Qk = None if s.get("Qk") is None else s["Qk"].to(dev)
