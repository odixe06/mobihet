"""Preflight for the KM-lite / participation knobs of methods/mobihet_sim.py on real fold-train rows (no training, no val/test).

    python scripts/check_km_lite.py [--n 40] [--frozen artifacts/km50/mh-k-km16q8-s42-r50/provenance/source/methods/mobihet_sim.py]

1. knobs off: new _kmeans == frozen source (b8/KM50, sha 2e92f1d) bit-for-bit on km_p/km_mu/km_L and bootstrap bytes;
2. lite 8 / 6 and participation .1 run end to end, finite, positive Cholesky diagonals; KM_LITE_BYTES printed;
3. em1/white/temp/gap-fill finite; temp and gap-fill leave the shared mixture unchanged; |km_L - km_L(KQ8)| reported.
Writes runs/km-codec-probe/check_km_lite.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from core import data as D  # noqa: E402
from scripts.sim_fold_val import fold_counts, pick  # noqa: E402

NB = 256


def env_for(ids, hp, dev):
    s = D.load(D.find_root(), "train", ids, cache="cache", folds=["train"]).to(dev)
    e = types.SimpleNamespace(train=s, hp=hp, dev=dev, rank=0, world=1, allreduce=lambda t: t)
    return e


def tables(env):
    """Core compact copula tables on env.train (same as Base._copula_bootstrap + WMixin compact decode)."""
    X, Y = env.train.X, env.train.Y.long()
    C, Dm = 16, X.shape[1]
    f = torch.arange(Dm, device=env.dev)
    bins = lambda x: ((x.float() + 4) * (NB / 8)).long().clamp(0, NB - 1)
    hist = torch.zeros(C * Dm * NB, dtype=torch.float64, device=env.dev)
    hist.index_add_(0, ((Y[:, None] * Dm + f) * NB + bins(X)).reshape(-1), torch.ones(X.numel(), dtype=torch.float64, device=env.dev))
    hist = hist.view(C, Dm, NB)
    mass = hist / hist.sum(-1, keepdim=True).clamp_min(1)
    cdf = mass.cumsum(-1)
    score = torch.distributions.Normal(0.0, 1.0).icdf((cdf - mass / 2).clamp(1e-4, 1 - 1e-4)).float()
    z = score[Y[:, None], f, bins(X)].double()
    mu = torch.stack([z[Y == c].mean(0) for c in range(C)])
    cov = torch.stack([(z[Y == c] - mu[c]).T @ (z[Y == c] - mu[c]) / (Y == c).sum() for c in range(C)]) + 1e-4 * torch.eye(Dm, device=env.dev, dtype=torch.float64)
    cdf = torch.round(cdf.double() * 65535) / 65535
    cdf[..., -1] = torch.where(cdf[..., -1] > 0, 1., 0.).double()
    return dict(cop_cdf=cdf.float().contiguous(), cop_mass=torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])).float(),
                cop_mu=mu.float().half().float(), cop_L=torch.linalg.cholesky(cov).float().half().float())


def run(cls, env, tab, K=16):
    m = object.__new__(cls)
    m.env = env
    for k, v in tab.items():
        setattr(m, k, v.clone())
    m.bootstrap_bytes = {"up": 0, "down": 0}
    m._kmeans(K, int(env.hp.get("cop_kmeans_iters", 4)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--frozen", default="artifacts/km50/mh-k-km16q8-s42-r50/provenance/source/methods/mobihet_sim.py")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = pick(200, cnt=fold_counts())[:: max(1, 200 // a.n)][:a.n].tolist()
    spec = importlib.util.spec_from_file_location("frozen_sim", os.path.join(ROOT, a.frozen))
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    from methods import mobihet_sim as new
    base = {"cop_kmeans": 16, "cop_kmeans_w0": 0.5, "cop_kmeans_q": "int8"}
    env = env_for(ids, dict(base), dev)
    tab = tables(env)
    out = {"vehicles": len(ids), "rows": int(env.train.off[-1])}
    old, cur = run(frozen.Method, env, tab), run(new.Method, env, tab)
    same = {k: bool(torch.equal(getattr(old, k), getattr(cur, k))) for k in ("km_p", "km_mu", "km_L")}
    same["bytes"] = old.bootstrap_bytes == cur.bootstrap_bytes
    out["knobs_off_equal"] = same
    print("knobs off == frozen:", same, flush=True)
    assert all(same.values())
    env.hp = dict(base, km_bytes=True)                                # accounting only: tensors must stay KM50 bit-exact
    acc = run(new.Method, env, tab)
    same_acc = {k: bool(torch.equal(getattr(old, k), getattr(acc, k))) for k in ("km_p", "km_mu", "km_L")}
    out["km_bytes_equal"], out["km_bytes"] = same_acc, acc.codec_bytes
    print("km_bytes == frozen:", same_acc, flush=True)
    assert all(same_acc.values())
    for tag, extra in [("ch8", {"km_chol_bits": 8, "km_bytes": True}), ("ch6", {"km_chol_bits": 6, "km_bytes": True}),
                       ("cm8", {"km_cls_q": "int8", "km_bytes": True}), ("ct8", {"km_cen_q": "int8", "km_bytes": True}),
                       ("ly2", {"cop_kmeans_iters": 2, "km_bytes": True}), ("ly0", {"cop_kmeans_iters": 0, "km_bytes": True}),
                       ("lite8_acct", {"cop_kmeans_lite": 8, "km_bytes": True}),
                       ("lite8", {"cop_kmeans_lite": 8}), ("lite6", {"cop_kmeans_lite": 6}),
                       ("lite6_part10", {"cop_kmeans_lite": 6, "cop_kmeans_part": 0.1}),
                       ("em1", {"cop_kmeans_em": 1}), ("white", {"cop_kmeans_white": True}),
                       ("temp05", {"cop_kmeans_temp": 0.5}), ("gapfill", {"synth_gapfill": True})]:
        env.hp = dict(base, **extra)
        m = run(new.Method, env, tab)
        ok = all(bool(torch.isfinite(getattr(m, k)).all()) for k in ("km_p", "km_mu", "km_L"))
        dg = torch.diagonal(m.km_L, dim1=-2, dim2=-1)
        d = (m.km_L - cur.km_L).abs()
        if tag == "gapfill":
            ok = ok and bool(torch.isfinite(m.km_pv).all()) and bool(torch.allclose(m.km_pv.sum(-1), torch.ones(1, device=dev)))
            assert torch.equal(m.km_L, cur.km_L) and torch.equal(m.km_p, cur.km_p), "gap-fill must not change the shared mixture"
        if tag == "temp05":
            assert torch.equal(m.km_L, cur.km_L) and not torch.equal(m.km_p, cur.km_p)
        out[tag] = {"finite": ok, "diag_min": float(dg.min()), "max_abs_dL": float(d.max()), "mean_abs_dL": float(d.mean()),
                    "bytes": m.bootstrap_bytes}
        if tag == "lite8_acct":                                          # same tensors as lite8 without accounting
            ref = run(new.Method, types.SimpleNamespace(**{**vars(env), "hp": dict(base, cop_kmeans_lite=8)}), tab)
            assert all(torch.equal(getattr(m, k), getattr(ref, k)) for k in ("km_p", "km_mu", "km_L")), "km_bytes changed lite8"
        if tag == "ch8":                                                 # same uplink/centres as KM50: only the factor differs
            assert torch.equal(m.km_p, cur.km_p) and torch.equal(m.km_mu, cur.km_mu)
            assert m.codec_bytes["up_mean"] == acc.codec_bytes["up_mean"]
        print(tag, out[tag], flush=True)
        assert ok and float(dg.min()) > 0
    os.makedirs("runs/km-codec-probe", exist_ok=True)
    json.dump(out, open("runs/km-codec-probe/check_km_lite.json", "w"), indent=1)
    print("PASS")


if __name__ == "__main__":
    main()
