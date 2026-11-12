"""B19 local gate: generator Bayes classifier before/after boundary components on real fold-val rows (sim panel).

    python scripts/run_local_guarded.py python scripts/diag_boundary.py

Uses the deployed code path (methods.mobihet_boundary bootstrap + GenScore), KM16 w0 .5 without quantisation (owner 12/11).
No training, no official test. Gate locked in docs/research/screening-boundary-b19.md. Output: runs/b19-boundary-diag/diag.json
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import data as D  # noqa: E402
from scripts import sim_fold_val as S  # noqa: E402

KNOBS = "cop_kmeans=16 cop_kmeans_w0=0.5 km_boundary=on".split()
WEAK = {"timeDelay": 13, "benign": 0, "positionMirroring": 7, "suddenConstantSpeed": 11}


def report(y, pred, C=16):
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "f1_macro": float(f1[cm.sum(1) > 0].mean()),
            "recall": np.round(rec, 4).tolist(), "precision": np.round(prec, 4).tolist(), "f1": np.round(f1, 4).tolist()}


def main():
    out_dir = "runs/b19-boundary-diag"
    cnt = S.fold_counts()
    ids = S.pick(200, cnt=cnt)
    orig = D.load
    D.load = lambda root, split, clients=None, cache=None, folds=None, with_fold=False: orig(
        root, "train", clients, cache, folds=["train"] if split == "train" else ["val"])
    from core import fl
    from core.dagsnet import width
    from scripts.run_fl import PROTOCOL
    fl.Env._sizes = staticmethod(lambda root, cache: cnt[:, 0].sum(1))
    hp = dict(PROTOCOL, rounds=20)
    for s in open("runs/sim/core_hp.txt").read().split() + KNOBS:
        k, v = s.split("=", 1)
        try:
            hp[k] = json.loads(v)
        except json.JSONDecodeError:
            hp[k] = v
    env = fl.Env(width(1.0), hp, clients=ids.tolist(), cache=os.environ.get("V4_CACHE", "cache"))
    from methods.mobihet_boundary import GenScore, Method
    M = Method(env)
    total = env.allreduce(M.cnt.sum(0).double())
    logpi = ((total + 1) / (total.sum() + 16)).log().float()
    gens = {"base": GenScore(M.cop_cdf, M.cop_mass, *M.bd_base), "boundary": GenScore(M.cop_cdf, M.cop_mass, M.km_p, M.km_mu, M.km_L)}
    Xv, yv = env.test.X, env.test.Y.long().cpu().numpy()
    res = {"ids_sha": hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()[:16], "km_boundary": M.bd_info,
           "bootstrap_bytes": M.bootstrap_bytes, "rows_val": int(len(yv))}
    with torch.no_grad():
        for name, g in gens.items():
            pred = torch.cat([(g(Xv[i:i + 2048]) + logpi).argmax(-1) for i in range(0, len(yv), 2048)]).cpu().numpy()
            res[name] = report(yv, pred)
    b, a = res["base"], res["boundary"]
    d_macro = a["f1_macro"] - b["f1_macro"]
    not_lower = {k: a["recall"][c] >= b["recall"][c] for k, c in WEAK.items()}
    res["gate"] = {"d_macro": d_macro, "d_macro_ge_0.010": d_macro >= .010, "weak_recall_not_lower": not_lower,
                   "weak_ge_3of4": sum(not_lower.values()) >= 3}
    res["gate"]["pass"] = bool(res["gate"]["d_macro_ge_0.010"] and res["gate"]["weak_ge_3of4"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "diag.json"), "w"), indent=1)
    print(json.dumps(res["gate"]))
    print("boundary info", json.dumps(M.bd_info))
    for k in ("base", "boundary"):
        print(k, "acc", round(res[k]["accuracy"], 4), "macro", round(res[k]["f1_macro"], 4), "recall", res[k]["recall"])


if __name__ == "__main__":
    main()
