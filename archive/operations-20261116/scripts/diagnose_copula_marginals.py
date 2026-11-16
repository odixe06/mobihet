"""Separate fitting error from train/val temporal shift; no refit or gate changes."""
import hashlib
import json

import numpy as np

from scripts.standard_copula_generator import Generator
from scripts.probe_pair_generator import ROOT, PAIRS, load, bins, hist


def tv(a, b):
    return float(np.abs(a-b).sum()/2)


def main():
    out = ROOT / "runs/copula-marginal-diagnosis"
    out.mkdir(exist_ok=True)
    order, ((xt, yt), (xv, yv)), _ = load()
    pairs = [(order.index(a), order.index(b)) for a,b in PAIRS]
    rows = []
    for c in range(16):
        real, val = xt[yt == c], xv[yv == c]
        model = Generator.fit(real, pairs)
        a = model.sample(np.random.default_rng(8100+c), 65536)
        b = model.sample(np.random.default_rng(8100+c), 65536, direct=True)
        rt, rv, base, standard = (bins(x) for x in (real, val, a, b))
        for j in range(77):
            target, future, old, new = (hist(x[:,j],16) for x in (rt,rv,base,standard))
            rows.append({"class": c, "feature": order[j], "train_val_tv": tv(target, future),
                         "base_train_tv": tv(old,target), "standard_train_tv": tv(new,target),
                         "base_val_tv": tv(old,future), "standard_val_tv": tv(new,future),
                         "latent_mu": float(model.mu[j]), "latent_variance": float(np.square(model.chol[j]).sum())})
    result = {"scope": "exploratory reused-panel attribution, no gate change or utility proof", "feature_class_rows": rows,
              "source_sha256": hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    result["means"] = {k: float(np.mean([r[k] for r in rows])) for k in
                       ("train_val_tv", "base_train_tv", "standard_train_tv", "base_val_tv", "standard_val_tv")}
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("COPULA_DIAG_DONE", json.dumps(result["means"]), flush=True)


if __name__ == "__main__":
    main()
