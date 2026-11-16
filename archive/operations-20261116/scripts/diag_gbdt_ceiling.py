"""N2 ceiling: histogram GBDT on the pooled sim panel fold-train rows (= federated histogram GBDT with full participation
and shared bins), evaluated on the same fold-val rows as the CBB sim. No official test.

    python scripts/run_local_guarded.py python scripts/diag_gbdt_ceiling.py

Locked rule (docs/research/method-building-brief.md §5, 12/11): continue N2 only if pooled macro-F1 >= .924
(CBB sim r20 C .9146/.9129 + .010). Output: runs/n2-gbdt-ceiling/diag.json
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import data as D  # noqa: E402
from scripts import sim_fold_val as S  # noqa: E402


def report(y, pred, C=16):
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    rec, prec = tp / np.maximum(cm.sum(1), 1), tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "f1_macro": float(f1[cm.sum(1) > 0].mean()),
            "recall": np.round(rec, 4).tolist(), "precision": np.round(prec, 4).tolist(), "f1": np.round(f1, 4).tolist()}


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    cnt = S.fold_counts()
    ids = S.pick(200, cnt=cnt)
    cache = os.environ.get("V4_CACHE", "cache")
    tr = D.load(D.find_root(), "train", ids.tolist(), cache, folds=["train"])
    va = D.load(D.find_root(), "train", ids.tolist(), cache, folds=["val"])
    Xt, yt = np.asarray(tr.X, np.float32), np.asarray(tr.Y, np.int64)
    Xv, yv = np.asarray(va.X, np.float32), np.asarray(va.Y, np.int64)
    assert len(yt) == cnt[ids, 0].sum() and len(yv) == cnt[ids, 1].sum()
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, max_bins=255,
                                         early_stopping=False, random_state=0)
    clf.fit(Xt, yt)
    fit_s = time.time() - t0
    res = {"ids_sha": hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()[:16], "rows": [len(yt), len(yv)],
           "params": {k: v for k, v in clf.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))},
           "fit_seconds": fit_s, "val": report(yv, clf.predict(Xv)), "train": report(yt, clf.predict(Xt))}
    res["gate"] = {"threshold": .924, "pass": res["val"]["f1_macro"] >= .924}
    os.makedirs("runs/n2-gbdt-ceiling", exist_ok=True)
    json.dump(res, open("runs/n2-gbdt-ceiling/diag.json", "w"), indent=1)
    print(json.dumps(res["gate"]), "fit", round(fit_s), "s")
    for k in ("val", "train"):
        print(k, "acc", round(res[k]["accuracy"], 4), "macro", round(res[k]["f1_macro"], 4), "f1", res[k]["f1"])


if __name__ == "__main__":
    main()
