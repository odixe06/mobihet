"""Read-only producer-quality diagnostic from retained V20 train/internal-val CMs.

Uses original temporal validation, not official test; no expert weights fitted.
Binary prior standardization does not fix shifts inside the negative-class mixture.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics


def quality(cm, prior):
    tp = cm.diagonal(axis1=1, axis2=2).astype(float)
    pos, pred = cm.sum(2), cm.sum(1)
    neg = cm.sum((1, 2))[:, None] - pos
    recall = tp / np.maximum(pos, 1)
    fpr = (pred - tp) / np.maximum(neg, 1)
    den = prior * recall + (1 - prior) * fpr
    precision = prior * recall / np.maximum(den, 1e-30)
    return pos, neg, recall, fpr, precision


def rho(a, b):
    return None if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0 else float(spearmanr(a, b).statistic)


def main():
    out = ROOT / "runs/temporal-quality-review-20261111"
    out.mkdir(exist_ok=True, parents=True)
    meta_path = ROOT / "runs/r-study/analysis/analysis.json"
    arms = json.loads(meta_path.read_text())["arms"]
    result = {"source_analysis_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(), "arms": {}}
    names = json.loads((ROOT / "knowledge/label_mapping.json").read_text())["id_to_label"]
    reference = None
    for arm in ("V2", "V3", "V4", "V5"):
        spec = arms[arm]
        assert spec["split"] == "val"
        run = ROOT / "artifacts/v20" / spec["slug"] / spec["slug"]
        config = json.loads((run / "config.json").read_text())
        assert config["n_params"] == 395408 and config["hp"]["rounds"] == 50
        paths = [run / name for name in ("train_cms_r020.npz", "cms_r020.npz")]
        with np.load(paths[0]) as t, np.load(paths[1]) as v:
            assert np.array_equal(t["ids"], v["ids"])
            train, val = t["private"].copy(), v["private"].copy()
        assert train.shape == val.shape == (5343, 16, 16)
        assert int(train.sum()) == 32685166 and int(val.sum()) == 4676107
        supports = (train.sum(2), val.sum(2))
        if reference is None:
            reference = supports
        assert all(np.array_equal(a, b) for a, b in zip(reference, supports))
        row = next(json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines()
                   if json.loads(x).get("round") == 20)
        for cm, prefix in ((train, "train_private/"), (val, "private/")):
            ten = metrics.summary(cm)
            assert all(abs(ten[k] - row[prefix + k]) < 1e-9 for k in metrics.TEN)
        prior = train.sum((0, 2)) / train.sum()
        qt, qv = quality(train, prior), quality(val, prior)
        detail = {"seed": spec["seed"], "mode": spec["mode"], "rows": [int(train.sum()), int(val.sum())],
                  "source_cm_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  "thresholds": {}}
        for threshold in (8, 32):
            # Retain abstaining experts (no predicted positives) with precision0;
            # eligibility depends only on true support, not model predictions.
            eligible = (qt[0] >= threshold) & (qv[0] >= threshold) & (qt[1] >= threshold) & (qv[1] >= threshold)
            classes = []
            for c in range(16):
                keep = eligible[:, c]
                classes.append({"class": c, "name": names[str(c)], "clients": int(keep.sum()),
                                "infit_precision_median": float(np.median(qt[4][keep, c])),
                                "future_precision_median": float(np.median(qv[4][keep, c])),
                                "rho_infit_future_precision": rho(qt[4][keep, c], qv[4][keep, c]),
                                "rho_count_future_precision": rho(qt[0][keep, c], qv[4][keep, c]),
                                "infit_recall_median": float(np.median(qt[2][keep, c])),
                                "future_recall_median": float(np.median(qv[2][keep, c]))})
            detail["thresholds"][str(threshold)] = classes
        result["arms"][arm] = detail
    result["limits"] = ["Historical V20 recipes, not current KM/lite6/Q8 CBB.",
                        "Hard CMs/own-client quality, not soft Q, cross-recipient transfer or utility of reweighting.",
                        "Same global TRAIN binary prior in both splits; negative-class mixture shifts remain.",
                        "No predicted positives -> precision0, retained if true support eligible; no prediction-dependent filtering.",
                        "Support thresholds 8 and 32 are descriptive sensitivity checks, not selected weights/HPs.",
                        "No official test or training launch; historical source audit retained, 10 metrics recomputed at r20."]
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Temporal producer-quality diagnostic", "", "Retained V20 r20 train/internal-val; no weights fitted or utility claims.", "",
             "One-vs-rest precision at same global TRAIN binary prior. Negative-mixture shift remains.", "",
             "| arm | class | clients (min32 each binary side/split) | median train P* | median future P* | Spearman train/future P* | Spearman count/future P* |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm, a in result["arms"].items():
        for c in a["thresholds"]["32"]:
            if c["class"] in (0, 1, 6, 7, 11, 13):
                fmt = lambda x: "NA" if x is None else f"{x:.4f}"
                lines.append(f"| {arm} | {c['name']} | {c['clients']} | {c['infit_precision_median']:.4f} | "
                             f"{c['future_precision_median']:.4f} | {fmt(c['rho_infit_future_precision'])} | {fmt(c['rho_count_future_precision'])} |")
    lines += ["", *["- " + s for s in result["limits"]]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("Temporal-quality CM/10metric/support diagnostic written, no training")


if __name__ == "__main__":
    main()
