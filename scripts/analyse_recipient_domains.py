"""Descriptive domain/error diagnostic from V20 CMs; no training or teacher fitting.

Forecast own-model hard output frequencies conditional on true class. Donor and
recipient clients are disjoint. Future-peer source is diagnostic, not deployable.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics
from scripts.sim_fold_val import pick


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecast(source, target, groups, halves, minimum):
    """Brier expectation for a recipient output, not classifier prediction loss."""
    global_scores, group_scores, support, fallback = [], [], [], []
    for half in (0, 1):
        donor, recipient = halves != half, halves == half
        assert not np.any(donor & recipient)
        total = source[donor].sum(0).astype(float)
        global_p = total / np.maximum(total.sum(1, keepdims=True), 1)
        assert np.all(total.sum(1) >= minimum)
        for group in np.unique(groups[recipient]):
            keep = recipient & (groups == group)
            pooled = source[donor & (groups == group)].sum(0).astype(float)
            enough = pooled.sum(1) >= minimum
            p = np.where(enough[:, None], pooled / np.maximum(pooled.sum(1, keepdims=True), 1), global_p)
            cm = target[keep].sum(0).astype(float)
            # Sum E[||onehot(model-output)-p||^2] by true class.
            n = cm.sum(1)
            global_scores.append(n * (1 + np.square(global_p).sum(1)) - 2 * (cm * global_p).sum(1))
            group_scores.append(n * (1 + np.square(p).sum(1)) - 2 * (cm * p).sum(1))
            support.append(n)
            fallback.append(n * ~enough)
    n = np.sum(support, axis=0)
    assert np.array_equal(n, target.sum((0, 2))) and np.all(n > 0)
    base, grouped = np.sum(global_scores, axis=0) / n, np.sum(group_scores, axis=0) / n
    delta = base - grouped  # positive means group forecasts better
    return {"global_macro_brier": float(base.mean()), "group_macro_brier": float(grouped.mean()),
            "macro_improvement": float(delta.mean()), "micro_improvement": float(np.dot(delta, n) / n.sum()),
            "improved_classes": int((delta > 0).sum()), "per_class_improvement": delta.tolist(),
            "per_class_fallback_mass": (np.sum(fallback, axis=0) / n).tolist()}


def group_support(counts, group):
    rows = []
    for key in np.unique(group):
        n = counts[group == key].astype(float)
        total = n.sum(0)
        rows.append({"group": str(key), "clients": len(n), "counts": total.astype(int).tolist(),
                     "experts_min8": (n >= 8).sum(0).tolist(),
                     "effective_experts": (np.square(total) / np.maximum(np.square(n).sum(0), 1)).tolist(),
                     "max_expert_share": (n.max(0) / np.maximum(total, 1)).tolist()})
    return rows


def main():
    out = ROOT / "runs/recipient-domain-review-20261111"
    out.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / "runs/r-study/analysis/analysis.json"
    meta = json.loads(meta_path.read_text())["arms"]
    clients_path = Path("/home/odixe/project/mobihet-features/data/v4-scaled/clients.parquet")
    clients = pq.ParquetFile(clients_path).read(columns=["client_id", "scenario", "run"]).to_pandas().set_index("client_id")
    assert clients.index.is_unique and len(clients) == 5343
    result = {"metadata_sha256": sha(clients_path), "source_analysis_sha256": sha(meta_path), "arms": {}}
    reference = None
    for arm in ("V2", "V3", "V4", "V5"):
        spec = meta[arm]
        assert spec["split"] == "val"
        run = ROOT / "artifacts/v20" / spec["slug"] / spec["slug"]
        paths = [run / s for s in ("train_cms_r020.npz", "cms_r020.npz", "config.json")]
        cfg = json.loads(paths[2].read_text())
        assert cfg["n_params"] == 395408 and cfg["hp"]["rounds"] == 50
        with np.load(paths[0]) as t, np.load(paths[1]) as v:
            ids, train, val = t["ids"].copy(), t["private"].copy(), v["private"].copy()
            assert np.array_equal(ids, v["ids"])
        assert len(np.unique(ids)) == 5343 and train.shape == val.shape == (5343, 16, 16)
        assert int(train.sum()) == 32685166 and int(val.sum()) == 4676107
        if reference is None:
            reference = (ids, train.sum(2), val.sum(2))
        assert all(np.array_equal(a, b) for a, b in zip(reference, (ids, train.sum(2), val.sum(2))))
        row = next(json.loads(s) for s in (run / "rounds.jsonl").read_text().splitlines() if json.loads(s).get("round") == 20)
        for cm, prefix in ((train, "train_private/"), (val, "private/")):
            ten = metrics.summary(cm)
            assert all(abs(ten[k] - row[prefix + k]) < 1e-9 for k in metrics.TEN)
        table = clients.loc[ids]
        keys = {"scenario": table.scenario.to_numpy(), "run": table.run.to_numpy(),
                "scenario_run": (table.scenario + ":" + table.run).to_numpy()}
        groups = {k: np.unique(v, return_inverse=True)[1] for k, v in keys.items()}
        halves = np.array([hashlib.sha256(f"recipient-diagnostic:{i}".encode()).digest()[0] % 2 for i in ids])
        a = {"seed": spec["seed"], "mode": spec["mode"], "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
             "forecasts": {}}
        for minimum in (8, 32):
            a["forecasts"][str(minimum)] = {}
            for name, group in groups.items():
                a["forecasts"][str(minimum)][name] = {
                    source_name: forecast(source, val, group, halves, minimum)
                    for source_name, source in (("infit_train", train), ("future_peer_diagnostic", val))}
        result["arms"][arm] = a
    result["group_client_counts"] = {name: {str(k): int((v == k).sum()) for k in np.unique(v)} for name, v in keys.items()}
    result["client_halves"] = [int((halves == h).sum()) for h in (0, 1)]
    count_paths = [ROOT / "cache/fold_counts.npy", ROOT / "evidence/research-data/train_counts.npy"]
    fold, full = np.load(count_paths[0]), np.load(count_paths[1])
    panel = pick(200, cnt=fold)
    assert int(full.sum()) == 37361273 and int(fold[panel, 0].sum()) == 1476064
    result["count_source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in count_paths}
    result["population_support"] = {}
    for name, selected, count in (("sim200_foldtrain", panel, fold[panel, 0]), ("full5343_trainval", ids, full[ids])):
        tab = clients.loc[selected]
        result["population_support"][name] = group_support(count, (tab.scenario + ":" + tab.run).to_numpy())
    result["limits"] = [
        "V20 historical DAGSNetW1, not current CBB; train/internal val only, no official test.",
        "Forecast own-model hard output distribution conditional on true class; not soft Q or cross-model transfer.",
        "Donor/recipient clients disjoint by fixed SHA256 halves; no grouping/threshold selected by utility.",
        "Future-peer source uses other clients' future validation labels/predictions, unavailable during train: diagnostic only.",
        "Brier improvement is forecast quality, not classifier macro F1. No new weights/teacher or training launch.",
        "No world/sender/trueclass metadata used to assign domains. All 16 conditional rows scored equally for macro.",
        "Thresholds8/32 apply donor class support only, with exact global fallback; no prediction-dependent filtering.",
        "Observed grouping association can reflect covariates, label mixture in donor training and temporal shift; not causal proof."]
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Recipient domain diagnostic", "", "Positive ΔBrier: group forecasts recipient error distribution better than global donors.",
             "Not classifier utility; future-peer source unavailable during training.", "",
             "| arm | group | train→future Δmacro Brier | future-peer→future Δmacro Brier | improved classes (train / peer) |",
             "|---|---|---:|---:|---:|"]
    for arm, a in result["arms"].items():
        for name, f in a["forecasts"]["32"].items():
            t, v = f["infit_train"], f["future_peer_diagnostic"]
            lines.append(f"| {arm} | {name} | {t['macro_improvement']:.8f} | {v['macro_improvement']:.8f} | {t['improved_classes']} / {v['improved_classes']} |")
    lines += ["", *["- " + s for s in result["limits"]]]
    lines += ["", "## Main-population support (not V20 train counts)", "",
              "| population | scenario/run | clients | class11 experts ≥8 | class11 effective experts | class11 max share |",
              "|---|---|---:|---:|---:|---:|"]
    for name, rows in result["population_support"].items():
        for r in rows:
            lines.append(f"| {name} | {r['group']} | {r['clients']} | {r['experts_min8'][11]} | {r['effective_experts'][11]:.2f} | {r['max_expert_share'][11]:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("Recipient domain support/CM/10metric/hash/forecast diagnostic written; no training")


if __name__ == "__main__":
    main()
