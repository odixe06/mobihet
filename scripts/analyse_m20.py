"""Fixed-endpoint M20 diagnosis: own-test curves, class-support cohorts, paired clients and teacher probes."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics
from scripts.research_metrics import client_f1, probe_summary

ROOT = Path("evidence/m20")
OUT = Path("runs/analysis-m20")
COHORTS = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 100000)]


def read_cm(slug, r):
    with np.load(ROOT / slug / slug / f"cms_r{r:03}.npz") as z:
        assert np.array_equal(z["ids"], np.arange(5343))
        return z["private"]


def conditional_accuracy(cm, count):
    support, correct = cm.sum(2), cm.diagonal(axis1=1, axis2=2)
    out = {}
    for name, mask in {"absent": count == 0, "rare1_16": (count > 0) & (count <= 16),
                       "seen_gt16": count > 16}.items():
        n = int(support[mask].sum())
        out[name] = {"rows": n, "accuracy": float(correct[mask].sum() / n) if n else None}
    return out


def main():
    global ROOT, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--batch", type=Path, default=Path("evidence/m20-summary/batch.json"))
    args = ap.parse_args()
    ROOT, OUT = args.root, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    counts = np.load("evidence/research-data/train_counts.npy")
    n = counts.sum(1)
    batch = json.loads(args.batch.read_text())
    arms, final_cms, csvrows = {}, {}, []
    reference_support = np.load("evidence/research-data/test_support.npy")
    for spec in batch:
        d = ROOT / spec["slug"] / spec["slug"]
        cfg = json.loads((d / "config.json").read_text())
        hp = cfg["hp"]
        rows = list(map(json.loads, (d / "rounds.jsonl").read_text().splitlines()))
        assert [x["round"] for x in rows] == list(range(1, 21))
        curve, endpoint = [], None
        for row in rows:
            r = row["round"]
            cm = read_cm(spec["slug"], r)
            assert np.array_equal(cm.sum(2), reference_support)
            f = client_f1(cm)
            pooled = metrics.ten(cm.sum(0))
            for k in metrics.TEN:
                np.testing.assert_allclose(pooled[k], row["private/" + k], atol=1e-12, rtol=0)
            np.testing.assert_allclose(f.mean(), row["private/client_f1_macro_mean"], atol=1e-12, rtol=0)
            point = {"round": r, "f1": float(pooled["f1_macro"]), "mean": float(f.mean()),
                     "p10": float(np.percentile(f, 10)), "median": float(np.median(f)), "min": float(f.min()),
                     "std": float(f.std()), "p90_minus_p10": float(np.percentile(f, 90) - np.percentile(f, 10)),
                     "low256_mean": float(f[n < 256].mean()), "timeDelay_f1": pooled["f1_per_class"][13],
                     "benign_f1": pooled["f1_per_class"][0], "q_change_l1": row.get("q_change_l1")}
            curve.append(point)
            csvrows.append({"arm": spec["arm"], "seed": hp["seed"], **point, **{k: pooled[k] for k in metrics.TEN}})
            if r == 20:
                final_cms[spec["arm"]] = cm
                cohort = []
                for lo, hi in COHORTS:
                    sel = (n >= lo) & (n < hi)
                    cohort.append({"lo": lo, "hi": hi, "clients": int(sel.sum()), "mean": float(f[sel].mean()),
                                   "p10": float(np.percentile(f[sel], 10)),
                                   "by_local_support": conditional_accuracy(cm[sel], counts[sel])})
                pool = cm.sum(0)
                endpoint = {**pooled, **point, "cohorts": cohort, "by_local_support": conditional_accuracy(cm, counts),
                            "benign_to_timeDelay": int(pool[0, 13]), "timeDelay_to_benign": int(pool[13, 0])}
        probe_records = [json.loads(x) for rank in (0, 1)
                         for x in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
        probes, train, external = {}, {}, {}
        for r in (1, 2, 5, 10, 15, 20):
            clients = [c for rec in probe_records if rec["round"] == r for c in rec["clients"]]
            assert sorted(c["client_id"] for c in clients) == sorted(hp["diagnostic_ids"])
            probes[r] = {"all": probe_summary(clients), "low256": probe_summary([c for c in clients if c["train_rows"] < 256])}
            with np.load(d / f"train_cms_r{r:03}.npz") as z:
                assert np.array_equal(z["private"].sum(2), counts)
                f = client_f1(z["private"])
                train[r] = {**metrics.ten(z["private"].sum(0)), "mean": float(f.mean()),
                            "low256_mean": float(f[n < 256].mean())}
        for r in (1, 10, 20):
            records = []
            for rank in (0, 1):
                with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                    records.extend({"client_id": int(cid), "f1": metrics.ten(cm)["f1_macro"],
                                    "support": cm.sum(1).tolist(), "selection_sha256": str(h)}
                                   for cid, cm, h in zip(z["ids"], z["private"], z["selection_sha256"]))
            external[r] = {"clients": records, "mean_f1": float(np.mean([x["f1"] for x in records])),
                           "low256_mean": float(np.mean([x["f1"] for x in records if n[x["client_id"]] < 256]))}
        boot = sum(cfg["payload_bytes"]["bootstrap"].values())
        arms[spec["arm"]] = {"slug": spec["slug"], "hp": hp, "final": endpoint, "curve": curve,
            "delta_f1_10_20": curve[19]["f1"] - curve[9]["f1"],
            "delta_f1_15_20": curve[19]["f1"] - curve[14]["f1"],
            "slope_f1_11_20": float(np.polyfit(np.arange(11, 21), [x["f1"] for x in curve[10:]], 1)[0]),
            "train": train, "probes": probes, "external": external,
            "bytes_total_20": boot + sum(x["bytes_up"] + x["bytes_down"] for x in rows),
            "train_seconds_mean_r2_20": float(np.mean([x["t_train"] for x in rows[1:]])),
            "train_seconds_total": sum(x["t_train"] for x in rows), "eval_seconds_total": sum(x["t_eval"] for x in rows),
            "diagnostic_seconds_total": sum(x.get("t_diagnostics", 0) for x in rows),
            "q_collect_seconds_total": sum(x.get("q_collect_seconds", 0) for x in rows),
            "max_skip": max(x["skipped_frac"] for x in rows), "contrasts": {}}
    effects = {}
    for seed, offset in ((42, 0), (43, 1)):
        for i in range(offset, 16, 2):
            name = f"M{i}"
            cm, f = final_cms[name], client_f1(final_cms[name])
            for label, j in (("augmentation", offset), ("noReal", 2 + offset), ("gate", 4 + offset),
                             ("uniform", 12 + offset), ("fallback", 6 + offset)):
                ref = final_cms[f"M{j}"]
                rf = client_f1(ref)
                delta = f - rf
                arms[name]["contrasts"][label] = {
                    "delta_f1": arms[name]["final"]["f1"] - arms[f"M{j}"]["final"]["f1"],
                    "delta_mean": float(delta.mean()), "delta_p10": arms[name]["final"]["p10"] - arms[f"M{j}"]["final"]["p10"],
                    "delta_low256": float(delta[n < 256].mean()), "better": int((delta > 1e-12).sum()),
                    "worse": int((delta < -1e-12).sum()),
                    "delta_high16384": float(delta[n >= 16384].mean())}
        value = lambda i: arms[f"M{i + offset}"]["final"]["f1"]
        effects[seed] = {"Q_on_synthetic": value(2) - value(0), "real_gate": value(4) - value(2),
                         "fallback": value(6) - value(4), "postepoch": value(8) - value(4),
                         "sqrt_server": value(10) - value(4), "uniform_mixture": value(12) - value(6),
                         "local_minus_uniform": value(14) - value(12)}
    result = {"scope": "Fixed20-round own-test diagnosis; confirm snapshot/source audit separately; two seeds, no significance or post20 proof",
              "arms": arms, "effects": effects}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    with (OUT / "metrics-all-rounds.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    for name, a in arms.items():
        f = a["final"]
        print(name, "F1", round(f["f1"], 6), "p10", round(f["p10"], 6), "low", round(f["low256_mean"], 6),
              "d10_20", round(a["delta_f1_10_20"], 6), "external", round(a["external"][20]["mean_f1"], 6))
    print(json.dumps(effects, indent=2))


if __name__ == "__main__":
    main()
