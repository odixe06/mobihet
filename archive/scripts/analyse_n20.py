"""Audit completion and report the locked N20 factorial; no training or test-driven selection."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics
from scripts.analyse_m20 import COHORTS, conditional_accuracy
from scripts.research_metrics import client_f1, probe_summary

ROOT = Path("artifacts/n20")
OUT = Path("runs/n-study/analysis")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batch_path = Path("runs/n-study/batch.json")
    assert sha(batch_path) == "f6f35bb881f251544fedb6c7d120eb7455533ab52d93bb73a0dba2a65f47f158"
    batch = json.loads(batch_path.read_text())
    receipts = {x["arm"]: x for x in json.loads(Path("runs/n-study/collection/diagnostic-audit.json").read_text())}
    counts = np.load("evidence/research-data/train_counts.npy")
    support = np.load("evidence/research-data/test_support.npy")
    n = counts.sum(1)
    arms, finals, tables, csvrows, verified = {}, {}, {}, [], []
    oldspec = json.loads(Path("evidence/m20-summary/batch.json").read_text())[2]
    oldsource = Path("evidence/m20") / oldspec["slug"] / "provenance/source"
    closure = sorted([str(p.relative_to(oldsource)) for folder in ("core", "methods")
                      for p in (oldsource / folder).glob("*.py")] +
                     ["scripts/run_fl.py", "knowledge/input_order.json", "knowledge/label_mapping.json", "knowledge/runtime.json"])
    for spec in batch:
        name, slug = spec["arm"], spec["slug"]
        folder = ROOT / slug
        d = folder / slug
        cfg = json.loads((d / "config.json").read_text())
        hp = cfg["hp"]
        manifest = json.loads((folder / "provenance/source_manifest.json").read_text())
        assert manifest["base_sha"] == spec["base_sha"] and manifest["overlay"] == spec["overlay_sha256"]
        source = folder / "provenance/source"
        for rel, digest in manifest["files"].items():
            assert sha(source / rel) == digest, (name, rel)
        for rel in closure:
            assert sha(source / rel) == sha(oldsource / rel), (name, rel, "M scientific closure changed")
        rows = list(map(json.loads, (d / "rounds.jsonl").read_text().splitlines()))
        assert [x["round"] for x in rows] == list(range(1, 21))
        curve, qhistory = [], []
        for row in rows:
            r = row["round"]
            with np.load(d / f"cms_r{r:03}.npz") as z:
                assert np.array_equal(z["ids"], np.arange(5343))
                cm = z["private"]
            assert (cm >= 0).all() and np.array_equal(cm.sum(2), support)
            pooled, f = metrics.ten(cm.sum(0)), client_f1(cm)
            for k in metrics.TEN:
                np.testing.assert_allclose(pooled[k], row["private/" + k], atol=1e-12, rtol=0)
            np.testing.assert_allclose([f.mean(), np.percentile(f, 10)],
                                       [row["private/client_f1_macro_mean"], row["private/client_f1_macro_p10"]], atol=1e-12, rtol=0)
            point = {"round": r, "f1": pooled["f1_macro"], "mean": float(f.mean()),
                     "p10": float(np.percentile(f, 10)), "median": float(np.median(f)), "std": float(f.std()),
                     "low256_mean": float(f[n < 256].mean()), "timeDelay_f1": pooled["f1_per_class"][13],
                     "other15_f1": float(np.delete(pooled["f1_per_class"], 13).mean())}
            curve.append(point)
            csvrows.append({"arm": name, "seed": hp["seed"], **point, **{k: pooled[k] for k in metrics.TEN}})
            snapshot = d / "models" / f"r{r:03}.pt"
            assert sha(snapshot) == receipts[name]["snapshots"][r - 1]["sha256"]
            state = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
            q = state["Q"].numpy().copy()
            assert q.shape == (16, 16) and np.isfinite(q).all() and (q >= 0).all()
            np.testing.assert_allclose(q.sum(1), 1, atol=1e-5, rtol=0)
            qhistory.append(q)
            if r == 20:
                finals[name] = cm
                endpoint = {**pooled, **point, "cohorts": [
                    {"lo": lo, "hi": hi, "clients": int(((n >= lo) & (n < hi)).sum()),
                     "mean": float(f[(n >= lo) & (n < hi)].mean()),
                     "by_local_support": conditional_accuracy(cm[(n >= lo) & (n < hi)], counts[(n >= lo) & (n < hi)])}
                    for lo, hi in COHORTS], "by_local_support": conditional_accuracy(cm, counts)}
        qs = np.stack(qhistory)
        tables[name] = qs
        correct = qs.argmax(2) == np.arange(16)
        # Q1..Q19 are consumed in rounds2..20; Q20 has not trained a client yet.
        flips = correct[1:19] != correct[:18]
        policy = qs.copy()
        if hp["synth_target"] == "fallback":
            policy[~correct] = np.broadcast_to(np.eye(16), qs.shape)[~correct]
        margin = qs[:, 13, 13] - np.delete(qs[:, 13], 13, axis=1).max(1)
        transitions = [{"teacher_round": t + 2, "first_consumed_round": t + 3,
                        "correct_after": bool(correct[t + 1, 13]), "margin": float(margin[t + 1]),
                        "synthetic_target_l1_jump": float(np.abs(policy[t + 1, 13] - policy[t, 13]).sum()),
                        "next_test_f1_delta": curve[t + 2]["timeDelay_f1"] - curve[t + 1]["timeDelay_f1"]}
                       for t in range(18) if flips[t, 13]]
        train, external, probes = {}, {}, {}
        records = [json.loads(line) for rank in (0, 1)
                   for line in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
        for r in (1, 2, 5, 10, 15, 20):
            clients = [c for rec in records if rec["round"] == r for c in rec["clients"]]
            assert sorted(c["client_id"] for c in clients) == sorted(hp["diagnostic_ids"])
            probes[r] = {"all": probe_summary(clients), "low256": probe_summary([c for c in clients if c["train_rows"] < 256])}
            with np.load(d / f"train_cms_r{r:03}.npz") as z:
                assert np.array_equal(z["private"].sum(2), counts)
                train[r] = metrics.ten(z["private"].sum(0))
                if r == 20:
                    train_recall = np.divide(z["private"].diagonal(axis1=1, axis2=2), counts,
                                             out=np.zeros_like(counts, dtype=float), where=counts > 0)
                    matched = {}
                    for label, sel in (("all", n >= 0), ("low256", n < 256), ("ge4096", n >= 4096)):
                        weight = support[:, 13] * sel * (counts[:, 13] > 0)
                        matched[label] = {"rows": int(weight.sum()),
                            "train_recall": float((train_recall[:, 13] * weight).sum() / weight.sum()),
                            "test_recall": float((finals[name][:, 13, 13] * sel * (counts[:, 13] > 0)).sum() / weight.sum())}
        for r in (1, 10, 20):
            clients = []
            for rank in (0, 1):
                with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                    clients.extend({"client_id": int(cid), "f1": metrics.ten(c)["f1_macro"]}
                                   for cid, c in zip(z["ids"], z["private"]))
            assert sorted(x["client_id"] for x in clients) == sorted(hp["diagnostic_ids"])
            external[r] = {"clients": clients, "mean_f1": float(np.mean([x["f1"] for x in clients]))}
        arms[name] = {"slug": slug, "cell": spec["cell"], "seed": hp["seed"], "final": endpoint, "curve": curve,
            "train": train, "probes": probes, "external": external,
            "timeDelay_matched_seen_train_test": matched,
            "delta_f1_15_20": curve[-1]["f1"] - curve[14]["f1"],
            "slope_f1_11_20": float(np.polyfit(np.arange(11, 21), [x["f1"] for x in curve[10:]], 1)[0]),
            "teacher": {"correct_argmax_flips_consumed_per_class": flips.sum(0).tolist(), "timeDelay_transitions": transitions,
                        "timeDelay_margin_collected": margin.tolist(), "timeDelay_diag_collected": qs[:, 13, 13].tolist(),
                        "timeDelay_correct_consumed_r2_20": correct[:19, 13].tolist()},
            "bytes_total_20": sum(cfg["payload_bytes"]["bootstrap"].values()) + sum(x["bytes_up"] + x["bytes_down"] for x in rows),
            "train_seconds_mean_r2_20": float(np.mean([x["t_train"] for x in rows[1:]])),
            "train_seconds_total": sum(x["t_train"] for x in rows), "eval_seconds_total": sum(x["t_eval"] for x in rows),
            "diagnostic_seconds_total": sum(x.get("t_diagnostics", 0) for x in rows), "max_skip": max(x["skipped_frac"] for x in rows)}
        log = json.loads((folder / f"{slug}.log").read_text())
        arms[name]["notebook_log_elapsed_seconds"] = max(x["time"] for x in log)
        files = [p for p in folder.rglob("*") if p.is_file()]
        verified.append({"arm": name, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                         "snapshot_sha_matches": 20, "own_cm_support_matches": 20, "M_scientific_closure_matches": closure})
    effects = {}
    for seed, offset in ((42, 0), (43, 1)):
        contrasts = {"B-A": (-1, 1, 0, 0), "D-C": (0, 0, -1, 1), "C-A": (-1, 0, 1, 0),
                     "D-B": (0, -1, 0, 1), "interaction": (1, -1, -1, 1)}
        names = [f"N{i + offset}" for i in (0, 2, 4, 6)]
        effects[seed] = {}
        for label, coeff in contrasts.items():
            delta = sum(w * client_f1(finals[name]) for w, name in zip(coeff, names))
            scalar = {k: sum(w * arms[name]["final"][k] for w, name in zip(coeff, names))
                      for k in ("f1", "mean", "p10", "timeDelay_f1", "other15_f1", "low256_mean")}
            scalar.update({"better": int((delta > 1e-12).sum()), "worse": int((delta < -1e-12).sum()),
                           "paired_delta_p10": float(np.percentile(delta, 10)),
                           "per_class_f1": sum(w * np.array(arms[name]["final"]["f1_per_class"]) for w, name in zip(coeff, names)).tolist()})
            effects[seed][label] = scalar
    result = {"scope": "Locked20round factorial, own-test primary; two seeds, no significance/causal gate-isolation/50round proof",
              "arms": arms, "effects": effects, "artifact_verification": verified}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / "teacher-tables.npz", **tables)
    with (OUT / "metrics-all-rounds.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    print(json.dumps({"effects": effects, "artifact_files": sum(x["files"] for x in verified),
                      "artifact_bytes": sum(x["bytes"] for x in verified)}, indent=2))
    print("N20_ANALYSIS_VERIFIED")


if __name__ == "__main__":
    main()
