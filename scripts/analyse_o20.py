"""Verify actual O20 artifacts and diagnose locked replay contrasts; supports the early terminal subset."""
import argparse
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
from methods.mobihet_replay import frozen, source_schedule, permutation


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--early", action="store_true")
    args = ap.parse_args()
    study = Path("runs/o-study")
    base = study / "early-collect-20261101" if args.early else study
    root = Path("artifacts/o20-early-20261101" if args.early else "artifacts/o20")
    out = base / "analysis"; out.mkdir(exist_ok=True)
    batch = json.loads((base / "batch.json").read_text())
    assert sha(study / "batch.json") == "aaad9baddc1b91d919be0a6eb119a52aa19ce18d37ef8a26a2921f0aa6beaad0"
    assert len(batch) == (6 if args.early else 8)
    receipt = {}
    for spec in batch:
        location = base / "audits" / spec["arm"] if args.early else study / "collection"
        for item in json.loads((location / "diagnostic-audit.json").read_text()): receipt[item["arm"]] = item
    count = np.load("evidence/research-data/train_counts.npy"); n = count.sum(1)
    support = np.load("evidence/research-data/test_support.npy")
    prior = json.loads(Path("runs/n-study/analysis/analysis.json").read_text())["arms"]
    n_batch = json.loads(Path("runs/n-study/batch.json").read_text())
    oldsource = Path("artifacts/n20") / n_batch[0]["slug"] / "provenance/source"
    closure = [str(p.relative_to(oldsource)) for folder in ("core", "methods") for p in (oldsource / folder).glob("*.py")]
    closure += ["scripts/run_fl.py", "knowledge/input_order.json", "knowledge/label_mapping.json", "knowledge/runtime.json"]
    arms, finals, csvrows, verification = {}, {}, [], []
    for spec in batch:
        name, slug = spec["arm"], spec["slug"]
        folder = root / slug; d = folder / slug; source = folder / "provenance/source"
        cfg = json.loads((d / "config.json").read_text()); hp = cfg["hp"]
        manifest = json.loads((folder / "provenance/source_manifest.json").read_text())
        assert manifest["base_sha"] == spec["base_sha"] and manifest["overlay"] == spec["overlay_sha256"]
        for rel, digest in manifest["files"].items(): assert sha(source / rel) == digest
        for rel in closure: assert sha(source / rel) == sha(oldsource / rel), (name, rel)
        raw = source_schedule(hp["seed"], torch.device("cpu"), frozen.SHA256).numpy()
        target = raw.copy()
        if hp["replay_mode"] != "full":
            for c, order in enumerate(permutation()):
                other = np.delete(np.arange(16), c)
                target[:, c, other] = raw[:, c, other].astype(np.float64).mean(-1, keepdims=True) if hp["replay_mode"] == "uniform" else raw[:, c, order]
        rows = list(map(json.loads, (d / "rounds.jsonl").read_text().splitlines()))
        assert [r["round"] for r in rows] == list(range(1, 21))
        curve = []
        for row in rows:
            r = row["round"]
            snapshot = d / "models" / f"r{r:03}.pt"
            assert sha(snapshot) == receipt[name]["snapshots"][r - 1]["sha256"]
            state = torch.load(snapshot, weights_only=True, map_location="cpu", mmap=True)
            assert np.array_equal(state["Q_raw"].numpy(), raw[r - 1])
            np.testing.assert_allclose(state["Q"].numpy(), target[r - 1], rtol=0, atol=1e-7)
            assert row["replay_consumed_round"] == (None if r == 1 else r - 1)
            with np.load(d / f"cms_r{r:03}.npz") as z:
                assert np.array_equal(z["ids"], np.arange(5343)); cm = z["private"]
            assert np.array_equal(cm.sum(2), support) and (cm >= 0).all()
            pooled, f = metrics.ten(cm.sum(0)), client_f1(cm)
            for k in metrics.TEN: np.testing.assert_allclose(pooled[k], row["private/" + k], atol=1e-12, rtol=0)
            np.testing.assert_allclose(f.mean(), row["private/client_f1_macro_mean"], atol=1e-12, rtol=0)
            point = {"round": r, "f1": pooled["f1_macro"], "mean": float(f.mean()), "p10": float(np.percentile(f, 10)),
                "low256_mean": float(f[n < 256].mean()), "std": float(f.std()), "timeDelay_f1": pooled["f1_per_class"][13],
                "other15_f1": float(np.delete(pooled["f1_per_class"], 13).mean())}
            curve.append(point); csvrows.append({"arm": name, "seed": hp["seed"], **point, **{k: pooled[k] for k in metrics.TEN}})
            if r == 20:
                finals[name] = cm
                endpoint = {**point, **pooled, "by_local_support": conditional_accuracy(cm, count), "cohorts": [
                    {"lo": lo, "hi": hi, "clients": int(((n >= lo) & (n < hi)).sum()),
                     "mean": float(f[(n >= lo) & (n < hi)].mean()),
                     "by_local_support": conditional_accuracy(cm[(n >= lo) & (n < hi)], count[(n >= lo) & (n < hi)])}
                    for lo, hi in COHORTS]}
        probes, external, train = {}, {}, {}
        records = [json.loads(line) for rank in (0, 1) for line in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
        for r in (1, 2, 5, 10, 15, 20):
            clients = [c for rec in records if rec["round"] == r for c in rec["clients"]]
            assert sorted(c["client_id"] for c in clients) == sorted(hp["diagnostic_ids"])
            probes[r] = {"all": probe_summary(clients), "low256": probe_summary([c for c in clients if c["train_rows"] < 256])}
            with np.load(d / f"train_cms_r{r:03}.npz") as z:
                assert np.array_equal(z["private"].sum(2), count)
                train[r] = metrics.ten(z["private"].sum(0))
                if r == 20:
                    tr = np.divide(z["private"].diagonal(axis1=1, axis2=2), count, out=np.zeros_like(count, dtype=float), where=count > 0)
                    matched = {}
                    for key, sel in (("all", n >= 0), ("low256", n < 256), ("ge4096", n >= 4096)):
                        w = support[:, 13] * sel * (count[:, 13] > 0)
                        matched[key] = {"rows": int(w.sum()), "train_recall": float((tr[:, 13] * w).sum() / w.sum()),
                            "test_recall": float((finals[name][:, 13, 13] * sel * (count[:, 13] > 0)).sum() / w.sum())}
        for r in (1, 10, 20):
            clients = []
            for rank in (0, 1):
                with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                    clients.extend({"client_id": int(cid), "f1": metrics.ten(c)["f1_macro"]} for cid, c in zip(z["ids"], z["private"]))
            external[r] = {"mean_f1": float(np.mean([c["f1"] for c in clients])), "clients": clients}
        log = json.loads((folder / (slug + ".log")).read_text())
        arms[name] = {"slug": slug, "seed": hp["seed"], "cell": spec["cell"], "final": endpoint, "curve": curve,
            "probes": probes, "external": external, "train": train, "timeDelay_matched_seen_train_test": matched,
            "delta_f1_15_20": curve[-1]["f1"] - curve[14]["f1"],
            "slope_f1_11_20": float(np.polyfit(np.arange(11, 21), [p["f1"] for p in curve[10:]], 1)[0]),
            "train_seconds_mean_r2_20": float(np.mean([r["t_train"] for r in rows[1:]])),
            "train_seconds_total": sum(r["t_train"] for r in rows), "eval_seconds_total": sum(r["t_eval"] for r in rows),
            "diagnostic_seconds_total": sum(r.get("t_diagnostics", 0) for r in rows),
            "notebook_log_elapsed_seconds": max(e["time"] for e in log), "max_skip": max(r["skipped_frac"] for r in rows),
            "packet_equivalent_bytes_20": 3333504, "offline_preload_bytes_20": 3402795}
        if name in ("O0", "O1"):
            parent = prior["N" + name[1]]
            delta = {k: endpoint[k] - parent["final"][k] for k in ("f1", "timeDelay_f1", "p10", "low256_mean")}
            limits = {"f1": .001, "timeDelay_f1": .01, "p10": .01, "low256_mean": .01}
            arms[name]["replay_control_tripwire"] = {"delta_vs_N": delta, "limits": limits, "pass": all(abs(delta[k]) <= limits[k] for k in limits)}
        files = [p for p in folder.rglob("*") if p.is_file()]
        verification.append({"arm": name, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "snapshot_sha_matches": 20, "CM_support_tenmetrics_matches": 20, "raw_teacher_matches": 20, "scientific_closure_vs_N": closure})
    effects = []
    for offset, seed in ((0, 42), (1, 43)):
        reference = f"O{offset}"
        if reference not in arms: continue
        for i, label in ((2, "real_gate"), (4, "uniform"), (6, "permuted")):
            name = f"O{i + offset}"
            if name not in arms: continue
            delta = client_f1(finals[name]) - client_f1(finals[reference])
            effects.append({"seed": seed, "contrast": name + "-" + reference, "label": label,
                **{k: arms[name]["final"][k] - arms[reference]["final"][k] for k in ("f1", "mean", "p10", "low256_mean", "timeDelay_f1", "other15_f1")},
                "better": int((delta > 1e-12).sum()), "worse": int((delta < -1e-12).sum()),
                "paired_delta_p10": float(np.percentile(delta, 10)),
                "per_class_f1": (np.array(arms[name]["final"]["f1_per_class"]) - arms[reference]["final"]["f1_per_class"]).tolist()})
    result = {"scope": "Early6of8, incomplete paired seeds" if args.early else "Full8run replay; no significance/liveFL/50round proof",
              "arms": arms, "effects": effects, "artifact_verification": verification}
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "metrics-all-rounds.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0])); w.writeheader(); w.writerows(csvrows)
    print(json.dumps({"effects": effects, "control_tripwires": {k: a["replay_control_tripwire"] for k, a in arms.items() if "replay_control_tripwire" in a},
        "files": sum(v["files"] for v in verification), "bytes": sum(v["bytes"] for v in verification)}, indent=2))
    print("O20_ANALYSIS_VERIFIED")


if __name__ == "__main__": main()
