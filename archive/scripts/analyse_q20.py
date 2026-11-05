"""Q20 fixed-endpoint contrasts and per-car diagnosis from fully collected, audited artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from core import metrics
from scripts.analyse_m20 import COHORTS, conditional_accuracy
from scripts.research_metrics import client_f1

STUDY = ROOT / "runs/q-study"
OUT = STUDY / "analysis"
PROBES = (1, 2, 5, 10, 15, 20)
CONTRASTS = [(1, 0, "query_control"), (2, 1, "mixed_support"), (3, 2, "input_resolution"),
             (4, 3, "sqrt_input_weight"), (5, 3, "real_KD"), (6, 5, "margin_vs_real"),
             (6, 7, "margin_vs_strength"), (3, 1, "mixed_input_total"), (6, 0, "margin_vs_stream")]
TRIPWIRE = {"f1": .001, "timeDelay_f1": .01, "p10": .01, "low256_mean": .01}


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    closure = json.loads((STUDY / "closure.json").read_text())
    inventory = json.loads((STUDY / "closure-verification.json").read_text())
    assert closure["full20_pass"] == 16 and closure["all16_classified"]
    assert sha(STUDY / "closure.json") == inventory["closure_sha256"]
    for path, value in inventory["artifact_sha256"].items():
        assert sha(ROOT / path) == value, path
    batch = json.loads((STUDY / "batch.json").read_text())
    assert len(batch) == 16 and closure["batch_sha256"] == sha(STUDY / "batch.json")
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    n = counts.sum(1)
    assert counts.shape == support.shape == (5343, 16)
    assert counts.sum() == 37361273 and support.sum() == 9347679
    arms, finals, csvrows, bank_hashes = {}, {}, [], {}
    for spec in batch:
        name = spec["arm"]
        rec = closure["runs"][name]
        assert rec["pulled"] and rec["audited"] and rec["state"]["status"] == "COMPLETE"
        report_dir = STUDY / "collection" / name
        for report, value in rec["report_sha256"].items():
            assert sha(report_dir / report) == value
        detail, = json.loads((report_dir / "diagnostic-audit.json").read_text())
        folder = ROOT / "artifacts/q20" / spec["slug"]
        d = folder / spec["slug"]
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
        assert [row["round"] for row in rows] == list(range(1, 21))
        curve, teacher = [], []
        for row, snapshot in zip(rows, detail["snapshots"]):
            r = row["round"]
            assert snapshot["round"] == r
            assert sha(d / "models" / f"r{r:03}.pt") == snapshot["sha256"]
            state = torch.load(d / "models" / f"r{r:03}.pt", weights_only=True, map_location="cpu")
            assert all(torch.isfinite(t).all() for t in state.values() if torch.is_tensor(t))
            with np.load(d / f"cms_r{r:03}.npz") as z:
                assert np.array_equal(z["ids"], np.arange(5343))
                cm = z["private"]
            assert np.array_equal(cm.sum(2), support)
            pooled, f = metrics.ten(cm.sum(0)), client_f1(cm)
            for key in metrics.TEN:
                np.testing.assert_allclose(pooled[key], row["private/" + key], rtol=0, atol=1e-12)
            point = {"round": r, "f1": float(pooled["f1_macro"]), "mean": float(f.mean()),
                "p10": float(np.percentile(f, 10)), "std": float(f.std()),
                "median": float(np.median(f)), "min": float(f.min()),
                "low256_mean": float(f[n < 256].mean()), "ge4096_mean": float(f[n >= 4096].mean()),
                "timeDelay_f1": pooled["f1_per_class"][13], "benign_f1": pooled["f1_per_class"][0],
                "other15_f1": float(np.delete(pooled["f1_per_class"], 13).mean())}
            curve.append(point)
            csvrows.append({"arm": name, "mode": spec["mode"], "seed": cfg["hp"]["seed"], **point,
                            **{key: pooled[key] for key in metrics.TEN}})
            q = state["Q"].double()
            td = {"round": r, "q_entropy": float(-(q * q.clamp_min(1e-30).log()).sum(1).mean()),
                  "q_diagonal_mean": float(q.diag().mean()), "q_argmax": q.argmax(1).tolist()}
            if state.get("bank_target") is not None:
                target = state["bank_target"].double()
                q_rows = q[state["bank_y"]]
                td.update(input_entropy=float(-(target * target.clamp_min(1e-30).log()).sum(1).mean()),
                    input_label_agreement=float((target.argmax(1) == state["bank_y"]).double().mean()),
                    input_to_class_KL=float((target * (target.clamp_min(1e-30).log() - q_rows.clamp_min(1e-30).log())).sum(1).mean()))
                bank_hashes[(cfg["hp"]["seed"], r)] = bank_hashes.get((cfg["hp"]["seed"], r), sha_tensor(state["bank_X"]))
                assert bank_hashes[(cfg["hp"]["seed"], r)] == sha_tensor(state["bank_X"])
            teacher.append(td)
            if r == 20:
                finals[name] = cm
                cohorts = []
                for lo, hi in COHORTS:
                    selected = (n >= lo) & (n < hi)
                    cohorts.append({"lo": lo, "hi": hi, "clients": int(selected.sum()),
                        "mean": float(f[selected].mean()), "p10": float(np.percentile(f[selected], 10)),
                        "by_local_support": conditional_accuracy(cm[selected], counts[selected])})
                final = {**point, **pooled, "cohorts": cohorts, "by_local_support": conditional_accuracy(cm, counts)}
        train, shadow, external = {}, {}, {}
        records = [json.loads(line) for rank in (0, 1) for line in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
        for r in PROBES:
            with np.load(d / f"train_cms_r{r:03}.npz") as z:
                tr = z["private"]
                assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(tr.sum(2), counts)
            train[r] = metrics.ten(tr.sum(0))
            if r == 20:
                recall = np.divide(tr.diagonal(axis1=1, axis2=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
                matched = {}
                for label, selected in [("all", n >= 0), ("low256", n < 256), ("ge4096", n >= 4096)]:
                    weights = support[:, 13] * selected * (counts[:, 13] > 0)
                    matched[label] = {"rows": int(weights.sum()), "train_recall": float((recall[:, 13] * weights).sum() / weights.sum()),
                        "test_recall": float((finals[name][:, 13, 13] * selected * (counts[:, 13] > 0)).sum() / weights.sum())}
            clients = [c for rec in records if rec["round"] == r for c in rec["clients"]]
            assert sorted(c["client_id"] for c in clients) == sorted(cfg["hp"]["diagnostic_ids"])
            shadow[r] = {"clients": len(clients), "median_gradient_norm": float(np.median([c["gradient_norm"] for c in clients])),
                         "exceeds_clip": sum(c["exceeds_clip"] for c in clients), "scope": "Shadow fp32/dropout0/next teacher; not actual AMP/clipping or gradient-conflict diagnosis"}
        for r in (1, 10, 20):
            values = []
            for rank in (0, 1):
                with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                    values.extend({"client_id": int(cid), "f1": metrics.ten(cm)["f1_macro"], "selection_sha256": str(h)}
                        for cid, cm, h in zip(z["ids"], z["private"], z["selection_sha256"]))
            external[r] = {"clients": values, "mean_f1": float(np.mean([v["f1"] for v in values])),
                          "low256_mean": float(np.mean([v["f1"] for v in values if n[v["client_id"]] < 256]))}
        log = json.loads((folder / (spec["slug"] + ".log")).read_text())
        timing = {key: float(sum(row.get(key, 0) for row in rows)) for key in
                  ["t_train", "t_eval", "t_diagnostics", "server_seconds", "scoring_seconds_local"]}
        timing["notebook_log_elapsed_seconds"] = max(entry["time"] for entry in log)
        timing["unattributed_seconds"] = timing["notebook_log_elapsed_seconds"] - sum(timing[key] for key in ["t_train", "t_eval", "t_diagnostics"])
        boot = cfg["payload_bytes"]["bootstrap"]
        costs = {"bootstrap_up": boot["up"], "bootstrap_down": boot["down"],
                 "rounds_up": sum(row["bytes_up"] for row in rows), "rounds_down": sum(row["bytes_down"] for row in rows)}
        costs["total"] = sum(costs.values())
        arms[name] = {"mode": spec["mode"], "seed": cfg["hp"]["seed"], "final": final, "curve": curve, "train": train,
            "shadow": shadow, "external": external, "teacher": teacher, "matched_timeDelay": matched,
            "delta_f1_15_20": curve[-1]["f1"] - curve[14]["f1"], "delta_low256_15_20": curve[-1]["low256_mean"] - curve[14]["low256_mean"],
            "bytes": costs, "timing": timing, "max_skip": max(row["skipped_frac"] for row in rows)}
        print("Q20_ANALYSED", name, flush=True)
    effects, harmed_both = [], {}
    for target, reference, label in CONTRASTS:
        paired = []
        for offset, seed in [(0, 42), (1, 43)]:
            name, ref = f"Q{2 * target + offset}", f"Q{2 * reference + offset}"
            delta = client_f1(finals[name]) - client_f1(finals[ref]); paired.append(delta)
            effect = {"label": label, "seed": seed, "target": name, "reference": ref,
                **{key: arms[name]["final"][key] - arms[ref]["final"][key] for key in
                   ["f1", "mean", "p10", "std", "low256_mean", "ge4096_mean", "timeDelay_f1", "benign_f1", "other15_f1"]},
                "per_class_delta": (np.asarray(arms[name]["final"]["f1_per_class"]) - arms[ref]["final"]["f1_per_class"]).tolist(),
                "better_clients": int((delta > 1e-12).sum()), "worse_clients": int((delta < -1e-12).sum()),
                "low256_better_fraction": float((delta[n < 256] > 1e-12).mean()), "paired_delta_p10": float(np.percentile(delta, 10)),
                "paired_delta_quantiles": np.percentile(delta, [0, 10, 25, 50, 75, 90, 100]).tolist(),
                "worse_by_0_01": int((delta < -.01).sum()), "worse_by_0_05": int((delta < -.05).sum())}
            if label == "query_control":
                effect["tripwire_limits"] = TRIPWIRE
                effect["tripwire_pass"] = all(abs(effect[key]) <= limit for key, limit in TRIPWIRE.items())
            effects.append(effect)
        harmed_both[label] = {"all": int(((paired[0] < -1e-12) & (paired[1] < -1e-12)).sum()),
            "low256": int(((paired[0] < -1e-12) & (paired[1] < -1e-12) & (n < 256)).sum()),
            "worse_by_0_01": int(((paired[0] < -.01) & (paired[1] < -.01)).sum()),
            "worse_by_0_05": int(((paired[0] < -.05) & (paired[1] < -.05)).sum())}
    old_path = ROOT / "runs/p-study/analysis/analysis.json"
    old_receipt = json.loads((old_path.parent / "completion-verification.json").read_text())
    assert sha(old_path) == old_receipt["report_sha256"][str(old_path.relative_to(ROOT))]
    old = json.loads(old_path.read_text())
    references = []
    for offset, seed in [(0, 42), (1, 43)]:
        for qcell, pcell, label in [(0, 2, "stream_vs_P_C2"), (6, 4, "margin_vs_P_C4")]:
            name, ref = f"Q{qcell * 2 + offset}", f"P{pcell * 2 + offset}"
            effect = {"label": label, "seed": seed, "target": name, "reference": ref,
                      **{key: arms[name]["final"][key] - old["arms"][ref]["final"][key] for key in TRIPWIRE}}
            if label == "stream_vs_P_C2": effect["tripwire_pass"] = all(abs(effect[key]) <= limit for key, limit in TRIPWIRE.items())
            references.append(effect)
    result = {"closure_sha256": sha(STUDY / "closure.json"), "inventory_sha256": sha(STUDY / "closure-verification.json"),
              "arms": arms, "effects": effects, "harmed_both_seeds": harmed_both, "cross_batch_references": references,
              "mixed_round1_pooled_spread": {str(seed): max(arms[f"Q{i}"]["curve"][0]["f1"] for i in range(4 + off, 16, 2))
                  - min(arms[f"Q{i}"]["curve"][0]["f1"] for i in range(4 + off, 16, 2)) for off, seed in [(0, 42), (1, 43)]},
              "delta_magnitude_scope": "Thresholds .01/.05 are post-hoc descriptive, not acceptance/HP gates. Query controls also differ per car; two seeds do not estimate a noise bound or prove each negative delta causal. Full remote private weights not retained.",
              "query_tripwires_pass": all(e["tripwire_pass"] for e in effects if e["label"] == "query_control"),
              "scope": "Fixed20/cosine50/full5343/W1, two exploratory seeds; live total effects, no significance, privatebank replay, validation-quality, mixedarchitecture, DP or contact proof"}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "metrics-all-rounds.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    print("Q20_ANALYSIS_VERIFIED_DONE", flush=True)


def sha_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    main()
