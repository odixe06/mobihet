"""Post-hoc B18 paired CM and measured source diagnostics; no training/tuning."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics


def main():
    out = ROOT / "runs/sim-kaggle-b18r1"
    verdict = json.loads((out / "verdict-r20.json").read_text())
    result = {}
    lines = ["# B18 source/CM diagnostics", "", "No new training. Source r19 influences main r20; source r20 is unused at the endpoint.", "",
             "| seed | phase | Δmacro r20 | Δmacro TB16–20 | Δbenign F1 | ΔTD F1 | Δclass11 F1 |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for spec in json.loads((out / "batch.json").read_text()):
        run = ROOT / "artifacts/sim-kaggle-b18r1" / spec["slug"] / spec["slug"]
        assert json.loads((run / "source-audit.json").read_text())["pass"]
        assert json.loads((out / f"resource-audit-s{spec['seed']}.json").read_text())["pass"]
        seed = str(spec["seed"])
        rows, cms, first, cfgs = {}, {}, {}, {}
        for phase in ("shadow", "C", "E", "H"):
            rows[phase] = [json.loads(s) for s in (run / phase / "rounds.jsonl").read_text().splitlines()]
            cfgs[phase] = json.loads((run / phase / "config.json").read_text())
            with np.load(run / phase / "cms_r020.npz") as z:
                cms[phase] = z["private"].sum(0)
            with np.load(run / phase / "cms_r001.npz") as z:
                first[phase] = z["private"].copy()
        seed_result = {"arms": {}, "source": {}, "bootstrap": cfgs["H"]["payload_bytes"],
                       "shadow_train_seconds_includes_both_probes": sum(r["t_train"] for r in rows["shadow"])}
        seed_result["actual_teacher_r19"] = {}
        for phase in ("E", "H"):
            with np.load(run / phase / "source/rank0_r019.npz") as z:
                q = z["Q"]
                seed_result["actual_teacher_r19"][phase] = {
                    "Q": q.tolist(), "argmax_disagrees_with_trueclass": np.flatnonzero(q.argmax(1) != np.arange(16)).tolist()}
        for phase in ("E", "H"):
            cm, base = cms[phase], cms["C"]
            f = lambda a: 2 * np.diag(a) / np.maximum(a.sum(0) + a.sum(1), 1)
            delta = f(cm) - f(base)
            trace = [a["private/f1_macro"] - b["private/f1_macro"] for a, b in zip(rows[phase], rows["C"])]
            sr = verdict["seeds"][seed]["stats"][phase]
            seed_result["arms"][phase] = {"delta_macro_trace": trace, "delta_class_f1": delta.tolist(),
                "delta_macro_mean16_20": float(np.mean(trace[-5:])),
                "r1_cm_L1_vs_C": int(abs(first[phase] - first["C"]).sum()),
                "delta_benign_to_TD": int(cm[0, 13] - base[0, 13]),
                "delta_TD_to_benign": int(cm[13, 0] - base[13, 0]),
                "delta_benign_to_attack_not_TD": int((cm[0, 1:13] - base[0, 1:13]).sum() + (cm[0, 14:] - base[0, 14:]).sum()),
                "main_train_seconds": sr["main_train_seconds"],
                "main_plus_shared_shadow_seconds_ratio_vs_C":
                    (sr["main_train_seconds"] + seed_result["shadow_train_seconds_includes_both_probes"]) /
                    verdict["seeds"][seed]["stats"]["C"]["main_train_seconds"]}
            lines.append(f"| {seed} | {phase} | {trace[-1]:+.8f} | {np.mean(trace[-5:]):+.8f} | {delta[0]:+.8f} | {delta[13]:+.8f} | {delta[11]:+.8f} |")
        for r in (1, 19, 20):
            shards = []
            for rank in (0, 1):
                with np.load(run / "shadow/source" / f"rank{rank}_r{r:03d}.npz") as z:
                    shards.append({k: z[k].copy() for k in z.files})
            ids = np.concatenate([s["ids"] for s in shards])
            probe_counts = np.concatenate([s["counts"] for s in shards])
            # Fixed population weighting of eligible cells, identical E/H.
            from scripts.sim_fold_val import fold_counts
            weights = fold_counts()[ids, 0] * (probe_counts >= 8)
            total = weights.sum(0)
            assert np.all(total > 0)
            detail = {}
            for phase in ("E", "H"):
                means = np.concatenate([s["mean_" + phase] for s in shards])
                soft = (means * weights[..., None]).sum(0) / total[:, None]
                cm = np.concatenate([s["cm_" + phase] for s in shards])
                ten = metrics.ten(cm.sum(0))
                detail[phase] = {"measured_probe_rows": int(probe_counts.sum()),
                                 "population_weighted_mean_true_probability": float(soft.diagonal().mean()),
                                 "population_weighted_mean_by_trueclass": soft.tolist(),
                                 "hard_probe_ten": {k: float(ten[k]) for k in metrics.TEN}}
            seed_result["source"][str(r)] = detail
        result[seed] = seed_result
    result["limits"] = ["Sim200 foldval reused, two seeds not significance.",
                        "Source hard metrics on matched early/heldout probe populations, not main future utility.",
                        "Raw soft means do not recover per-input soft Brier/NLL/variance; none inferred.",
                        "Outside-fit source changes temporal covariates too; no claim of event independence.",
                        "Measured shadow phase includes both probe passes; ratio is experimental elapsed cost, not FLOPs or a single-probe deployment estimate.",
                        "r1 CM agreement is bounded observation, not proof of whole-trajectory bit exactness."]
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines += ["", *["- " + s for s in result["limits"]]]
    (out / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print("B18_DIAGNOSTICS_WRITTEN")


if __name__ == "__main__":
    main()
