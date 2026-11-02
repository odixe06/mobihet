"""Verify recovered P21 and supplement sealed P20 analysis, without training or replacing originals."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics
from core.dagsnet import DAGSNet, width
from core.flat import Layout
from scripts.analyse_m20 import conditional_accuracy
from scripts.audit_p20 import extra
from scripts.audit_screening import audit
from scripts.research_metrics import client_f1

RECOVERY = ROOT / "runs/p-study/recovery-p21"
OUT = ROOT / "runs/p-study/analysis-p21-recovery"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    source_names = ["scripts/analyse_p21_recovery.py", "scripts/audit_p20.py", "scripts/audit_screening.py",
                    "scripts/research_metrics.py", "core/metrics.py", "core/flat.py", "core/dagsnet.py"]
    source_before = {name: sha(ROOT / name) for name in source_names}
    state_path = Path("/home/odixe/.codex/long-jobs/jobs/kaggle-p21-recovery20-20261101/state.json")
    state = json.loads(state_path.read_text())
    assert state["status"] == "succeeded" and state["exit_code"] == 0 and state["marker_seen"]
    assert "P21_RECOVERY20_PULL_AUDIT_DONE" in Path(state["log"]).read_text().splitlines()
    complete = json.loads((RECOVERY / "complete.json").read_text())
    assert complete["rounds"] == 20 and complete["status"] == "PASS_BOUNDED"
    assert complete["batch_sha256"] == sha(RECOVERY / "batch.json")
    assert complete["preflight_sha256"] == sha(RECOVERY / "preflight.json")
    for name, value in complete["reports"].items():
        assert sha(RECOVERY / "audit" / name) == value
    spec, = json.loads((RECOVERY / "batch.json").read_text())
    batch = {s["arm"]: s for s in json.loads((ROOT / "runs/p-study/batch.json").read_text())}
    assert spec["hp"] == batch["P21"]["hp"] and spec["base_sha"] == batch["P21"]["base_sha"]
    assert all(spec["overlay_sha256"][name] == value for name, value in batch["P21"]["overlay_sha256"].items())
    assert complete["ref"] == f'{spec["account"]}/{spec["slug"]}/1'
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    n = counts.sum(1)
    root = ROOT / "artifacts/p20"
    d = root / spec["slug"] / spec["slug"]
    common, detail = audit(root, spec, support, endpoint=20), extra(root, spec, counts)
    assert common == json.loads((RECOVERY / "audit/audit.json").read_text())[0]
    assert detail == json.loads((RECOVERY / "audit/diagnostic-audit.json").read_text())[0]
    for rank in (0, 1):
        records = [json.loads(line) for line in (d / "recovery" / f"rank{rank}.jsonl").read_text().splitlines()]
        assert records[0] == {"rank": rank, "event": "compile_guard", "compiled_ranks": 2, "world": 2}
        assert len(records) == 21
        for r, record in enumerate(records[1:], 1):
            assert record == {"rank": rank, "event": "round_finite", "round": r, "groups": 42,
                              "clients": 2672 if rank == 0 else 2671}
    cfg = json.loads((d / "config.json").read_text())
    lay, model = Layout(width(1)), DAGSNet(width(1))
    assert cfg["cfg"] == width(1) and lay.P == cfg["n_params"] == 395408 and lay.B == cfg["n_buffers"]
    for r in range(1, 21):
        snapshot = torch.load(d / "models" / f"r{r:03}.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(lay.state_dict(snapshot["teacher_p"], snapshot["teacher_b"]), strict=True)
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    old_dir = ROOT / "runs/p-study/analysis"
    old_receipt = json.loads((old_dir / "completion-verification.json").read_text())
    old_path = old_dir / "analysis.json"
    assert sha(old_path) == old_receipt["report_sha256"][str(old_path.relative_to(ROOT))]
    old = json.loads(old_path.read_text())
    rows = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
    curve = []
    for row in rows:
        r = row["round"]
        with np.load(d / f"cms_r{r:03}.npz") as z:
            cm = z["private"]
        pooled, f = metrics.ten(cm.sum(0)), client_f1(cm)
        point = {"round": r, **{key: pooled[key] for key in metrics.TEN}, "mean": float(f.mean()),
                 "p10": float(np.percentile(f, 10)), "std": float(f.std()), "low256_mean": float(f[n < 256].mean()),
                 "ge4096_mean": float(f[n >= 4096].mean())}
        curve.append(point)
        if r == 20:
            final = {**point, **pooled, "by_local_support": conditional_accuracy(cm, counts),
                     "low256_by_local_support": conditional_accuracy(cm[n < 256], counts[n < 256])}
    teacher = {}
    for r in (1, 2, 5, 10, 15, 20):
        paired = [v for rank in (0, 1) for v in json.loads((d / f"teacher_probe_rank{rank}_r{r:03}.json").read_text())]
        teacher[r] = {}
        for fold in ("train", "test"):
            t = np.asarray([v[fold]["teacher_cm"] for v in paired]).sum(0)
            s = np.asarray([v[fold]["student_cm"] for v in paired]).sum(0)
            opportunity = sum(v[fold]["teacher_correct_student_wrong"] for v in paired)
            risk = sum(v[fold]["student_correct_teacher_wrong"] for v in paired)
            assert int(t.trace() - s.trace()) == opportunity - risk
            teacher[r][fold] = {"clients": len(paired), "rows": int(t.sum()), "teacher": metrics.ten(t),
                                "student": metrics.ten(s), "teacher_correct_student_wrong": opportunity,
                                "student_correct_teacher_wrong": risk}
    effects, paired_deltas = [], {}
    for seed, name in [(42, "P20"), (43, "P21R1")]:
        target = old["arms"][name]["final"] if seed == 42 else final
        target_spec = batch["P20"] if seed == 42 else spec
        with np.load(root / target_spec["slug"] / target_spec["slug"] / "cms_r020.npz") as z:
            target_cm = z["private"]
        assert np.array_equal(target_cm.sum(2), support)
        for key, value in metrics.ten(target_cm.sum(0)).items():
            np.testing.assert_allclose(value, target[key], rtol=0, atol=1e-12)
        for cell, label in [(8, "hybrid_vs_proxy"), (7, "hybrid_vs_input_ensemble"),
                            (2, "hybrid_vs_stream_class"), (1, "hybrid_vs_augmentation")]:
            reference = f"P{2 * cell + (seed == 43)}"
            refspec = batch[reference]
            with np.load(root / refspec["slug"] / refspec["slug"] / "cms_r020.npz") as z:
                refcm = z["private"]
            assert np.array_equal(refcm.sum(2), support)
            ref = old["arms"][reference]["final"]
            for key in metrics.TEN:
                assert abs(metrics.ten(refcm.sum(0))[key] - ref[key]) < 1e-12
            delta = client_f1(target_cm) - client_f1(refcm)
            paired_deltas.setdefault(label, []).append(delta)
            effects.append({"seed": seed, "target": name, "reference": reference, "label": label,
                "pooled_f1": target["f1_macro"] - ref["f1_macro"],
                "mean": target["mean"] - ref["mean"], "p10": target["p10"] - ref["p10"],
                "low256_mean": target["low256_mean"] - ref["low256_mean"],
                "per_class_delta": (np.asarray(target["f1_per_class"]) - ref["f1_per_class"]).tolist(),
                "improved": int((delta > 1e-12).sum()), "harmed": int((delta < -1e-12).sum()),
                "low256_improved": int(((delta > 1e-12) & (n < 256)).sum()),
                "paired_delta_p10": float(np.percentile(delta, 10))})
    log = json.loads((d.parent / (spec["slug"] + ".log")).read_text())
    timing = {key: sum(row.get(key, 0) for row in rows) for key in
              ["t_train", "t_eval", "t_diagnostics", "server_seconds", "scoring_seconds_local", "bank_seconds_local"]}
    timing["notebook_log_elapsed_seconds"] = max(entry["time"] for entry in log)
    timing["unattributed_seconds"] = timing["notebook_log_elapsed_seconds"] - sum(timing[k] for k in ["t_train", "t_eval", "t_diagnostics"])
    boot = cfg["payload_bytes"]["bootstrap"]
    costs = {"bootstrap_up": boot["up"], "bootstrap_down": boot["down"],
             "rounds_up": sum(row["bytes_up"] for row in rows), "rounds_down": sum(row["bytes_down"] for row in rows)}
    costs["total"] = sum(costs.values())
    result = {"logical_arm": "P21", "recovery_arm": "P21R1", "ref": complete["ref"], "final": final,
              "curve": curve, "effects": effects, "teacher_probe": teacher, "bytes": costs, "timing": timing,
              "harmed_both_seeds": {label: {"all": int(((pair[0] < -1e-12) & (pair[1] < -1e-12)).sum()),
                  "low256": int(((pair[0] < -1e-12) & (pair[1] < -1e-12) & (n < 256)).sum())}
                  for label, pair in paired_deltas.items()},
              "max_skip": max(row["skipped_frac"] for row in rows),
              "delta_pooled_15_20": curve[-1]["f1_macro"] - curve[14]["f1_macro"],
              "delta_low256_15_20": curve[-1]["low256_mean"] - curve[14]["low256_mean"],
              "old_analysis_sha256": sha(old_path), "scope": "Supplement: original P21 remains failed; fresh recovery at fixed20, two exploratory seeds, live total effects; no rootcause fix, significance, privatecheckpoint, DP or heterogeneous proof"}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "metrics-all-rounds.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
    lines = ["# P21 recovery — đủ10metric own-test, 20round", "",
             f"`{complete['ref']}`; logicalP21/C10seed43, fresh r1–20/cosine50, PASS_BOUNDED.",
             "5343DAGSNetW1; mỗi xe chấm own-test, gộp9347679row. Headline r20, không chọn bestround.",
             "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro.", "",
             "| round | " + " | ".join(metrics.TEN) + " |", "|---|" + "---|" * 10]
    for point in curve:
        lines.append(f"| {point['round']} | " + " | ".join(f"{point[k]:.6f}" for k in metrics.TEN) + " |")
    lines += ["", "Nguồn CM/rounds đã đối chiếu lại với worker audit; không prediction→CM replay.",
              "[Kết quả và chi phí](results-p21-recovery.md); fullprecision CSV: `runs/p-study/analysis-p21-recovery/metrics-all-rounds.csv`."]
    metric_path = ROOT / "docs/research/metrics-p21-recovery.md"
    metric_path.write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cell, label in [(2, "C2 stream classQ"), (8, "C8 proxy bank"), (10, "C10 hybrid")]:
        pair = [old["arms"][f"P{2 * cell}"]["curve"], curve if cell == 10 else old["arms"][f"P{2 * cell + 1}"]["curve"]]
        for ax, key in zip(axes, ["f1_macro", "low256_mean"]):
            values = np.array([[p[key if key in p else "f1"] for p in arm] for arm in pair])
            ax.plot(range(1, 21), values.mean(0), label=label)
            ax.fill_between(range(1, 21), values.min(0), values.max(0), alpha=.12)
            ax.set_xlabel("Round (cosine50)"); ax.grid(alpha=.2)
    axes[0].set(title="Own-test pooled macro F1", ylabel="F1")
    axes[1].set(title="848 clients with <256 train rows", ylabel="Mean client F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("P21 recovery completes C10: bands = two-seed range, not confidence intervals")
    fig.tight_layout()
    figures = ROOT / "docs/research/figures"
    for ext in ["png", "pdf"]:
        fig.savefig(figures / ("p21-recovery." + ext), dpi=180)
    plt.close(fig)
    assert {name: sha(ROOT / name) for name in source_names} == source_before
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(d.parent.rglob("*")) if p.is_file()}
    receipt = {"verified_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
               "state_sha256": sha(state_path), "complete_sha256": sha(RECOVERY / "complete.json"),
               "worker_reports_match_reaudit": True, "rounds": 20, "proxy_strict_rebuilds": 20,
               "train_rows": int(counts.sum()), "test_rows": int(support.sum()), "effects": len(effects),
               "source_sha256": source_before, "artifact_sha256": manifest,
               "report_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                   [OUT / "analysis.json", OUT / "metrics-all-rounds.csv", metric_path,
                    figures / "p21-recovery.png", figures / "p21-recovery.pdf"]}}
    (OUT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("P21_RECOVERY_ANALYSIS_VERIFIED_DONE", flush=True)


if __name__ == "__main__":
    main()
