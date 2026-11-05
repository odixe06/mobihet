"""Bind Q20 completion, executed base/overlays, CM-derived CSV, rendered3200metrics and reports."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.metrics import TEN

STUDY = ROOT / "runs/q-study"
BASE = STUDY / "analysis"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    job = "kaggle-q-v1-auditfix-collect-20m-20261102"
    state_path = Path("/home/odixe/.codex/long-jobs/jobs") / job / "state.json"
    state = json.loads(state_path.read_text())
    assert state["status"] == "succeeded" and state["exit_code"] == 0 and state["marker_seen"]
    assert "Q20_LAUNCH_COLLECT_AUDIT_DONE" in Path(state["log"]).read_text().splitlines()
    closure = json.loads((STUDY / "closure.json").read_text())
    analysis = json.loads((BASE / "analysis.json").read_text())
    inventory = json.loads((STUDY / "closure-verification.json").read_text())
    assert closure["full20_pass"] == len(closure["runs"]) == 16
    assert analysis["closure_sha256"] == inventory["closure_sha256"] == sha(STUDY / "closure.json")
    assert analysis["inventory_sha256"] == sha(STUDY / "closure-verification.json")
    batch = json.loads((STUDY / "batch.json").read_text())
    rows = list(csv.DictReader((BASE / "metrics-all-rounds.csv").open()))
    assert len(rows) == 320 and len({(r["arm"], r["round"]) for r in rows}) == 320
    rendered = [line for line in (ROOT / "docs/research/metrics-q20.md").read_text().splitlines()
                if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert len(rendered) == 320
    reference_blobs, checked_rows, checked_sources = {}, 0, 0
    for spec in batch:
        folder = ROOT / "artifacts/q20" / spec["slug"]
        d = folder / spec["slug"]
        manifest = json.loads((folder / "provenance/source_manifest.json").read_text())
        for path, value in manifest["files"].items():
            if path in spec["overlay_sha256"]:
                expected = spec["overlay_sha256"][path]
            else:
                key = (spec["base_sha"], path)
                if key not in reference_blobs:
                    blob = subprocess.run(["git", "show", spec["base_sha"] + ":" + path], cwd=ROOT,
                                          capture_output=True, check=True).stdout
                    reference_blobs[key] = hashlib.sha256(blob).hexdigest()
                expected = reference_blobs[key]
            assert value == sha(folder / "provenance/source" / path) == expected, (spec["arm"], path)
            checked_sources += 1
        assert set(spec["overlay_sha256"]) <= set(manifest["files"])
        passed, = json.loads((STUDY / "collection" / spec["arm"] / "audit.json").read_text())
        actual = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
        selected = [r for r in rows if r["arm"] == spec["arm"]]
        assert len(actual) == len(selected) == 20
        for row, cached, raw, point in zip(selected, passed["rounds"], actual, analysis["arms"][spec["arm"]]["curve"]):
            columns = [col.strip() for col in rendered[checked_rows].strip("|").split("|")]
            assert len(columns) == 11
            assert int(row["round"]) == cached["round"] == raw["round"] == point["round"] == int(columns[0])
            for key, shown in zip(TEN, columns[1:]):
                value = float(row[key])
                assert np.isfinite(value) and 0 <= value <= 1
                assert abs(value - raw["private/" + key]) < 1e-12 and abs(value - cached[key]) < 1e-12
                assert shown == f"{value:.6f}"
            for key in ["precision_micro", "recall_micro", "recall_weighted", "f1_micro"]:
                assert abs(float(row[key]) - float(row["accuracy"])) < 1e-12
            checked_rows += 1
    limits = {"f1": .001, "timeDelay_f1": .01, "p10": .01, "low256_mean": .01}
    controls = [e for e in analysis["effects"] if e["label"] == "query_control"]
    assert len(controls) == 2 and all(e["tripwire_pass"] and e["tripwire_limits"] == limits for e in controls)
    assert all(abs(e[k]) <= limit for e in controls for k, limit in limits.items())
    old = json.loads((ROOT / "runs/p-study/analysis/completion-verification.json").read_text())
    assert all(sha(ROOT / p) == value for p, value in old["report_sha256"].items())
    reports = [BASE / name for name in ["analysis.json", "metrics-all-rounds.csv", "costs.csv", "endpoint-table.md", "contrast-table.md"]]
    reports += [ROOT / "docs/research" / name for name in ["results-q20.md", "metrics-q20.md", "cost-q20.md",
                "figures/q20-summary.png", "figures/q20-summary.pdf", "figures/q20-cost.png"]]
    sources = [ROOT / name for name in ["scripts/analyse_q20.py", "scripts/report_q20.py", "scripts/verify_q20_results.py",
               "scripts/audit_q20.py", "scripts/audit_screening.py", "core/metrics.py", "scripts/research_metrics.py"]]
    receipt = {"verified_at": datetime.now().astimezone().isoformat(), "job": job, "state_sha256": sha(state_path),
        "closure_sha256": sha(STUDY / "closure.json"), "inventory_sha256": sha(STUDY / "closure-verification.json"),
        "full20_pass": 16, "metric_rows": checked_rows, "rendered_metric_values": checked_rows * 10,
        "source_files_checked_against_base_and_overlay": checked_sources, "snapshots": 320, "train_cms": 96,
        "paired_contrasts": len(analysis["effects"]), "query_tripwires_pass": True,
        "original_P20_reports_unchanged": len(old["report_sha256"]),
        "report_sha256": {str(p.relative_to(ROOT)): sha(p) for p in reports},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "scope": analysis["scope"]}
    (BASE / "completion-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Q20_RESULTS_VERIFIED", checked_rows, "rows", checked_rows * 10, "values", checked_sources, "sourcefiles", flush=True)


if __name__ == "__main__": main()
