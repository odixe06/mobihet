"""Post-hoc descriptive per-car transfer, against C2; never choose a method by a car's test."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.research_metrics import client_f1

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "runs/p-study/analysis"
    analysis_path = base / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    verified = json.loads((base / "completion-verification.json").read_text())
    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == verified["report_sha256"]["runs/p-study/analysis/analysis.json"]
    batch = {s["arm"]: s for s in json.loads((ROOT / "runs/p-study/batch.json").read_text())}
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy").sum(1)
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    values = {}
    for name, a in analysis["arms"].items():
        spec = batch[name]
        with np.load(ROOT / "artifacts/p20" / spec["slug"] / spec["slug"] / "cms_r020.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), support)
            values[name] = client_f1(z["private"])
        assert abs(values[name].mean() - a["final"]["mean"]) < 1e-12
    records = []
    for cell in range(16):
        if cell == 2: continue
        rows, deltas = [], []
        for offset, seed in [(0, 42), (1, 43)]:
            name, ref = f"P{2 * cell + offset}", f"P{4 + offset}"
            if name not in values: continue
            d = values[name] - values[ref]; deltas.append(d)
            row = {"arm": name, "reference": ref, "seed": seed,
                   "delta_pooled": analysis["arms"][name]["final"]["f1"] - analysis["arms"][ref]["final"]["f1"]}
            for label, selected in [("all", counts >= 0), ("low256", counts < 256), ("ge4096", counts >= 4096)]:
                v = d[selected]
                row[label] = {"clients": len(v), "improved": int((v > 1e-12).sum()), "harmed": int((v < -1e-12).sum()),
                              "mean_delta": float(v.mean()), "paired_delta_p10": float(np.percentile(v, 10))}
            rows.append(row)
        rec = {"cell": cell, "mode": batch[f"P{2 * cell}"]["mode"], "seeds": rows}
        if len(deltas) == 2:
            rec["harmed_both_seeds"] = int(((deltas[0] < -1e-12) & (deltas[1] < -1e-12)).sum())
            rec["low256_harmed_both_seeds"] = int(((deltas[0] < -1e-12) & (deltas[1] < -1e-12) & (counts < 256)).sum())
        records.append(rec)
    result = {"analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(), "reference": "C2 streaming classQ",
              "scope": "Post-hoc descriptive, same own-test, no test-driven per-car selector or significance; C10 one seed",
              "records": records}
    (base / "client-transfer.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# P20 — tri thức có lợi không đồng đều giữa các xe", "",
        "Chẩn đoán mô tả sau khi đọc officialtest; so cùng seed với C2. Không dùng test chọn method cho từng xe.",
        "F1 mỗi xe tính lớp có mặt; dấu delta dùng tolerance1e-12. Hai seed không là significance.", "",
        "| cell | xe tăng s42/s43 | xe giảm s42/s43 | xe giảm cả2seed | xe<256 giảm cả2seed |", "|---|---|---|---:|---:|"]
    for rec in records:
        cells = ['/'.join(str(row["all"][key]) for row in rec["seeds"]) for key in ["improved", "harmed"]]
        lines.append(f'| C{rec["cell"]} {rec["mode"]} | {cells[0]} | {cells[1]} | {rec.get("harmed_both_seeds", "thiếu")} | {rec.get("low256_harmed_both_seeds", "thiếu")} |')
    lines += ["", "C10 chỉ seed42, không gọi stable gain. Số lượng xe không cân theo testrows hoặc độ lớn delta; xem mean/p10/absent/perclass cùng lúc.",
              "Không có cơ chế chung đã chứng minh tăng mọi xe; personalized reception cần validation riêng, không oracle theo test.",
              "Nguồn: `runs/p-study/analysis/client-transfer.json`; [phân tích chính](results-p20.md)."]
    (ROOT / "docs/research/p20-client-transfer.md").write_text('\n'.join(lines) + '\n')
    print("P20_CLIENT_TRANSFER_DIAGNOSIS_DONE", flush=True)


if __name__ == "__main__": main()
