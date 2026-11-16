"""Describe audited oracle-vs-Core r50; no fitting or hyperparameter selection."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import ten
from scripts.research_metrics import client_f1
from scripts.y50_common import CONTROL


def main():
    folder = ROOT / "artifacts/y50/mh-y0-oracle-s42-r50/mh-y0-oracle-s42-r50"
    audit = json.loads((ROOT / "runs/y-study/wave1/collection/Y0-r50/audit.json").read_text())[0]
    assert len(audit["rounds"]) == 50
    cms = {"core": np.load(CONTROL[42] / "cms_r050.npz")["private"],
           "oracle": np.load(folder / "cms_r050.npz")["private"]}
    assert np.array_equal(cms["core"].sum(2), cms["oracle"].sum(2))
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy").sum(1)
    per_class = []
    for c in range(16):
        row = {"class": c}
        for name, cm in cms.items():
            pooled = cm.sum(0)
            tp = pooled[c, c]
            row[name] = {"precision": float(tp / max(pooled[:, c].sum(), 1)),
                         "recall": float(tp / max(pooled[c].sum(), 1)),
                         "f1": float(2 * tp / max(pooled[:, c].sum() + pooled[c].sum(), 1))}
        row["delta_f1"] = row["oracle"]["f1"] - row["core"]["f1"]
        per_class.append(row)
    cohorts = []
    for low, high in ((0, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 1000000)):
        mask = (counts >= low) & (counts < high)
        row = {"train_rows": [low, high], "clients": int(mask.sum())}
        for name, cm in cms.items():
            f = client_f1(cm[mask])
            row[name] = {"pooled_f1": float(ten(cm[mask].sum(0))["f1_macro"]),
                         "client_mean": float(f.mean()), "p10": float(np.percentile(f, 10))}
        cohorts.append(row)
    out = ROOT / "runs/generator-study"
    out.mkdir(exist_ok=True)
    result = {"scope": "post-hoc audited own-test diagnostic, seed42 only; no causal isolation",
              "per_class": per_class, "cohorts": cohorts}
    (out / "oracle-gap.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
