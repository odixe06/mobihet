"""Read audited F50 own-test CMs at r20/30/40/50; apply the locked bank-vs-core rule of screening-f50. No training."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, ten
from scripts.research_metrics import client_f1
from scripts.analyse_m20 import conditional_accuracy

RUNS = {"F0": "mh-f0-bank-s42-r50", "F1": "mh-f1-bank-s43-r50", "F2": "mh-f2-core-s42-r50"}


def main():
    closure = json.loads((ROOT / "runs/f-study/closure.json").read_text()); assert closure["full50_pass"] == 3
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy"); small = counts.sum(1) < 256
    res = {}
    for arm, slug in RUNS.items():
        d = ROOT / "artifacts/f50" / slug / slug; res[arm] = {"rounds": {}}
        rows = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
        res[arm]["n_rounds"] = len(rows); res[arm]["last_row_keys"] = sorted(rows[-1])
        for r in (20, 30, 40, 50):
            cm = np.load(d / f"cms_r{r:03d}.npz")["private"]; f = client_f1(cm); p = ten(cm.sum(0)); c = conditional_accuracy(cm[small], counts[small])
            res[arm]["rounds"][r] = {**{k: float(p[k]) for k in TEN}, "client_mean": float(f.mean()), "p10": float(np.percentile(f, 10)),
                                     "min": float(f.min()), "low256": float(f[small].mean()), "timeDelay": float(p["f1_per_class"][13]),
                                     "absent": c["absent"]["accuracy"], "rare1_16": c["rare1_16"]["accuracy"]}
        res[arm]["f1_per_class"] = [float(x) for x in p["f1_per_class"]]
    res["bank_minus_core_s42"] = {r: {k: res["F0"]["rounds"][r][k] - res["F2"]["rounds"][r][k] for k in ("f1_macro", "p10", "low256", "client_mean", "timeDelay")} for r in (20, 30, 40, 50)}
    out = ROOT / "runs/f-study/analysis"; out.mkdir(exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=0))


if __name__ == "__main__": main()
