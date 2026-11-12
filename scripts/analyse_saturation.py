"""Saturation review 12/11/2026: CBB/CQ8 r50 learning curves vs LR, per-class trajectory.

Read-only on audited rounds.jsonl (codec-combo50). No training, no official-test selection.
Output: runs/saturation-review-20261112/analysis.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/codec-combo50"
RUNS = {
    "CBB42": A / "mh-k-cbb-s42-r50-c2/mh-k-cbb-s42-p20",
    "CBB43": A / "mh-k-cbb-s43-r50-c2/mh-k-cbb-s43-p20",
    "CQ8_42": A / "mh-k-cq8-s42-r50-c2/mh-k-cq8-s42-p20",
    "CQ8_43": A / "mh-k-cq8-s43-r50/mh-k-cq8-s43-r50",
}
WINDOWS = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50)]


def main():
    out = {"source": {}, "runs": {}}
    for name, d in RUNS.items():
        p = d / "rounds.jsonl"
        out["source"][name] = str(p.relative_to(ROOT))
        rows = [json.loads(line) for line in p.open()]
        assert [r["round"] for r in rows] == list(range(1, 51)), name
        f = np.array([r["private/f1_macro"] for r in rows])
        lr = np.array([r["lr"] for r in rows])
        cum = np.cumsum(lr)
        loss = np.array([r["train_loss"] for r in rows])
        pc = np.array([r["private/f1_per_class"] for r in rows])
        win = []
        for a, b in WINDOWS:
            df, dl = f[b - 1] - f[a - 1], cum[b - 1] - cum[a - 1]
            win.append({"from": a, "to": b, "d_f1_macro": round(float(df), 5),
                        "sum_lr": float(dl), "d_f1_per_sum_lr": round(float(df / dl), 3),
                        "train_loss": [round(float(loss[a - 1]), 4), round(float(loss[b - 1]), 4)]})
        out["runs"][name] = {
            "f1_macro": {f"r{r}": round(float(f[r - 1]), 5) for r in (10, 20, 30, 40, 50)},
            "windows": win,
            "per_class_r50": np.round(pc[49], 4).tolist(),
            "per_class_r50_minus_r30": np.round(pc[49] - pc[29], 4).tolist(),
            "client_p10": {"r30": round(rows[29]["private/client_f1_macro_p10"], 4),
                           "r50": round(rows[49]["private/client_f1_macro_p10"], 4)},
        }
    g = {k: [out["runs"]["CBB" + k]["f1_macro"][f"r{r}"] - out["runs"]["CQ8_" + k]["f1_macro"][f"r{r}"]
             for r in (10, 20, 30, 40, 50)] for k in ("42", "43")}
    out["cbb_minus_cq8_r10_20_30_40_50"] = {k: [round(x, 5) for x in v] for k, v in g.items()}
    dst = ROOT / "runs/saturation-review-20261112/analysis.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    for k, v in out["runs"].items():
        print(k, v["f1_macro"], [w["d_f1_per_sum_lr"] for w in v["windows"]])
    print("CBB-CQ8", out["cbb_minus_cq8_r10_20_30_40_50"])


if __name__ == "__main__":
    main()
