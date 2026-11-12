"""B20 trajectory read (owner 12/11 addendum): is an arm's learning still healthy where a gate/guard fails?

    python scripts/analyse_b20_trajectory.py [batch=b20] [rounds=20]
      -> runs/sim-kaggle-<batch>/trajectory.{json,md}

Reads the audited CMs only (run after audit_sim_batch + analyse_b13). Per arm/seed and metric (macro + report-only guards):
delta(r) = arm - C(same seed) for every round, OLS slope of delta over the last half (closing > 0, widening < 0),
delta mean of rounds [R/2+1, 3R/4] vs [3R/4+1, R], and the arm's own macro slope over the last quarter.
Descriptive: two seeds are no significance; a cosine schedule flattens the end by design (sim LR -> 1e-5 at r20).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyse_b13 import stats  # noqa: E402

BATCH = sys.argv[1] if len(sys.argv) > 1 else "b20"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT, ART = ROOT / f"runs/sim-kaggle-{BATCH}", ROOT / f"artifacts/sim-kaggle-{BATCH}"
KEYS = ("f1_macro", "p10", "low256", "benign", "timeDelay", "absent")


def series(spec, counts):
    run = next((ART / spec["slug"]).glob("**/config.json")).parent
    return {k: np.array([stats(run, r, counts)[k] for r in range(1, R + 1)]) for k in KEYS}


def slope(y):
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def main():
    assert json.loads((OUT / "audit.json").read_text())["pass"], "audit first"
    counts = np.load(ROOT / "cache/fold_counts.npy")
    batch = json.loads((OUT / "batch.json").read_text())
    s = {spec["arm"]: series(spec, counts) for spec in batch}
    h, q = R // 2, (3 * R) // 4
    out, lines = {}, ["| arm | metric | Δ r%d | slope Δ r%d–%d | Δ TB r%d–%d → r%d–%d | trend |" % (R, h + 1, R, h + 1, q, q + 1, R),
                      "|---|---|---:|---:|---:|---|"]
    for spec in batch:
        arm = spec["arm"]
        if arm in {"C", "CS43"}:
            continue
        c = s["C" if spec["seed"] == 42 else "CS43"]
        out[arm] = {"own_macro_slope_last_quarter": slope(s[arm]["f1_macro"][q:])}
        for k in KEYS:
            d = s[arm][k] - c[k]
            sl = slope(d[h:])
            early, late = float(d[h:q].mean()), float(d[q:].mean())
            trend = "closing" if sl > 0 and late > early else "widening" if sl < 0 and late < early else "flat/mixed"
            out[arm][k] = {"delta": d.tolist(), "delta_end": float(d[-1]), "slope_last_half": sl,
                           "mean_q3": early, "mean_q4": late, "trend": trend}
            lines.append(f"| {arm} | {k} | {d[-1]:+.4f} | {sl:+.5f} | {early:+.4f} → {late:+.4f} | {trend} |")
    (OUT / "trajectory.json").write_text(json.dumps(out, indent=1) + "\n")
    (OUT / "trajectory.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
