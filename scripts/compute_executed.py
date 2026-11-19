"""Rows through the model per real row and round: useful (rows a method trains on) vs executed by the grouped engine (padded
and masked rows included), full-scale vehicle sizes (cache/train_sizes.npy), batch 64, group_rows 4096 (docs/research/compute-harness.md).

    python scripts/compute_executed.py [--panel 200]      # --panel: the sim fold-train panel (scripts/sim_fold_val.py)

Per-row model cost is shape-independent (DAGSNet / FeatureBERT act row by row), so FLOPs = rows x the FedAvg MFLOP per row
measured with FlopCounterMode (13.297 / 40.022). Executed: every grouped step runs Gfix clients x (slot + synthetic width of the
round) rows, padding clients and exhausted / masked rows included (core/local.py). "compact" = hp compact=true of
methods/mobihet_budget.py: a group runs the widest real and synthetic row count any of its vehicles needs, and a step past
every vehicle's real rows (synthetic floor) runs no real slot. Checked against rows counted inside the engine by
scripts/count_executed.py (docs/research/compute-harness.md).
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import local  # noqa: E402

TIER, FLOOR, B, GROUP = [4096, 16384], 16, 64, 4096
ARMS = {"FedAvg": (None, (64, 64, 64)),
        "T3": ([[64, 64], [64, 32], [48, 16]], (64, 64, 64)),
        "Lite2 B+U1": ([[64, 32], [32, 16], [16, 16]], (64, 64, 64)),
        "FB N4 / FB-L x4": ([[64, 64], [16, 16], [16, 16]], (64, 37, 37)),
        "FB N2 / FB-L x2": ([[64, 32], [16, 16], [16, 16]], (64, 40, 40))}


def rows(n, groups, synth, real, late, compact):
    useful = executed = 0
    col = int(late)
    for g in groups:
        nm = max(n[k] for k in g)
        b = local.slot(nm, B)
        steps = max(math.ceil(nm / b), FLOOR if synth else 0)
        gfix = max(len(g), GROUP // b)
        tiers = [int(np.searchsorted(TIER, n[k], side="right")) for k in g]
        real_steps = math.ceil(nm / b)
        if compact:
            ws = max(synth[t][col] for t in tiers) if synth else 0
            executed += gfix * (real_steps * (max(min(b, real[t]) for t in tiers) + ws) + (steps - real_steps) * ws)
        else:
            executed += steps * gfix * (b + (max(s[col] for s in synth) if synth else 0))
        for k, t in zip(g, tiers):
            useful += (n[k] // B) * real[t] + min(n[k] % B, real[t])
            if synth:
                useful += max(math.ceil(n[k] / B), FLOOR) * synth[t][col]
    return useful / n.sum(), executed / n.sum()


def main():
    if "--panel" in sys.argv:
        from scripts.sim_fold_val import fold_counts, pick
        cnt = fold_counts()
        n = cnt[pick(int(sys.argv[sys.argv.index("--panel") + 1]), cnt=cnt), 0].sum(1).astype(np.int64)
    else:
        n = np.load(ROOT / "cache/train_sizes.npy").astype(np.int64)
    groups = local.group_schedule(n, [k for k in range(len(n)) if n[k] > 0], B, GROUP)
    fed = rows(n, groups, None, (64, 64, 64), False, False)[1]
    print(f"{'arm':16s} {'useful':>8s} {'executed (mask)':>16s} {'executed (compact)':>19s}   (x FedAvg executed {fed:.3f} rows/real row;"
          f" early / late in brackets)")
    for arm, (synth, real) in ARMS.items():
        u = [rows(n, groups, synth, real, late, False)[0] for late in (False, True)]
        e = [rows(n, groups, synth, real, late, False)[1] / fed for late in (False, True)]
        c = [rows(n, groups, synth, real, late, True)[1] / fed for late in (False, True)]
        mean = lambda v: sum(v) / 2                                      # half the schedule late, as the FB estimator
        print(f"{arm:16s} {mean(u):8.3f} {mean(e):16.3f} {mean(c):19.3f}   [{e[0]:.4f}/{e[1]:.4f}  {c[0]:.4f}/{c[1]:.4f}]")


if __name__ == "__main__":
    main()
