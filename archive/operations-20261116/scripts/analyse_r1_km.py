"""b5: KM16 in the R1 mixed cell (KR1, widths 1,0.25) vs Core mixed b4 M1 (same widths/split) and homogeneous; KS43 vs Core S1.

    python scripts/analyse_r1_km.py [round]   -> runs/sim-kaggle-b5/r1_km.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyse_r1 as A  # noqa: E402  (runs the b4 table on import; reuses run/group)

ids, cm_w1, _ = A.run("artifacts/sim-kaggle-b2", "mh-sim-b2core-s42-r20")
_, cm_w25, _ = A.run("artifacts/sim-kaggle-b4", "mh-sim-homog-w025-s42-r20")
_, cm_m1, arch_m1 = A.run("artifacts/sim-kaggle-b4", "mh-sim-mix-w1-w025-s42-r20")
_, cm_km16, _ = A.run("artifacts/sim-kaggle-b3", "mh-sim-km16-s42-r20")
i2, cm, arch = A.run("artifacts/sim-kaggle-b5", "mh-sim-km16-mix-s42-r20")
assert np.array_equal(i2, ids)
out = {}
print("--- KR1 (KM16 mixed) per width group")
for r, a in sorted(arch.items()):
    assert sorted(a["client_ids"]) == sorted(arch_m1[r]["client_ids"]) and a["width"] == arch_m1[r]["width"]
    mem = a["client_ids"]
    g = A.group(ids, cm, mem)
    ref = {"core_mixed_M1": A.group(ids, cm_m1, mem),
           "core_homog": A.group(ids, cm_w1 if a["width"] == 1.0 else cm_w25, mem)}
    if a["width"] == 1.0:
        ref["km16_homog_W1"] = A.group(ids, cm_km16, mem)
    out[f"rank{r}/W{a['width']}"] = {"km_mixed": g, **ref}
    print(f"rank{r} W{a['width']:<4} n={g['n']} KR1 macro {g['f1_macro']:.4f} p10 {g['p10']:.4f}  " + "  ".join(
        f"vs {k} Δmacro {g['f1_macro'] - v['f1_macro']:+.4f} Δp10 {g['p10'] - v['p10']:+.4f}" for k, v in ref.items()))
json.dump(out, open(ROOT / "runs/sim-kaggle-b5/r1_km.json", "w"), indent=1, default=float)
