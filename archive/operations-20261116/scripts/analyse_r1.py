"""R1 mixed-architecture cell in the simulator: per-width-group metrics, mixed vs homogeneous (fold val, r20).

Groups = the vehicles of each rank (arch_rank<r>.json). Homogeneous W1 = b2 S0; homogeneous W0.25 = b4 H25.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
cnt = np.load(ROOT / "cache/fold_counts.npy")


def run(art, slug):
    d = Path(glob.glob(str(ROOT / art / slug / "**" / "rounds.jsonl"), recursive=True)[0]).parent
    z = np.load(d / f"cms_r{R:03d}.npz")
    arch = {}
    for f in d.glob("arch_rank*.json"):
        a = json.loads(f.read_text()); arch[a["rank"]] = a
    return z["ids"], z["private"].astype(np.int64), arch


def group(ids, cm, members):
    m = np.isin(ids, members)
    s = metrics.summary(cm[m])
    f1c = np.array([metrics.ten(c)["f1_macro"] for c in cm[m]])
    return {"n": int(m.sum()), "f1_macro": s["f1_macro"], "accuracy": s["accuracy"], "client_mean": float(f1c.mean()),
            "p10": s["client_f1_macro_p10"]}


ids, cm_w1, _ = run("artifacts/sim-kaggle-b2", "mh-sim-b2core-s42-r20")
_, cm_w25, a25 = run("artifacts/sim-kaggle-b4", "mh-sim-homog-w025-s42-r20")
out = {}
for arm, slug in (("M1", "mh-sim-mix-w1-w025-s42-r20"), ("M2", "mh-sim-mix-w025-w1-s42-r20")):
    i2, cm, arch = run("artifacts/sim-kaggle-b4", slug)
    assert np.array_equal(i2, ids)
    for r, a in sorted(arch.items()):
        mem = a["client_ids"]
        homog = cm_w1 if a["width"] == 1.0 else cm_w25
        g, h = group(ids, cm, mem), group(ids, homog, mem)
        out[f"{arm}/rank{r}/W{a['width']}"] = {"mixed": g, "homog": h, "delta": {k: g[k] - h[k] for k in g if k != "n"}}
        print(f"{arm} rank{r} W{a['width']:<4} n={g['n']:3d} mixed macro {g['f1_macro']:.4f} homog {h['f1_macro']:.4f} "
              f"Δ{g['f1_macro'] - h['f1_macro']:+.4f}  client mean Δ{g['client_mean'] - h['client_mean']:+.4f} p10 Δ{g['p10'] - h['p10']:+.4f}")
print("pooled mixed M1 / M2:", [round(metrics.summary(run("artifacts/sim-kaggle-b4", s)[1])["f1_macro"], 4)
                                for s in ("mh-sim-mix-w1-w025-s42-r20", "mh-sim-mix-w025-w1-s42-r20")],
      "homog W1 / W0.25:", round(metrics.summary(cm_w1)["f1_macro"], 4), round(metrics.summary(cm_w25)["f1_macro"], 4))
json.dump(out, open(ROOT / "runs/sim-kaggle-b4/r1.json", "w"), indent=1, default=float)
