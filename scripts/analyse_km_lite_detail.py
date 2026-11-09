"""Post-hoc detail of sim b9/b10 (fold val, audited CMs; read-only, verdict unchanged): per-class deltas, round stability, noise.

    python scripts/analyse_km_lite_detail.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics  # noqa: E402

for b in ("b9", "b10"):
    for r in (16, 18, 20):
        v = json.loads((ROOT / f"runs/sim-kaggle-{b}/verdict-r{r}.json").read_text())["verdict"]
        print(b, r, " ".join(f"{a}:{x['delta']['f1_macro']:+.4f}{'P' if x['pass'] else 'F'}" for a, x in v.items()))
S = {**json.loads((ROOT / "runs/sim-kaggle-b9/verdict-r20.json").read_text())["stats"],
     **json.loads((ROOT / "runs/sim-kaggle-b10/verdict-r20.json").read_text())["stats"]}
pairs = [("LT6", "KQ8"), ("LT6S43", "KQ8S43"), ("PK16", "LT6"), ("PK32", "PK16"), ("PP10", "PK16"), ("E1", "KQ8"), ("E1S43", "KQ8S43"),
         ("E2", "KQ8"), ("WH", "KQ8"), ("SB", "KQ8"), ("ST", "KQ8"), ("GF", "KQ8"), ("QS", "KQ8")]
for a, c in pairs:
    d = np.array(S[a]["f1_class"]) - np.array(S[c]["f1_class"])
    top = np.argsort(np.abs(d))[::-1][:5]
    print(f"{a:6s}-{c:6s} macro {S[a]['f1_macro'] - S[c]['f1_macro']:+.4f} p10 {S[a]['p10'] - S[c]['p10']:+.4f} "
          f"absent {S[a]['absent_acc'] - S[c]['absent_acc']:+.3f} | " + " ".join(f"c{i}:{d[i]:+.4f}" for i in top))


def cm_at(art, r):
    return np.load(glob.glob(str(ROOT / art / "**" / f"cms_r{r:03d}.npz"), recursive=True)[0])["private"].astype(np.int64)


B8, B10 = "artifacts/sim-kaggle-b8/mh-sim-km16-w05-int8-", "artifacts/sim-kaggle-b10/mh-sim-km16q8-"
for name, art in [("KQ8", B8 + "s42-r20"), ("KQ8S43", B8 + "s43-r20"), ("LT6", "artifacts/sim-kaggle-b9/mh-sim-km16-lite6-s42-r20")]:
    s = np.array([metrics.summary(cm_at(art, r))["f1_macro"] for r in range(16, 21)])
    print(name, "macro r16-20", s.round(4), "round-diff std", np.diff(s).std().round(4))


def pr(art):
    cm = cm_at(art, 20).sum(0)
    tp = np.diag(cm)
    return tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1), cm


P0, R0, C0 = pr(B8 + "s42-r20")
for n in ("benign19", "qsynth", "gapfill"):
    P, R, C = pr(B10 + n + "-s42-r20")
    print(n, " ".join(f"c{c}: P{P[c] - P0[c]:+.3f} R{R[c] - R0[c]:+.3f}" for c in (0, 11, 13)),
          "| pred c0", int(C.sum(0)[0] - C0.sum(0)[0]), "c13", int(C.sum(0)[13] - C0.sum(0)[13]), "| c11->c15", int(C0[11, 15]), "->", int(C[11, 15]))
