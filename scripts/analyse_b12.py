"""Locked rule of sim b12 (docs/research/improvement-proposals.md §b12; fold val only; controls C/CS43 of this batch).

    python scripts/analyse_b12.py [round=20]   -> runs/sim-kaggle-b12/verdict-r<round>.json

Guards (every family): ΔtimeDelay >= -.0035, Δbenign >= -.0005, Δabsent >= -.01 when both have >= 2000 absent rows.
Fusion (eval-only; fuse view vs "private" of the SAME run): PASS per seed Δmacro >= +.002, Δp10 >= -.006 + guards; w* = highest-macro
  guard-passing window on C seed42; PASS = w* passes on C and CS43. Reported on every arm too.
Floor family (vs C same seed): PASS per seed Δp10 >= +.006 and Δmacro >= -.0015 + guards. Synthetic-rows family: Δmacro >= +.002,
  Δp10 >= -.006 + guards. Candidate = PASS at both seeds. Choice per family = lowest-compute candidate; a higher dose replaces it only
  if its two-seed mean gain exceeds it by >= +.006 p10 (floor) / >= +.001 macro (rows). Compute % from measured train counts.
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
RUNS, ART = ROOT / "runs/sim-kaggle-b12", ROOT / "artifacts/sim-kaggle-b12"
batch = {b["arm"]: b for b in json.loads((RUNS / "batch.json").read_text())}
COMPUTE = {"F24": 1.71, "F32": 3.70, "F48": 8.17, "F64": 13.10, "R80": 12.74, "R96": 25.49, "R112": 38.23, "R128": 50.98}
VIEWS = ("fuse004", "fuse016", "fuseall")


def stats(arm, view="private"):
    f = glob.glob(str(ART / batch[arm]["slug"] / "**" / f"cms_r{R:03d}.npz"), recursive=True)
    if not f:
        return None
    z = np.load(f[0])
    ids, cm = z["ids"], z[view].astype(np.int64)
    s = metrics.summary(cm)
    tot = cm.sum(0)
    tp = np.diag(tot)
    prec, rec = tp / np.maximum(tot.sum(0), 1), tp / np.maximum(tot.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    ntr = cnt[ids, 0]
    f1c = np.array([metrics.ten(c)["f1_macro"] for c in cm])
    absent = ntr == 0
    rows_abs = int((cm.sum(2) * absent).sum())
    corr_abs = int((np.einsum("kcc->kc", cm) * absent).sum())
    return {"f1_macro": s["f1_macro"], "p10": s["client_f1_macro_p10"], "low256": float(f1c[ntr.sum(1) < 256].mean()),
            "timeDelay": float(f1[13]), "benign": float(f1[0]), "absent_acc": corr_abs / max(rows_abs, 1), "absent_rows": rows_abs,
            "f1_class": f1.round(4).tolist(), "precision": prec.round(4).tolist(), "ten": {k: float(s[k]) for k in metrics.TEN}}


def delta(a, c):
    return {k: a[k] - c[k] for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent_acc")}


def guards(d, a, c):
    return {"timeDelay": bool(d["timeDelay"] >= -.0035), "benign": bool(d["benign"] >= -.0005),
            "absent": bool(d["absent_acc"] >= -.01 or min(a["absent_rows"], c["absent_rows"]) < 2000)}


res = {a: stats(a) for a in batch}
ctrl = {42: "C", 43: "CS43"}
fam, fuse = {}, {}
for arm, spec in batch.items():
    if not res.get(arm):
        continue
    for v in VIEWS:                                                   # fusion on every arm (same-run comparison)
        s = stats(arm, v)
        res[f"{arm}:{v}"] = s
        d = delta(s, res[arm])
        g = guards(d, s, res[arm]) | {"macro": bool(d["f1_macro"] >= .002), "p10": bool(d["p10"] >= -.006)}
        fuse[f"{arm}:{v}"] = {"delta": d, "gate": g, "pass": bool(all(g.values())),
                              "guards": bool(all(x for k, x in g.items() if k != "macro")),
                              "df1": {i: round(s["f1_class"][i] - res[arm]["f1_class"][i], 4) for i in (0, 1, 6, 7, 11, 13, 14)}}
    if arm in ctrl.values():
        continue
    c = res.get(ctrl[spec["seed"]])
    if not c:
        continue
    d = delta(res[arm], c)
    g = guards(d, res[arm], c)
    if arm.startswith("F"):
        g |= {"p10_gain": bool(d["p10"] >= .006), "macro_noninf": bool(d["f1_macro"] >= -.0015)}
    else:
        g |= {"macro": bool(d["f1_macro"] >= .002), "p10": bool(d["p10"] >= -.006)}
    fam[arm] = {"delta": d, "gate": g, "pass": bool(all(g.values()))}
ok42 = [v for v in VIEWS if fuse.get(f"C:{v}", {}).get("guards")]
w_star = max(ok42, key=lambda v: fuse[f"C:{v}"]["delta"]["f1_macro"]) if ok42 else None
fuse_pass = bool(w_star and fuse[f"C:{w_star}"]["pass"] and fuse.get(f"CS43:{w_star}", {}).get("pass"))
choice = {}
for prefix, key, step in (("F", "p10", .006), ("R", "f1_macro", .001)):
    levels = sorted({a for a in fam if a.startswith(prefix) and not a.endswith("S43")}, key=lambda a: COMPUTE[a])
    cand = [a for a in levels if fam[a]["pass"] and fam.get(a + "S43", {}).get("pass")]
    mean = {a: np.mean([fam[a]["delta"][key], fam[a + "S43"]["delta"][key]]) for a in cand}
    pick = cand[0] if cand else None
    for a in cand[1:]:
        if mean[a] >= mean[pick] + step:
            pick = a
    choice[prefix] = {"candidates": cand, "pick": pick, "mean_gain": {a: float(v) for a, v in mean.items()}}
print("== family (vs C same seed) ==")
for arm, v in fam.items():
    print(f"{arm:7s} +{COMPUTE[arm.replace('S43', '')]:5.1f}% " + " ".join(f"{k} {x:+.4f}" for k, x in v["delta"].items()),
          "PASS" if v["pass"] else "FAIL " + str([k for k, x in v["gate"].items() if not x]))
print("== fusion (fuse view - private, same run) ==")
for k, v in fuse.items():
    if k.split(":")[0] in ("C", "CS43") or k.endswith("fuse016"):
        print(f"{k:16s} " + " ".join(f"{a} {x:+.4f}" for a, x in v["delta"].items()), "PASS" if v["pass"] else "FAIL", "dF1", v["df1"])
print("fusion w*:", w_star, "PASS both seeds:", fuse_pass)
print("choice:", choice)
json.dump({"round": R, "stats": res, "family": fam, "fusion": fuse, "fusion_w_star": w_star, "fusion_pass": fuse_pass,
           "choice": choice, "compute_pct": COMPUTE}, open(RUNS / f"verdict-r{R}.json", "w"), indent=1)
