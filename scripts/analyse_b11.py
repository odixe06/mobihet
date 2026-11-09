"""Locked rule of sim b11 (docs/research/improvement-proposals.md §b11; fold val only, controls C/CS43 of the same batch).

    python scripts/analyse_b11.py [round=20]   -> runs/sim-kaggle-b11/verdict-r<round>.json

Arm PASS (per seed): Δmacro >= +.002 vs C of the same seed and Δp10 >= -.006, ΔtimeDelay >= -.0035, Δbenign >= -.0005,
Δabsent >= -.01 when both arms have >= 2000 absent rows. Candidate for full-scale = PASS at seed42 AND seed43.
PH: views ph025..ph100 of C; a* = highest macro among guard-passing a at seed42; PASS if a* also passes at seed43.
EM: if EM5 and EM8 both qualify, keep the higher two-seed mean Δmacro. Report only: per-class F1/precision deltas.
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
RUNS, ART = ROOT / "runs/sim-kaggle-b11", ROOT / "artifacts/sim-kaggle-b11"
batch = {b["arm"]: b for b in json.loads((RUNS / "batch.json").read_text())}


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
            "precision": prec.round(4).tolist(), "recall": rec.round(4).tolist(), "f1_class": f1.round(4).tolist(),
            "ten": {k: float(s[k]) for k in metrics.TEN}}


def judge(a, c):
    d = {k: a[k] - c[k] for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent_acc")}
    gate = {"macro": bool(d["f1_macro"] >= .002), "p10": bool(d["p10"] >= -.006), "timeDelay": bool(d["timeDelay"] >= -.0035),
            "benign": bool(d["benign"] >= -.0005),
            "absent": bool(d["absent_acc"] >= -.01 or min(a["absent_rows"], c["absent_rows"]) < 2000)}
    guards = all(v for k, v in gate.items() if k != "macro")
    return {"delta": d, "gate": gate, "pass": bool(all(gate.values())), "guards": bool(guards),
            "dprec": {i: round(a["precision"][i] - c["precision"][i], 4) for i in (0, 1, 6, 7, 11, 13)},
            "df1": {i: round(a["f1_class"][i] - c["f1_class"][i], 4) for i in (0, 1, 6, 7, 11, 13)}}


ctrl = {42: "C", 43: "CS43"}
res, verdict = {}, {}
for arm, spec in batch.items():
    res[arm] = stats(arm)
for arm, spec in batch.items():
    if arm in ctrl.values():
        continue
    c = res.get(ctrl[spec["seed"]])
    verdict[arm] = judge(res[arm], c) if res.get(arm) and c else "missing"
ph = {}
for seed, c in ctrl.items():
    if not res.get(c):
        continue
    for v in ("ph025", "ph050", "ph075", "ph100"):
        s = stats(c, v)
        res[f"{c}:{v}"] = s
        ph[(seed, v)] = judge(s, res[c])
ok42 = [(v, ph[(42, v)]) for v in ("ph025", "ph050", "ph075", "ph100") if (42, v) in ph and ph[(42, v)]["guards"]]
a_star = max(ok42, key=lambda t: t[1]["delta"]["f1_macro"])[0] if ok42 else None
ph_pass = bool(a_star and ph[(42, a_star)]["pass"] and (43, a_star) in ph and ph[(43, a_star)]["pass"])
two = lambda base: [verdict.get(base), verdict.get(base + "S43")]
cand = {b: all(isinstance(v, dict) and v["pass"] for v in two(b)) for b in ("EM5", "EM8", "QK", "QT2")}
if cand["EM5"] and cand["EM8"]:
    mean = {b: np.mean([v["delta"]["f1_macro"] for v in two(b)]) for b in ("EM5", "EM8")}
    cand["EM5" if mean["EM5"] < mean["EM8"] else "EM8"] = False
for arm, v in verdict.items():
    if isinstance(v, dict):
        print(f"{arm:7s} " + " ".join(f"{k} {x:+.4f}" for k, x in v["delta"].items()),
              "PASS" if v["pass"] else "FAIL " + str([k for k, x in v["gate"].items() if not x]), "dP", v["dprec"])
for (seed, v), j in sorted(ph.items()):
    print(f"PH s{seed} {v} " + " ".join(f"{k} {x:+.4f}" for k, x in j["delta"].items()), "PASS" if j["pass"] else "FAIL",
          "dP", j["dprec"])
print("PH a* (seed42, guard-passing max macro):", a_star, "| PH PASS both seeds:", ph_pass)
print("full-scale candidates (PASS both seeds):", [b for b, ok in cand.items() if ok] + (["PH:" + a_star] if ph_pass else []))
json.dump({"round": R, "stats": res, "verdict": verdict, "ph": {f"s{s}:{v}": j for (s, v), j in ph.items()}, "ph_a_star": a_star,
           "ph_pass": ph_pass, "candidates": cand}, open(RUNS / f"verdict-r{R}.json", "w"), indent=1)
