"""Apply the locked simulator rule (docs/research/screening-sim-foldval.md) to pulled sim outputs (fold val only).

    python scripts/analyse_sim.py [round]        # default 20 -> runs/sim-kaggle/verdict.json + table on stdout
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
import os
B = os.environ.get("SIM_BATCH", "b1")
SUF = "" if B == "b1" else "-" + B
ART = ROOT / ("artifacts/sim-kaggle" + SUF)
RUNS = ROOT / ("runs/sim-kaggle" + SUF)
batch = json.loads((RUNS / "batch.json").read_text())
CTRL = os.environ.get("SIM_CONTROL", "")          # e.g. b2: take S0/S1 from that batch
if CTRL:
    cs = "" if CTRL == "b1" else "-" + CTRL
    ctrl = [dict(b, _art=str(ROOT / ("artifacts/sim-kaggle" + cs))) for b in json.loads((ROOT / ("runs/sim-kaggle" + cs) / "batch.json").read_text())
            if b["arm"] in ("S0", "S1")]
    batch = ctrl + batch
cnt = np.load(ROOT / "cache/fold_counts.npy")
WATCH = [1, 3, 6, 7, 10]


def stats(slug, art=None):
    f, = glob.glob(str(Path(art or ART) / slug / "**" / f"cms_r{R:03d}.npz"), recursive=True)
    z = np.load(f)
    ids, cm = z["ids"], z["private"].astype(np.int64)
    s = metrics.summary(cm)
    tot = cm.sum(0)
    tp, pred, true = np.diag(tot), tot.sum(0), tot.sum(1)
    prec, rec = tp / np.maximum(pred, 1), tp / np.maximum(true, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    ntr = cnt[ids, 0]
    f1c = np.array([metrics.ten(c)["f1_macro"] for c in cm])
    rows_abs = (cm * (ntr == 0)[:, :, None]).sum()
    corr_abs = sum(cm[k, c, c] for k in range(len(ids)) for c in range(16) if ntr[k, c] == 0)
    return {"f1_macro": s["f1_macro"], "accuracy": s["accuracy"], "p10": s["client_f1_macro_p10"],
            "low256": float(f1c[ntr.sum(1) < 256].mean()), "n_low256": int((ntr.sum(1) < 256).sum()),
            "timeDelay": float(f1[13]), "benign": float(f1[0]), "absent_acc": float(corr_abs / max(rows_abs, 1)),
            "absent_rows": int(rows_abs), "precision": {c: float(prec[c]) for c in WATCH},
            "ten": {k: float(s[k]) for k in metrics.TEN}}


out = {spec["arm"]: stats(spec["slug"], spec.get("_art")) for spec in batch
       if glob.glob(str(Path(spec.get("_art") or ART) / spec["slug"] / "**" / f"cms_r{R:03d}.npz"), recursive=True)}
assert "S0" in out, "control S0 missing"
base = out["S0"]
noise = abs(out["S1"]["f1_macro"] - base["f1_macro"]) if "S1" in out else None
verdict = {}
for arm, s in out.items():
    d = {k: s[k] - base[k] for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent_acc")}
    ok = (d["f1_macro"] >= max(.003, noise or 0) and d["p10"] >= -.006 and d["low256"] >= -.003 and d["timeDelay"] >= -.0035
          and d["benign"] >= -.0005 and (s["absent_rows"] < 2000 or d["absent_acc"] >= -.01))
    verdict[arm] = {"delta": d, "pass": bool(ok) if arm not in ("S0", "S1") else None,
                    "delta_precision": {c: s["precision"][c] - base["precision"][c] for c in WATCH}}
    print(f"{arm:6s} macro {s['f1_macro']:.4f} Δ{d['f1_macro']:+.4f} p10 {d['p10']:+.4f} low256 {d['low256']:+.4f} "
          f"td {d['timeDelay']:+.4f} ben {d['benign']:+.4f} abs {d['absent_acc']:+.4f} ({s['absent_rows']}) "
          f"ΔP1/3/6/7/10 " + " ".join(f"{verdict[arm]['delta_precision'][c]:+.3f}" for c in WATCH)
          + ("" if verdict[arm]["pass"] is None else f"  {'PASS' if verdict[arm]['pass'] else 'FAIL'}"))
A = [a for a in ("A256", "A1024", "A4096") if verdict.get(a, {}).get("pass")]
L = [a for a in verdict if a.startswith("L") and verdict[a]["pass"]]
print("lambda PASS:", L, "best:", max(L, key=lambda a: verdict[a]["delta"]["f1_macro"]) if L else None)
T = None
if A:
    best = max(verdict[a]["delta"]["f1_macro"] for a in A)
    T = min(int(a[1:]) for a in A if verdict[a]["delta"]["f1_macro"] >= best - .001)
print("noise |S1-S0| macro", noise, "chosen T", T)
json.dump({"round": R, "stats": out, "verdict": verdict, "noise_macro": noise, "chosen_T": T},
          open(RUNS / "verdict.json", "w"), indent=1, default=float)
