"""Locked rule of docs/research/screening-km-lite.md on sim batch b9 (fold val only; controls: b8 KQ8/KQ8S43, b9 PK16).

    python scripts/analyse_km_lite.py [round]     # default 20 -> runs/sim-kaggle-b9/verdict-r<round>.json
    KM_BATCH=b10 python scripts/analyse_km_lite.py [round]   # b10 rule (same doc, section b10)
"""
import os
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
cnt = np.load(ROOT / "cache/fold_counts.npy")
BATCH = os.environ.get("KM_BATCH", "b9")
B8, B9 = ROOT / "artifacts/sim-kaggle-b8", ROOT / f"artifacts/sim-kaggle-{BATCH}"
RUNS = ROOT / f"runs/sim-kaggle-{BATCH}"
SLUG = {"KQ8": (B8, "mh-sim-km16-w05-int8-s42-r20"), "KQ8S43": (B8, "mh-sim-km16-w05-int8-s43-r20")}
SLUG.update({b["arm"]: (B9, b["slug"]) for b in json.loads((RUNS / "batch.json").read_text())})
if BATCH == "b9":
    CTRL = {"LT8": "KQ8", "LT6": "KQ8", "LT6S43": "KQ8S43", "PK32": "PK16", "PP10": "PK16"}
    NONINF = {"LT8", "LT6", "LT6S43", "PP10"}
else:
    CTRL = {a: ("KQ8S43" if a.endswith("S43") else "KQ8") for a in SLUG if a not in ("KQ8", "KQ8S43")}
    NONINF = set()


def stats(art, slug):
    f = glob.glob(str(art / slug / "**" / f"cms_r{R:03d}.npz"), recursive=True)
    if not f:
        return None
    z = np.load(f[0])
    ids, cm = z["ids"], z["private"].astype(np.int64)
    s = metrics.summary(cm)
    tot = cm.sum(0)
    tp, pred, true = np.diag(tot), tot.sum(0), tot.sum(1)
    prec, rec = tp / np.maximum(pred, 1), tp / np.maximum(true, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    ntr = cnt[ids, 0]
    f1c = np.array([metrics.ten(c)["f1_macro"] for c in cm])
    absent = ntr == 0
    rows_abs = int((cm.sum(2) * absent).sum())
    corr_abs = int((np.einsum("kcc->kc", cm) * absent).sum())
    out = {"f1_macro": s["f1_macro"], "p10": s["client_f1_macro_p10"], "low256": float(f1c[ntr.sum(1) < 256].mean()),
           "timeDelay": float(f1[13]), "benign": float(f1[0]), "absent_acc": corr_abs / max(rows_abs, 1), "absent_rows": rows_abs,
           "precision": {c: float(prec[c]) for c in (1, 2, 3, 6, 7, 10)}, "f1_class": f1.round(4).tolist(),
           "ten": {k: float(s[k]) for k in metrics.TEN}}
    log = glob.glob(str(art / slug / "*.log"))
    if log:
        m = re.findall(r"KM_LITE_BYTES (\{.*?\})(?:\\n|KM_LITE_BYTES|$)", Path(log[0]).read_text())
        if m:
            out["bytes"] = json.loads(m[0].replace('\\"', '"'))
        k = re.findall(r"extra bootstrap up (\d+) B down (\d+) B", Path(log[0]).read_text())
        if k:
            out["km_extra_bytes"] = {"up": int(k[0][0]), "down": int(k[0][1])}
    return out


res = {a: stats(*p) for a, p in SLUG.items()}
verdict = {}
for arm, ctrl in CTRL.items():
    a, c = res.get(arm), res.get(ctrl)
    if not (a and c):
        verdict[arm] = "missing"
        continue
    d = {k: a[k] - c[k] for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent_acc")}
    gate = {"macro": bool(d["f1_macro"] >= (-.0015 if arm in NONINF else .002)), "p10": bool(d["p10"] >= -.006),
            "timeDelay": bool(d["timeDelay"] >= -.0035), "benign": bool(d["benign"] >= -.0005),
            "absent": bool(d["absent_acc"] >= -.01 or min(a["absent_rows"], c["absent_rows"]) < 2000)}
    verdict[arm] = {"vs": ctrl, "delta": d, "gate": gate, "pass": bool(all(gate.values())),
                    "dprec": {k: float(a["precision"][k] - c["precision"][k]) for k in a["precision"]}}
    print(f"{arm:7s} vs {ctrl:7s} " + " ".join(f"{k} {v:+.4f}" for k, v in d.items()),
          "PASS" if all(gate.values()) else "FAIL " + str([k for k, v in gate.items() if not v]))
if res.get("E1") and res.get("E2"):
    print("report E2-E1 macro", round(res["E2"]["f1_macro"] - res["E1"]["f1_macro"], 4), "(E2 replaces E1 only if >= +.001)")
if res.get("PK16") and res.get("LT6"):
    print("report PK16-LT6 (statistic volume) macro", round(res["PK16"]["f1_macro"] - res["LT6"]["f1_macro"], 4))
lt = verdict.get("LT6", {}), verdict.get("LT6S43", {}), verdict.get("LT8", {})
ok = [isinstance(v, dict) and bool(v.get("pass")) for v in lt]
codec = ("lite6" if ok[0] and ok[1] else "lite8" if ok[2] else "none") if BATCH == "b9" else None
if BATCH == "b9":
    print("codec selected:", codec)
if BATCH == "b10":
    em = [isinstance(verdict.get(a), dict) and bool(verdict[a].get("pass")) for a in ("E1", "E1S43")]
    print("EM selected:", all(em))
for a, s in res.items():
    if s and "bytes" in s:
        b = s["bytes"]
        print(f"bytes {a}: up {b['total_up']:.0f} down {b['total_down']:.0f} total {b['total_up'] + b['total_down']:.0f}")
json.dump({"round": R, "stats": res, "verdict": verdict, "codec": codec}, open(RUNS / f"verdict-r{R}.json", "w"), indent=1)
