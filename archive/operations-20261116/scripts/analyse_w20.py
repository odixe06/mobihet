"""W20/X20 cell readout beyond the endpoint table: size cohorts, step-floor touched/untouched cars, timeDelay P/R, absent
classes and the live teacher row Q[13]. Reads analysis.json from analyse_r20_v20.py plus audited CMs. Post-hoc only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.analyse_m20 import conditional_accuracy
from scripts.research_metrics import client_f1

OUT = ROOT / "runs/w-study/analysis.json"
BINS = [(1, 64), (64, 256), (256, 961), (961, 4096), (4096, 10**6)]   # 961: step floor 16 touches ceil(n/64) < 16
REF = {"replay_real_nckd": {42: ["R4", "R6"], 43: ["R5", "R7"]}, "live_real_nckd": {42: ["Q6", "Q10"], 43: ["Q7", "Q11"]},
       "bank": {42: ["Q6"], 43: ["Q7"]}, "live_real_nckd_local": {42: ["Q6"], 43: ["Q7"]}}
STREAM = {42: ["Q0", "R0", "R2"], 43: ["Q1", "R1", "R3"]}


def cm20(name, arms):
    d = ROOT / arms[name]["dir"]
    with np.load(d / "cms_r020.npz") as z:
        return z["private"]


def teacher_row(d):
    pt = sorted((d / "models").glob("r020.pt"))
    if not pt:
        return None
    q = torch.as_tensor(torch.load(pt[0], map_location="cpu", weights_only=False)["Q"]).float()
    return {"Q13_13": float(q[13, 13]), "Q13_benign": float(q[13, 0]), "diag_mean": float(q.diag().mean())}


def class13(cm):
    c = cm.sum(0)
    tp, pred, true = c[13, 13], c[:, 13].sum(), c[13].sum()
    return {"recall": float(tp / true), "precision": float(tp / pred), "td_to_benign": float(c[13, 0] / true),
            "benign_to_td": float(c[0, 13] / c[0].sum())}


def main():
    a = json.loads((ROOT / "runs/r-study/analysis/analysis.json").read_text())
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    n = counts.sum(1)
    folder = {"W": "w20", "X": "x20", "R": "r20", "Q": "q20"}
    arms = {k: {"dir": f"artifacts/{folder[k[0]]}/{v['slug']}/{v['slug']}",
                "mode": v["mode"], "seed": v["seed"]} for k, v in a["arms"].items() if v["split"] == "test"}
    cms = {k: cm20(k, arms) for k in arms if k[0] in "WQRX" and (ROOT / arms[k]["dir"] / "cms_r020.npz").exists()}
    f = {k: client_f1(cm) for k, cm in cms.items()}
    noise = {42: np.abs(f["R0"] - f["R2"]), 43: np.abs(f["R1"] - f["R3"])}
    res = {"arms": {}, "pairs": {}}
    for k in cms:
        res["arms"][k] = {"mode": arms[k]["mode"], "seed": arms[k]["seed"], "class13": class13(cms[k]),
                          "teacher": teacher_row(ROOT / arms[k]["dir"]) if k[0] in "WX" else None}
    cells = {(v["mode"], v["seed"]): k for k, v in arms.items() if k[0] in "WX"}
    for k in sorted(x for x in cms if x[0] in "WX"):
        cell, seed = arms[k]["mode"], arms[k]["seed"]
        extra = {"core": [cells.get(("step_floor", seed))], "bank": [cells.get(("core", seed))],
                 "live_real_nckd_local": [cells.get(("live_real_nckd", seed))], "full": [cells.get(("bank", seed))]}.get(cell, [])
        base = REF.get(cell, {}).get(seed, STREAM[seed] if cell not in {"live_real_nckd_local", "full"} else [])
        for ref in filter(None, [*base, *extra]):
            delta, base = f[k] - f[ref], np.maximum(noise[seed], .01)
            row = {"bins": {}}
            for lo, hi in BINS:
                sel = (n >= lo) & (n < hi)
                ca, cb = conditional_accuracy(cms[k][sel], counts[sel]), conditional_accuracy(cms[ref][sel], counts[sel])
                row["bins"][f"{lo}-{hi}"] = {
                    "cars": int(sel.sum()), "mean_delta": float(delta[sel].mean()),
                    "down_beyond": int((delta[sel] < -base[sel]).sum()), "up_beyond": int((delta[sel] > base[sel]).sum()),
                    "absent_delta": (ca["absent"]["accuracy"] - cb["absent"]["accuracy"]) if ca["absent"]["rows"] else None}
            row["down_beyond"] = int((delta < -base).sum())
            row["up_beyond"] = int((delta > base).sum())
            # repeated-session reference for the same counts (R0 vs R2 / R1 vs R3 has base = |delta| by construction)
            row["class13_delta"] = {m: res["arms"][k]["class13"][m] - res["arms"][ref]["class13"][m] for m in res["arms"][k]["class13"]}
            res["pairs"][f"{k}-{ref}"] = row
    # session spread of the same per-bin statistic: Q0 vs R0/R2, Q1 vs R1/R3 against each other
    spread = {}
    for x, y in [("R0", "R2"), ("R1", "R3"), ("R0", "Q0"), ("R2", "Q0"), ("R1", "Q1"), ("R3", "Q1")]:
        delta, base = f[x] - f[y], np.maximum(np.abs(f["R0"] - f["R2"]) if arms[x]["seed"] == 42 else np.abs(f["R1"] - f["R3"]), .01)
        spread[f"{x}-{y}"] = {f"{lo}-{hi}": float(delta[(n >= lo) & (n < hi)].mean()) for lo, hi in BINS}
        if (x, y) not in [("R0", "R2"), ("R1", "R3")]:
            spread[f"{x}-{y}"] |= {"down_beyond": int((delta < -base).sum()), "up_beyond": int((delta > base).sum())}
    res["spread_bins"] = spread
    res["scope"] = "official own-test r20, 2 seeds, repeated exploratory reads; beyond = |delta| > max(R repeat |delta| of the car, .01)"
    OUT.write_text(json.dumps(res, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
