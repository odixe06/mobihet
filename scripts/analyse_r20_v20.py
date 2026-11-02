"""R20 repeat/replay and V20 validation contrasts from audited artifacts. Post-hoc only; no selection or training."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from core import metrics
from scripts.analyse_m20 import COHORTS, conditional_accuracy
from scripts.research_metrics import client_f1

OUT = ROOT / "runs/r-study/analysis"
TRIPWIRE = {"f1": .001, "timeDelay_f1": .01, "p10": .01, "low256_mean": .01}
KEYS = ["f1", "mean", "p10", "std", "low256_mean", "ge4096_mean", "timeDelay_f1", "other15_f1",
        "absent_acc", "rare_acc", "low256_absent_acc"]


def arms():
    """(name, artifact dir, closure record, split) for every audited run used here."""
    out = {}
    for study, art in [("q-study", "q20"), ("r-study", "r20"), ("r-repeat", "r20")]:
        closure = json.loads((ROOT / "runs" / study / "closure.json").read_text())
        for spec in json.loads((ROOT / "runs" / study / "batch.json").read_text()):
            rec = closure["runs"][spec["arm"]]
            if spec["arm"] == "R8":  # original budget-stopped at r19; resumed kernel audited separately
                rec = json.loads((ROOT / "runs/r-study/recovery-r8/closure.json").read_text())
                assert rec["status"] == "PASS_BOUNDED" and rec["endpoint"] == 20
            else:
                assert rec["pulled"] and rec["audited"] and rec["state"]["status"] == "COMPLETE", spec["arm"]
            out[spec["arm"]] = (ROOT / "artifacts" / art / spec["slug"] / spec["slug"], spec, "test")
    for study, art in [("v-study", "v20"), ("v-study/recovery-ce", "v20-recovery")]:
        closure = json.loads((ROOT / "runs" / study / "closure.json").read_text())
        for spec in json.loads((ROOT / "runs" / study / "batch.json").read_text()):
            rec = closure["runs"][spec["arm"]]
            if not rec.get("audited"):
                continue  # V0/V1 original ERROR before r1 commit; V0R/V1R are the logical cells
            out[spec["arm"]] = (ROOT / "artifacts" / art / spec["slug"] / spec["slug"], spec, "val")
    for study, art in [("w-study", "w20"), ("x-study", "x20")]:   # W20/X20: only audited arms enter the analysis
        w = ROOT / "runs" / study / "closure.json"
        if not w.exists(): continue
        closure = json.loads(w.read_text())
        for spec in json.loads((ROOT / "runs" / study / "batch.json").read_text()):
            if closure["runs"].get(spec["arm"], {}).get("audited"):
                out[spec["arm"]] = (ROOT / "artifacts" / art / spec["slug"] / spec["slug"], spec | {"mode": spec["cell"]}, "test")
    return out


def summarize(cm, counts):
    n = counts.sum(1)
    pooled, f = metrics.ten(cm.sum(0)), client_f1(cm)
    cond = conditional_accuracy(cm, counts)
    small = conditional_accuracy(cm[n < 256], counts[n < 256])
    return {"f1": float(pooled["f1_macro"]), "mean": float(f.mean()), "p10": float(np.percentile(f, 10)),
            "std": float(f.std()), "low256_mean": float(f[n < 256].mean()), "ge4096_mean": float(f[n >= 4096].mean()),
            "timeDelay_f1": float(pooled["f1_per_class"][13]),
            "other15_f1": float(np.delete(pooled["f1_per_class"], 13).mean()),
            "absent_acc": cond["absent"]["accuracy"], "rare_acc": cond["rare1_16"]["accuracy"],
            "low256_absent_acc": small["absent"]["accuracy"], "low256_absent_rows": small["absent"]["rows"],
            "accuracy": float(pooled["accuracy"]), "f1_per_class": [float(x) for x in pooled["f1_per_class"]]}, f


def load(d, split, counts, support):
    rows = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == list(range(1, 21)), d
    curve, percar = {}, None
    for r in (1, 5, 10, 15, 20):
        with np.load(d / f"cms_r{r:03}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343))
            cm = z["private"]
        assert np.array_equal(cm.sum(2), support)
        point, f = summarize(cm, counts)
        np.testing.assert_allclose(point["f1"], rows[r - 1]["private/f1_macro"], rtol=0, atol=1e-12)
        curve[r] = point
        if r == 20:
            percar, final_cm = f, cm
    with np.load(d / "train_cms_r020.npz") as z:
        tr = z["private"]
    assert np.array_equal(tr.sum(2), counts)
    n = counts.sum(1)
    recall = np.divide(tr.diagonal(axis1=1, axis2=2), counts, out=np.zeros(counts.shape), where=counts > 0)
    gap = {}
    for label, sel in [("all", n >= 0), ("low256", n < 256), ("ge4096", n >= 4096)]:
        w = support[:, 13] * sel * (counts[:, 13] > 0)
        gap[label] = {"rows": int(w.sum()), "train_recall": float((recall[:, 13] * w).sum() / w.sum()),
                      "eval_recall": float((final_cm[:, 13, 13] * sel * (counts[:, 13] > 0)).sum() / w.sum())}
    train_pooled = metrics.ten(tr.sum(0))
    timing = {k: float(sum(r.get(k, 0) for r in rows)) for k in ["t_train", "t_eval", "t_diagnostics"]}
    return {"split": split, "curve": curve, "train_f1_r20": float(train_pooled["f1_macro"]),
            "train_timeDelay_f1_r20": float(train_pooled["f1_per_class"][13]), "timeDelay_gap": gap,
            "skipped_frac_max": max(r.get("skipped_frac", 0) for r in rows),
            "bytes_rounds": sum(r["bytes_up"] + r["bytes_down"] for r in rows), "timing": timing}, percar


def contrast(a, b, fa, fb, n):
    d = {k: (a["curve"][20][k] - b["curve"][20][k]) if a["curve"][20][k] is not None else None for k in KEYS}
    delta = fa - fb
    d["percar"] = {"down_gt_001": int((delta < -.01).sum()), "down_gt_005": int((delta < -.05).sum()),
                   "up_gt_001": int((delta > .01).sum()), "up_gt_005": int((delta > .05).sum()),
                   "abs_median": float(np.median(np.abs(delta))), "abs_p90": float(np.percentile(np.abs(delta), 90)),
                   "low256_up_frac": float((delta[n < 256] > 1e-12).mean()),
                   "low256_down_gt_005": int((delta[n < 256] < -.05).sum())}
    return d, delta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    test_support = np.load(ROOT / "evidence/research-data/test_support.npy")
    fold_counts = np.load(ROOT / "runs/v-study/train_counts.npy")
    val_support = np.load(ROOT / "runs/v-study/val_counts.npy")
    res, percar = {}, {}
    for name, (d, spec, split) in arms().items():
        counts, support = (test_counts, test_support) if split == "test" else (fold_counts, val_support)
        res[name], percar[name] = load(d, split, counts, support)
        res[name].update(mode=spec["mode"], seed=json.loads((d / "config.json").read_text())["hp"]["seed"], slug=spec["slug"])
        print(name, spec["mode"], res[name]["curve"][20]["f1"], flush=True)
    pairs = {
        # repeated control: same recipe/seed, new session
        "repeatA_vs_Q": [("R0", "Q0"), ("R1", "Q1")], "repeatB_vs_Q": [("R2", "Q0"), ("R3", "Q1")],
        "repeatA_vs_B": [("R0", "R2"), ("R1", "R3")], "query_vs_stream": [("Q2", "Q0"), ("Q3", "Q1")],
        # frozen-teacher replay
        "replay_control_vs_live": [("R4", "Q6"), ("R5", "Q7")],
        "replay_real": [("R6", "R4"), ("R7", "R5")], "replay_margin": [("R8", "R4"), ("R9", "R5")],
        "replay_strength": [("R10", "R4"), ("R11", "R5")], "replay_margin_vs_real": [("R8", "R6"), ("R9", "R7")],
        "replay_margin_vs_strength": [("R8", "R10"), ("R9", "R11")],
        "live_real": [("Q10", "Q6"), ("Q11", "Q7")], "live_margin": [("Q12", "Q6"), ("Q13", "Q7")],
        "live_strength": [("Q14", "Q6"), ("Q15", "Q7")],
        # validation split
        "val_augment_vs_CE": [("V2", "V0R"), ("V3", "V1R")], "val_mixed_vs_augment": [("V4", "V2"), ("V5", "V3")],
        "val_margin_vs_mixed": [("V6", "V4"), ("V7", "V5")], "val_margin_vs_CE": [("V6", "V0R"), ("V7", "V1R")],
    }
    cells = {(v["mode"], v["seed"]): k for k, v in res.items() if k[0] in "WX"}
    stream = {42: ["Q0", "R0", "R2"], 43: ["Q1", "R1", "R3"]}
    for (cell, seed), arm in cells.items():
        s42 = seed == 42
        refs = {"replay_real_nckd": ["R4" if s42 else "R5", "R6" if s42 else "R7"],
                "live_real_nckd": ["Q6" if s42 else "Q7", "Q10" if s42 else "Q11"],
                "core": [*stream[seed], cells.get(("step_floor", seed)), cells.get(("boot_compact", seed))],
                "bank": ["Q6" if s42 else "Q7", cells.get(("core", seed))],
                "live_real_nckd_local": ["Q6" if s42 else "Q7", cells.get(("live_real_nckd", seed))],
                "full": [cells.get(("bank", seed))]}.get(cell, stream[seed])
        for ref in filter(None, refs):
            pairs.setdefault(f"{arm[0].lower()}_{cell}_vs_{res[ref]['mode']}", []).append((arm, ref))
    contrasts = {}
    for label, items in pairs.items():
        out = []
        for a, b in items:
            n = (fold_counts if res[a]["split"] == "val" else test_counts).sum(1)
            c, _ = contrast(res[a], res[b], percar[a], percar[b], n)
            out.append({"a": a, "b": b, **c})
        contrasts[label] = out
    for item in contrasts["replay_control_vs_live"]:
        item["tripwire_pass"] = all(abs(item[k]) <= v for k, v in TRIPWIRE.items())
    # per-car: does a reception effect exceed the repeated-session spread for the same car?
    noise = {42: np.abs(percar["R0"] - percar["R2"]), 43: np.abs(percar["R1"] - percar["R3"])}
    n = test_counts.sum(1)
    beyond = {}
    for label in ["replay_real", "replay_margin", "replay_strength"]:
        rows = []
        for a, b in pairs[label]:
            delta, base = percar[a] - percar[b], noise[res[a]["seed"]]
            rows.append({"a": a, "b": b, "down_beyond_repeat": int((delta < -np.maximum(base, .01)).sum()),
                         "up_beyond_repeat": int((delta > np.maximum(base, .01)).sum()),
                         "low256_down_beyond": int(((delta < -np.maximum(base, .01)) & (n < 256)).sum()),
                         "low256_up_beyond": int(((delta > np.maximum(base, .01)) & (n < 256)).sum())})
        beyond[label] = rows
    slopes = {k: {m: v["curve"][20][m] - v["curve"][15][m] for m in ("f1", "low256_mean", "p10")} for k, v in res.items()}
    (OUT / "analysis.json").write_text(json.dumps({"arms": res, "contrasts": contrasts, "beyond_repeat": beyond,
        "slopes_15_20": slopes, "scope": "R test=official own-test (repeated exploratory reads); V val=foldval train-only fit. "
        "Two seeds, no CI/significance; R8 r20 from resumed kernel takids1909/mh-r8-replay-margin-s42-resume1/1."}, indent=1))
    with (OUT / "endpoint.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mode", "seed", "split", *KEYS, "train_f1_r20", "td_train_recall_ge4096", "td_eval_recall_ge4096"])
        for k, v in res.items():
            c = v["curve"][20]
            w.writerow([k, v["mode"], v["seed"], v["split"], *[c[x] for x in KEYS], v["train_f1_r20"],
                        v["timeDelay_gap"]["ge4096"]["train_recall"], v["timeDelay_gap"]["ge4096"]["eval_recall"]])
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
