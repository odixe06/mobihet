"""Locked CBB module screen. Audit all rounds/CM metrics first; r20 is the utility endpoint."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import fl, metrics  # noqa: E402
from scripts.research_metrics import client_f1  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json  # noqa: E402

BATCH = sys.argv[1] if len(sys.argv) > 1 else "b13"
OUT, ART = ROOT / f"runs/sim-kaggle-{BATCH}", ROOT / f"artifacts/sim-kaggle-{BATCH}"


def stats(run, r, counts):
    with np.load(run / f"cms_r{r:03d}.npz") as z:
        ids, cm = z["ids"], z["private"]
    assert np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
    assert np.array_equal(cm.sum(2), counts[ids, 1])
    s = metrics.summary(cm)
    pooled = cm.sum(0)
    f1 = 2 * np.diag(pooled) / np.maximum(pooled.sum(0) + pooled.sum(1), 1)
    absent = counts[ids, 0] == 0
    rows = int((cm.sum(2) * absent).sum())
    return {"f1_macro": float(s["f1_macro"]), "p10": float(s["client_f1_macro_p10"]),
            "low256": float(client_f1(cm)[counts[ids, 0].sum(1) < 256].mean()),
            "benign": float(f1[0]), "timeDelay": float(f1[13]),
            "absent": float((cm.diagonal(axis1=1, axis2=2) * absent).sum() / max(rows, 1)),
            "absent_rows": rows, "f1_class": f1.tolist(), "ten": {k: float(s[k]) for k in metrics.TEN}}


def main():
    audit = json.loads((OUT / "audit.json").read_text())
    assert audit["pass"] and audit["round"] == 20, "Require source/config/support audit first"
    counts = np.load(ROOT / "cache/fold_counts.npy")
    batch = json.loads((OUT / "batch.json").read_text())
    results = {}
    for spec in batch:
        matches = list((ART / spec["slug"]).glob("**/config.json"))
        assert len(matches) == 1
        run = matches[0].parent
        config = json.loads(matches[0].read_text())
        assert config["n_params"] == 395408 and config["hp"]["rounds"] == 20
        rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()]
        assert [x["round"] for x in rows] == list(range(1, 21))
        history = []
        for row in rows:
            s = stats(run, row["round"], counts)
            for key in metrics.TEN:
                assert math.isclose(s["ten"][key], row["private/" + key], rel_tol=1e-7, abs_tol=1e-9), (spec["arm"], key)
            lr = fl.cosine_lr(row["round"], 20, .001, .00001)
            assert math.isclose(row["lr"], lr, rel_tol=1e-7)
            assert row["clients_without_applied"] == 0
            assert row["teacher_consumed_round"] == (None if row["round"] == 1 else row["round"] - 1)
            history.append(s)
        results[spec["arm"]] = {"endpoint": history[-1], "mean16_20": {
            k: float(np.mean([s[k] for s in history[-5:]]))
            for k in ("f1_macro", "p10", "low256", "benign", "timeDelay", "absent")},
            "bytes_per_round": float(np.mean([r["bytes_up"] + r["bytes_down"] for r in rows])),
            "train_seconds": sum(r["t_train"] for r in rows)}
    verdict = {}
    for spec in batch:
        if spec["arm"] in {"C", "CS43"}:
            continue
        arm = spec["arm"]
        a, c = results[arm]["endpoint"], results["C" if spec["seed"] == 42 else "CS43"]["endpoint"]
        delta = {k: a[k] - c[k] for k in ("f1_macro", "p10", "low256", "benign", "timeDelay", "absent")}
        gates = {"macro": delta["f1_macro"] >= .002, "p10": delta["p10"] >= -.006,
                 "benign": delta["benign"] >= -.0005, "timeDelay": delta["timeDelay"] >= -.0035,
                 "absent": delta["absent"] >= -.01 or min(a["absent_rows"], c["absent_rows"]) < 2000}
        if BATCH == "b17":
            gates = {"macro": gates["macro"]}  # Owner11/11: secondary utility metrics are report-only.
        if BATCH == "b20":  # compute screen: macro non-inferiority (G1/codec rule), secondary metrics report-only
            m, mc = results[arm]["mean16_20"], results["C" if spec["seed"] == 42 else "CS43"]["mean16_20"]
            delta["f1_macro_mean16_20"] = m["f1_macro"] - mc["f1_macro"]
            gates = {"macro_mean16_20": delta["f1_macro_mean16_20"] >= -.0015, "macro_r20": delta["f1_macro"] >= -.003}
        verdict[arm] = {"delta": delta, "gates": gates, "pass": all(gates.values())}
        print(arm, "PASS" if all(gates.values()) else "FAIL", delta, flush=True)
    candidates = {s["arm"]: all(verdict[s["arm"] + suffix]["pass"] for suffix in ("", "S43"))
                  for s in batch if s["seed"] == 42 and s["arm"] != "C"}
    atomic_json(OUT / "verdict-r20.json", {"stats": results, "verdict": verdict, "candidates": candidates,
                                           "audit_all_round_metrics": True})
    print("Candidates:", [arm for arm, passed in candidates.items() if passed], flush=True)


if __name__ == "__main__":
    main()
