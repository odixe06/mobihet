"""Locked FB-L rule across DAGSNet and FeatureBERT (docs/research/screening-fbl.md) -> runs/sim-kaggle-fbl/verdict-r20.json.

    python scripts/audit_sim_batch.py fbl && python scripts/audit_sim_batch.py fbl-bert && python scripts/gate_fbl.py

Per model, vs its own controls at the same rounds: candidate PASS if dTB r16-20 vs S0 >= -.007, d r20 vs S0 >= -.008,
compute <= 1.01 x FedAvg on both measures -- useful-work estimator and executed rows of the engine (scripts/compute_executed.py,
full-scale sizes, compact; checked against rows counted in the engine, docs/research/compute-harness.md, added 19/11 before
push) -- and dTB vs its paired base (N4c or N2c: same model, same compact engine) >= +.001. An arm qualifies only if it passes
on both models; several -> highest mean over the two models of dTB vs S0. Guards are report-only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_sim_budget import rounds  # noqa: E402
from scripts import compute_executed as CE  # noqa: E402

BERT = ROOT.parent / "fl_veremi_bert"
GATE_TB, GATE_END, GATE_COST, GATE_EXEC, GATE_BASE = -.007, -.008, 1.01, 1.01, .001
BASES = ("S0", "N4c", "N2c")
OUT = ROOT / "runs/sim-kaggle-fbl"
CLASSES = {"timeDelayAttack": 13, "suddenConstantSpeed": 11, "accelerationMultiplication": 1, "benign": 0}


def audited(runs):
    a = json.loads((runs / "audit.json").read_text())["arms"]
    return {k for k, v in a.items() if v.get("pass")}


def executed(hp):
    """Executed client rows x FedAvg executed, full-scale sizes, mean over a half-late schedule (compute_executed.py)."""
    v = dict(h.split("=", 1) for h in hp)
    assert v.get("compact") == "true", "FB-L arms run the compact engine"
    synth, real = json.loads(v["synth_tier_rows"]), json.loads(v["real_tier_rows"])
    n = CE.np.load(ROOT / "cache/train_sizes.npy").astype(CE.np.int64)
    groups = CE.local.group_schedule(n, [k for k in range(len(n)) if n[k] > 0], CE.B, CE.GROUP)
    fed = CE.rows(n, groups, None, (64, 64, 64), False, False)[1]
    return sum(CE.rows(n, groups, synth, real, late, True)[1] for late in (False, True)) / 2 / fed


def dagsnet():
    lite = {s["arm"]: s for s in json.loads((ROOT / "runs/sim-kaggle-lite2/batch.json").read_text())}
    assert {"S0"} <= audited(ROOT / "runs/sim-kaggle-lite2")
    rows = {"S0": rounds(ROOT / "artifacts/sim-kaggle-lite2" / lite["S0"]["slug"])}
    ok = audited(OUT)
    for s in json.loads((OUT / "batch.json").read_text()):
        if s["arm"] in ok:
            rows[s["arm"]] = dict(rounds(ROOT / "artifacts/sim-kaggle-fbl" / s["slug"]), cost=s["cost_x_fedavg"],
                                  executed=executed(s["hp"]))
    return rows


def bert(cost, ex):
    runs, art = BERT / "runs/sim-fbl", BERT / "artifacts/sim-fbl"
    ok = audited(runs)
    rows = {}
    for p in sorted(runs.glob("*.pushed")):
        s = json.loads(p.read_text())
        if s["arm"] in ok:
            arm = s["arm"].removesuffix("B")
            rows[arm] = dict(rounds(art / s["slug"]), **({"cost": cost[arm], "executed": ex[arm]} if arm in cost else {}))
    return rows


def judge(rows, model):
    c = rows.get("S0")
    out = {}
    for arm, x in rows.items():
        if arm in BASES or c is None:
            continue
        base = rows.get("N4c" if arm.endswith("4") else "N2c")
        x["d_tb"], x["d_r20"] = x["macro_tb"] - c["macro_tb"], x["macro_r20"] - c["macro_r20"]
        x["d_base_tb"] = x["macro_tb"] - base["macro_tb"] if base else None
        pc = json.loads(x["f1_per_class_r20"]) if isinstance(x["f1_per_class_r20"], str) else x["f1_per_class_r20"]
        bc = (json.loads(base["f1_per_class_r20"]) if isinstance(base["f1_per_class_r20"], str) else base["f1_per_class_r20"]) if base else None
        x["d_base_class_r20"] = {k: pc[i] - bc[i] for k, i in CLASSES.items()} if bc else None
        x["pass"] = (x["d_tb"] >= GATE_TB and x["d_r20"] >= GATE_END and x["cost"] <= GATE_COST and x["executed"] <= GATE_EXEC
                     and x["d_base_tb"] is not None and x["d_base_tb"] >= GATE_BASE)
        out[arm] = x
        print(f'{model:7s} {arm} r20 {x["macro_r20"]:.4f} dTB {x["d_tb"]:+.4f} dr20 {x["d_r20"]:+.4f} exec {x["executed"]:.3f} '
              f'dTB-base {x["d_base_tb"] if x["d_base_tb"] is None else round(x["d_base_tb"], 4)} '
              f'{"PASS" if x["pass"] else "FAIL"}', flush=True)
    return out


def main():
    d = dagsnet()
    b = bert({a: x["cost"] for a, x in d.items() if "cost" in x}, {a: x["executed"] for a, x in d.items() if "executed" in x})
    jd, jb = judge(d, "DAGSNet"), judge(b, "BERT")
    both = [a for a in jd if jd[a]["pass"] and jb.get(a, {}).get("pass")]
    both.sort(key=lambda a: -(jd[a]["d_tb"] + jb[a]["d_tb"]) / 2)
    cand = ("L4", "P4", "Q4", "L2", "P2", "Q2")
    missing = {"dagsnet": [a for a in ("N4c", "N2c", *cand) if a not in d],
               "bert": [a for a in (*BASES, *cand) if a not in b]}
    json.dump({"rule": {"d_tb_vs_s0_min": GATE_TB, "d_r20_vs_s0_min": GATE_END, "cost_max": GATE_COST,
                        "executed_max": GATE_EXEC, "bases": "N4c/N2c compact, same model",
                        "d_tb_vs_base_min": GATE_BASE, "both_models": True, "select": "highest mean dTB vs S0 over models"},
               "missing": missing, "selected": both[0] if both else None, "dagsnet": d, "bert": b},
              open(OUT / "verdict-r20.json", "w"), indent=1)
    print("SELECTED", both[0] if both else None, "missing", missing, flush=True)


if __name__ == "__main__":
    main()
