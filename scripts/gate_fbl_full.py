"""Best FB-L arm after the full-scale pilots (docs/research/screening-fbl.md, rule locked 19/11 before push).

    python scripts/run_fbl_full.py audit                           # DAGSNet  -> runs/fbl-full/audit.json
    (cd ../fl_veremi_bert && python scripts/run_fbl_pilot.py audit ARM)   # BERT, each arm -> runs/fbl-pilot/<ARM>/audit.json
    python scripts/gate_fbl_full.py                                -> runs/fbl-full/verdict-pilot.json

An arm qualifies if its pilot verdict is PASS on both models (locked pilot gate: DAGSNet -.006/-.006, BERT -.010/-.010 vs T3,
bootstrap <= 382 000 B, real_frac == estimator, artifact audit). Among those: highest mean over the two models of dTB r16-20
vs T3. Arms not yet audited on a model are listed as missing, never as FAIL.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT.parent / "fl_veremi_bert"
ARMS = ("L4", "P4", "Q4", "L2", "P2", "Q2")


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    dags = load(ROOT / "runs/fbl-full/audit.json")
    rows, missing = {}, {"dagsnet": [], "bert": []}
    for a in ARMS:
        d = dags.get(a + "P")
        chain = BERT / "runs/fbl-pilot" / (a + "p2")                # owner 19/11: short-quota arms resume in a session-2 kernel
        b = load((chain if (chain / "batch.json").exists() else BERT / "runs/fbl-pilot" / a) / "audit.json").get(a + "B")
        if not d or d.get("pulled") is False:
            missing["dagsnet"].append(a)
        if not b:
            missing["bert"].append(a)
        if d and b and d.get("pulled") is not False:
            rows[a] = {"dagsnet": {k: d[k] for k in ("verdict", "delta_mean_last5", "delta_end", "delta_p10_end", "t_train_hours")},
                       "bert": {k: b[k] for k in ("verdict", "delta_mean_last5", "delta_end", "delta_p10_end", "t_train_hours")}}
            rows[a]["both_pass"] = d["verdict"] == "PASS" and b["verdict"] == "PASS"
            rows[a]["mean_d_tb"] = (d["delta_mean_last5"] + b["delta_mean_last5"]) / 2
    ok = sorted((a for a in rows if rows[a]["both_pass"]), key=lambda a: -rows[a]["mean_d_tb"])
    for a, r in rows.items():
        print(a, "DAGSNet %s dTB %+.4f | BERT %s dTB %+.4f | mean %+.4f" % (r["dagsnet"]["verdict"], r["dagsnet"]["delta_mean_last5"],
              r["bert"]["verdict"], r["bert"]["delta_mean_last5"], r["mean_d_tb"]), flush=True)
    out = {"rule": "PASS both models; highest mean dTB r16-20 vs T3", "selected": ok[0] if ok else None, "ranking": ok,
           "missing": missing, "arms": rows}
    (ROOT / "runs/fbl-full/verdict-pilot.json").write_text(json.dumps(out, indent=1))
    print("SELECTED", out["selected"], "missing", missing, flush=True)
    return 0 if not (missing["dagsnet"] or missing["bert"]) else 3


if __name__ == "__main__":
    sys.exit(main())
