"""Attribution probe: can marginal transport alone explain the forest-hybrid gain?"""
import hashlib
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from core.metrics import ten
from scripts.marginal_copula_generator import Generator
from scripts.direct_generator import Generator as Base
from scripts.probe_pair_generator import ROOT, PAIRS, load, evaluate

OUT = ROOT / "runs/marginal-copula"


def main():
    OUT.mkdir(exist_ok=True)
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               ("scripts/probe_marginal_copula.py", "scripts/marginal_copula_generator.py",
                "scripts/direct_generator.py", "scripts/probe_pair_generator.py")}
    plan = {"sources": sources, "purpose": "attribute hybrid TSTR gain dominated by zeroSpeed, not tune bins/potentials",
            "calibration_rows": 32768, "grid": 4097, "calibration_seed": 9200, "draw_seed": 8100,
            "draws_per_class": 16384, "tstr_rows_per_class": 4096, "classifier": "HGB40/31leaf/seed42",
            "extra_wire_statistics": 0, "statistical_gates": "pairTV>=20% lower; singletonincrease<=.02 each class",
            "tstr_gate": "macro gain>=.003; no relaxation", "tstr_runs_even_if_stats_fail": "attribution only, not promotion"}
    path = OUT / "plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        path.write_text(json.dumps(plan, indent=2)+"\n")
    start = time.time()
    order, ((xt, yt), (xv, yv)), _ = load()
    digest = hashlib.sha256()
    for a in (xt, yt, xv, yv):
        digest.update(a.tobytes())
    previous = json.loads((ROOT / "runs/marginal-direct-generator/result.json").read_text())
    assert digest.hexdigest() == previous["sample_sha256"]
    pairs = [(order.index(a), order.index(b)) for a, b in PAIRS]
    rows, train = [], []
    for c in range(16):
        real, val = xt[yt == c], xv[yv == c]
        model = Generator.fit(real, pairs)
        control = model.sample(np.random.default_rng(8100+c), 16384)
        np.testing.assert_array_equal(control[:256], Base.sample(model, np.random.default_rng(8100+c), 16384)[:256])
        x = model.sample(np.random.default_rng(8100+c), 16384, direct=True)
        a, _ = evaluate(val, control, pairs); b, _ = evaluate(val, x, pairs)
        rows.append({"class": c, "baseline": a, "marginal_only": b})
        train.append(x[:4096])
        print("MARGINAL_ONLY_CLASS", c, a["pair_tv"], b["pair_tv"], flush=True)
    model = HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=31, early_stopping=False, random_state=42)
    pred = model.fit(np.concatenate(train), np.repeat(np.arange(16), 4096)).predict(xv)
    cm = confusion_matrix(yv, pred, labels=np.arange(16))
    np.save(OUT / "marginal-val-cm.npy", cm)
    scores = ten(cm)
    before, after = (float(np.mean([r[k]["pair_tv"] for r in rows])) for k in ("baseline", "marginal_only"))
    gates = {"pair_improvement": bool(after <= .8*before),
             "singleton": bool(all(r["marginal_only"]["singleton_tv"] <= r["baseline"]["singleton_tv"]+.02 for r in rows)),
             "tstr": bool(scores["f1_macro"] >= previous["tstr"]["copula"]["f1_macro"]+.003)}
    result = {"classes": rows, "gates": gates, "pass_feasibility": all(gates.values()),
              "scores": scores, "copula_reference": previous["tstr"]["copula"], "hybrid_reference": previous["tstr"]["direct"],
              "pair_tv_before": before, "pair_tv_after": after, "sample_sha256": digest.hexdigest(),
              "reference_result_sha256": hashlib.sha256((ROOT/"runs/marginal-direct-generator/result.json").read_bytes()).hexdigest(),
              "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seconds": time.time()-start}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in sources.items())
    (OUT / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print("MARGINAL_ONLY_DONE", json.dumps(gates), json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
