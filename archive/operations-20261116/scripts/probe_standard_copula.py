"""Fix latent marginal mismatch; same exploratory panel and TSTR harness, no pair tables needed."""
import hashlib
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from core.metrics import ten
from scripts.standard_copula_generator import Generator
from scripts.probe_pair_generator import ROOT, PAIRS, load, evaluate

OUT = ROOT / "runs/standard-copula"


def main():
    OUT.mkdir(exist_ok=True)
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               ("scripts/probe_standard_copula.py", "scripts/standard_copula_generator.py",
                "scripts/direct_generator.py", "scripts/probe_pair_generator.py")}
    plan = {"sources": sources, "hypothesis": "Nonstandard normal-score Gaussian composed with Phi distorts marginal CDF",
            "change": "zero mean, row-normalize decoded Cholesky; correlation unchanged; no quantile calibration",
            "sample_seed": 8100, "draws_per_class": 16384, "tstr_rows_per_class": 4096,
            "classifier": "HGB40/31leaf/seed42", "extra_wire_statistics": 0,
            "gates": {"singleton_tv_mean_relative_reduction": .20, "singleton_tv_max_class_increase": .02,
                      "tstr_macro_gain": .003, "benign_timedelay_tstr_f1_max_drop": .02},
            "pair_tv": "diagnostic only: this arm repairs marginal, not nonlinear pairs; prior arms retain old verdicts"}
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
    previous = json.loads((ROOT/"runs/marginal-direct-generator/result.json").read_text())
    assert digest.hexdigest() == previous["sample_sha256"]
    pairs = [(order.index(a), order.index(b)) for a, b in PAIRS]
    rows, train = [], []
    for c in range(16):
        model = Generator.fit(xt[yt == c], pairs)
        baseline = model.sample(np.random.default_rng(8100+c), 16384)
        x = model.sample(np.random.default_rng(8100+c), 16384, direct=True)
        a, _ = evaluate(xv[yv == c], baseline, pairs); b, _ = evaluate(xv[yv == c], x, pairs)
        rows.append({"class": c, "baseline": a, "standard": b,
                     "latent_mu_max_abs": float(np.abs(model.mu).max()),
                     "latent_variance_min": float(np.square(model.chol).sum(1).min())})
        train.append(x[:4096])
        print("STANDARD_CLASS", json.dumps(rows[-1]), flush=True)
    model = HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=31, early_stopping=False, random_state=42)
    pred = model.fit(np.concatenate(train), np.repeat(np.arange(16), 4096)).predict(xv)
    cm = confusion_matrix(yv, pred, labels=np.arange(16)); np.save(OUT/"standard-val-cm.npy", cm)
    score = ten(cm); reference = previous["tstr"]["copula"]
    before, after = (float(np.mean([row[k]["singleton_tv"] for row in rows])) for k in ("baseline", "standard"))
    gates = {"singleton_improvement": bool(after <= .8*before),
             "singleton_guard": bool(all(row["standard"]["singleton_tv"] <= row["baseline"]["singleton_tv"]+.02 for row in rows)),
             "tstr_macro": bool(score["f1_macro"] >= reference["f1_macro"]+.003),
             "class_guard": bool(all(score["f1_per_class"][c] >= reference["f1_per_class"][c]-.02 for c in (0,13)))}
    result = {"classes": rows, "gates": gates, "pass_feasibility": all(gates.values()), "scores": score,
              "reference": reference, "singleton_tv_before": before, "singleton_tv_after": after,
              "sample_sha256": digest.hexdigest(), "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "seconds": time.time()-start}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in sources.items())
    (OUT/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print("STANDARD_PROBE_DONE", json.dumps(gates), json.dumps(score), flush=True)


if __name__ == "__main__":
    main()
