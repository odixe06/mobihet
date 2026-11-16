"""Train->val forest/conditional-Gaussian probe; no official-test or GPU launch."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.direct_generator import Generator
from scripts.probe_pair_generator import PAIRS, ROOT, evaluate, load

OUT = ROOT / "runs/direct-generator"


def main():
    OUT.mkdir(exist_ok=True)
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ("scripts/probe_direct_generator.py", "scripts/direct_generator.py", "scripts/probe_pair_generator.py")}
    plan = {"sources": sources, "pairs": PAIRS, "coarse_bins": 16, "fine_bins": 256,
            "draws_per_class": 16384, "tstr_rows_per_class": 4096, "seed": 8100,
            "roots": "smallest feature index, independent components", "outside_pairs_seed": 8110,
            "outside_pairs_count": 64, "gates": {"pair_tv_reduction": .20, "singleton_increase_max_per_class": .02,
                                                "tstr_macro_gain": .003},
            "scope": "bounded local CPU approximation; reused exploratory train/val panel, not production FL"}
    path = OUT / "plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(plan)), "Preserve frozen plan"
    else:
        path.write_text(json.dumps(plan, indent=2)+"\n")
    start = time.time()
    order, ((xt, yt), (xv, yv)), files = load()
    digest = hashlib.sha256()
    for a in (xt, yt, xv, yv):
        digest.update(a.tobytes())
    pairs = [(order.index(a), order.index(b)) for a, b in PAIRS]
    outside = [(i, j) for i in range(77) for j in range(i+1, 77) if (i,j) not in pairs and (j,i) not in pairs]
    rng = np.random.default_rng(8110)
    outside = [outside[k] for k in rng.choice(len(outside), 64, replace=False)]
    rows, base_x, direct_x = [], [], []
    for c in range(16):
        real, val = xt[yt == c], xv[yv == c]
        assert len(real) >= 256 and len(val) >= 128
        model = Generator.fit(real, pairs)
        base = model.sample(np.random.default_rng(8100+c), 16384)
        direct = model.sample(np.random.default_rng(8100+c), 16384, direct=True)
        a, _ = evaluate(val, base, pairs); b, _ = evaluate(val, direct, pairs)
        a_out, _ = evaluate(val, base, outside); b_out, _ = evaluate(val, direct, outside)
        row = {"class": c, "train_rows": len(real), "val_rows": len(val), "baseline": a, "direct": b,
               "outside_tv_baseline": a_out["pair_tv"], "outside_tv_direct": b_out["pair_tv"]}
        rows.append(row); base_x.append(base[:4096]); direct_x.append(direct[:4096])
        print("DIRECT_CLASS", json.dumps(row), flush=True)
    before, after = (float(np.mean([row[k]["pair_tv"] for row in rows])) for k in ("baseline", "direct"))
    gates = {"pair_improvement": after <= .8*before,
             "singleton": all(row["direct"]["singleton_tv"] <= row["baseline"]["singleton_tv"]+.02 for row in rows)}
    report = {"classes": rows, "gates": gates, "pair_tv_before": before, "pair_tv_after": after,
              "sample_sha256": digest.hexdigest(), "files": files, "train_rows": len(yt), "val_rows": len(yv),
              "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "statistical-result.json").write_text(json.dumps(report, indent=2)+"\n")
    if all(gates.values()):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import confusion_matrix
        from core.metrics import ten
        scores = {}
        synth_y = np.repeat(np.arange(16), 4096)
        for name, x, y in (("real", xt, yt), ("copula", np.concatenate(base_x), synth_y),
                           ("direct", np.concatenate(direct_x), synth_y)):
            model = HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=31, early_stopping=False, random_state=42)
            pred = model.fit(x, y).predict(xv)
            cm = confusion_matrix(yv, pred, labels=np.arange(16))
            np.save(OUT / (name+"-val-cm.npy"), cm)
            scores[name] = ten(cm)
            print("DIRECT_TSTR", name, json.dumps(scores[name]), flush=True)
        report["tstr"] = scores
        gates["tstr"] = scores["direct"]["f1_macro"] >= scores["copula"]["f1_macro"]+.003
    for name, sha in sources.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
    report.update(pass_feasibility=bool(all(gates.values())), seconds=time.time()-start)
    (OUT / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print("DIRECT_PROBE_DONE", json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
