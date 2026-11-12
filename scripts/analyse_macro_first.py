"""Owner macro-first review: retained pilot curves, no changed historical verdicts or training."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import fl, metrics


def main():
    out = ROOT / "runs/macro-first-review-20261111"
    out.mkdir(exist_ok=True, parents=True)
    history = json.loads((ROOT / "runs/r-study/analysis/analysis.json").read_text())["arms"]
    sources, candidates = {}, {}

    def read_json(name):
        path = ROOT / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    read_json("runs/r-study/analysis/analysis.json")
    prior_delta = read_json("runs/y-study/wave1/prior-r20.json")["deltas"]
    for arm, seed, base_arm in (("Y1", 42, "F2"), ("Y2", 43, "F3")):
        audit = read_json(f"runs/y-study/wave1/collection/{arm}-r20/audit.json")[0]
        assert audit["hp"]["rounds"] == 50
        prior = ROOT / f"artifacts/y50/mh-{arm.lower()}-prior-s{seed}-r50/mh-{arm.lower()}-prior-s{seed}-r50"
        control = ROOT / f"artifacts/f50/mh-{base_arm.lower()}-core-s{seed}-r50/mh-{base_arm.lower()}-core-s{seed}-r50"
        values = []
        for r, row in enumerate(audit["rounds"], 1):
            assert row["round"] == r
            with np.load(prior / f"cms_r{r:03d}.npz") as a, np.load(control / f"cms_r{r:03d}.npz") as b:
                assert np.array_equal(a["private"].sum(2), b["private"].sum(2))
                ten, base = metrics.summary(a["private"]), metrics.summary(b["private"])
            assert all(abs(ten[k] - row[k]) < 1e-9 for k in metrics.TEN)
            values.append(dict(round=r, macro=float(ten["f1_macro"]), control=float(base["f1_macro"]),
                               delta=float(ten["f1_macro"] - base["f1_macro"])))
        assert abs(values[-1]["delta"] - prior_delta[arm]["f1_macro"]) < 1e-12
        candidates[arm] = dict(family="prior-coordinate", seed=seed, control=base_arm, schedule=50, curve=values,
                               scope="Recomputed 20 retained prior/control CMs and prior ten metrics; historical source audit retained")
    for arm, control in (("W2", "Q6"), ("W3", "Q7"), ("X6", "X2"), ("X7", "X3")):
        assert history[arm]["split"] == history[control]["split"] == "test"
        group = "w-study" if arm.startswith("W") else "x-study"
        audit = read_json(f"runs/{group}/collection/{arm}/audit.json")[0]
        assert audit["hp"]["rounds"] == 50
        ac, bc = history[arm]["curve"], history[control]["curve"]
        rr = sorted(set(map(int, ac)) & set(map(int, bc)))
        values = [dict(round=r, macro=ac[str(r)]["f1"], control=bc[str(r)]["f1"],
                       delta=ac[str(r)]["f1"] - bc[str(r)]["f1"]) for r in rr]
        assert abs(values[-1]["macro"] - audit["rounds"][-1]["f1_macro"]) < 1e-12
        candidates[arm] = dict(family=history[arm]["mode"], seed=history[arm]["seed"], control=control,
                               schedule=50, curve=values, scope="Retained sampled curve + historical audited endpoint; no new private-state replay")
    for arm, a in candidates.items():
        curve = {v["round"]: v for v in a["curve"]}
        a["macro_delta_r20"] = curve[20]["delta"]
        a["own_gain_15_20"] = curve[20]["macro"] - curve[15]["macro"]
        a["relative_gain_change_15_20"] = curve[20]["delta"] - curve[15]["delta"]
    recent = {}
    for b in range(13, 17):
        audit = read_json(f"runs/sim-kaggle-b{b}/audit.json")
        assert audit["pass"]
        j = read_json(f"runs/sim-kaggle-b{b}/verdict-r20.json")
        recent[f"b{b}"] = {a: v["delta"]["f1_macro"] for a, v in j["verdict"].items()}
        assert not any(all(j["verdict"][arm + sfx]["delta"]["f1_macro"] >= .002 for sfx in ("", "S43"))
                       for arm in j["candidates"])
    result = dict(primary_review_margin=.002, secondary_utility_guards="report-only per owner; historical verdicts unchanged",
                  lr20_cosine50=fl.cosine_lr(20, 50, .001, .00001), lr20_cosine20=fl.cosine_lr(20, 20, .001, .00001),
                  pilots=candidates, recent_sim_macro_deltas=recent, source_sha256=sources,
                  scope="Post-hoc selection for new objective; repeated official test, two seeds, no significance or r50 extrapolation")
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    for ax, arms, title in zip(axes, (("Y1", "Y2"), ("W2", "W3")),
                               ("Prior coordinate vs Core", "Real NCKD vs matched bank control")):
        for arm in arms:
            a = candidates[arm]
            ax.plot([v["round"] for v in a["curve"]], [100 * v["delta"] for v in a["curve"]],
                    marker="o", markersize=3, label=f"seed {a['seed']}")
        ax.axhline(.2, color="gray", linestyle="--", linewidth=1, label="+0.2 percentage points")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(title=title, xlabel="Round (cosine schedule: 50)", xlim=(1, 20))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Macro-F1 gain (percentage points)")
    fig.suptitle("Historical pilots; different recipes from current CBB. No forecast beyond r20.", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(out / ("pilot-curves." + suffix), dpi=180)
    plt.close(fig)
    print("Macro-first review written; no historical verdict changed")


if __name__ == "__main__":
    main()
