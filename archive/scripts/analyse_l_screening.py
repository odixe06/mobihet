"""Fixed-r10 paired L contrasts and descriptive shadow diagnostics; no hyperparameter selection."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metrics import summary

ROOT = Path("runs/pulls")
OUT = Path("runs/l-screening")


def cm(slug):
    with np.load(ROOT / slug / slug / "cms_r010.npz") as z:
        assert np.array_equal(z["ids"], np.arange(5343))
        return z["private"]


def client_f1(c):
    support = c.sum(2)
    den = support + c.sum(1)
    f = np.divide(2 * c.diagonal(axis1=1, axis2=2), den, out=np.zeros_like(den, dtype=float), where=den > 0)
    return (f * (support > 0)).sum(1) / (support > 0).sum(1)


def quantiles(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    return None if not len(v) else {"n": len(v), "min": float(v.min()), "median": float(np.median(v)),
                                   "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                                   "negative_fraction": float((v < 0).mean())}


def probe_summary(clients):
    out = {key: quantiles([c[key] for c in clients]) for key in (
        "ce_real", "ce_teacher", "adaptive_weight", "used_weight", "cos_real_synth", "cos_real_kd_real",
        "grad_real_norm", "grad_synth_norm", "grad_kd_real_norm", "grad_total_norm")}
    out["synth_to_real_grad_ratio"] = quantiles([c["grad_synth_norm"] / max(c["grad_real_norm"], 1e-12) for c in clients])
    out["shadow_exceeds_clip_count"] = sum(c["total_exceeds_clip"] for c in clients)
    out["client_count"] = len(clients)
    out["per_class_logit"] = []
    for label in range(16):
        rows = [r for c in clients for r in c["classes"] if r["class"] == label]
        valid = sum(r["logit_cosine_count"] for r in rows)
        out["per_class_logit"].append({"class": label, "valid_rows": valid,
            "negative_before_gate": sum(r["logit_negative_count_before_gate"] for r in rows),
            "negative_fraction": sum(r["logit_negative_count_before_gate"] for r in rows) / valid if valid else None})
    return out


def main():
    audit = json.loads((OUT / "audit.json").read_text())
    diagnostic = {x["arm"]: x for x in json.loads((OUT / "diagnostic-audit.json").read_text())}
    n = np.load("runs/j-screening/train_counts.npy").sum(1)
    refs = {seed: {name: cm(slug) for name, slug in {
        "base": "mh-g4-copula" if seed == 42 else "mh-h0-copula-s43",
        "augmentation": f"mh-k{4 if seed == 42 else 5}-aug-s{seed}",
        "old_combo": f"mh-k{0 if seed == 42 else 1}-combo-s{seed}",
        "old_gate": f"mh-k{8 if seed == 42 else 9}-gate-s{seed}"}.items()} for seed in (42, 43)}
    arms = {}
    for x in audit:
        arm, seed = x["arm"], x["hp"]["seed"]
        s, c = x["rounds"][-1], cm(x["slug"])
        f = client_f1(c)
        np.testing.assert_allclose(f.mean(), s["client_f1_macro_mean"], atol=1e-12, rtol=0)
        contrasts = {}
        for name, ref in refs[seed].items():
            rs = summary(ref)
            delta = f - client_f1(ref)
            contrasts[name] = {"delta_f1": s["f1_macro"] - rs["f1_macro"],
                "delta_p10": s["client_f1_macro_p10"] - rs["client_f1_macro_p10"],
                "clients_better": int((delta > 1e-12).sum()), "clients_worse": int((delta < -1e-12).sum()),
                "mean_delta_low256": float(delta[n < 256].mean()),
                "mean_delta_high16384": float(delta[n >= 16384].mean()),
                "class_delta": (np.array(s["f1_per_class"]) - rs["f1_per_class"]).tolist()}
        d = diagnostic[arm]
        probe = {}
        for r in (1, 2, 5, 10):
            clients = [v for rank in d["probes"] for rec in rank if rec["round"] == r for v in rec["clients"]]
            probe[r] = {"all": probe_summary(clients),
                        "low256": probe_summary([v for v in clients if v["train_rows"] < 256]),
                        "high16384": probe_summary([v for v in clients if v["train_rows"] >= 16384])}
        histories = []
        for r in range(1, 10):
            q = torch.load(ROOT / x["slug"] / x["slug"] / "models" / f"r{r:03}.pt",
                           weights_only=True, map_location="cpu")["Q"].numpy()
            histories.append({"teacher_round": r, "used_at_round": r + 1,
                "rejected_classes": np.flatnonzero(q.argmax(1) != np.arange(16)).tolist(),
                "timeDelay_true_probability": float(q[13, 13]), "timeDelay_to_benign": float(q[13, 0])})
        pool = c.sum(0)
        train = d["train_rounds"][-1]
        arms[arm] = {"seed": seed, "slug": x["slug"], "test": s, "train_r10": train,
            "train_test_f1_gap": train["f1_macro"] - s["f1_macro"], "contrasts": contrasts,
            "gate": "PASS" if contrasts["base"]["delta_f1"] >= .010 and contrasts["base"]["delta_p10"] >= -.010 else "FAIL",
            "benign_to_timeDelay": int(pool[0, 13]), "timeDelay_to_benign": int(pool[13, 0]),
            "cohorts": [{"lo": lo, "hi": hi, "clients": int(((n >= lo) & (n < hi)).sum()),
                "mean_f1": float(f[(n >= lo) & (n < hi)].mean()),
                "p10": float(np.percentile(f[(n >= lo) & (n < hi)], 10))}
                for lo, hi in [(0,256),(256,1024),(1024,4096),(4096,16384),(16384,100000)]],
            "train_seconds_mean_r2_10": x["train_seconds_mean_r2_10"],
            "eval_seconds_mean_r2_10": float(np.mean([z["t_eval"] for z in x["rounds"][1:]])),
            "diagnostic_seconds_total": d["t_diagnostics_total"],
            "total_bytes_10": x["total_bytes_10"], "max_skip": max(z["skipped_frac"] for z in x["rounds"]),
            "probe": probe, "teacher_history": histories}
    effects = {}
    for seed, k in [(42,0),(43,1)]:
        f = lambda i: arms[f"L{i + k}"]["test"]["f1_macro"]
        aug = summary(refs[seed]["augmentation"])["f1_macro"]
        effects[seed] = {"soft_minus_hard_fixed": f(0)-aug, "soft_minus_hard_adaptive": f(8)-f(2),
            "adaptive_minus_fixed_hard": f(2)-aug, "adaptive_minus_fixed_soft": f(8)-f(0),
            "target_scale_interaction": f(8)-f(2)-f(0)+aug,
            "real_gate_on_hard_fixed": f(4)-aug, "real_gate_on_soft_fixed": f(6)-f(0),
            "adaptive_minus_fixed_soft_gate": f(10)-f(6),
            "codec_combo": f(8)-summary(refs[seed]["old_combo"])["f1_macro"],
            "codec_gate": f(10)-summary(refs[seed]["old_gate"])["f1_macro"]}
    warmup = {seed: {"min": min(x["rounds"][0]["f1_macro"] for x in audit if x["hp"]["seed"]==seed),
                    "max": max(x["rounds"][0]["f1_macro"] for x in audit if x["hp"]["seed"]==seed)} for seed in (42,43)}
    out = {"scope": "Fixed r10, paired nominal seeds, descriptive 34-client shadow probes; no independent heldout or significance proof",
           "arms": arms, "effects": effects, "r1_shared_math_spread": warmup}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"gates": {k: [v["test"]["f1_macro"],v["test"]["client_f1_macro_p10"],v["gate"]] for k,v in arms.items()},
                      "effects": effects, "r1_shared_math_spread": warmup}, indent=2))


if __name__ == "__main__":
    main()
