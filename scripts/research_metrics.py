"""Shared post-hoc client-F1 and shadow-probe summaries. No training or model selection."""
import numpy as np


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
