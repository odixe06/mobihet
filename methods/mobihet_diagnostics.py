"""Post-round shadow probes. Never used for updates, teacher aggregation or model selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from core import evaluate, grouped, metrics


def _cosine(a, b):
    denom = a.norm() * b.norm()
    return float(a.dot(b) / denom) if denom > 0 else None


def probe_client(method, k, teacher):
    """Current private weights; fixed train minibatch/synthetic seed, train-mode BN, dropout off, fp32.

    Parameter gradients are on independent cloned weights before clipping. Teacher is the table consumed during
    the just-finished round, not the newly produced table. This is a shadow probe, not a replay of a training step.
    """
    env = method.env
    cid = int(env.ids[k])
    start, stop = int(env.train.off[k]), int(env.train.off[k + 1])
    rng = np.random.default_rng(20261031 + cid)
    rows = torch.as_tensor(start + rng.choice(stop - start, min(64, stop - start), replace=False), device=env.dev)
    x, y = env.train.X[rows][None].float(), env.train.Y[rows][None].long()
    b = x.shape[1]
    ctx = SimpleNamespace(Gfix=1, x=x, y=y, mask=torch.ones(1, b, dtype=torch.bool, device=env.dev),
                          clients=torch.tensor([k], device=env.dev))
    previous_gen = method.gen_g
    try:
        method.gen_g = torch.Generator(device=env.dev).manual_seed(env.hp["seed"] * 1000003 + cid)
        xg, lab = method._synth(ctx)
    finally:
        method.gen_g = previous_gen
    flat = method.priv_p[k:k + 1].detach().float().clone().requires_grad_(True)
    bufs = method.priv_b[k:k + 1].detach().float().clone()
    xp = torch.cat([x, xg.float()], 1)
    mask = torch.ones(xp.shape[:2], dtype=torch.bool, device=env.dev)
    z, _ = grouped.forward(env.lay.params(flat), env.lay.bufs(bufs), xp,
                           dict(env.cfg, dropout=0.0), mask=mask, train=True)
    lsp = F.log_softmax(z, -1)
    ce_rows = F.cross_entropy((z[:, :b] + method.la_shift[k:k + 1, None]).reshape(-1, 16),
                             y.reshape(-1), reduction="none")
    ce = ce_rows.mean()
    hard = -lsp[:, b:].gather(-1, lab[..., None]).mean()
    real = z.sum() * 0
    ce_t, adaptive, used = None, 1.0, 1.0
    keep = torch.zeros_like(y, dtype=torch.bool)
    if teacher is None or method.no_kd:
        synth = hard
    else:
        lt = teacher.clamp_min(1e-6).log()
        qr, qg = lt[y], lt[lab]
        ce_t = -qr.gather(-1, y[..., None]).mean()
        adaptive = float(1 / (ce.detach() + ce_t).clamp_min(1e-6))
        used = 1.0 if method.kd_weight == "fixed" else adaptive
        keep = qr.argmax(-1) == y if method.real_kd_gate else torch.ones_like(y, dtype=torch.bool)
        kr = (qr.exp() * (qr - lsp[:, :b])).sum(-1)
        if method.kd_support != "off":
            kr = kr * method._support_weight(y, ctx.clients)
        if method.real_kd:
            real = torch.where(keep, kr, 0.0).mean()
        if method.synth_target == "fallback":
            rho = method._support_weight(lab, ctx.clients)
            rho = torch.where(qg.argmax(-1) == lab, rho, 0.0)
            target = rho[..., None] * qg.exp() + (1 - rho[..., None]) * F.one_hot(lab, 16)
            synth = (target * (target.clamp_min(1e-6).log() - lsp[:, b:])).sum(-1).mean()
        else:
            synth = hard if method.synth_target == "onehot" else (qg.exp() * (qg - lsp[:, b:])).sum(-1).mean()
    terms = (ce, used * real, used * synth)
    grads = [torch.autograd.grad(t, flat, retain_graph=True)[0].reshape(-1).detach() for t in terms]
    total = sum(grads)
    record = {"client_id": cid, "train_rows": stop - start, "probe_rows": b,
              "ce_real": float(ce.detach()), "kl_real_gated": float(real.detach()),
              "synthetic_loss": float(synth.detach()), "ce_teacher": None if ce_t is None else float(ce_t),
              "adaptive_weight": adaptive, "used_weight": used, "real_gate_kept_fraction": float(keep.float().mean()),
              "grad_real_norm": float(grads[0].norm()), "grad_kd_real_norm": float(grads[1].norm()),
              "grad_synth_norm": float(grads[2].norm()), "grad_total_norm": float(total.norm()),
              "cos_real_kd_real": _cosine(grads[0], grads[1]),
              "cos_real_synth": _cosine(grads[0], grads[2]),
              "total_exceeds_clip": bool(total.norm() > env.hp["clip"]), "classes": []}
    if teacher is not None and not method.no_kd:
        record["real_support_weight_mean"] = float(method._support_weight(y, ctx.clients).mean())
        if method.synth_target == "fallback":
            record["synthetic_teacher_weight_mean"] = float(rho.mean())
    # Per-class logit gradient alignment is a separate diagnostic, not parameter-gradient alignment.
    p = lsp[:, :b].detach().exp()[0]
    gc = (z[:, :b].detach() + method.la_shift[k:k + 1, None]).softmax(-1)[0] - F.one_hot(y[0], 16)
    if teacher is not None and not method.no_kd:
        q_clamped = teacher[y[0]].clamp_min(1e-6)
        gk = q_clamped.sum(-1, keepdim=True) * p - q_clamped
        for c in y.unique().tolist():
            sel = y[0] == c
            denom = gc[sel].norm(dim=1) * gk[sel].norm(dim=1)
            valid = denom > 0
            cosine = (gc[sel] * gk[sel]).sum(-1)[valid] / denom[valid]
            record["classes"].append({"class": c, "rows": int(sel.sum()), "logit_cosine_count": int(valid.sum()),
                                      "logit_negative_count_before_gate": int((cosine < 0).sum()),
                                      "teacher_true_probability": float(teacher[c, c]),
                                      "real_gate_keeps_class": bool(keep[0, sel].all()) and method.real_kd})
    assert all(np.isfinite(v) for v in record.values() if isinstance(v, (float, int)))
    return record


def run(method, r, teacher, out):
    """Every rank participates. Full train CM is descriptive (train+val were both fitted)."""
    env = method.env
    assert method.arm == "none" and method.anchor == "off" and method.kd_bins == 1
    if r not in env.hp.get("diagnostic_rounds", (1, 2, 5, 10)):
        return {}
    devices = [env.dev.index or 0] if env.dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        records = [probe_client(method, k, teacher) for k, cid in enumerate(env.ids)
                   if int(cid) in env.hp["diagnostic_ids"]]
        cm = evaluate.own_cm(env.lay, method.priv_p, method.priv_b, env.train, env.cfg,
                             torch.arange(env.K, device=env.dev), max_rows=env.hp.get("eval_rows", 1 << 15))
        cm = env.gather_cms(cm)
        from methods import mobihet_external_diagnostic
        external = mobihet_external_diagnostic.run(method, r, out)
    out = Path(out)
    with (out / f"diagnostics_rank{env.rank}.jsonl").open("a") as f:
        f.write(json.dumps({"round": r, "teacher_round": r - 1 if teacher is not None else None,
                            "probe_mode": "shadow_postepoch_fp32_trainBN_dropout0", "clients": records}) + "\n")
    s = metrics.summary(cm)
    if env.rank == 0:
        np.savez_compressed(out / f"train_cms_r{r:03}.npz", ids=env.all_ids, private=cm)
    return {**external, **{f"train_private/{key}": s[key] for key in metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10",
             "client_f1_macro_median", "client_f1_macro_min", "f1_per_class"]}}
