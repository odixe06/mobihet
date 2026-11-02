"""P20 evaluator-only probes. No diagnostic result enters messages, losses or server weighting."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from core import evaluate, grouped, local, metrics


def shadow(method, k):
    env = method.env
    cid = int(env.ids[k])
    rng = np.random.default_rng(20261101 + cid)
    start, stop = int(env.train.off[k]), int(env.train.off[k + 1])
    rows = torch.as_tensor(start + rng.choice(stop - start, min(64, stop - start), replace=False), device=env.dev)
    flat = method.priv_p[k:k + 1].clone().requires_grad_(True)
    bufs = method.priv_b[k:k + 1].clone()
    x, y = env.train.X[rows][None].float(), env.train.Y[rows][None].long()
    ctx = SimpleNamespace(Gfix=1, G=1, P=env.lay.params(flat), B=env.lay.bufs(bufs), x=x, y=y,
                          clients=torch.tensor([k], device=env.dev), mask=torch.ones_like(y, dtype=torch.bool))
    cfg = dict(env.cfg)
    keys = ("_shift", "_csum", "receive_weight", "bank_index")
    old = {key: getattr(method, key, None) for key in keys}
    generator = method.gen_g.get_state()
    train_function = grouped.TRAIN
    try:
        # fp32/dropout0/eager cloned forward, not actual AMP/clipped training gradient.
        env.cfg = dict(cfg, dropout=0.)
        grouped.TRAIN = grouped.forward
        method.gen_g.manual_seed(env.hp["seed"] * 1000003 + cid)
        method._shift = method.la_shift[k:k + 1]
        method._csum = torch.zeros(1, 16, 16, device=env.dev)
        if method.mode in {"margin", "strength"} and method.Q is not None:
            from methods.mobihet_study import reception
            method.receive_weight = reception(method.Q, method.cnt, method.mode)
        loss, _ = method._loss_private(ctx)
        grad = torch.autograd.grad(loss.sum(), flat)[0]
        assert torch.isfinite(grad).all()
        return {"client_id": cid, "rows": len(rows), "loss": float(loss.detach()), "gradient_norm": float(grad.norm()),
                "exceeds_clip": bool(grad.norm() > env.hp["clip"]), "train_rows": stop - start}
    finally:
        env.cfg = cfg
        grouped.TRAIN = train_function
        method.gen_g.set_state(generator)
        for key, value in old.items():
            if value is None: method.__dict__.pop(key, None)
            else: setattr(method, key, value)


@torch.no_grad()
def proxy_probe(method, r, out):
    env = method.env
    records = []
    if method.teacher_p is not None:
        for k, cid in enumerate(env.ids):
            if int(cid) not in env.hp["diagnostic_ids"]: continue
            record = {"client_id": int(cid)}
            for label, source in (("train", env.train), ("test", env.test)):
                start, stop = int(source.off[k]), int(source.off[k + 1])
                rng = np.random.default_rng(20261102 + int(cid))
                rows = torch.as_tensor(start + rng.choice(stop - start, min(512, stop - start), replace=False), device=env.dev)
                x, y = source.X[rows][None], source.Y[rows].long()
                t = method.proxy_probability(x)[0].argmax(1)
                with torch.autocast("cuda", dtype=torch.float16, enabled=env.dev.type == "cuda"):
                    z, _ = grouped.EVAL(env.lay.params(method.priv_p[k:k + 1]), env.lay.bufs(method.priv_b[k:k + 1]), x, env.cfg, train=False)
                p = z[0].argmax(1)
                record[label] = {"teacher_cm": torch.bincount(y * 16 + t, minlength=256).view(16, 16).cpu().tolist(),
                                 "student_cm": torch.bincount(y * 16 + p, minlength=256).view(16, 16).cpu().tolist(),
                                 "teacher_correct_student_wrong": int(((t == y) & (p != y)).sum()),
                                 "student_correct_teacher_wrong": int(((p == y) & (t != y)).sum())}
            records.append(record)
    (Path(out) / f"teacher_probe_rank{env.rank}_r{r:03}.json").write_text(json.dumps(records) + "\n")


def run(method, r, out):
    env = method.env
    if r not in env.hp.get("diagnostic_rounds", (1, 2, 5, 10, 15, 20)): return {}
    devices = [env.dev.index or 0] if env.dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        probes = [shadow(method, k) for k, cid in enumerate(env.ids) if int(cid) in env.hp["diagnostic_ids"]]
        cm = evaluate.own_cm(env.lay, method.priv_p, method.priv_b, env.train, env.cfg,
                             torch.arange(env.K, device=env.dev), max_rows=env.hp.get("eval_rows", 1 << 15))
        cm = env.gather_cms(cm)
        proxy_probe(method, r, out)
        from methods import mobihet_external_diagnostic
        external = mobihet_external_diagnostic.run(method, r, out)
    with (Path(out) / f"diagnostics_rank{env.rank}.jsonl").open("a") as f:
        f.write(json.dumps({"round": r, "probe_teacher_round": r if not method.no_kd else None,
            "probe_mode": "shadow_next_teacher_fp32_dropout0", "clients": probes}) + "\n")
    if env.rank == 0:
        np.savez_compressed(Path(out) / f"train_cms_r{r:03}.npz", ids=env.all_ids, private=cm)
    summary = metrics.summary(cm)
    return external | {f"train_private/{key}": summary[key] for key in metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10",
            "client_f1_macro_median", "client_f1_macro_min", "f1_per_class"]}
