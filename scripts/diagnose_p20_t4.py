"""Reproduce P21 round1 with unchanged training; capture numerical evidence, never real rows."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from core import fl, grouped, local
from core.dagsnet import width
from methods.mobihet_study import Method
from scripts.p20_common import settings
from scripts.run_fl import PROTOCOL

OUT = Path("/kaggle/working/p20-numerics-t4")


def summary(tensor):
    finite = torch.isfinite(tensor)
    return {"shape": list(tensor.shape), "nonfinite": int((~finite).sum()),
            "absmax_finite": float(torch.where(finite, tensor.abs(), 0).max())}


@torch.no_grad()
def score_fp32(env, flat, bufs, xs):
    result = []
    for begin in range(0, len(xs), 128):
        x = xs[begin:begin + 128][None].expand(len(flat), -1, -1).float()
        logits, _ = grouped.forward(env.lay.params(flat), env.lay.bufs(bufs), x, env.cfg, train=False)
        result.append(logits.softmax(-1))
    return torch.cat(result, 1)


def main():
    hp = dict(PROTOCOL, **settings("proxy_hybrid"), seed=43, group_rows=4096)
    env = fl.Env(width(1), hp, cache=None)
    method = Method(env)
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / f"rank{env.rank}"
    grouped.compile_all()
    ok, report = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
    print(f"compile gate rank{env.rank}: {'PASS' if ok else 'FAIL -> eager'} {report}", flush=True)
    if not ok:
        grouped.TRAIN = grouped.EVAL = grouped.forward
        from core import optim
        optim.UPDATE = optim._update
    context = {"rank": env.rank, "world": env.world, "seed": 43, "backend": "compiled" if ok else "eager",
               "device": torch.cuda.get_device_name(env.dev), "capability": torch.cuda.get_device_capability(env.dev),
               "torch": torch.__version__, "compile_gate": report, "clients": env.K,
               "hp": hp, "scope": "Diagnostic round1, same P21 method; no own-test evaluation or raw rows/private weights exported"}
    prefix.with_suffix(".json").write_text(json.dumps(context, indent=2) + "\n")
    original = method._probs
    group_number = 0

    def score(flat, bufs, xs, G, chunk=512):
        nonlocal group_number
        idx = env.groups[group_number]
        group_number += 1
        probability = original(flat, bufs, xs, G, chunk)
        finite = bool(torch.isfinite(probability).all())
        row = {"group": group_number, "client_ids": env.ids[idx].tolist(), "probability": summary(probability)}
        if not finite:
            row.update(parameters=summary(flat), buffers=summary(bufs), synthetic=summary(xs))
            row["affected_client_ids"] = env.ids[idx][(~torch.isfinite(probability)).any(2).any(1).cpu().numpy()].tolist()
            # Read-only counterfactual at the same private weights/BN and synthetic rows.
            row["fp32_eager_probability"] = summary(score_fp32(env, flat, bufs, xs))
            original_bn = grouped._bn
            trace = []
            def traced_bn(ctx, name, value):
                result = original_bn(ctx, name, value)
                if not bool(torch.isfinite(result).all()):
                    trace.append({"layer": name, "input": summary(value), "output": summary(result)})
                return result
            grouped._bn = traced_bn
            try:
                first_bad_query = int(torch.nonzero((~torch.isfinite(probability)).any(0).any(1))[0, 0])
                begin = (first_bad_query // 128) * 128
                row["traced_query_begin"] = begin
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    x = xs[begin:begin + 128][None].expand(G, -1, -1).to(env.train.X.dtype)
                    logits, _ = grouped.forward(env.lay.params(flat), env.lay.bufs(bufs), x, env.cfg, train=False)
                row["fp16_eager_logits"] = summary(logits)
                row["first_bad_bn"] = trace[:1]
            finally:
                grouped._bn = original_bn
        with prefix.with_suffix(".jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
            file.flush()
        print("P20_T4_GROUP", env.rank, group_number, finite, flush=True)
        return probability  # Original codec still decides whether this packet is admissible.

    method._probs = score
    info = method.round(1, fl.cosine_lr(1, 50, .001, .00001))
    prefix.with_name(prefix.name + "-round1.json").write_text(json.dumps(info, indent=2) + "\n")
    if env.world > 1: torch.distributed.barrier()
    print("P20_T4_DIAGNOSTIC_ROUND1_DONE", env.rank, flush=True)


if __name__ == "__main__": main()
