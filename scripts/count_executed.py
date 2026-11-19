"""Executed client compute, measured inside the real engine: the same counter for FedAvg and every method.

    python scripts/count_executed.py OUT.json -- scripts/sim_fold_val.py run --n 200 --method fedavg --out ... [args]

Wraps grouped.TRAIN while a local_epoch runs (client training only: compile gate, eval, server/bootstrap excluded) and
counts per round the rows the grouped step executes (Gfix clients x row width: padding clients and masked rows included)
and the rows it trains on (mask true). FLOPs = rows x MFLOP per row measured with FlopCounterMode (per-row model cost
is shape-independent, docs/research/compute-harness.md). Real rows = the vehicles' local rows (one epoch).
"""
import json
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import fl, grouped, local  # noqa: E402

out, script, argv = sys.argv[1], sys.argv[3], sys.argv[3:]
assert sys.argv[2] == "--"
cnt = {"depth": 0, "executed": 0, "calls": 0, "useful": None}
rounds = []


def wrap(f):
    def counted(P, B, x, cfg, mask=None, train=False, **k):
        if cnt["depth"]:
            cnt["executed"] += x.shape[0] * x.shape[1]
            cnt["calls"] += 1
            u = mask.sum()                                             # on device: no host sync per step
            cnt["useful"] = u if cnt["useful"] is None else cnt["useful"] + u
        return f(P, B, x, cfg, mask=mask, train=train, **k)
    return counted


grouped.TRAIN = wrap(grouped.TRAIN)
compile_all = grouped.compile_all


def compile_and_wrap():
    compile_all()
    grouped.TRAIN = wrap(grouped.TRAIN)


grouped.compile_all = compile_and_wrap
epoch = local.local_epoch


def counted_epoch(*a, **k):
    cnt["depth"] += 1
    try:
        return epoch(*a, **k)
    finally:
        cnt["depth"] -= 1


local.local_epoch = counted_epoch
init = fl.Env.__init__


def env_init(self, *a, **k):
    init(self, *a, **k)
    cnt["real"] = int(sum(int(v) for v in self.train.sizes))


fl.Env.__init__ = env_init
cosine = fl.cosine_lr


def lr_and_mark(r, *a, **k):                                           # called once at the start of each round
    flush()
    cnt["round"] = r
    return cosine(r, *a, **k)


def flush():
    if cnt.get("round") is not None:
        rounds.append({"round": cnt["round"], "real_rows": cnt["real"], "executed_rows": cnt["executed"],
                       "useful_rows": int(cnt["useful"]) if cnt["useful"] is not None else 0, "train_calls": cnt["calls"]})
        Path(out).write_text(json.dumps(rounds, indent=1))
    cnt.update(executed=0, calls=0, useful=None, round=None)


fl.cosine_lr = lr_and_mark
sys.argv = argv
try:
    runpy.run_path(str(ROOT / script), run_name="__main__")
finally:
    flush()
