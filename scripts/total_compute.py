"""Total compute of one method run, split by phase, one harness for every method and repo (owner 19/11/2026,
docs/research/compute-harness.md).

    python scripts/total_compute.py count|time OUT.json --repo REPO [--panel 200] -- SCRIPT ARGS...

SCRIPT runs inside REPO exactly as it trains (scripts/sim_fold_val.py of MobiHet / fl_veremi_bert, or scripts/run_fl.py of a
baseline repo with --panel N: the same fold-train panel and fold-val split as sim_fold_val, applied here).
Phases: bootstrap = the method's constructor (copula / KM statistics, teacher, proxy setup ...); train = every local_epoch of a
round (client forward/backward/optimizer, synthetic rows, per-step loss terms); round_other = the rest of the round (server
aggregation, codecs, packets, prototypes ...); eval = the method's evaluate (reported apart, owner 19/11). Data loading and
the compile gate are outside every phase.
  count: eager (compile replaced by an eager TRAIN wrapper so method guards still see a non-reference TRAIN), every op
         counted by scripts/op_counter.py; also rows executed / trained by the grouped engine per round.
  time:  production path (--compile), wall time per phase with a CUDA synchronize at every phase boundary.
"""
import importlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

MH = Path(__file__).resolve().parents[1]
mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
argv = sys.argv[3:]
cut = argv.index("--")
opts, script_argv = argv[:cut], argv[cut + 1:]
repo = Path(opts[opts.index("--repo") + 1]).resolve()
panel = int(opts[opts.index("--panel") + 1]) if "--panel" in opts else 0
method = script_argv[script_argv.index("--method") + 1]
assert mode in ("count", "time")
os.chdir(repo)
sys.path.insert(0, str(repo))
sys.path.insert(1, str(MH / "scripts"))                               # op_counter only (no repo package shadowing)

import torch  # noqa: E402
from op_counter import OpCounter  # noqa: E402

torch.backends.mha.set_fastpath_enabled(False)                       # fused MHA has no FLOP formula; decompose it
counter = OpCounter() if mode == "count" else None
stack, acc, t0 = [], {}, [None]
rows = {}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def push(ph):
    if mode == "time":
        sync()
        now = time.perf_counter()
        if stack:
            acc[stack[-1]] = acc.get(stack[-1], 0.0) + now - t0[0]
        t0[0] = now
    stack.append(ph)
    if counter is not None:
        counter.phase = ph


def pop():
    if mode == "time":
        sync()
        now = time.perf_counter()
        acc[stack[-1]] = acc.get(stack[-1], 0.0) + now - t0[0]
        t0[0] = now
    stack.pop()
    if counter is not None:
        counter.phase = stack[-1] if stack else None


def save(status):
    for v in rows.values():
        if isinstance(v, dict) and v.get("useful") is not None and not isinstance(v["useful"], int):
            v["useful"] = int(v["useful"])
    res = {"mode": mode, "repo": str(repo), "method": method, "argv": script_argv, "panel": panel, "status": status,
           "torch": torch.__version__, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
           "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "rows": rows}
    if counter is not None:
        res["ops"] = counter.table()
    else:
        res["seconds"] = acc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))


cur = {"round": 0}


def wrap_instance(M):
    """Per-instance wrappers: run_fl calls M.round / M.evaluate; super() calls inside stay unwrapped."""
    rnd, ev = M.round, M.evaluate

    def round_(r, lr):
        cur["round"] = r
        push(f"r{r}.round_other")
        try:
            return rnd(r, lr)
        finally:
            pop()
            save("running")

    def evaluate(*a, **k):
        push(f"r{cur['round']}.eval")
        try:
            return ev(*a, **k)
        finally:
            pop()
    M.round, M.evaluate = round_, evaluate


def patch_local():
    """The engine module the methods actually use (fl_veremi_bert swaps core.local for core.local_t3)."""
    loc = sys.modules["core.local"]
    if getattr(loc, "_tc_patched", False):
        return
    epoch = loc.local_epoch

    def counted(*a, **k):
        top = stack[-1] if stack else ""
        push(top.replace("round_other", "train") if top.endswith("round_other") else top or "unphased.train")
        try:
            return epoch(*a, **k)
        finally:
            pop()
    loc.local_epoch, loc._tc_patched = counted, True
    grouped = sys.modules["core.grouped"]

    def wrap_train(f):
        def train(P, B, x, cfg, mask=None, train=False, **k):
            if stack and stack[-1].endswith(".train"):
                r = rows.setdefault(stack[-1], {"executed": 0, "useful": None})
                if mask is None:                                      # teacher / auxiliary forward (fd_ids, pfedes): ops
                    r["aux_executed"] = r.get("aux_executed", 0) + x.shape[0] * x.shape[1]   # counted, rows kept apart
                else:
                    r["executed"] += x.shape[0] * x.shape[1]
                    u = mask.sum()
                    r["useful"] = u if r["useful"] is None else r["useful"] + u
            return f(P, B, x, cfg, mask=mask, train=train, **k)
        return train
    if mode == "count":                                               # eager, and TRAIN is not grouped.forward
        ev_rows = [int(h.split("=", 1)[1]) for h in script_argv if h.startswith("eval_rows=")]
        evm = sys.modules.get("core.evaluate")
        if ev_rows and evm is not None and hasattr(evm, "class_sums"):
            # eager prototype pass (fedtgp ...) OOMs 4 GB at its default 32768-row chunk (19/11): chunk at eval_rows, as the
            # count run already does for evaluate; same ops per row, only the chunk padding differs (rows kept in JSON)
            cs = evm.class_sums

            def class_sums(*a, max_rows=1 << 15, **k):
                rows["class_sums_max_rows"] = min(max_rows, ev_rows[-1])
                return cs(*a, max_rows=min(max_rows, ev_rows[-1]), **k)
            evm.class_sums = class_sums
        fwd = grouped.forward
        grouped.compile_all = lambda: setattr(grouped, "TRAIN", wrap_train(fwd))
        loc.compile_gate = lambda *a, **k: (True, {"total_compute": "eager count, gate skipped"})
        grouped.TRAIN = wrap_train(fwd)
    else:
        comp = grouped.compile_all

        def compile_and_wrap():
            comp()
            grouped.TRAIN = wrap_train(grouped.TRAIN)
        grouped.compile_all = compile_and_wrap


real_import = importlib.import_module


def import_module(name, *a, **k):
    mod = real_import(name, *a, **k)
    if name == f"methods.{method}" and not getattr(mod.Method, "_tc_wrapped", False):
        init = mod.Method.__init__

        def __init__(self, env, *a2, **k2):
            patch_local()
            rows["real_rows"] = int(sum(int(v) for v in env.train.sizes))
            push("bootstrap")
            try:
                init(self, env, *a2, **k2)
            finally:
                pop()
            wrap_instance(self)
        mod.Method.__init__, mod.Method._tc_wrapped = __init__, True
    return mod


importlib.import_module = import_module


def apply_panel(n):
    """scripts/sim_fold_val.py of MobiHet: fold-train panel of n vehicles, fold val as the test split."""
    import numpy as np
    from core import data as D
    from core import fl
    cnt = np.load(MH / "cache/fold_counts.npy")
    tr, va = cnt[:, 0].sum(1), cnt[:, 1].sum(1)                       # = sim_fold_val.pick(n, seed=7)
    ok = np.flatnonzero((tr > 0) & (va >= 32))
    edges = np.quantile(tr[ok], np.linspace(0, 1, 11))
    strata = np.clip(np.searchsorted(edges, tr[ok], side="right") - 1, 0, 9)
    rng = np.random.default_rng(7)
    ids = np.sort(np.concatenate([rng.choice(ok[strata == s], n // 10, replace=False) for s in range(10)]))
    orig = D.load

    def load(root, split, clients=None, cache=None, folds=None, with_fold=False):
        return orig(root, "train", clients, cache, folds=["train" if split == "train" else "val"])
    D.load = load
    fl.Env._sizes = staticmethod(lambda root, cache: cnt[:, 0].sum(1))
    init = fl.Env.__init__

    def env_init(self, cfg, hp, clients=None, cache=None, root=None):
        init(self, cfg, hp, sorted(int(c) for c in ids), cache, root)
    fl.Env.__init__ = env_init


if panel:
    apply_panel(panel)
sys.argv = script_argv
status = "error"
try:
    if counter is not None:
        with counter:
            runpy.run_path(str(repo / script_argv[0]), run_name="__main__")
    else:
        runpy.run_path(str(repo / script_argv[0]), run_name="__main__")
    status = "done"
finally:
    save(status)
