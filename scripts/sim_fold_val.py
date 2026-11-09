"""Local per-vehicle simulator: Core-style FL on fold train, own-model evaluation on fold val (never the official test).

    python scripts/sim_fold_val.py counts                                   # once: cache/fold_counts.npy (5343, 2, 16)
    python scripts/sim_fold_val.py run --n 200 --out runs/sim/core --rounds 20 [--until 10] [--set k=v ...]

The subset is stratified by fold-train size (10 quantile strata, seed 7) among vehicles with >= 32 val rows, so every
selected vehicle has both splits. Global statistics (copula, LA prior, Q) come from the subset only: a caveat versus the
5343-vehicle protocol, identical for every arm. Remaining args go to scripts/run_fl.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import data as D  # noqa: E402

CACHE = os.environ.get("V4_CACHE", "cache")
COUNTS = os.path.join(CACHE, "fold_counts.npy")


def counts():
    import pyarrow.parquet as pq
    out = np.zeros((D.N_CLIENTS, 2, D.N_CLS), np.int64)
    for f in D.files(D.find_root(), "train"):
        t = pq.ParquetFile(f).read(columns=["client_id", "fold", "label"])
        c, y = t.column("client_id").to_numpy(), t.column("label").to_numpy()
        fo = np.where(t.column("fold").to_numpy(zero_copy_only=False) == "val", 1, 0)
        np.add.at(out, (c, fo, y), 1)
    os.makedirs(CACHE, exist_ok=True)
    np.save(COUNTS, out)
    print("rows train/val", out[:, 0].sum(), out[:, 1].sum())


def fold_counts():
    """(5343, 2, 16) fold-train / fold-val rows per class: from V4_NPY (Kaggle, fold code per row) or the local cache."""
    npy = os.environ.get("V4_NPY")
    if npy and os.path.exists(os.path.join(npy, "done")):
        Y, F, off, ids = (np.load(os.path.join(npy, f"train_{k}.npy")) for k in ("Y", "F", "off", "ids"))
        client = np.repeat(ids, np.diff(off))
        out = np.zeros((D.N_CLIENTS, 2, D.N_CLS), np.int64)
        sel = F <= 1
        np.add.at(out, (client[sel], F[sel].astype(np.int64), Y[sel].astype(np.int64)), 1)
        return out
    return np.load(COUNTS)


def pick(n, seed=7, cnt=None):
    cnt = fold_counts() if cnt is None else cnt
    tr, va = cnt[:, 0].sum(1), cnt[:, 1].sum(1)
    ok = np.flatnonzero((tr > 0) & (va >= 32))
    edges = np.quantile(tr[ok], np.linspace(0, 1, 11))
    strata = np.clip(np.searchsorted(edges, tr[ok], side="right") - 1, 0, 9)
    rng = np.random.default_rng(seed)
    sel = [rng.choice(ok[strata == s], n // 10, replace=False) for s in range(10)]
    return np.sort(np.concatenate(sel))


def main():
    if sys.argv[1] == "counts":
        return counts()
    assert sys.argv[1] == "run"
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--widths", default="", help="R1 cell: DAGSNet width per rank, e.g. 1,0.25 (vehicles split by rank)")
    a, rest = ap.parse_known_args(sys.argv[2:])
    cnt = fold_counts()
    ids = pick(a.n, cnt=cnt)
    out = rest[rest.index("--out") + 1]
    os.environ.setdefault("MOBIHET_RECOVERY_DIR", os.path.join(out, "recovery"))
    orig = D.load

    def load(root, split, clients=None, cache=None, folds=None, with_fold=False):
        if split == "train":
            return orig(root, "train", clients, cache, folds=["train"])
        return orig(root, "train", clients, cache, folds=["val"])     # "test" = fold val of the train split
    D.load = load
    from core import fl
    fl.Env._sizes = staticmethod(lambda root, cache: cnt[:, 0].sum(1))
    if a.widths:   # mixed architectures: Core exchanges only class statistics/Q, so each rank may hold another model size
        from core.dagsnet import width
        ws = [float(w) for w in a.widths.split(",")]
        init = fl.Env.__init__

        def env_init(self, cfg, hp, clients=None, cache=None, root=None):
            rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
            assert len(ws) == world, (ws, world)
            init(self, width(ws[rank]), hp, clients, cache, root)
            import json
            os.makedirs(out, exist_ok=True)
            json.dump({"rank": rank, "width": ws[rank], "n_params": self.lay.P, "client_ids": self.ids.tolist()},
                      open(os.path.join(out, f"arch_rank{rank}.json"), "w"))
            print(f"R1 rank{rank}: width {ws[rank]} params {self.lay.P} vehicles {self.K}", flush=True)
        fl.Env.__init__ = env_init
    from scripts import run_fl
    sys.argv = ["run_fl.py", "--clients", ",".join(map(str, ids)), *rest]
    import hashlib
    print("sim ids sha256", hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()[:16], flush=True)
    print(f"sim: {len(ids)} vehicles, fold-train rows {cnt[ids, 0].sum()}, fold-val rows {cnt[ids, 1].sum()}", flush=True)
    run_fl.main()


if __name__ == "__main__":
    main()
