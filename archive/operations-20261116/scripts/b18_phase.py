"""One isolated B18 training process: shadow-only A, or full-train C/E/H."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import data as D, fl
from scripts.b18_partition import read, subset
from scripts.sim_fold_val import fold_counts, pick

CBB_HP = (ROOT / "runs/sim/core_hp.txt").read_text().split() + [
    "cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--phase", choices=["shadow", "C", "E", "H"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--until", type=int, default=20)
    ap.add_argument("--max-hours", type=float, default=0)
    a = ap.parse_args()
    part, meta = read(a.root)
    counts = fold_counts()
    assert np.array_equal(part["ids"], pick(200, cnt=counts))
    assert np.array_equal(part["early_counts"] + part["heldout_counts"], counts[part["ids"], 0])
    original = D.load
    D.load_full_foldtrain = original

    def load(root, split, clients=None, cache=None, folds=None, with_fold=False):
        if a.phase == "shadow":
            full = original(root, "train", clients, cache, folds=["train"])
            return subset(full, part, 1)
        return original(root, "train", clients, cache, folds=["train" if split == "train" else "val"])

    sizes = counts[:, 0].sum(1).copy()
    if a.phase == "shadow":
        sizes[part["ids"]] = part["early_counts"].sum(1)
    D.load = load
    fl.Env._sizes = staticmethod(lambda root, cache: sizes)
    out = Path(a.root) / a.phase
    os.environ["MOBIHET_RECOVERY_DIR"] = str(out / "recovery")
    hp = [h for h in CBB_HP if not h.startswith("seed=")] + [f"seed={a.seed}",
          f"source_mode={'S' if a.phase == 'shadow' else a.phase}", f"source_root={a.root}",
          f"source_partition_sha={meta['logical_sha256']}"]
    from scripts import run_fl
    sys.argv = ["run_fl.py", "--method", "mobihet_source", "--clients", ",".join(map(str, part["ids"])),
                "--out", str(out), "--rounds", "20", "--until", str(a.until), "--compile", "--max-hours", str(a.max_hours),
                "--set", *hp]
    run_fl.main()


if __name__ == "__main__":
    main()
