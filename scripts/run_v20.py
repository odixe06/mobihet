"""Diagnostic-only foldtrain -> foldval. Headline core data/protocol remain unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core import data, fl


def build_cache(root, out):
    """Cache original train/ only; never decode officialtest for this diagnostic."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    s = data.load(root, "train", with_fold=True)
    for key in ("X", "Y", "off", "ids", "F"): np.save(out / f"train_{key}.npy", getattr(s, key))
    (out / "done").write_text("diagnostic original-foldtrain/foldval only\n")


class DiagnosticEnv(fl.Env):
    @staticmethod
    def _sizes(root, cache):
        npy = os.environ.get("V4_NPY")
        if npy and (Path(npy) / "done").exists():
            folds = np.load(Path(npy) / "train_F.npy", mmap_mode="r")
            off = np.load(Path(npy) / "train_off.npy")
            ids = np.load(Path(npy) / "train_ids.npy")
            sizes = np.zeros(5343, np.int64)
            sizes[ids] = [np.count_nonzero(folds[off[i]:off[i + 1]] == 0) for i in range(len(ids))]
            return sizes
        import pyarrow.parquet as pq
        sizes = np.zeros(5343, np.int64)
        for name in data.files(root, "train"):
            t = pq.ParquetFile(name).read(columns=["client_id", "fold"])
            c, f = (t.column(k).to_numpy(zero_copy_only=False) for k in ("client_id", "fold"))
            sizes += np.bincount(c[f == "train"], minlength=5343)
        return sizes

    def __init__(self, cfg, hp, clients=None, cache=None, root=None):
        original = data.load
        def fold_load(root, split, clients=None, cache=None, folds=None, with_fold=False):
            assert folds is None and not with_fold and split in {"train", "test"}
            return original(root, "train", clients, cache, folds=["train" if split == "train" else "val"])
        try:
            data.load = fold_load
            super().__init__(cfg, hp, clients=clients, cache=cache, root=root)
        finally:
            data.load = original
        root = Path(root or data.find_root())
        scaler = json.loads((root / "scaler.json").read_text())
        assert scaler["fit_on"] == "fold == train" and scaler["train_rows"] == 32685166
        assert np.array_equal(self.train.sizes, self.n_all[np.searchsorted(self.all_ids, self.ids)])
        self.fold_receipt = {"training_fold": "train", "evaluation_fold": "val", "officialtest_read": False,
            "local_train_rows": int(len(self.train.Y)), "local_val_rows": int(len(self.test.Y)),
            "scaler_fit_on": scaler["fit_on"], "scaler_sha256": hashlib.sha256((root / "scaler.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--out", required=True)
    a, _ = parser.parse_known_args()
    os.environ["MOBIHET_RECOVERY_DIR"] = str(Path(a.out) / "recovery")
    from scripts import run_fl
    class RecordedEnv(DiagnosticEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            folder = Path(a.out); folder.mkdir(parents=True, exist_ok=True)
            (folder / f"fold_rank{self.rank}.json").write_text(json.dumps(self.fold_receipt, indent=2) + "\n")
    run_fl.fl.Env = RecordedEnv
    run_fl.main()


if __name__ == "__main__": main()
