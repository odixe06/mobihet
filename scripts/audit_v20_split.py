"""Stream original folds: class support, temporal separation and train-only scaler provenance."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from core import data


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    out = ROOT / "runs/v-study"; out.mkdir(exist_ok=True)
    root = Path(data.find_root())
    scaler = json.loads((root / "scaler.json").read_text())
    assert scaler["fit_on"] == "fold == train" and scaler["train_rows"] == 32685166
    counts = {f: np.zeros((5343, 16), dtype=np.int64) for f in ("train", "val")}
    lo = {f: np.full(5343, np.iinfo(np.int64).max, dtype=np.int64) for f in counts}
    hi = {f: np.full(5343, np.iinfo(np.int64).min, dtype=np.int64) for f in counts}
    sources = []
    for name in data.files(root, "train"):
        path = Path(name)
        frame = pq.ParquetFile(path).read(columns=["client_id", "label", "fold", "rcv_time_ns"])
        c, y, folds, t = (frame.column(k).to_numpy(zero_copy_only=False) for k in ("client_id", "label", "fold", "rcv_time_ns"))
        assert np.isin(folds, ["train", "val"]).all() and (c >= 0).all() and (c < 5343).all()
        assert (y >= 0).all() and (y < 16).all()
        for f in counts:
            pick = folds == f
            cid, label, time = c[pick].astype(np.int64), y[pick].astype(np.int64), t[pick].astype(np.int64)
            counts[f] += np.bincount(cid * 16 + label, minlength=5343 * 16).reshape(5343, 16)
            np.minimum.at(lo[f], cid, time); np.maximum.at(hi[f], cid, time)
        sources.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha(path), "rows": len(c)})
    assert counts["train"].sum() == 32685166 and counts["val"].sum() == 4676107
    assert all((v.sum(1) > 0).all() for v in counts.values())
    assert (hi["train"] <= lo["val"]).all(), "Original folds not temporally separated for every client"
    combined = np.load(ROOT / "evidence/research-data/train_counts.npy")
    assert np.array_equal(counts["train"] + counts["val"], combined)
    hashes = {}
    for f, value in counts.items():
        path = out / f"{f}_counts.npy"; np.save(path, value); hashes[path.name] = file_sha(path)
    receipt = {"train_rows": int(counts["train"].sum()), "val_rows": int(counts["val"].sum()), "clients": 5343,
        "train_plus_val_counts_exact": True, "all_clients_temporal_train_before_val": True,
        "scaler_fit_on": scaler["fit_on"], "scaler_sha256": file_sha(root / "scaler.json"),
        "input_order_sha256": file_sha(ROOT / "knowledge/input_order.json"), "counts_sha256": hashes,
        "sources": sources, "scope": "All original train-part content SHA/fold counts/time bounds; no officialtest read, raw rows not shared"}
    (out / "split-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("V20_SPLIT_AUDIT_PASS", flush=True)


if __name__ == "__main__": main()
