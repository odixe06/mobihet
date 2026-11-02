"""Offline byte measurement of the copula bootstrap uplink: dense fp32 (current) vs lossless sparse per-client counts.

Reads train/ parquet (train+val, the FL train set), bins exactly like methods/mobihet.py `_copula_bootstrap` (256 bins on
[-4, 4]), and counts the non-zero (class, feature, bin) cells each client would upload. No training, no test rows.
    python scripts/measure_bootstrap_codec.py --root ../mobihet-features/data/v4-scaled --out runs/bootstrap-codec
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq

from core import data as D

C, NB = 16, 256


def varint_len(v):
    """Byte length of the unsigned LEB128 varint of each value."""
    v = np.asarray(v, dtype=np.int64)
    return 1 + sum((v >= (1 << (7 * k))).astype(np.int64) for k in range(1, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/bootstrap-codec")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cols = D.ORDER
    D_ = len(cols)
    assert D_ == 77
    nnz = np.zeros(D.N_CLIENTS, np.int64)          # non-zero (class, feature, bin) cells per client
    entry_bytes = np.zeros(D.N_CLIENTS, np.int64)  # delta-index + count varints per client
    rows = np.zeros(D.N_CLIENTS, np.int64)
    B = 32                                           # clients span files: unique per file, merge per client bucket
    parts = [[] for _ in range(B)]
    for f in D.files(str(a.root), "train"):
        t = pq.ParquetFile(f).read(columns=["client_id", "label", *cols])
        cid = t.column("client_id").to_numpy().astype(np.int64)
        y = t.column("label").to_numpy().astype(np.int64)
        x = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float16) for c in cols], 1)
        bins = np.clip(((x.astype(np.float32) + 4) * (NB / 8)).astype(np.int64), 0, NB - 1)   # fp16 cache values, as in training
        k, cnt = np.unique((((cid[:, None] * C + y[:, None]) * D_ + np.arange(D_)) * NB + bins).ravel(), return_counts=True)
        bucket = (k // (C * D_ * NB)) % B
        for i in range(B):
            sel = bucket == i
            parts[i].append((k[sel], cnt[sel].astype(np.int64)))
        np.add.at(rows, cid, 1)
    assert rows.sum() == 37361273, rows.sum()
    for i in range(B):
        k = np.concatenate([p[0] for p in parts[i]]); c = np.concatenate([p[1] for p in parts[i]])
        parts[i] = None
        order = np.argsort(k, kind="stable"); k, c = k[order], c[order]
        head = np.r_[True, k[1:] != k[:-1]]
        k, cnt = k[head], np.add.reduceat(c, np.nonzero(head)[0])
        owner = k // (C * D_ * NB)
        np.add.at(nnz, owner, 1)
        cell = k % (C * D_ * NB)
        start = np.r_[True, owner[1:] != owner[:-1]]
        delta = np.where(start, cell, np.diff(cell, prepend=0))
        np.add.at(entry_bytes, owner, varint_len(delta) + varint_len(cnt))
    dense = 4 * C * D_ * NB
    sparse_fixed = nnz * 6                          # uint32 flat index + uint16 count (counts < 65536 per cell per client)
    out = {"clients": int(D.N_CLIENTS), "dense_fp32_bytes_per_client": dense,
           "nnz": {"median": float(np.median(nnz)), "p90": float(np.percentile(nnz, 90)), "max": int(nnz.max())},
           "sparse_fixed6_bytes": {"mean": float(sparse_fixed.mean()), "median": float(np.median(sparse_fixed)),
                                   "p90": float(np.percentile(sparse_fixed, 90)), "max": int(sparse_fixed.max())},
           "sparse_varint_bytes": {"mean": float(entry_bytes.mean()), "median": float(np.median(entry_bytes)),
                                   "p90": float(np.percentile(entry_bytes, 90)), "max": int(entry_bytes.max())},
           "small_lt256": {"clients": int((rows < 256).sum()), "varint_mean": float(entry_bytes[rows < 256].mean())},
           "scope": "Uplink histogram part only (dense 1261568 B of the 1655872 B copula uplink); lossless, server sums identical."}
    (a.out / "uplink-hist.json").write_text(json.dumps(out, indent=1))
    np.save(a.out / "nnz.npy", nnz)
    np.save(a.out / "varint_bytes.npy", entry_bytes)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
