"""Sender-session sidecar for decision fusion (docs/research/improvement-proposals.md §Hướng a). No model input changes.

    python scripts/build_sessions.py full            -> cache/sessions/{train,test}_{sess,t,y}.npy  (core.data loader row order)
    python scripts/build_sessions.py sim [--n 200]   -> cache/sessions/sim_val200.npz (sim panel, fold val rows, per-client blocks)

Session = (client, world, sender_alias): the pseudonym a receiver actually sees, inside one simulation replica. sender_id (true
vehicle) is never used: it would link Sybil pseudonyms (label leak). (client, alias) alone mixes the benign twins drawn from other
worlds with the attack rows (17 % rows in label-pure sessions vs 99 % with world, fold val 09/11).
v3-raw rows are positionally identical to v4-scaled (same 410 files per split, same row order): asserted per file on
client_id/sender_id/message_id/rcv_time_ns. Row order replicates core.data.load: sorted files, stable sort by client per file.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from core import data as D  # noqa: E402

V4 = "/home/odixe/project/mobihet-features/data/v4-scaled"
V3 = "/home/odixe/project/mobihet-features/data/v3-raw"
OUT = os.path.join(ROOT, "cache", "sessions")
KEYS = ["client_id", "sender_id", "message_id", "rcv_time_ns"]


def v3_path(f4):
    rel = os.path.relpath(f4, V4).replace("scenario=", "").replace("world=", "")
    return os.path.join(V3, rel)


def build(split):
    fs = D.files(V4, split)
    cid = [pq.ParquetFile(f).read(columns=["client_id"]).column("client_id").to_numpy() for f in fs]
    counts = np.zeros(D.N_CLIENTS, np.int64)
    for c in cid:
        counts += np.bincount(c, minlength=D.N_CLIENTS)
    off = np.concatenate([[0], np.cumsum(counts)])
    N = int(off[-1])
    world, alias, t, y = np.empty(N, np.int8), np.empty(N, np.int64), np.empty(N, np.int64), np.empty(N, np.int8)
    client = np.empty(N, np.int32)
    cur = off[:-1].copy()
    wnames = {}
    for f, c in zip(fs, cid):
        a = pq.ParquetFile(f).read(columns=KEYS + ["label", "world"]).to_pandas()
        b = pq.ParquetFile(v3_path(f)).read(columns=KEYS + ["sender_alias"]).to_pandas()
        assert len(a) == len(b) and all((a[k].values == b[k].values).all() for k in KEYS), f
        w = wnames.setdefault(str(a["world"].iloc[0]), len(wnames))
        assert (a["world"].astype(str) == list(wnames)[w]).all(), f
        o = np.argsort(c, kind="stable")
        cs = c[o]
        u, first, cnt = np.unique(cs, return_index=True, return_counts=True)
        al, tt, yy = b["sender_alias"].to_numpy()[o], a["rcv_time_ns"].to_numpy()[o], a["label"].to_numpy()[o]
        for k, i0, m in zip(u, first, cnt):
            s = slice(cur[k], cur[k] + m)
            world[s], alias[s], t[s], y[s], client[s] = w, al[i0:i0 + m], tt[i0:i0 + m], yy[i0:i0 + m], k
            cur[k] += m
    assert (cur == off[1:]).all()
    sess = pd.DataFrame({"c": client, "w": world, "a": alias}).groupby(["c", "w", "a"], sort=False).ngroup().to_numpy().astype(np.int64)
    os.makedirs(OUT, exist_ok=True)
    for k, v in (("sess", sess), ("t", t), ("y", y), ("off", off)):
        np.save(os.path.join(OUT, f"{split}_{k}.npy"), v)
    print(split, N, "rows", int(sess.max()) + 1, "sessions", flush=True)


def sim(n):
    from scripts.sim_fold_val import fold_counts, pick
    ids = np.sort(pick(n, cnt=fold_counts()))
    sess, t, y, off = (np.load(os.path.join(OUT, f"train_{k}.npy"), mmap_mode="r") for k in ("sess", "t", "y", "off"))
    fold = np.empty(len(y), np.int8)
    fs = D.files(V4, "train")                                        # fold code in loader order (as build_npy F)
    cid = [pq.ParquetFile(f).read(columns=["client_id", "fold"]) for f in fs]
    cur = np.asarray(off[:-1]).copy()
    for tb in cid:
        c = tb.column("client_id").to_numpy()
        fo = (tb.column("fold").to_numpy(zero_copy_only=False) == "val").astype(np.int8)
        o = np.argsort(c, kind="stable")
        cs, fo = c[o], fo[o]
        u, first, cnt = np.unique(cs, return_index=True, return_counts=True)
        for k, i0, m in zip(u, first, cnt):
            fold[cur[k]:cur[k] + m] = fo[i0:i0 + m]
            cur[k] += m
    blocks, boff = [], [0]
    for k in ids:
        r = np.arange(off[k], off[k + 1])
        r = r[fold[r] == 1]
        blocks.append(r)
        boff.append(boff[-1] + len(r))
    rows = np.concatenate(blocks)
    s = np.unique(np.asarray(sess)[rows], return_inverse=True)[1].astype(np.int32)
    tt, yy = np.asarray(t)[rows], np.asarray(y)[rows].astype(np.int8)
    o = np.lexsort((tt, s))                                          # compact: time rank inside the session (no int64 time)
    ss = s[o]
    start = np.maximum.accumulate(np.where(np.concatenate([[True], ss[1:] != ss[:-1]]), np.arange(len(ss)), 0))
    rank = np.empty(len(s), np.int64)
    rank[o] = np.arange(len(ss)) - start
    rank = rank.astype(np.uint16 if rank.max() < 65535 else np.int32)
    out = os.path.join(OUT, f"sim_val{n}.npz")                        # notebook overlay: SaveKernel rejected a 1.47 MB notebook
    np.savez_compressed(out, ids=ids, off=np.asarray(boff), sess=s, rank=rank, y=yy)
    print(out, len(rows), "rows", s.max() + 1, "sessions", os.path.getsize(out), "bytes", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "full":
        for sp in ("train", "test"):
            build(sp)
    else:
        sim(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200)
