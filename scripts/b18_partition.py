"""Nested temporal split in canonical within-client FILE order, sim fold-train only."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from core import data as D
from scripts.sim_fold_val import fold_counts, pick


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def roles_for(y, t, client):
    cut = np.partition(t, math.ceil(.8 * len(t)) - 1)[math.ceil(.8 * len(t)) - 1]
    early = t <= cut
    assert early.any() and (~early).any() and t[early].max() < t[~early].min()
    a, h = (np.bincount(y[mask], minlength=16) for mask in (early, ~early))
    eligible = (a >= 8) & (h >= 8)
    roles = early.astype(np.uint8)
    # SplitMix64 hash of stable (client, canonical local row ordinal), no model seed.
    with np.errstate(over="ignore"):
        z = np.arange(len(y), dtype=np.uint64) + np.uint64(client + 1) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        z ^= z >> 31
    for c in np.flatnonzero(eligible):
        m = min(a[c], h[c])
        for mask, bit in ((early, 2), (~early, 4)):
            rows = np.flatnonzero((y == c) & mask)
            chosen = rows[np.argsort(z[rows], kind="stable")[:m]]
            roles[chosen] |= bit
    assert not np.any((roles & 2 > 0) & (roles & 4 > 0))
    assert np.all(early[roles & 2 > 0]) and not np.any(early[roles & 4 > 0])
    return roles, a, h, int(cut)


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "partition.json").exists(), "Preserve existing partition"
    counts = fold_counts()
    ids = pick(200, cnt=counts)
    pieces = {int(i): [] for i in ids}
    files = D.files(D.find_root(), "train")
    assert len(files) == 410
    for p in files:
        tab = pq.ParquetFile(p).read(columns=["client_id", "fold", "label", "rcv_time_ns"])
        tab = tab.filter(pc.and_(pc.is_in(tab["client_id"], value_set=pa.array(ids)), pc.equal(tab["fold"], "train")))
        c, y, t = (tab[k].to_numpy() for k in ("client_id", "label", "rcv_time_ns"))
        for k in np.unique(c):
            keep = c == k
            pieces[int(k)].append((y[keep].astype(np.int8), t[keep].astype(np.int64)))
    all_roles, ys, early, held, cuts = [], [], [], [], []
    for k in ids:
        y = np.concatenate([s[0] for s in pieces[int(k)]])
        t = np.concatenate([s[1] for s in pieces[int(k)]])
        assert np.array_equal(np.bincount(y, minlength=16), counts[k, 0])
        r, a, h, cut = roles_for(y, t, int(k))
        all_roles.append(r); ys.append(y); early.append(a); held.append(h); cuts.append(cut)
    off = np.r_[0, np.cumsum([len(y) for y in ys])].astype(np.int64)
    roles, labels = np.concatenate(all_roles), np.concatenate(ys)
    early, held = np.array(early), np.array(held)
    probes = np.zeros_like(early)
    for j in range(len(ids)):
        y, r = labels[off[j]:off[j + 1]], roles[off[j]:off[j + 1]]
        e, h = (np.bincount(y[r & bit > 0], minlength=16) for bit in (2, 4))
        assert np.array_equal(e, h)
        probes[j] = e
    assert int(early.sum()) == 1181080 and int(held.sum()) == 294984
    label_hashes = np.array([hashlib.sha256(y.tobytes()).hexdigest().encode() for y in ys], dtype="S64")
    arrays = dict(ids=ids.astype(np.int64), off=off, roles=roles, label_sha256=label_hashes,
                  early_counts=early, heldout_counts=held, probe_counts=probes, cutoffs_ns=np.array(cuts, dtype=np.int64))
    np.savez_compressed(out / "partition.npz", **arrays)
    meta = dict(logical_sha256=digest(*arrays.values()), array_order=list(arrays), rows=len(roles),
                early_rows=int(early.sum()), heldout_rows=int(held.sum()), matched_probe_rows=int(probes.sum()),
                eligible_population_mass=float(((early + held) * (probes > 0)).sum() / (early + held).sum()),
                source="410 train parquets; fold=train only; canonical per-client file order",
                limits="No embargo/event independence; model-out-of-fit conditional on fixed v4 preprocessing")
    (out / "partition.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("B18_PARTITION", json.dumps(meta), flush=True)


def read(path):
    with np.load(Path(path) / "partition.npz") as z:
        a = {k: z[k].copy() for k in z.files}
    meta = json.loads((Path(path) / "partition.json").read_text())
    assert digest(*(a[k] for k in meta["array_order"])) == meta["logical_sha256"]
    return a, meta


def subset(split, part, bit):
    """Validate row labels/length before applying sidecar positions to resident data."""
    assert not __import__("torch").is_tensor(split.X), "Filter CPU data before device transfer"
    masks = []
    for j, cid in enumerate(split.ids):
        k = np.searchsorted(part["ids"], cid)
        assert part["ids"][k] == cid
        roles = part["roles"][part["off"][k]:part["off"][k + 1]]
        labels = split.Y[split.off[j]:split.off[j + 1]]
        assert len(labels) == len(roles), "Cache/metadata row count mismatch"
        assert hashlib.sha256(labels.astype(np.int8).tobytes()).hexdigest().encode() == part["label_sha256"][k], "Cache/metadata row order mismatch"
        masks.append(roles & bit > 0)
    sizes = np.array([int(m.sum()) for m in masks])
    keep = np.concatenate(masks)
    # Probe subsets may contain zero rows for an individual client; preserve its slot.
    return D.Split(np.ascontiguousarray(split.X[keep]), split.Y[keep], np.r_[0, sizes.cumsum()], split.ids.copy())
