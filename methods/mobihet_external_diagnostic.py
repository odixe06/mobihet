"""Simulation evaluator only: fixed test rows of other vehicles, never used by training or FL messages."""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from core import grouped, metrics

_POOLS = {}


def test_pool(npy):
    """Fixed class-stratified candidates from the laboratory's test memmap. No train-time fitting."""
    key = str(Path(npy).resolve())
    if key not in _POOLS:
        root = Path(npy)
        X = np.load(root / "test_X.npy", mmap_mode="r")
        y = np.load(root / "test_Y.npy", mmap_mode="r")
        off = np.load(root / "test_off.npy")
        ids = np.load(root / "test_ids.npy")
        rng = np.random.default_rng(20261101)
        pools = []
        for c in range(16):
            rows = np.flatnonzero(y == c)
            rows = rng.choice(rows, min(1024, len(rows)), replace=False)
            source = ids[np.searchsorted(off[1:], rows, side="right")]
            pools.append((np.asarray(X[rows]).copy(), source, rows))
        _POOLS[key] = pools
    return _POOLS[key]


@torch.no_grad()
def run(method, r, out):
    """Score only fixed diagnostic clients; same external rows across seeds/methods, own source excluded.

    The simulator can read the full test memmap; this evaluator is separate from the method protocol.
    Persist CM/support and selection hashes, never raw test rows or predictions in FL packets.
    """
    env = method.env
    if not env.hp.get("external_diagnostics", False) or r not in (1, 10, 20):
        return {}
    pools = test_pool(os.environ["V4_NPY"])
    cms, ids, hashes = [], [], []
    for k, cid in enumerate(env.ids):
        if int(cid) not in env.hp["diagnostic_ids"]:
            continue
        cm = torch.zeros(16 * 16, dtype=torch.int64, device=env.dev)
        digest = hashlib.sha256()
        P, B = env.lay.params(method.priv_p[k:k + 1]), env.lay.bufs(method.priv_b[k:k + 1])
        for c, (x, source, rows) in enumerate(pools):
            sel = np.flatnonzero(source != int(cid))[:256]
            digest.update(rows[sel].astype("<i8").tobytes())
            for start in range(0, len(sel), 256):
                batch = torch.from_numpy(x[sel[start:start + 256]]).to(env.dev)[None]
                with torch.autocast("cuda", dtype=torch.float16, enabled=env.dev.type == "cuda"):
                    logits, _ = grouped.EVAL(P, B, batch, env.cfg, train=False)
                cm += torch.bincount(c * 16 + logits[0].argmax(-1), minlength=256)
        cms.append(cm.view(16, 16).cpu().numpy())
        ids.append(int(cid))
        hashes.append(digest.hexdigest())
    cm = np.asarray(cms, dtype=np.int64).reshape(-1, 16, 16)
    np.savez_compressed(Path(out) / f"external_cms_rank{env.rank}_r{r:03}.npz",
                        ids=np.asarray(ids, dtype=np.int64), private=cm, selection_sha256=np.asarray(hashes))
    if not len(cm):
        return {"external_diagnostic_clients_local": 0}
    # Rank-local interim scalars only; merged CM is the reporting source after collection.
    s = metrics.summary(cm)
    return {"external_diagnostic_clients_local": len(cm), "external_diagnostic_f1_local": s["f1_macro"]}
