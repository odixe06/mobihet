"""Actual fold-train rows: compiled W1 AMP/dropout replay at identical parameters.

Checks the RNG/stride prerequisite of SAM. CPU tests separately verify its nonzero
perturbation against an explicit two-pass reference; this is not trajectory proof.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import data, grouped  # noqa: E402
from core.dagsnet import width  # noqa: E402
from core.flat import Layout  # noqa: E402


def main():
    assert torch.cuda.is_available()
    source = data.files(data.find_root(), "train")[0]
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096, columns=[*data.ORDER, "label", "fold"]):
        table = batch.to_pydict()
        keep = np.array(table["fold"]) == "train"
        if keep.sum() >= 256:
            x = np.column_stack([np.asarray(table[k])[keep][:256] for k in data.ORDER]).astype(np.float32)
            y = np.asarray(table["label"])[keep][:256].astype(np.int64)
            break
    else:
        raise AssertionError("Need 256 real fold-train fixture rows")
    cfg, device = width(1), torch.device("cuda")
    lay = Layout(cfg)
    flat, buffers = lay.init(42, 2, device)
    xx = torch.from_numpy(x).to(device).view(2, 128, 77)
    yy = torch.from_numpy(y).to(device).view(2, 128)
    mask = torch.ones(2, 128, dtype=torch.bool, device=device)
    compiled = torch.compile(grouped.forward, dynamic=False)

    def step(weights):
        p = {k: v.detach().requires_grad_(True) for k, v in lay.params(weights).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            z, b = compiled(p, lay.bufs(buffers), xx, cfg, mask=mask, train=True)
            loss = F.cross_entropy(z.reshape(-1, 16), yy.reshape(-1))
        g = torch.autograd.grad(loss * 4096, tuple(p[k] for k, _ in lay.p))
        grad = torch.cat([v.reshape(2, -1) for v in g], 1) / 4096
        assert torch.isfinite(grad).all() and torch.isfinite(z).all()
        return z.detach(), lay.flat_bufs(b), grad

    step(flat)  # compile before the measured replay
    rng = torch.cuda.get_rng_state(device)
    first = step(flat)
    after = torch.cuda.get_rng_state(device)
    torch.cuda.set_rng_state(rng, device)
    repeated = step(flat.detach() + torch.zeros_like(flat))
    assert all(torch.equal(a, b) for a, b in zip(first, repeated)), "Compiled dropout/stride replay differs"
    assert torch.equal(after, torch.cuda.get_rng_state(device))
    print(json.dumps({"pass": True, "scope": "Compiled W1 AMP/dropout/gradient/BN RNG replay, one local GPU",
                      "sample_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                      "source": str(Path(source).relative_to(data.find_root())), "fold": "train", "rows": 256,
                      "n_params": lay.P, "device": torch.cuda.get_device_name(device),
                      "capability": torch.cuda.get_device_capability(device), "torch": torch.__version__}))


if __name__ == "__main__":
    main()
