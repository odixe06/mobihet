"""Train-only numerical probe of P21's initial eager compile-gate forward; no training."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import torch

from core import data, grouped, local
from core.dagsnet import width
from core.flat import Layout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study/numerics-probe"


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = np.load(ROOT / "evidence/research-data/train_counts.npy").sum(1)
    order = np.argsort(-sizes, kind="stable")
    selected = []
    for rank in (0, 1):
        mine = np.sort(order[rank::2])
        selected.append(mine[np.argsort(-sizes[mine], kind="stable")[:64]])
    ids = np.unique(np.concatenate(selected))
    rows = {int(k): [] for k in ids}
    labels = {int(k): [] for k in ids}
    remaining = set(rows)
    files_read = []
    for file in data.files(data.find_root(), "train"):
        pf = pq.ParquetFile(file)
        cid = pf.read(columns=["client_id"]).column("client_id").to_numpy()
        present = remaining.intersection(np.unique(cid).tolist())
        if not present: continue
        table = pf.read(columns=data.ORDER + ["label"])
        files_read.append(str(file))
        for k in present:
            take = np.flatnonzero(cid == k)[:64 - len(rows[k])]
            x = np.stack([table.column(col).to_numpy()[take] for col in data.ORDER], 1).astype(np.float16)
            y = table.column("label").to_numpy()[take]
            rows[k].extend(x); labels[k].extend(y)
        remaining = {k for k in remaining if len(rows[k]) < 64}
        if not remaining: break
    assert not remaining
    cfg = width(1, dropout=0.)
    lay = Layout(cfg)
    records, fixtures = [], []
    original_bn = grouped._bn
    device = torch.device("cuda:0")
    for rank, selected_ids in enumerate(selected):
        x = np.stack([rows[int(k)] for k in selected_ids])
        y = np.stack([labels[int(k)] for k in selected_ids])
        assert x.shape == (64, 64, 77) and np.isfinite(x).all()
        fixture = OUT / f"rank{rank}-train64.npz"
        np.savez_compressed(fixture, ids=selected_ids, X=x, Y=y)
        fixtures.append({"rank": rank, "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         "ids": selected_ids.tolist(), "input_abs_max": float(np.abs(x).max())})
        for seed in (42, 43):
            flat, bufs = lay.init(seed, 64, device)
            for fp16 in (True, False):
                trace = []
                def bn(ctx, name, value):
                    result = original_bn(ctx, name, value)
                    if not bool(torch.isfinite(result).all()):
                        trace.append({"layer": name, "input_finite": bool(torch.isfinite(value).all()),
                                      "output_nonfinite": int((~torch.isfinite(result)).sum()),
                                      "input_abs_max": float(value.float().abs().nan_to_num().max())})
                    return result
                grouped._bn = bn
                try:
                    xx = torch.as_tensor(x, device=device).float()
                    mask = torch.ones(64, 64, dtype=torch.bool, device=device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                        z, nb = grouped.forward(lay.params(flat), lay.bufs(bufs), xx, cfg, mask=mask, train=True)
                    bad = ~torch.isfinite(z)
                    records.append({"rank": rank, "seed": seed, "amp_fp16": fp16,
                                    "nonfinite_logits": int(bad.sum()),
                                    "affected_clients": selected_ids[bad.any(2).any(1).cpu().numpy()].tolist(),
                                    "bn_buffers_finite": bool(torch.isfinite(lay.flat_bufs(nb)).all()),
                                    "first_bad_bn": trace[:1]})
                    print("P20_NUMERICS", rank, seed, fp16, int(bad.sum()), flush=True)
                finally:
                    grouped._bn = original_bn
    report = {"device": torch.cuda.get_device_name(device), "capability": torch.cuda.get_device_capability(device),
              "torch": torch.__version__, "fixtures": fixtures, "files_read": files_read, "records": records,
              "scope": "Local train-only forward probe; stable tie selection/row order, no backward or P21 crash-state replay, not T4 proof"}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("P20_NUMERICS_PROBE_DONE", flush=True)


if __name__ == "__main__": main()
