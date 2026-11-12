"""b14 actual packet replay: every sender/round, two-rank sums, Q8 and teacher hash.

FP32 log transforms have a prelocked 1e-5 envelope; LG transmitted scores require
1e-10. Hybrid LG cells use their count-weighted component envelope, since small
Q8 cells also execute an FP32 log transform. Q8 uncertainty is allowed only across a 1e-6
probability rounding-boundary interval, never a blanket integer-code tolerance.
"""
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.mobihet_next import signature  # noqa: E402
from scripts.sim_fold_val import pick  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json  # noqa: E402


def raw_body(frame, flag):
    assert frame[flag] in (0, 1)
    return zlib.decompress(frame[flag + 1:]) if frame[flag] else frame[flag + 1:]


def replay(frame, count, mode):
    assert frame[:32] == bytes(32)
    mask = lambda b: np.array([bool(int.from_bytes(b, "little") & (1 << c)) for c in range(16)])
    present, eligible = mask(frame[32:34]), mask(frame[34:36])
    assert np.array_equal(present, count > 0) and np.array_equal(eligible, count >= 8)
    raw = raw_body(frame, 36)
    n = int(eligible.sum()) * 16 * 2
    assert len(raw) == n + int((present & ~eligible).sum()) * 16
    large = torch.from_numpy(np.frombuffer(raw[:n], "<f2").astype(np.float32)).view(-1, 16)
    assert torch.isfinite(large).all()
    if mode == "gp16":
        assert ((large >= 0) & (large <= 1)).all()
        assert torch.all((large.sum(-1) - 1).abs() <= .001)
        large = large.clamp_min(1e-6).log()
        large -= large.mean(-1, keepdim=True)
    small = torch.from_numpy(np.frombuffer(raw[n:], np.uint8).copy()).float().view(-1, 16)
    assert torch.all((small.sum(-1) - 255).abs() <= 8)
    small = (small / 255).clamp_min(1e-6).log()
    small -= small.mean(-1, keepdim=True)
    result = torch.zeros(16, 16)
    result[eligible], result[present & ~eligible] = large, small
    return result.double().numpy() * count[:, None]


def check_aggregate(total, stored, count, fallback_count, mode):
    assert np.all((fallback_count >= 0) & (fallback_count <= count)) and np.all(count > 0)
    error = np.abs((total - stored) / count[:, None])
    fraction = fallback_count / count
    bound = np.full_like(count, 1e-5, dtype=float) if mode == "gp16" else (1 - fraction) * 1e-10 + fraction * 1e-5
    assert np.all(error <= bound[:, None]), float(error.max())
    return float(error.max())


def main():
    local_run = Path(sys.argv[2]) if len(sys.argv) == 3 and sys.argv[1] == "--local" else None
    out, art = ROOT / "runs/sim-kaggle-b14", ROOT / "artifacts/sim-kaggle-b14"
    if not local_run:
        assert json.loads((out / "audit.json").read_text())["pass"]
    counts = np.load(ROOT / "cache/fold_counts.npy")
    panel = pick(200, cnt=counts)
    records = {}
    batch = ([{"arm": "local", "hp": ["uplink_mode=lg16"]}] if local_run else json.loads((out / "batch.json").read_text()))
    for spec in batch:
        mode = dict(h.split("=", 1) for h in spec["hp"]).get("uplink_mode", "prob")
        if mode == "prob":
            continue
        run = local_run if local_run else next((art / spec["slug"]).glob("**/config.json")).parent
        rows = [json.loads(s) for s in (run / "rounds.jsonl").read_text().splitlines() if s.strip()]
        worst = 0.
        for row in rows:
            shards = [np.load(run / "wire" / f"rank{k}_r{row['round']:03d}.npz") for k in range(1 if local_run else 2)]
            ids = np.concatenate([s["ids"] for s in shards])
            assert np.array_equal(np.sort(ids), panel) and len(np.unique(ids)) == 200
            total, size = np.zeros((16, 16), np.float64), 0
            fallback_count = np.zeros(16, np.int64)
            for s in shards:
                assert np.array_equal(s["counts"], counts[s["ids"], 0])
                data, offsets = s["frames"].tobytes(), s["offsets"]
                assert offsets[0] == 0 and offsets[-1] == len(data) and len(offsets) == len(s["ids"]) + 1
                for j, count in enumerate(s["counts"]):
                    total += replay(data[offsets[j]:offsets[j + 1]], count, mode)
                    fallback_count += np.where((count > 0) & (count < 8), count, 0)
                size += len(data)
            n = counts[panel, 0].sum(0)
            assert np.array_equal(shards[0]["global_count"], n)
            assert all(np.array_equal(shards[0][k], s[k]) for s in shards for k in ("global_sum", "global_count", "down", "Q"))
            error = check_aggregate(total, shards[0]["global_sum"], n, fallback_count, mode)
            worst = max(worst, error)
            down = shards[0]["down"].tobytes()
            codes = np.frombuffer(raw_body(down, 32), np.uint8).reshape(16, 16)
            assert np.all(codes.sum(-1) > 0)
            score = torch.from_numpy((shards[0]["global_sum"] / n[:, None]).astype(np.float32))
            p = score.softmax(-1).numpy()
            lo, hi = np.rint(np.clip(p - 1e-6, 0, 1) * 255), np.rint(np.clip(p + 1e-6, 0, 1) * 255)
            assert np.all((codes >= lo) & (codes <= hi))
            np.testing.assert_allclose(shards[0]["Q"], codes / codes.sum(-1, keepdims=True), rtol=2e-7, atol=1e-8)
            q = torch.from_numpy(shards[0]["Q"])
            assert signature(q, None, None, "stream") == row["teacher_produced_sha256"]
            assert abs(row["bytes_up"] - size / 200) < 1e-8 and row["bytes_down"] == len(down)
            for s in shards:
                s.close()
        records[spec["arm"]] = {"rounds": len(rows), "max_mean_logscore_error": worst, "pass": True}
        print(spec["arm"], "WIRE PASS", records[spec["arm"]], flush=True)
    assert len(records) == (1 if local_run else 4)
    dest = local_run.parent / "local-wire-audit.json" if local_run else out / "wire-audit.json"
    atomic_json(dest, {"arms": records, "pass": True,
                                          "scope": "Actual frames, aggregate, quantizer, Q/hash/bytes; no private-weight replay"})


if __name__ == "__main__":
    main()
