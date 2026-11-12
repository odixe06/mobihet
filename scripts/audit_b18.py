"""Replay B18 stored packets/aggregation and audit all committed phase CMs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import fl, metrics
from methods.mobihet_next import signature
from scripts.b18_partition import read
from scripts.sim_fold_val import fold_counts


def decode(frame, with_counts=False):
    assert frame[:32] == bytes(32) and frame[34] in (0, 1)
    present = np.array([bool(int.from_bytes(frame[32:34], "little") & (1 << c)) for c in range(16)])
    raw = zlib.decompress(frame[35:]) if frame[34] else frame[35:]
    n = int(present.sum())
    assert len(raw) == n * (20 if with_counts else 16)
    codes = np.zeros((16, 16), np.uint8)
    codes[present] = np.frombuffer(raw[n * 4 if with_counts else 0:], np.uint8).reshape(n, 16)
    counts = np.zeros(16, np.int64)
    if with_counts:
        counts[present] = np.frombuffer(raw[:n * 4], "<i4")
    return codes, present, counts


def shards(path, r, world):
    paths = [path / f"rank{k}_r{r:03d}.npz" for k in range(world)]
    data = []
    for p in paths:
        with np.load(p) as z:
            data.append({k: z[k].copy() for k in z.files})
    return paths, data


def audit(root, rounds, world):
    root = Path(root)
    part, meta = read(root)
    cnt = fold_counts()
    ids = part["ids"]
    assert np.array_equal(part["early_counts"] + part["heldout_counts"], cnt[ids, 0])
    configs, histories, ten = {}, {}, {}
    for phase in ("shadow", "C", "E", "H"):
        p = root / phase
        cfg = json.loads((p / "config.json").read_text())
        assert cfg["method"] == "mobihet_source" and cfg["n_params"] == 395408
        assert cfg["backend"] == "compiled" and cfg["hp"]["rounds"] == 20
        assert cfg["hp"]["source_partition_sha"] == meta["logical_sha256"]
        assert cfg["hp"]["source_mode"] == ("S" if phase == "shadow" else phase)
        configs[phase] = cfg
        rows = [json.loads(s) for s in (p / "rounds.jsonl").read_text().splitlines()]
        assert [x["round"] for x in rows] == list(range(1, rounds + 1))
        histories[phase] = rows
        for row in rows:
            r = row["round"]
            assert row["clients_without_applied"] == 0 and row["skipped_frac"] <= .01
            assert np.isfinite(row["train_loss"])
            assert abs(row["lr"] - fl.cosine_lr(r, 20, .001, .00001)) < 1e-12
            assert row["teacher_consumed_round"] == (None if r == 1 else r - 1)
            if r > 1:
                assert row["teacher_consumed_sha256"] == rows[r - 2]["teacher_produced_sha256"]
            with np.load(p / f"cms_r{r:03d}.npz") as z:
                assert np.array_equal(z["ids"], ids)
                cm = z["private"]
            assert np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
            assert np.array_equal(cm.sum(2), part["early_counts"] if phase == "shadow" else cnt[ids, 1])
            s = metrics.summary(cm)
            assert all(abs(s[k] - row["private/" + k]) < 1e-9 for k in metrics.TEN)
        ten[phase] = {k: s[k] for k in metrics.TEN}
        log = (root / f"{phase}.log").read_text()
        assert all(f"compile gate rank{k}: PASS" in log for k in range(world))
    main_boot = configs["C"]["payload_bytes"]["bootstrap"]
    shadow_boot = configs["shadow"]["payload_bytes"]["bootstrap"]
    for phase in ("E", "H"):
        body = configs[phase]["payload_bytes"]
        assert body["bootstrap_main"] == main_boot and body["bootstrap_shadow"] == shadow_boot
        assert body["bootstrap"] == {k: main_boot[k] + shadow_boot[k] for k in ("up", "down")}
    for r in range(1, rounds + 1):
        paths, ss = shards(root / "shadow/source", r, world)
        srcids = np.concatenate([s["ids"] for s in ss])
        assert np.array_equal(np.sort(srcids), ids) and len(np.unique(srcids)) == len(ids)
        order = np.argsort(srcids)
        n = np.concatenate([s["counts"] for s in ss])[order]
        assert np.array_equal(n, part["probe_counts"])
        eligible = n >= 8
        for phase in ("E", "H"):
            costs = 0
            for shard in ss:
                codes, mean, cm = (shard[k] for k in (phase, "mean_" + phase, "cm_" + phase))
                assert np.isfinite(mean).all() and (mean >= 0).all() and (mean <= 1).all()
                assert np.array_equal(np.rint(mean * 255).astype(np.uint8), codes)
                assert np.array_equal(cm.sum(2), shard["counts"])
                blob, off = shard[phase + "_frames"].tobytes(), shard[phase + "_offsets"]
                assert off[0] == 0 and off[-1] == len(blob)
                for j in range(len(codes)):
                    q, present, c = decode(blob[off[j]:off[j + 1]], True)
                    assert np.array_equal(q, codes[j]) and np.array_equal(c, shard["counts"][j])
                    assert np.array_equal(present, c > 0)
                costs += len(blob)
            shadow_row = histories["shadow"][r - 1]
            assert abs(costs / len(ids) - shadow_row[f"source_probe_{phase}_up"]) < 1e-9
            assert shadow_row["source_probe_rows_each"] == int(n.sum())
            _, ms = shards(root / phase / "source", r, world)
            mids = np.concatenate([s["ids"] for s in ms])
            assert np.array_equal(np.sort(mids), ids) and len(np.unique(mids)) == len(ids)
            morder = np.argsort(mids)
            counts = np.concatenate([s["counts"] for s in ms])[morder]
            mixed = np.concatenate([s["mixed"] for s in ms])[morder]
            online = np.concatenate([s["online"] for s in ms])[morder]
            source = np.concatenate([s[phase] for s in ss])[order]
            assert np.array_equal(counts, cnt[ids, 0])
            assert np.array_equal(mixed, np.where(eligible[..., None], source, online))
            total = ((mixed.astype(np.float32) / np.float32(255)).astype(np.float16).astype(np.float64)
                     * counts[..., None]).sum(0)
            norm = counts.sum(0)
            up = 0
            by_id = {int(cid): i for i, cid in enumerate(ids)}
            for shard in ms:
                np.testing.assert_allclose(total, shard["global_sum"], rtol=0, atol=1e-9)
                assert np.array_equal(norm, shard["global_count"])
                off, blob = shard["offsets"], shard["frames"].tobytes()
                assert off[0] == 0 and off[-1] == len(blob)
                for j, cid in enumerate(shard["ids"]):
                    q, present, _ = decode(blob[off[j]:off[j + 1]])
                    pos = by_id[int(cid)]
                    want = (counts[pos] > 0) & ~eligible[pos]
                    assert np.array_equal(present, want) and np.array_equal(q[want], online[pos][want])
                up += len(blob)
                down = shard["down"].tobytes()
                qcode, present, _ = decode(down)
                assert present.all()
                p = (total / norm[:, None]).astype(np.float32).astype(np.float16).astype(np.float32)
                p /= p.sum(-1, keepdims=True)
                assert np.all((qcode >= np.rint(np.clip(p - 1e-6, 0, 1) * 255)) &
                              (qcode <= np.rint(np.clip(p + 1e-6, 0, 1) * 255)))
                np.testing.assert_allclose(shard["Q"], qcode / qcode.sum(-1, keepdims=True), rtol=2e-7, atol=1e-8)
                assert signature(torch.from_numpy(shard["Q"]), None, None, "stream") == histories[phase][r - 1]["teacher_produced_sha256"]
            row = histories[phase][r - 1]
            assert row["source_files_sha256"] == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            assert abs(row["source_main_up"] - up / len(ids)) < 1e-9
            assert row["source_main_down"] == len(down)
            assert row["source_eligible_rows"] == int((counts * eligible).sum())
            assert row["bytes_up"] == row["source_main_up"] + shadow_row["bytes_up"] + shadow_row[f"source_probe_{phase}_up"]
            assert row["bytes_down"] == row["source_main_down"] + shadow_row["bytes_down"]
    result = {"pass": True, "rounds": rounds, "world": world, "partition_sha256": meta["logical_sha256"],
              "endpoint_ten": ten, "bootstrap": {"main": main_boot, "shadow": shadow_boot},
              "scope": "Actual source/online/mixed/fallback packets, aggregation, teacher hash/lag, bootstrap and round bytes; all phase CMs/10metrics. No full private-gradient replay."}
    (root / "source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("B18_SOURCE_AUDIT_PASS", rounds, world, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    a = p.parse_args()
    audit(a.root, a.rounds, a.world)
