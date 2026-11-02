"""Q20 executed source, fixed20round CM, class/input aggregates, teacher chain and guards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from methods.mobihet_next import signature
from methods.mobihet_study import decode_probability
from scripts.audit_screening import audit
from scripts.audit_m20 import check_summary, PROBE_ROUNDS


def check_decoded_probability(probability, body):
    """Bound fp32 sum/division of stored fp16 rows, independent of CPU/CUDA reduction order.

    fp16 -> fp32 is exact. A positive sum of16 terms has relative error <=gamma16;
    one fp32 division contributes <=u. Intervals use the exact stored body, not a
    uniform absolute tolerance. This certifies rounding consistency, not CUDA replay.
    """
    assert body.dtype == torch.float16 and probability.dtype == torch.float32
    assert body.shape == probability.shape and body.ndim == 2 and body.shape[1] == 16
    assert torch.isfinite(body).all() and (body >= 0).all() and torch.isfinite(probability).all()
    exact = body.double()
    total = exact.sum(-1, keepdim=True)
    assert (total > 1e-12).all(), "Probability row requires positive mass above clamp"
    u = 2. ** -24
    gamma16 = 16 * u / (1 - 16 * u)
    lower = exact / (total * (1 + gamma16)) * (1 - u)
    upper = exact / (total * (1 - gamma16)) * (1 + u)
    assert ((probability.double() >= lower) & (probability.double() <= upper)).all(), "Decoded probability outside fp32 normalization interval"
    return {"max_interval_width": float((upper - lower).max()),
            "max_cpu_difference": float((probability - decode_probability(body)).abs().max()),
            "scope": "fp16 body/fp32 normalization consistency; not exact CUDA replay"}


def extra(root, spec, counts):
    folder = root / spec["slug"]; d = folder / spec["slug"]
    manifest = json.loads((folder / "provenance/source_manifest.json").read_text())
    assert manifest["overlay"] == spec["overlay_sha256"]
    cfg = json.loads((d / "config.json").read_text()); hp = cfg["hp"]; mode = hp["next_mode"]
    assert cfg["method"] == "mobihet_next" and cfg["backend"] == "compiled" and mode == spec["mode"]
    assert hp["study_mode"] == "class" and hp["q_agg"] == "count" and hp["teacher_rows"] == 1024
    assert hp["synth_rows"] == 64 and hp["kd_weight"] == "fixed" and hp["q_collect"] == "online"
    assert hp["real_kd"] == (mode in {"real", "margin", "strength"})
    assert sum(cfg["payload_bytes"]["bootstrap"].values()) == 3311744
    rows = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
    expected_bytes = 544 if mode == "stream" else 33344
    previous, snapshots, bank_hashes, normalization_checks = None, [], [], []
    totals = torch.as_tensor(counts.sum(0), dtype=torch.float64)
    ys = torch.arange(1024) % 16
    for row in rows:
        r = row["round"]; path = d / "models" / f"r{r:03}.pt"
        s = torch.load(path, weights_only=True, map_location="cpu")
        assert "priv_p" not in s and all(torch.isfinite(t).all() for t in s.values() if torch.is_tensor(t))
        assert row["bytes_up"] == row["bytes_down"] == expected_bytes
        assert row["teacher_consumed_round"] == (None if r == 1 else r - 1)
        assert row["teacher_produced_round"] == r and row["teacher_consumed_sha256"] == previous
        previous = signature(s["Q"], s.get("bank_X"), s.get("bank_target"), mode)
        assert row["teacher_produced_sha256"] == previous
        assert s["Q"].shape == (16, 16) and bool((s["Q"] >= 0).all())
        torch.testing.assert_close(s["Q"].sum(1), torch.ones(16), atol=1e-5, rtol=0)
        if mode not in {"stream", "query"}:
            assert torch.equal(s["class_totals"], totals)
            body = (s["class_sums"] / totals[:, None]).float().half()
            normalization_checks.append({"round": r, "kind": "class", **check_decoded_probability(s["Q"], body)})
        if mode == "stream": assert s.get("bank_X") is None and s.get("bank_target") is None
        else:
            assert s["bank_X"].shape == (1024, 77) and torch.equal(s["bank_y"], ys)
            bank_hashes.append(hashlib.sha256(s["bank_X"].numpy().tobytes()).hexdigest())
            assert torch.equal(s["bank_body"], (s["aggregate_sums"] / s["aggregate_weights"][:, None]).float().half())
            normalization_checks.append({"round": r, "kind": "input", **check_decoded_probability(s["bank_target"], s["bank_body"])})
            if mode == "input_sqrt": expected_weight = torch.as_tensor(counts, dtype=torch.float64).sqrt().sum(0)[ys]
            else: expected_weight = totals[ys]
            torch.testing.assert_close(s["aggregate_weights"], expected_weight, atol=1e-7, rtol=1e-12)
            assert row["input_up_bytes"] == row["input_down_bytes"] == 32800
        assert s.get("teacher_p") is None and s.get("teacher_b") is None
        snapshots.append({"round": r, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(snapshots) == 20
    if bank_hashes: assert len(set(bank_hashes)) == 20
    for r in PROBE_ROUNDS:
        with np.load(d / f"train_cms_r{r:03}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), counts)
            check_summary(z["private"], rows[r - 1], "train_private")
        ids = []
        for rank in (0, 1):
            records = [json.loads(line) for line in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
            assert [v["round"] for v in records] == list(PROBE_ROUNDS)
            rec = next(v for v in records if v["round"] == r)
            assert rec["probe_teacher_round"] == r and rec["probe_mode"] == "shadow_next_teacher_fp32_dropout0"
            ids.extend(c["client_id"] for c in rec["clients"])
            assert json.loads((d / f"teacher_probe_rank{rank}_r{r:03}.json").read_text()) == []
        assert sorted(ids) == sorted(hp["diagnostic_ids"])
    for r in (1, 10, 20):
        ids = []
        for rank in (0, 1):
            with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                assert z["private"].shape == (len(z["ids"]), 16, 16) and (z["private"] >= 0).all()
                ids.extend(z["ids"].tolist())
        assert sorted(ids) == sorted(hp["diagnostic_ids"])
    for rank in (0, 1):
        records = [json.loads(line) for line in (d / "recovery" / f"rank{rank}.jsonl").read_text().splitlines()]
        assert records[0] == {"rank": rank, "event": "compile_guard", "compiled_ranks": 2, "world": 2}
        assert len(records) == 21 and [v["round"] for v in records[1:]] == list(range(1, 21))
        assert all(v["event"] == "round_finite" and v["clients"] == (2672 if rank == 0 else 2671)
                   and v["groups"] == (0 if mode == "stream" else 42) for v in records[1:])
    text = (folder / (spec["slug"] + ".log")).read_text()
    assert "Tesla T4" in text and all(f"compile gate rank{rank}: PASS" in text for rank in (0, 1))
    assert "Traceback (most recent call last)" not in text
    return {"arm": spec["arm"], "mode": mode, "snapshots": snapshots, "bank_hashes": bank_hashes,
            "normalization_checks": normalization_checks,
            "scope": "CM/source/aggregate/teacher/guard bounded audit, not privateprediction replay/fullresume/DP/contact proof"}


def main():
    p = argparse.ArgumentParser()
    for key in ("batch", "root", "out"): p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    common, detail = [], []
    for spec in json.loads(a.batch.read_text()):
        common.append(audit(a.root, spec, support, endpoint=20)); detail.append(extra(a.root, spec, counts))
    for name, data in [("audit.json", common), ("diagnostic-audit.json", detail)]:
        (a.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print("Q20_ARTIFACT_AUDIT_PASS", len(common), flush=True)


if __name__ == "__main__": main()
