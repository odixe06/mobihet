"""Audit P20 executed source,20round own/train/external CM, teacher chain, packets and proxy replica."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.mobihet_study import BANK_MODES, PROXY_MODES, decode_probability, teacher_signature
from scripts.audit_screening import audit
from scripts.audit_m20 import check_summary, PROBE_ROUNDS


def check_class_mean(q, body, labels):
    """Bound fp32 reduction/division before fp16 rounding; CPU and CUDA can select adjacent bins.

    Positive sums of16/64 terms have relative error at most gamma_n=n*u/(1-n*u).
    Propagate intervals through both normalizations and the actual fp16 quantizer.
    This checks consistency, not bit-exact reproduction of CUDA atomic reduction order.
    """
    assert body.dtype == torch.float16 and body.shape == (1024, 16)
    assert torch.isfinite(body).all() and (body >= 0).all()
    assert torch.equal(torch.bincount(labels, minlength=16), torch.full((16,), 64))
    assert q.shape == (16, 16) and torch.isfinite(q).all() and (q >= 0).all()
    u = 2. ** -24
    gamma16, gamma64 = 16 * u / (1 - 16 * u), 64 * u / (1 - 64 * u)
    exact = body.double()
    denominator = exact.sum(1, keepdim=True)
    assert (denominator > 0).all()
    lo = exact / denominator * (1 - u) / (1 + gamma16)
    hi = exact / denominator * (1 + u) / (1 - gamma16)
    lo = (torch.zeros(16, 16, dtype=torch.float64).index_add_(0, labels, lo) / 64 * (1 - gamma64)).half().double()
    hi = (torch.zeros(16, 16, dtype=torch.float64).index_add_(0, labels, hi) / 64 * (1 + gamma64)).half().double()
    lower = lo / (hi.sum(1, keepdim=True) * (1 + gamma16)) * (1 - u)
    upper = hi / (lo.sum(1, keepdim=True) * (1 - gamma16)) * (1 + u)
    assert ((q.double() >= lower) & (q.double() <= upper)).all(), "Class Q outside reduction/quantization interval"
    return {"ambiguous_fp16_bins": int((lo != hi).sum()), "max_interval_width": float((upper - lower).max()),
            "scope": "fp32 reduction/fp16 rounding consistency; not exact CUDA replay"}


def extra(root, spec, counts):
    folder = root / spec["slug"]; d = folder / spec["slug"]
    manifest = json.loads((folder / "provenance/source_manifest.json").read_text())
    assert manifest["overlay"] == spec["overlay_sha256"]
    config = json.loads((d / "config.json").read_text()); hp = config["hp"]; mode = hp["study_mode"]
    assert config["backend"] == "compiled" and hp["kd_weight"] == "fixed" and hp["teacher_rows"] == 1024
    assert hp["proxy_steps"] == 200 and mode == spec["mode"]
    rows = list(map(json.loads, (d / "rounds.jsonl").read_text().splitlines()))
    expected_boot = 0 if mode == "real_no_la" else 128 if mode == "real" else 3311744
    assert sum(config["payload_bytes"]["bootstrap"].values()) == expected_boot
    no = hp["no_kd"]
    up = 0 if no else 32800 if mode in BANK_MODES else 544
    down = 0 if no else (32 + 2 * (config["n_params"] + config["n_buffers"]) if mode in {"proxy_stream", "proxy_weights_bank"}
                       else 544 if mode not in BANK_MODES or mode == "bank_class" else 32800)
    previous = None; snapshots = []; bank_hashes = []; rounding_checks = []
    assert sorted(p.name for p in (d / "models").glob("*.pt")) == [f"r{r:03}.pt" for r in range(1, 21)]
    for row in rows:
        r = row["round"]; path = d / "models" / f"r{r:03}.pt"
        s = torch.load(path, weights_only=True, map_location="cpu")
        assert "priv_p" not in s and all(torch.isfinite(t).all() for t in s.values() if torch.is_tensor(t))
        assert row["bytes_up"] == up and row["bytes_down"] == down
        assert row["teacher_consumed_round"] == (None if no or r == 1 else r - 1)
        assert row["teacher_produced_round"] == (None if no else r)
        assert row["teacher_consumed_sha256"] == previous
        signature = None if no else teacher_signature(mode, s.get("Q"), s.get("bank_X"), s.get("bank_target"), s.get("teacher_p"), s.get("teacher_b"))
        assert row["teacher_produced_sha256"] == signature
        previous = signature
        if not no:
            assert s["Q"].shape == (16, 16)
            assert bool((s["Q"] >= 0).all())
            torch.testing.assert_close(s["Q"].sum(1), torch.ones(16), atol=1e-5, rtol=0)
        if mode in BANK_MODES:
            assert s["bank_X"].shape == (1024, 77) and torch.equal(s["bank_y"], torch.arange(1024) % 16)
            bank_hashes.append(hashlib.sha256(s["bank_X"].numpy().tobytes()).hexdigest())
            if not no:
                expected = (s["aggregate_sums"] / s["aggregate_weights"][:, None]).float().half()
                assert torch.equal(s["bank_body"], expected)
                aggregate = decode_probability(s["bank_body"])
                assert bool((s["bank_target"] >= 0).all())
                torch.testing.assert_close(s["bank_target"].sum(1), torch.ones(1024), atol=1e-5, rtol=0)
                rounding_checks.append({"round": r, **check_class_mean(s["Q"], s["bank_body"], s["bank_y"])})
                if mode == "bank_input":
                    torch.testing.assert_close(s["bank_target"], aggregate, atol=2e-7, rtol=0)
                elif mode == "bank_class":
                    assert torch.equal(s["bank_target"], s["Q"][s["bank_y"]])
                if mode in PROXY_MODES:
                    assert s["teacher_p"].shape == (config["n_params"],) and s["teacher_b"].shape == (config["n_buffers"],)
                    assert torch.equal(s["teacher_p"], s["server_p"].half().float())
                    assert torch.equal(s["teacher_b"], s["server_b"].half().float())
                    assert 0 <= row["proxy_fit_kl"] < 100 and 0 < row["server_seconds"] < row["t_train"]
        snapshots.append({"round": r, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if bank_hashes: assert len(set(bank_hashes)) == 20, "bank was not refreshed"
    for r in PROBE_ROUNDS:
        with np.load(d / f"train_cms_r{r:03}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), counts)
            check_summary(z["private"], rows[r - 1], "train_private")
        clients, proxy_ids = [], []
        for rank in (0, 1):
            records = list(map(json.loads, (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()))
            assert [v["round"] for v in records] == list(PROBE_ROUNDS)
            record = next(v for v in records if v["round"] == r)
            assert record["probe_mode"] == "shadow_next_teacher_fp32_dropout0"
            assert record["probe_teacher_round"] == (None if no else r)
            clients.extend(c["client_id"] for c in record["clients"])
            teacher_probes = json.loads((d / f"teacher_probe_rank{rank}_r{r:03}.json").read_text())
            proxy_ids.extend(v["client_id"] for v in teacher_probes)
            for rec in teacher_probes:
                for fold in ("train", "test"):
                    cm = np.asarray(rec[fold]["teacher_cm"]); student = np.asarray(rec[fold]["student_cm"])
                    assert cm.shape == student.shape == (16, 16) and (cm >= 0).all() and (student >= 0).all()
                    assert np.array_equal(cm.sum(1), student.sum(1)) and 0 < cm.sum() <= 512
        assert sorted(clients) == sorted(hp["diagnostic_ids"])
        assert sorted(proxy_ids) == (sorted(hp["diagnostic_ids"]) if mode in PROXY_MODES else [])
    external = []
    for r in (1, 10, 20):
        ids = []
        for rank in (0, 1):
            with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                assert z["private"].shape == (len(z["ids"]), 16, 16) and (z["private"] >= 0).all()
                ids.extend(z["ids"].tolist())
                external.extend({"round": r, "client_id": int(cid), "selection_sha256": str(h)} for cid, h in zip(z["ids"], z["selection_sha256"]))
        assert sorted(ids) == sorted(hp["diagnostic_ids"])
    text = (folder / (spec["slug"] + ".log")).read_text()
    assert "Tesla T4" in text and all(f"compile gate rank{rank}: PASS" in text for rank in (0, 1))
    assert "Traceback (most recent call last)" not in text
    return {"arm": spec["arm"], "mode": mode, "snapshots": snapshots, "bank_hashes": bank_hashes, "external": external,
            "class_mean_rounding_checks": rounding_checks,
            "bytes_up": up, "bytes_down": down, "bootstrap": expected_boot,
            "scope": "Executed source,CM,teacherlag/replica/aggregate audit; no rawprediction/fullprivatebank/DP/contact proof"}


def main():
    ap = argparse.ArgumentParser()
    for name in ("batch", "root", "out"): ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    common, details = [], []
    for spec in json.loads(args.batch.read_text()):
        common.append(audit(args.root, spec, support, endpoint=20))
        details.append(extra(args.root, spec, counts))
    (args.out / "audit.json").write_text(json.dumps(common, indent=2) + "\n")
    (args.out / "diagnostic-audit.json").write_text(json.dumps(details, indent=2) + "\n")
    print("P20_ARTIFACT_AUDIT_PASS", len(common), flush=True)


if __name__ == "__main__": main()
