"""CE recovery only: complete clean-val CM/source/fold/telemetry and finite guard audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from methods.mobihet_vrecover import admissible
from scripts.audit_r20 import check_executed_source
from scripts.audit_v20 import check_fold_receipts
from scripts.audit_screening import audit
from scripts.audit_m20 import check_summary, PROBE_ROUNDS


def check_telemetry(record, rank, mine, counts):
    assert record["rank"] == rank and record["accepted_ranks"] == record["world"] == 2
    assert record["guard_policy"] == "common_global_skip_le_0.01_finite_positive_applied_per_client"
    clients = record["clients"]
    assert sorted(c["client_id"] for c in clients) == mine.tolist()
    grouped = [{"client_id": cid, "applied": applied, "skipped": skipped} for g in record["groups"]
        for cid, applied, skipped in zip(g["client_ids"], g["applied"], g["skipped"])]
    assert grouped == clients
    assert all(len(g["client_ids"]) == len(g["applied"]) == len(g["skipped"]) for g in record["groups"])
    assert record["all_finite"] == all(g[k] for g in record["groups"] for k in ("weights_finite", "buffers_finite", "loss_finite"))
    for client in clients:
        assert client["applied"] > 0 and client["skipped"] >= 0
        assert client["applied"] + client["skipped"] == int(np.ceil(counts[client["client_id"]].sum() / 64))
    assert record["applied"] == sum(c["applied"] for c in clients)
    assert record["skipped"] == sum(c["skipped"] for c in clients)
    assert record["zero_applied_client_ids"] == [] and admissible(record, record["global_info"])
    return record["applied"], record["skipped"]


def main():
    ap = argparse.ArgumentParser()
    for key in ("batch", "root", "out"): ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    counts, support = (np.load(ROOT / f"runs/v-study/{f}_counts.npy") for f in ("train", "val"))
    split = json.loads((ROOT / "runs/v-study/split-audit.json").read_text()); common, details = [], []
    for spec in json.loads(a.batch.read_text()):
        common.append(audit(a.root, spec, support, endpoint=20))
        folder = a.root / spec["slug"]; d = folder / spec["slug"]; check_executed_source(folder, spec)
        cfg = json.loads((d / "config.json").read_text()); hp = cfg["hp"]
        assert cfg["method"] == "mobihet_vrecover" and cfg["backend"] == "compiled"
        assert hp["study_mode"] == "real" and hp["synth"] == "off" and hp["no_kd"]
        assert hp["validation_diagnostic"] is True and hp["external_diagnostics"] is False
        assert sum(cfg["payload_bytes"]["bootstrap"].values()) == 128
        ranks = check_fold_receipts(d, counts, support, split["scaler_sha256"])
        rows = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
        telemetry = []
        for row in rows:
            r = row["round"]; applied = skipped = 0
            assert row["bytes_up"] == row["bytes_down"] == 0
            assert row["teacher_consumed_round"] is None and row["teacher_produced_round"] is None
            for rank, mine in enumerate(ranks):
                path = d / f"numerics_rank{rank}_r{r:03}.json"; record = json.loads(path.read_text())
                assert record["round"] == r and record["loss_scale"] == 4096
                ap_count, sk_count = check_telemetry(record, rank, mine, counts)
                applied += ap_count; skipped += sk_count
                assert record["global_info"]["applied_steps"] == row["applied_steps"]
                assert record["global_info"]["skipped_steps"] == row["skipped_steps"]
                assert record["global_info"]["skipped_frac"] == row["skipped_frac"]
            assert applied == row["applied_steps"] and skipped == row["skipped_steps"]
            assert np.isclose(row["skipped_frac"], skipped / (applied + skipped), rtol=1e-6, atol=1e-12)
            path = d / "models" / f"r{r:03}.pt"; state = torch.load(path, weights_only=True, map_location="cpu")
            assert "priv_p" not in state and all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
            assert all(state.get(k) is None for k in ("Q", "bank_X", "bank_target", "teacher_p", "teacher_b"))
            telemetry.append({"round": r, "applied": applied, "skipped": skipped,
                "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for r in PROBE_ROUNDS:
            with np.load(d / f"train_cms_r{r:03}.npz") as z:
                assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), counts)
                check_summary(z["private"], rows[r - 1], "train_private")
            ids = []
            for rank in (0, 1):
                records = [json.loads(line) for line in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
                assert [v["round"] for v in records] == list(PROBE_ROUNDS)
                record = next(v for v in records if v["round"] == r)
                assert record["probe_teacher_round"] is None and record["probe_mode"] == "shadow_next_teacher_fp32_dropout0"
                ids.extend(c["client_id"] for c in record["clients"])
                assert json.loads((d / f"teacher_probe_rank{rank}_r{r:03}.json").read_text()) == []
            assert sorted(ids) == sorted(hp["diagnostic_ids"])
        assert not list(d.glob("external_cms*.npz"))
        for rank, mine in enumerate(ranks):
            events = [json.loads(line) for line in (d / "recovery" / f"rank{rank}.jsonl").read_text().splitlines()]
            assert events[0] == {"rank": rank, "event": "compile_guard", "compiled_ranks": 2, "world": 2}
            assert len(events) == 21 and [v["round"] for v in events[1:]] == list(range(1, 21))
            assert all(e["event"] == "round_finite" and e["clients"] == len(mine) and e["groups"] == 0 for e in events[1:])
        log = (folder / (spec["slug"] + ".log")).read_text()
        assert "Tesla T4" in log and all(f"compile gate rank{r}: PASS" in log for r in (0, 1))
        assert "Traceback (most recent call last)" not in log
        details.append({"arm": spec["arm"], "rounds": telemetry,
            "scope": "CE guard-only fresh recovery, originalfoldtrain/val; source/CM/all-client applied+skip/finite evidence; no privateprediction replay or mixedarchitecture proof"})
    for name, value in (("audit.json", common), ("diagnostic-audit.json", details)):
        (a.out / name).write_text(json.dumps(value, indent=2) + "\n")
    print("V20_CE_RECOVERY_ARTIFACT_AUDIT_PASS", len(common), flush=True)


if __name__ == "__main__": main()
