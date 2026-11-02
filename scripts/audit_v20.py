"""V20: clean-fold receipts, own-val/train confusion, exact source and live teacher chain."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core import local
from methods.mobihet_next import signature
from scripts.audit_r20 import check_executed_source
from scripts.audit_q20 import check_decoded_probability
from scripts.audit_screening import audit
from scripts.audit_m20 import check_summary, PROBE_ROUNDS


def check_fold_receipts(d, counts, support, scaler_sha):
    order = np.argsort(-counts.sum(1), kind="stable")
    ranks = []
    for rank in (0, 1):
        mine = np.sort(order[rank::2])
        receipt = json.loads((d / f"fold_rank{rank}.json").read_text())
        assert receipt["training_fold"] == "train" and receipt["evaluation_fold"] == "val" and receipt["officialtest_read"] is False
        assert receipt["local_train_rows"] == int(counts[mine].sum()) and receipt["local_val_rows"] == int(support[mine].sum())
        assert receipt["scaler_fit_on"] == "fold == train" and receipt["scaler_sha256"] == scaler_sha
        ranks.append(mine)
    return ranks


def extra(root, spec, counts, support, split):
    folder = root / spec["slug"]; d = folder / spec["slug"]
    check_executed_source(folder, spec)
    cfg = json.loads((d / "config.json").read_text()); hp = cfg["hp"]; mode = spec["mode"]
    assert cfg["method"] == "mobihet_vstudy" and cfg["backend"] == "compiled"
    assert hp["validation_diagnostic"] is True and hp["external_diagnostics"] is False
    mixed = mode in {"mixed_input", "margin"}
    assert hp["study_mode"] == ("class" if mixed else mode)
    assert not mixed or hp["next_mode"] == mode
    assert hp["real_kd"] == (mode == "margin") and hp["no_kd"] == (not mixed)
    ranks = check_fold_receipts(d, counts, support, split["scaler_sha256"])
    expected_bytes = 33344 if mixed else 0
    expected_boot = 128 if mode == "real" else 3311744
    assert sum(cfg["payload_bytes"]["bootstrap"].values()) == expected_boot
    rows = [json.loads(line) for line in (d / "rounds.jsonl").read_text().splitlines()]
    totals = torch.as_tensor(counts.sum(0), dtype=torch.float64); ys = torch.arange(1024) % 16
    snapshots, banks, normalizations, previous = [], [], [], None
    for row in rows:
        r = row["round"]; path = d / "models" / f"r{r:03}.pt"; state = torch.load(path, weights_only=True, map_location="cpu")
        assert "priv_p" not in state and all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
        assert row["skipped_frac"] == 0 and row["bytes_up"] == row["bytes_down"] == expected_bytes
        assert state.get("teacher_p") is None and state.get("teacher_b") is None
        if mixed:
            assert row["teacher_consumed_round"] == (None if r == 1 else r - 1) and row["teacher_produced_round"] == r
            assert row["teacher_consumed_sha256"] == previous
            previous = signature(state["Q"], state["bank_X"], state["bank_target"], mode)
            assert row["teacher_produced_sha256"] == previous and torch.equal(state["class_totals"], totals)
            body = (state["class_sums"] / totals[:, None]).float().half()
            normalizations.append({"round": r, "kind": "class", **check_decoded_probability(state["Q"], body)})
            assert torch.equal(state["bank_body"], (state["aggregate_sums"] / state["aggregate_weights"][:, None]).float().half())
            normalizations.append({"round": r, "kind": "input", **check_decoded_probability(state["bank_target"], state["bank_body"])})
            torch.testing.assert_close(state["aggregate_weights"], totals[ys], rtol=0, atol=0)
            assert state["bank_X"].shape == (1024, 77) and torch.equal(state["bank_y"], ys)
            banks.append(hashlib.sha256(state["bank_X"].numpy().tobytes()).hexdigest())
            assert row["input_up_bytes"] == row["input_down_bytes"] == 32800
        else:
            assert state.get("Q") is None and state.get("bank_X") is None and state.get("bank_target") is None
            assert row["teacher_consumed_round"] is None and row["teacher_produced_round"] is None
        snapshots.append({"round": r, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(snapshots) == 20
    if mixed: assert len(set(banks)) == 20
    for r in PROBE_ROUNDS:
        with np.load(d / f"train_cms_r{r:03}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), counts)
            check_summary(z["private"], rows[r - 1], "train_private")
        ids = []
        for rank in (0, 1):
            records = [json.loads(line) for line in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
            assert [v["round"] for v in records] == list(PROBE_ROUNDS)
            rec = next(v for v in records if v["round"] == r)
            assert rec["probe_teacher_round"] == (r if mixed else None) and rec["probe_mode"] == "shadow_next_teacher_fp32_dropout0"
            ids.extend(c["client_id"] for c in rec["clients"])
            assert json.loads((d / f"teacher_probe_rank{rank}_r{r:03}.json").read_text()) == []
        assert sorted(ids) == sorted(hp["diagnostic_ids"])
    assert not list(d.glob("external_cms*.npz")), "No external officialtest diagnostic in V"
    for rank, mine in enumerate(ranks):
        events = [json.loads(line) for line in (d / "recovery" / f"rank{rank}.jsonl").read_text().splitlines()]
        assert events[0] == {"rank": rank, "event": "compile_guard", "compiled_ranks": 2, "world": 2}
        assert [v["round"] for v in events[1:]] == list(range(1, 21))
        groups = len(local.group_schedule(counts[mine].sum(1), list(range(len(mine))), 64, 4096)) if mixed else 0
        assert all(e["event"] == "round_finite" and e["clients"] == len(mine) and e["groups"] == groups for e in events[1:])
    text = (folder / (spec["slug"] + ".log")).read_text()
    assert "Tesla T4" in text and all(f"compile gate rank{rank}: PASS" in text for rank in (0, 1))
    assert "Traceback (most recent call last)" not in text
    return {"arm": spec["arm"], "mode": mode, "snapshots": snapshots, "bank_hashes": banks,
        "normalization_checks": normalizations, "scope": "Separate originalfoldtrain/val; bounded CM/source/aggregate/receipt audit, no fullremote input/prediction/privatebank replay"}


def main():
    p = argparse.ArgumentParser()
    for key in ("batch", "root", "out"): p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    counts, support = (np.load(ROOT / f"runs/v-study/{f}_counts.npy") for f in ("train", "val"))
    split = json.loads((ROOT / "runs/v-study/split-audit.json").read_text()); common, detail = [], []
    for spec in json.loads(a.batch.read_text()):
        common.append(audit(a.root, spec, support, endpoint=20)); detail.append(extra(a.root, spec, counts, support, split))
    for name, data in (("audit.json", common), ("diagnostic-audit.json", detail)):
        (a.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print("V20_ARTIFACT_AUDIT_PASS", len(common), flush=True)


if __name__ == "__main__": main()
