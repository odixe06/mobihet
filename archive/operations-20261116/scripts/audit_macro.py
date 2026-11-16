"""B17 actual source-coordinate and row/step counters; local also checks state/CM."""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(run, rounds, local=False):
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["method"] == "mobihet_macro" and cfg["n_params"] == 395408
    hp = cfg["hp"]
    assert hp["batch"] == 64 and hp["synth_step_floor"] == 16 and hp["q_bits"] == 8
    mode = hp["macro_mode"]
    assert mode in {"cbb", "prior", "real_nckd"}
    rows = [json.loads(s) for s in (run / "rounds.jsonl").read_text().splitlines() if s.strip()]
    assert [r["round"] for r in rows] == list(range(1, rounds + 1))
    counts = np.load(ROOT / "cache/fold_counts.npy")
    with np.load(run / "cms_r001.npz") as z:
        ids = z["ids"]
    n, total = counts[ids, 0].sum(1), counts[ids, 0].sum(0).astype(np.float64)
    expected_shift = (-math.log(16) - np.log((total + 1) / (total.sum() + 16))).astype(np.float32)
    steps = int(np.maximum((n + 63) // 64, 16).sum())
    for row in rows:
        r = row["round"]
        assert row["clients_without_applied"] == 0 and row["skipped_frac"] <= .01
        assert row["teacher_consumed_round"] == (None if r == 1 else r - 1)
        if mode != "cbb":
            assert row["macro_active_steps"] == steps and row["macro_real_rows"] == int(n.sum())
            assert row["macro_real_kd_rows"] == (int(n.sum()) if mode == "real_nckd" and r > 1 else 0)
            keep = np.asarray(row["macro_real_kd_keep"])
            assert keep.shape == (16,) and keep.dtype == np.bool_
            expected_kept = int(total[keep].sum()) if mode == "real_nckd" and r > 1 else 0
            assert row["macro_real_kd_kept_rows"] == expected_kept
            if mode != "real_nckd" or r == 1:
                assert not keep.any()
            assert np.isfinite(row["macro_real_kd_loss"]) and row["macro_real_kd_loss"] >= -1e-6
            shift = np.asarray(row["macro_prior_shift"], dtype=np.float32)
            assert shift.shape == (16,) and np.isfinite(shift).all()
            if mode == "prior":
                assert np.max(abs(shift - expected_shift)) <= 2e-7
                assert row["macro_real_kd_loss"] == 0
            else:
                assert np.count_nonzero(shift) == 0
        else:
            assert "macro_active_steps" not in row
        if local:
            from core import metrics
            with np.load(run / f"cms_r{r:03d}.npz") as z:
                assert np.array_equal(z["ids"], ids)
                assert np.array_equal(z["private"].sum(2), counts[ids, 1])
                ten = metrics.summary(z["private"])
                assert all(abs(ten[k] - row["private/" + k]) < 1e-9 for k in metrics.TEN)
    if local:
        import torch
        state = torch.load(run / "state.pt", map_location="cpu", weights_only=True)
        assert state["round"] == rounds and state["method"] == "mobihet_macro"
        tensors = [v for v in state["method_state"].values() if torch.is_tensor(v)]
        assert tensors and all(bool(torch.isfinite(t).all()) for t in tensors)
    return {"pass": True, "mode": mode, "rounds": rounds, "active_steps": steps, "real_rows": int(n.sum()),
            "scope": "Actual counters/coordinate and executed source; not private-gradient or prediction replay"}


def main():
    out = ROOT / "runs/sim-kaggle-b17"
    if len(sys.argv) == 3 and sys.argv[1] == "--local":
        run = Path(sys.argv[2])
        result = {"local": check(run, 2, local=True)}
        target = out / (run.name + "-audit.json")
    else:
        assert json.loads((out / "audit.json").read_text())["pass"]
        batch = json.loads((out / "batch.json").read_text())
        result = {s["arm"]: check(next((ROOT / "artifacts/sim-kaggle-b17" / s["slug"]).glob("**/config.json")).parent, 20)
                  for s in batch}
        assert len(result) == 6
        target = out / "macro-audit.json"
    target.write_text(json.dumps({"pass": True, "arms": result}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
