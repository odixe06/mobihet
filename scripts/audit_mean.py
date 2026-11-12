"""B16 actual loss/coverage counters; optional local W1 state/CM checks."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(run, rounds, local=False):
    config = json.loads((run / "config.json").read_text())
    assert config["method"] == "mobihet_mean" and config["n_params"] == 395408
    hp = config["hp"]
    assert hp["batch"] == 64 and hp["synth_step_floor"] == 16 and hp["q_bits"] == 8
    mode = hp["mean_kd"]
    assert mode in {"row", "mean", "half"}
    rows = [json.loads(s) for s in (run / "rounds.jsonl").read_text().splitlines() if s.strip()]
    assert [r["round"] for r in rows] == list(range(1, rounds + 1))
    counts = np.load(ROOT / "cache/fold_counts.npy")
    with np.load(run / "cms_r001.npz") as z:
        ids = z["ids"]
    n = counts[ids, 0].sum(1)
    steps = int(np.maximum((n + 63) // 64, 16).sum())
    for row in rows:
        r = row["round"]
        assert row["clients_without_applied"] == 0 and row["skipped_frac"] <= .01
        assert row["teacher_consumed_round"] == (None if r == 1 else r - 1)
        if mode != "row":
            assert row["mean_kd_active_steps"] == steps and row["mean_kd_real_rows"] == int(n.sum())
            assert row["mean_kd_steps"] == (0 if r == 1 else steps)
            assert row["mean_kd_violations"] == 0
            a, b, gap = [row["mean_kd_" + k] for k in ("row_loss", "mean_loss", "jensen_gap")]
            assert np.isfinite([a, b, gap]).all() and gap >= -1e-5
            assert abs(a - b - gap) < 1e-5
        else:
            assert "mean_kd_steps" not in row
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
        assert state["round"] == rounds and state["method"] == "mobihet_mean"
        tensors = [v for v in state["method_state"].values() if torch.is_tensor(v)]
        assert tensors and all(bool(torch.isfinite(t).all()) for t in tensors)
    return dict(pass_=True, mode=mode, rounds=rounds, active_steps=steps, real_rows=int(n.sum()),
                scope="Actual counters/source; not full private-gradient replay")


def main():
    out = ROOT / "runs/sim-kaggle-b16"
    if len(sys.argv) == 3 and sys.argv[1] == "--local":
        run = Path(sys.argv[2])
        result = {"local": check(run, 2, local=True)}
        target = out / (run.name + "-audit.json")
    else:
        assert json.loads((out / "audit.json").read_text())["pass"]
        batch = json.loads((out / "batch.json").read_text())
        result = {s["arm"]: check(next((ROOT / "artifacts/sim-kaggle-b16" / s["slug"]).glob("**/config.json")).parent, 20)
                  for s in batch}
        assert len(result) == 6
        target = out / "mean-audit.json"
    target.write_text(json.dumps({"pass": True, "arms": result}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
