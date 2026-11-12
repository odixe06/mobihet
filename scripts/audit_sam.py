"""B15 SAM counters from the actual executed rows; source/CM audit is separate."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check(run, expected_rounds):
    config = json.loads((run / "config.json").read_text())
    assert config["method"] == "mobihet_sam" and config["n_params"] == 395408
    assert config["hp"]["batch"] == 64 and config["hp"]["synth_step_floor"] == 16
    rows = [json.loads(s) for s in (run / "rounds.jsonl").read_text().splitlines() if s.strip()]
    assert [r["round"] for r in rows] == list(range(1, expected_rounds + 1))
    with np.load(run / "cms_r001.npz") as z:
        ids = z["ids"]
    counts = np.load(ROOT / "cache/fold_counts.npy")[ids, 0].sum(1)
    steps = int(np.maximum((counts + 63) // 64, 16).sum())
    rho = config["hp"].get("sam_rho", 0)
    assert rho in (0, .05)
    for row in rows:
        assert row["clients_without_applied"] == 0 and row["skipped_frac"] <= .01
        if rho:
            assert row["sam_active_steps"] == steps and row["sam_real_rows"] == int(counts.sum())
            assert 0 <= row["sam_nonfinite_first"] <= steps * .01
            assert 0 <= row["sam_nonfinite_second"] <= steps * .01
            assert 0 <= row["sam_perturb_norm_error_bound"] < 1e-6
            assert np.isfinite([row["sam_loss_first"], row["sam_loss_second"]]).all()
        else:
            assert "sam_active_steps" not in row
    return {"pass": True, "rho": rho, "rounds": expected_rounds,
            "real_rows": int(counts.sum()), "expected_active_steps": steps if rho else None,
            "scope": "Actual counters + frozen method source, not full private-gradient replay"}


def main():
    out = ROOT / "runs/sim-kaggle-b15"
    if len(sys.argv) == 3 and sys.argv[1] == "--local":
        result = {"local": check(Path(sys.argv[2]), 2)}
        target = out / "local-sam-audit.json"
    else:
        assert json.loads((out / "audit.json").read_text())["pass"]
        batch = json.loads((out / "batch.json").read_text())
        result = {s["arm"]: check(next((ROOT / "artifacts/sim-kaggle-b15" / s["slug"]).glob("**/config.json")).parent, 20)
                  for s in batch}
        assert len(result) == 4
        target = out / "sam-audit.json"
    target.write_text(json.dumps({"pass": True, "arms": result}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
