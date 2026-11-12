"""B16 paired first-round CM and within-forward Jensen diagnostics. No tuning."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "runs/sim-kaggle-b16"
    assert json.loads((out / "audit.json").read_text())["pass"]
    assert json.loads((out / "mean-audit.json").read_text())["pass"]
    batch = json.loads((out / "batch.json").read_text())
    paths = {s["arm"]: next((ROOT / "artifacts/sim-kaggle-b16" / s["slug"]).glob("**/config.json")).parent
             for s in batch}
    histories = {a: [json.loads(x) for x in (p / "rounds.jsonl").read_text().splitlines() if x.strip()]
                 for a, p in paths.items()}
    result = {}
    for spec in batch:
        arm = spec["arm"]
        base = "C" if spec["seed"] == 42 else "CS43"
        if arm == base:
            continue
        rows, control = histories[arm], histories[base]
        with np.load(paths[arm] / "cms_r001.npz") as z, np.load(paths[base] / "cms_r001.npz") as b:
            assert np.array_equal(z["ids"], b["ids"])
            cm, bc = z["private"].astype(np.int64), b["private"].astype(np.int64)
        # This is CM disagreement count, not per-row prediction disagreement.
        r1 = {"cm_exact": bool(np.array_equal(cm, bc)), "cm_l1_difference": int(abs(cm - bc).sum()),
              "delta_macro": rows[0]["private/f1_macro"] - control[0]["private/f1_macro"],
              "delta_train_loss": rows[0]["train_loss"] - control[0]["train_loss"]}
        loss = [{"round": r["round"], "row": r["mean_kd_row_loss"], "mean": r["mean_kd_mean_loss"],
                 "jensen_gap": r["mean_kd_jensen_gap"],
                 "mean_over_row": r["mean_kd_mean_loss"] / r["mean_kd_row_loss"]} for r in rows[1:]]
        result[arm] = {"r1": r1, "loss": loss,
                       "scope": "Row/mean losses on same arm/forward/Q; not cross-arm gradient or teacher replay"}
    (out / "reception-diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# B16 reception diagnostics", "", "Post-hoc; no utility gate changes.", "",
             "| arm | r1 CM exact vs C | r1 CM L1 | r1 delta macro | r20 row KL | r20 mean KL | r20 mean/row |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm, data in result.items():
        r1, loss = data["r1"], data["loss"][-1]
        lines.append(f"| {arm} | {r1['cm_exact']} | {r1['cm_l1_difference']} | {r1['delta_macro']:+.7f} | "
                     f"{loss['row']:.6f} | {loss['mean']:.6f} | {loss['mean_over_row']:.4f} |")
    lines += ["", "CPU exact r1 proof does not establish whole T4 trajectory identity. Actual r1 CMs differ before soft Q is consumed;",
              "do not attribute all later differences solely to mean/half KD. Causes not isolated by this diagnostic.",
              "CM L1 counts changed confusion entries, not changed individual predictions.",
              "Jensen gap is algebraic within one arm. Neither its magnitude nor lower loss establishes utility."]
    (out / "reception-diagnostics.md").write_text("\n".join(lines) + "\n")
    print("B16 reception diagnostics written")


if __name__ == "__main__":
    main()
