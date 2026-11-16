"""KM16-copula full-scale pilot r50 vs Core F2/F3 r50 under the locked rule docs/research/screening-km-pilot50.md (official own-test)."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, summary  # noqa: E402
from scripts import y50_common as Y  # noqa: E402
import numpy as np  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 50
batch = json.loads((ROOT / "runs/km-pilot50/batch.json").read_text())
out, ok_all = {}, True
for spec in batch:
    run, = glob.glob(str(ROOT / "artifacts/km-pilot50" / spec["slug"] / "**" / "rounds.jsonl"), recursive=True)
    run = Path(run).parent
    d = Y.delta(run, spec["seed"], R)
    with np.load(run / f"cms_r{R:03d}.npz") as z:
        ten = {k: float(v) for k, v in summary(z["private"]).items() if k in TEN}
    gate = {"macro": d["f1_macro"] >= .003, "p10": d["p10"] >= -.006, "low256": d["low256"] >= -.003,
            "timeDelay": d["timeDelay"] >= -.0035, "benign": d["benign"] >= -.0005, "absent": d["absent"] >= -.01}
    ok_all &= all(gate.values())
    out[spec["arm"]] = {"seed": spec["seed"], "ten": ten, "delta": d, "gate": gate}
    print(spec["arm"], {k: round(v, 4) for k, v in d.items()}, "PASS" if all(gate.values()) else "FAIL",
          [k for k, v in gate.items() if not v])
out["verdict"] = "PASS" if ok_all and len(batch) == 2 else "FAIL"
print("pilot verdict (both seeds):", out["verdict"])
json.dump(out, open(ROOT / f"runs/km-pilot50/verdict-r{R}.json", "w"), indent=1)
for spec in batch:                                                    # report only (not gate): absent guard trajectory r41-r50
    run = Path(glob.glob(str(ROOT / "artifacts/km-pilot50" / spec["slug"] / "**" / "rounds.jsonl"), recursive=True)[0]).parent
    tr = [(r, Y.read(run, r)["absent"] - Y.read(Y.CONTROL[spec["seed"]], r)["absent"]) for r in range(41, R + 1)
          if (run / f"cms_r{r:03d}.npz").exists()]
    print(spec["arm"], "Δabsent r41..", " ".join(f"r{r}:{d:+.3f}" for r, d in tr))
