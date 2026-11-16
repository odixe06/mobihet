"""KM16-copula full-scale pilot r20 vs Core F2/F3 r20 under the locked generator pilot20 gate (official own-test)."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, summary  # noqa: E402
from scripts import y50_common as Y  # noqa: E402
import numpy as np  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
batch = json.loads((ROOT / "runs/km-pilot20/batch.json").read_text())
out, ok_all = {}, True
for spec in batch:
    run, = glob.glob(str(ROOT / "artifacts/km-pilot20" / spec["slug"] / "**" / "rounds.jsonl"), recursive=True)
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
json.dump(out, open(ROOT / f"runs/km-pilot20/verdict-r{R}.json", "w"), indent=1)
