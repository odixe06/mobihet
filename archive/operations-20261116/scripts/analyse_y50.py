"""Read audited Y50 arms against their paired F2/F3 control and apply the locked rules of docs/research/screening-y50.md. No training."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import y50_common as Y


def main():
    batch = [r for w in (1, 2) for r in json.loads((ROOT / f"runs/y-study/wave{w}/batch.json").read_text())]
    done = {}
    for w in (1, 2):
        path = ROOT / f"runs/y-study/wave{w}/closure.json"
        if path.exists(): done |= json.loads(path.read_text())["runs"]
    res, cells = {}, {}
    for spec in batch:
        if not done.get(spec["arm"], {}).get("audited"): continue
        d = ROOT / "artifacts/y50" / spec["slug"] / spec["slug"]
        rounds = [r for r in (20, 30, 40, 50) if (d / f"cms_r{r:03d}.npz").exists()]
        res[spec["arm"]] = {"cell": spec["cell"], "seed": spec["seed"], "rounds": {r: {"value": Y.read(d, r), "delta": Y.delta(d, spec["seed"], r)} for r in rounds}}
        if 50 in rounds: cells.setdefault(spec["cell"], []).append(res[spec["arm"]]["rounds"][50]["delta"])
    rules = {}
    if len(cells.get("prior", [])) == 2: rules["prior"] = Y.prior_rule(cells["prior"])
    for cell in ("noq", "nofloor"):
        if len(cells.get(cell, [])) == 2: rules[cell] = Y.ablation_rule(cells[cell])
    if cells.get("oracle"): rules["oracle"] = Y.oracle_rule(cells["oracle"][0])
    out = ROOT / "runs/y-study/analysis"; out.mkdir(exist_ok=True)
    (out / "analysis.json").write_text(json.dumps({"arms": res, "rules": rules}, indent=1))
    print(json.dumps(rules, indent=1))


if __name__ == "__main__": main()
