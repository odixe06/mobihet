"""W20 20-round audit: R20 checks (source/CM/aggregates/teacher chain/guards) with the W method, cell knobs and bootstrap bytes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_r20 import extra
from scripts.audit_screening import audit
from scripts.w20_common import CELLS


def main():
    p = argparse.ArgumentParser()
    for key in ("batch", "root", "out"): p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    common, detail = [], []
    for spec in json.loads(a.batch.read_text()):
        d = a.root / spec["slug"] / spec["slug"]
        hp = json.loads((d / "config.json").read_text())["hp"]
        cell = CELLS[spec["cell"]]
        assert hp["r_kind"] == cell["r_kind"] and all(hp[k] == v for k, v in cell["w"].items()), spec["arm"]
        assert not ({"w_real", "w_synth", "w_boot", "synth_step_floor"} - set(cell["w"])) & set(hp), spec["arm"]
        log = (a.root / spec["slug"] / (spec["slug"] + ".log")).read_text()
        assert "npy cache verified at" in log and "npy cache built" not in log
        common.append(audit(a.root, spec, support, endpoint=20))
        detail.append(extra(a.root, spec, counts, method="mobihet_wstudy", bootstrap_total=spec["bootstrap_total"]) | {"cell": spec["cell"]})
    for name, data in [("audit.json", common), ("diagnostic-audit.json", detail)]:
        (a.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print("W20_ARTIFACT_AUDIT_PASS", len(common), flush=True)


if __name__ == "__main__": main()
