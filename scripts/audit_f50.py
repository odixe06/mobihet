"""F50 50-round audit: the X20 checks (source/CM/aggregates/teacher chain/guards) over rounds 1..50, every session's log.

A resumed arm's last session holds the whole run directory (the resume copies it in), so CMs, snapshots and the teacher
chain are read once over 50 rounds; the chain row r-1 -> r also spans each session boundary.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_r20 import extra
from scripts.audit_screening import audit
from scripts.f50_common import CELLS, ENDPOINT


def main():
    p = argparse.ArgumentParser()
    for key in ("batch", "root", "out"): p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    common, detail = [], []
    for spec in json.loads(a.batch.read_text()):
        folder = a.root / spec["slug"]; d = folder / spec["slug"]
        hp = json.loads((d / "config.json").read_text())["hp"]
        cell = CELLS[spec["cell"]]
        assert hp["r_kind"] == cell["r_kind"] and all(hp[k] == v for k, v in cell["w"].items()), spec["arm"]
        assert not ({"w_real", "w_synth", "w_boot", "synth_step_floor"} - set(cell["w"])) & set(hp), spec["arm"]
        sessions = spec.get("sessions", [spec["slug"]])
        for i, kernel in enumerate(sessions):
            log = (folder / (kernel + ".log")).read_text()
            assert "npy cache verified at" in log and "npy cache built" not in log, kernel
            assert "Tesla T4" in log and all(f"compile gate rank{r}: PASS" in log for r in (0, 1)), kernel
            assert "Traceback (most recent call last)" not in log, kernel
            assert (i == 0) == ("resumed from" not in log), kernel
            assert i == len(sessions) - 1 or "budget stop before round" in log, kernel
        common.append(audit(a.root, spec, support, endpoint=ENDPOINT))
        detail.append(extra(a.root, spec, counts, method="mobihet_wstudy", bootstrap_total=spec["bootstrap_total"],
                            endpoint=ENDPOINT) | {"cell": spec["cell"], "sessions": sessions})
    for name, data in [("audit.json", common), ("diagnostic-audit.json", detail)]:
        (a.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print("F50_ARTIFACT_AUDIT_PASS", len(common), flush=True)


if __name__ == "__main__": main()
