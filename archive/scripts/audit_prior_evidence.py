"""Read-only sibling/archive provenance and v4 train-only missingness audit; no training or test fitting."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT.parent / "mobihet-features"
BENCH = ROOT.parent / "fl_dagsnet_veremi_v2"
OLD = Path("/tmp/mobihet-v1-review-20261101")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "runs/p-study"; out.mkdir(exist_ok=True)
    source = {}
    for base, names in ((BENCH, ["core/dagsnet.py", "core/grouped.py", "docs/research/protocol.md"]),
            (FEATURES, ["src/v2/scale.py", "src/v2/features.py", "src/v2/p4.py", "src/v2/build.py", "docs/research/feature-processing-v4.md"]),
            (OLD, ["CONTEXT.md", "LESSONS.md", "docs/research/methods/synthetic-weight-dynamics-2026-10-23.md",
                   "docs/research/methods/synthetic-learning-recovery-2026-10-22.md", "docs/research/methods/teacher-support-two-seed-2026-10-22.md"])):
        source[str(base)] = {"head": subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip(),
                              "files": {name: sha(base / name) for name in names}}
    identical = {name: sha(ROOT / name) == sha(BENCH / name) for name in ["core/dagsnet.py", "core/grouped.py", "docs/research/protocol.md"]}
    assert all(identical.values())
    root = FEATURES / "data/v4-scaled"
    scaler = json.loads((root / "scaler.json").read_text())
    indicators = scaler["nan_indicator"]
    flags = sorted(set(indicators.values()))
    columns = sorted(set(flags) | set(indicators))
    rows = violations = nonbinary = 0
    # Fixed first1024 rows/file, train directory only. Bounded audit, not a full-data identity proof.
    for path in sorted((root / "train").rglob("*.parquet")):
        table = next(pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns)).to_pandas()
        rows += len(table)
        nonbinary += int((~np.isin(table[flags].to_numpy(), [0, 1])).sum())
        for feature, flag in indicators.items():
            violations += int(((table[flag] == 1) & (table[feature] != 0)).sum())
    result = {"sources": source, "benchmark_byte_identical": identical, "scaler_sha256": sha(root / "scaler.json"),
              "train_sample_rows": rows, "nan_flags": flags, "nan_indicator": indicators,
              "nonbinary_flag_cells": nonbinary, "missing_zero_constraint_violations": violations,
              "scope": "First1024 train-directory rows/file; includes originaltrain+val, no officialtest. Source provenance and bounded consistency, not full content hash."}
    assert nonbinary == violations == 0
    (out / "prior-evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print("P20_PRIOR_EVIDENCE_PASS", rows, flush=True)


if __name__ == "__main__": main()
