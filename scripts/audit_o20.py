"""N-style full artifact audit plus source-to-consumed-target replay proof for all20rounds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_m20 import audit, extra
from methods.mobihet_replay import frozen, source_schedule, transform


def main():
    ap = argparse.ArgumentParser()
    for name in ("root", "out", "batch"): ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    counts = np.load("evidence/research-data/train_counts.npy")
    support = np.load("evidence/research-data/test_support.npy")
    common, details, replay, selections = [], [], [], {}
    for spec in json.loads(args.batch.read_text()):
        common.append(audit(args.root, spec, support, endpoint=20))
        detail = extra(args.root, spec, counts)
        for rec in detail["external"]:
            key = (rec["round"], rec["client_id"])
            value = (rec["selection_sha256"], rec["support"])
            assert selections.setdefault(key, value) == value
        details.append(detail)
        d = args.root / spec["slug"] / spec["slug"]
        hp = json.loads((d / "config.json").read_text())["hp"]
        raw = source_schedule(hp["seed"], torch.device("cpu"), hp["replay_source_sha256"])
        target = transform(raw, hp["replay_mode"])
        assert hp["replay_source_sha256"] == spec["replay_source_sha256"] == frozen.SHA256
        rows = list(map(json.loads, (d / "rounds.jsonl").read_text().splitlines()))
        for r, row in enumerate(rows, 1):
            s = torch.load(d / "models" / f"r{r:03}.pt", weights_only=True, map_location="cpu", mmap=True)
            assert torch.equal(s["Q_raw"], raw[r - 1])
            torch.testing.assert_close(s["Q"], target[r - 1], rtol=0, atol=1e-7)
            torch.testing.assert_close(s["Q_live"].sum(1), torch.ones(16), rtol=0, atol=1e-5)
            assert row["replay_consumed_round"] == (None if r == 1 else r - 1)
            assert row["replay_next_round"] == r and row["replay_source_sha256"] == frozen.SHA256
        module = args.root / spec["slug"] / "provenance/source/methods/mobihet_replay_data.py"
        assert module.stat().st_size == spec["replay_setup_module_bytes"]
        replay.append({"arm": spec["arm"], "mode": hp["replay_mode"], "source_sha256": frozen.SHA256,
                       "source_seed": hp["seed"], "checked_raw_and_target_snapshots": 20,
                       "packet_equivalent_bytes_20": 3333504,
                       "offline_preload_bytes_20": 3311744 + 20 * 544 + module.stat().st_size,
                       "accounting_note": "Offline total charges whole frozen module once plus uplinks; do not also charge cached per-round downlinks",
                       "scope": "Raw Q from N train-only outputs; consumed lag verified with snapshot/probe, not raw prediction replay"})
    for filename, content in (("audit.json", common), ("diagnostic-audit.json", details), ("replay-audit.json", replay)):
        (args.out / filename).write_text(json.dumps(content, indent=2) + "\n")
    print("O20_ARTIFACT_AUDIT_PASS", flush=True)


if __name__ == "__main__": main()
