"""Audit numerical observations; success means evidence collected, not healthy20round training."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.audit_r20 import check_executed_source
from scripts.audit_v20 import check_fold_receipts
from scripts.watch_kaggle_batch import atomic_json

OUT = ROOT / "runs/v-study/numerics-guard"


def main():
    collected = json.loads((OUT / "collected.json").read_text())
    counts, support = (np.load(ROOT / f"runs/v-study/{f}_counts.npy") for f in ("train", "val"))
    split = json.loads((ROOT / "runs/v-study/split-audit.json").read_text()); reports = []
    assert len(collected) == 2
    for item in collected:
        row = item["spec"]; folder = Path(item["root"]); d = folder / row["slug"]
        assert item["status"]["status"] == "COMPLETE", "Probe ERROR collected: inspect actual log"
        check_executed_source(folder, row)
        ranks = check_fold_receipts(d, counts, support, split["scaler_sha256"])
        records = [json.loads((d / f"numerics_rank{r}.json").read_text()) for r in (0, 1)]
        total_applied = total_skipped = 0; zero = []; finite = True
        for rank, (record, mine) in enumerate(zip(records, ranks)):
            assert record["rank"] == rank and record["round"] == 1 and record["compiled_ranks"] == record["world"] == 2
            assert record["n_params"] == 395408 and record["parent_arm"] == row["parent_arm"] and record["loss_scale"] == 4096
            for setting in row["hp"]:
                k, v = setting.split("=", 1)
                try: v = json.loads(v)
                except json.JSONDecodeError: pass
                assert record["hp"][k] == v
            clients = record["clients"]
            assert sorted(c["client_id"] for c in clients) == mine.tolist()
            for client in clients:
                assert client["applied"] >= 0 and client["skipped"] >= 0
                expected = int(np.ceil(counts[client["client_id"]].sum() / 64))
                assert client["applied"] + client["skipped"] == expected
            assert record["applied"] == sum(c["applied"] for c in clients)
            assert record["skipped"] == sum(c["skipped"] for c in clients)
            assert record["zero_applied_client_ids"] == [c["client_id"] for c in clients if not c["applied"]]
            total_applied += record["applied"]; total_skipped += record["skipped"]
            zero.extend(record["zero_applied_client_ids"]); finite &= record["all_finite"]
        observed_fraction = total_skipped / (total_applied + total_skipped)
        for record in records:
            assert np.isclose(record["global_info"]["skipped_frac"], observed_fraction, rtol=1e-6, atol=1e-12)
        log = (folder / (row["slug"] + ".log")).read_text()
        assert "Tesla T4" in log and all(f"compile gate rank{r}: PASS" in log for r in (0, 1))
        assert "Traceback (most recent call last)" not in log
        reports.append({"arm": row["arm"], "parent_arm": row["parent_arm"], "applied": total_applied,
            "skipped": total_skipped, "skipped_fraction": observed_fraction, "zero_applied_client_ids": zero,
            "all_reported_finite": bool(finite), "compile_both_ranks": True,
            "files_sha256": {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in d.glob("*.json")},
            "scope": "Source/fold/count/firstepoch telemetry verified; not accepted V20, warm-up/overflow cause or full20round utility proof"})
    atomic_json(OUT / "audit.json", reports)
    print("V20_NUMERICS_OBSERVATIONS_VERIFIED", flush=True)


if __name__ == "__main__": main()
