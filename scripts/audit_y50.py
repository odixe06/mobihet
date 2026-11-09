"""Y50 audit: the F50 checks over the arm's endpoint. noq has no teacher, so its teacher chain is replaced by checks of absence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.audit_m20 import PROBE_ROUNDS
from scripts.audit_r20 import check_executed_source, extra
from scripts.audit_screening import audit
from scripts.y50_common import CELLS


def extra_noq(root, spec, counts, endpoint):
    folder = root / spec["slug"]; d = folder / spec["slug"]
    check_executed_source(folder, spec)
    cfg = json.loads((d / "config.json").read_text()); hp = cfg["hp"]
    assert cfg["method"] == spec["method"] and cfg["backend"] == "compiled" and hp["no_kd"] is True and hp["next_mode"] == "stream"
    assert sum(cfg["payload_bytes"]["bootstrap"].values()) == spec["bootstrap_total"]
    rows = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    assert len(rows) == endpoint
    for row in rows:
        s = torch.load(d / "models" / f'r{row["round"]:03}.pt', weights_only=True, map_location="cpu")
        assert "priv_p" not in s and "Q" not in s and all(torch.isfinite(t).all() for t in s.values())
        assert row["bytes_up"] == row["bytes_down"] == 0
        assert row["teacher_consumed_sha256"] is None and row["teacher_produced_sha256"] is None
    for r in (r for r in PROBE_ROUNDS if r <= endpoint):
        with np.load(d / f"train_cms_r{r:03}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), counts)
    for rank in (0, 1):
        records = [json.loads(x) for x in (d / "recovery" / f"rank{rank}.jsonl").read_text().splitlines()]
        assert records[0]["event"] == "compile_guard" and records[0]["compiled_ranks"] == 2
        assert [v["round"] for v in records[1:]] == list(range(1, endpoint + 1))
    return {"arm": spec["arm"], "mode": "stream", "scope": "CM/source/no-teacher/guard bounded audit"}


def main():
    p = argparse.ArgumentParser()
    for key in ("batch", "root", "out"): p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--endpoint", type=int, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    common, detail = [], []
    for spec in json.loads(a.batch.read_text()):
        folder = a.root / spec["slug"]; d = folder / spec["slug"]
        hp = json.loads((d / "config.json").read_text())["hp"]
        cell = CELLS[spec["cell"]]
        assert all(hp[k] == v for k, v in cell["w"].items()), spec["arm"]
        assert ("synth_step_floor" in hp) == ("synth_step_floor" in cell["w"]), spec["arm"]
        assert hp["synth"] == cell["set"].get("synth", "copula") and hp["no_kd"] == (spec["cell"] == "noq"), spec["arm"]
        sessions = spec["sessions"]
        for i, kernel in enumerate(sessions):
            log = (folder / (kernel + ".log")).read_text()
            assert "npy cache verified at" in log and "npy cache built" not in log, kernel
            assert "Tesla T4" in log and all(f"compile gate rank{r}: PASS" in log for r in (0, 1)), kernel
            assert "Traceback (most recent call last)" not in log, kernel
            assert (i == 0) == ("resumed from" not in log), kernel
        common.append(audit(a.root, spec, support, endpoint=a.endpoint))
        check = extra_noq(a.root, spec, counts, a.endpoint) if spec["cell"] == "noq" else extra(
            a.root, spec, counts, method=spec["method"], bootstrap_total=spec["bootstrap_total"], endpoint=a.endpoint)
        detail.append(check | {"cell": spec["cell"], "sessions": sessions})
    for name, data in [("audit.json", common), ("diagnostic-audit.json", detail)]:
        (a.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print("Y50_ARTIFACT_AUDIT_PASS", len(common), a.endpoint, flush=True)


if __name__ == "__main__": main()
