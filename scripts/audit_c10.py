"""Audit exact C10 outputs and decoded Q, then report paired r10 metrics; no fresh-test claim."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.audit_screening import audit
from methods.mobihet_next import signature
from scripts.watch_kaggle_batch import atomic_json


def main():
    out, art = ROOT / "runs/c10-study", ROOT / "artifacts/c10"
    batch = json.loads((out / "batch.json").read_text())
    progress = json.loads((out / "progress.json").read_text())
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    reports = []
    for spec in batch:
        if progress[spec["arm"]]["state"]["status"] != "COMPLETE":
            reports.append(dict(arm=spec["arm"], cell=spec["cell"], status="ERROR", scope="Inspect pulled logs; no utility conclusion"))
            continue
        report = audit(art, spec, support, endpoint=10)
        d = art / spec["slug"] / spec["slug"]
        manifest = json.loads((art / spec["slug"] / "provenance/source_manifest.json").read_text())
        assert all(manifest["overlay"][k] == v for k, v in spec["overlay_sha256"].items())
        quant = spec["cell"] in ("q8", "combo")
        rows = [json.loads(s) for s in (d / "rounds.jsonl").read_text().splitlines()]
        previous = signature(None, None, None, "stream")
        for r in range(1, 11):
            snapshot = torch.load(d / f"models/r{r:03d}.pt", map_location="cpu", weights_only=False)
            assert all(torch.isfinite(t).all() for t in snapshot.values() if torch.is_tensor(t) and t.is_floating_point())
            teacher = snapshot["Q"]
            assert rows[r-1]["teacher_consumed_sha256"] == previous
            previous = signature(teacher, None, None, "stream")
            assert rows[r-1]["teacher_produced_sha256"] == previous
            assert rows[r-1]["bytes_up"] == 544 and rows[r-1]["bytes_down"] == (288 if quant else 544)
            torch.testing.assert_close(teacher.sum(-1), torch.ones(16), atol=1e-5, rtol=0)
            if quant:
                for rank in (0, 1):
                    packet = torch.load(d / f"qcodec_rank{rank}_r{r:03d}.pt", map_location="cpu", weights_only=True)
                    assert torch.equal(packet["body"], torch.round(packet["original"] * 255).to(torch.uint8))
                    decoded = packet["body"].float()
                    decoded = decoded / decoded.sum(-1, keepdim=True).clamp_min(1)
                    torch.testing.assert_close(decoded, packet["decoded"], atol=0, rtol=2 * torch.finfo(torch.float32).eps)
                    assert torch.equal(teacher, packet["decoded"])
        boots = [json.loads((d / f"bootstrap_codec_rank{rank}.json").read_text()) for rank in (0, 1)]
        assert boots[0] == boots[1] and boots[0]["roundtrip"]
        report.update(cell=spec["cell"], status="AUDIT_PASS", bootstrap_codec=boots[0])
        reports.append(report)
    atomic_json(out / "audit.json", reports)
    atomic_json(out / "closure.json", dict(runs=reports, audited=sum(x["status"] == "AUDIT_PASS" for x in reports)))
    print("C10_ARTIFACT_AUDIT_DONE", flush=True)


if __name__ == "__main__": main()
