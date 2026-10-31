"""Audit frozen L outputs: common screening gates, train CM, shadow records and small snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics
from scripts.audit_screening import audit

PROBE_ROUNDS = (1, 2, 5, 10)


def finite_values(value):
    if isinstance(value, dict):
        for x in value.values(): finite_values(x)
    elif isinstance(value, list):
        for x in value: finite_values(x)
    elif isinstance(value, (float, int)):
        assert np.isfinite(value)


def extra(root, spec, counts):
    d = root / spec["slug"] / spec["slug"]
    manifest = root / spec["slug"] / "provenance/source_manifest.json"
    source = json.loads(manifest.read_text())
    assert source["overlay"] == spec["overlay_sha256"]
    for name in ("input_order.json", "label_mapping.json", "runtime.json"):
        assert source["files"]["knowledge/" + name] == hashlib.sha256((Path("knowledge") / name).read_bytes()).hexdigest()
    cfg = json.loads((d / "config.json").read_text())
    hp = cfg["hp"]
    assert cfg["backend"] == "compiled" and hp["q_codec"] == "mean_fp16" and hp["diagnostics"] is True
    rows = {x["round"]: x for x in map(json.loads, (d / "rounds.jsonl").read_text().splitlines())}
    assert all(x["bytes_up"] == x["bytes_down"] == 544 for x in rows.values())
    assert sum(cfg["payload_bytes"]["bootstrap"].values()) == 3311744
    q, snapshots = {}, []
    assert sorted(p.name for p in (d / "models").glob("*.pt")) == [f"r{r:03}.pt" for r in range(1, 11)]
    for r in range(1, 11):
        p = d / "models" / f"r{r:03}.pt"
        s = torch.load(p, map_location="cpu", weights_only=True)
        assert "priv_p" not in s, "Small snapshots are not full private checkpoints"
        assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
        q[r] = s["Q"].double().numpy()
        assert q[r].shape == (16, 16) and (q[r] >= 0).all()
        np.testing.assert_allclose(q[r].sum(1), 1, atol=1e-5, rtol=0)
        snapshots.append({"round": r, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size})
    train, probes = [], []
    for rank in (0, 1):
        records = [json.loads(x) for x in (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()]
        assert [x["round"] for x in records] == list(PROBE_ROUNDS)
        for record in records:
            assert record["probe_mode"] == "shadow_postepoch_fp32_trainBN_dropout0"
            assert record["teacher_round"] == (None if record["round"] == 1 else record["round"] - 1)
            finite_values(record)
        probes.append(records)
    for r in PROBE_ROUNDS:
        path = d / f"train_cms_r{r:03}.npz"
        with np.load(path) as z:
            assert np.array_equal(z["ids"], np.arange(5343))
            cm = z["private"]
        assert cm.shape == (5343, 16, 16) and np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
        assert np.array_equal(cm.sum(2), counts) and cm.sum() == 37361273
        summary = metrics.summary(cm)
        for key in metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10", "client_f1_macro_median",
                                  "client_f1_macro_min", "f1_per_class"]:
            np.testing.assert_allclose(rows[r]["train_private/" + key], summary[key], atol=1e-12, rtol=0)
        train.append({"round": r, **summary, "t_diagnostics": rows[r]["t_diagnostics"]})
        clients = [c for rank in probes for rec in rank if rec["round"] == r for c in rec["clients"]]
        assert sorted(c["client_id"] for c in clients) == sorted(hp["diagnostic_ids"])
        for c in clients:
            assert c["train_rows"] == counts[c["client_id"]].sum() and c["probe_rows"] == 64
            assert all(c[key] >= 0 for key in ("grad_real_norm", "grad_kd_real_norm", "grad_synth_norm", "grad_total_norm"))
            for key in ("cos_real_kd_real", "cos_real_synth"):
                assert c[key] is None or -1.00001 <= c[key] <= 1.00001
            if r == 1:
                assert c["ce_teacher"] is None and c["used_weight"] == 1 and c["classes"] == []
            else:
                teacher = q[r - 1]
                assert sum(x["rows"] for x in c["classes"]) == 64
                ct = 0
                for x in c["classes"]:
                    label = x["class"]
                    assert 0 <= x["logit_negative_count_before_gate"] <= x["logit_cosine_count"] <= x["rows"]
                    np.testing.assert_allclose(x["teacher_true_probability"], teacher[label, label], atol=1e-8, rtol=0)
                    ct -= x["rows"] / 64 * np.log(max(teacher[label, label], 1e-6))
                    expected = hp.get("real_kd", True) and (not hp.get("real_kd_gate", False) or teacher[label].argmax() == label)
                    assert x["real_gate_keeps_class"] == expected
                np.testing.assert_allclose(c["ce_teacher"], ct, atol=3e-6, rtol=1e-5)
                np.testing.assert_allclose(c["adaptive_weight"], 1 / (c["ce_real"] + c["ce_teacher"]), atol=3e-6, rtol=1e-5)
                np.testing.assert_allclose(c["used_weight"], 1 if hp.get("kd_weight") == "fixed" else c["adaptive_weight"], atol=1e-8, rtol=0)
            if not hp.get("real_kd", True) or r == 1:
                assert c["kl_real_gated"] == c["grad_kd_real_norm"] == 0 and c["cos_real_kd_real"] is None
    log = json.loads((root / spec["slug"] / (spec["slug"] + ".log")).read_text())
    text = "".join(x.get("data", "") for x in log)
    assert "['Tesla T4', 'Tesla T4']" in text
    assert all(f"compile gate rank{rank}: PASS" in text for rank in (0, 1))
    assert "Traceback (most recent call last)" not in text
    return {"arm": spec["arm"], "train_rounds": train, "probes": probes, "snapshots": snapshots,
            "t_diagnostics_total": sum(x["t_diagnostics"] for x in rows.values()),
            "scope": "Frozen source/CM/shadow/table audit; no full bank, prediction replay or independent heldout proof"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("runs/pulls"))
    ap.add_argument("--batch", type=Path, default=Path("runs/l-screening/batch.json"))
    ap.add_argument("--counts", type=Path, default=Path("runs/j-screening/train_counts.npy"))
    ap.add_argument("--out", type=Path, default=Path("runs/l-screening"))
    args = ap.parse_args()
    counts = np.load(args.counts)
    assert counts.shape == (5343, 16) and counts.sum() == 37361273
    support = np.load(args.root / "mh-g4-copula/mh-g4-copula/cms_r010.npz")["private"].sum(2)
    common, diagnostic = [], []
    for spec in json.loads(args.batch.read_text()):
        common.append(audit(args.root, spec, support))
        diagnostic.append(extra(args.root, spec, counts))
        print(spec["arm"], "CM/source/snapshots/train/probe PASS", common[-1]["rounds"][-1]["f1_macro"], flush=True)
    (args.out / "audit.json").write_text(json.dumps(common, indent=2))
    (args.out / "diagnostic-audit.json").write_text(json.dumps(diagnostic, indent=2))
    print("L12_ARTIFACT_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
