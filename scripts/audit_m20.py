"""Require all20 rounds, frozen source, own/train/external CM, teacher lag and local support weights."""
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

PROBE_ROUNDS = (1, 2, 5, 10, 15, 20)


def check_summary(cm, row, prefix):
    summary = metrics.summary(cm)
    for k in metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10", "client_f1_macro_median",
                            "client_f1_macro_min", "f1_per_class"]:
        np.testing.assert_allclose(summary[k], row[prefix + "/" + k], atol=1e-12, rtol=0)


def extra(root, spec, counts):
    folder = root / spec["slug"]
    d = folder / spec["slug"]
    manifest = json.loads((folder / "provenance/source_manifest.json").read_text())
    assert manifest["overlay"] == spec["overlay_sha256"]
    cfg = json.loads((d / "config.json").read_text())
    hp = cfg["hp"]
    assert cfg["backend"] == "compiled" and hp["kd_weight"] == "fixed"
    assert hp["q_codec"] == "mean_fp16" and hp["diagnostic_rounds"] == list(PROBE_ROUNDS)
    rows = {x["round"]: x for x in map(json.loads, (d / "rounds.jsonl").read_text().splitlines())}
    no_q = hp.get("no_kd", False)
    assert all(x["bytes_up"] == x["bytes_down"] == (0 if no_q else 544) for x in rows.values())
    assert sum(cfg["payload_bytes"]["bootstrap"].values()) == 3311744
    snapshots, tables = [], {}
    assert sorted(p.name for p in (d / "models").glob("*.pt")) == [f"r{r:03}.pt" for r in range(1, 21)]
    for r in range(1, 21):
        p = d / "models" / f"r{r:03}.pt"
        s = torch.load(p, map_location="cpu", weights_only=True)
        assert "priv_p" not in s
        assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
        snapshots.append({"round": r, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        if not no_q:
            q = s["Q"].numpy()
            assert q.shape == (16, 16) and (q >= 0).all()
            np.testing.assert_allclose(q.sum(1), 1, atol=1e-5, rtol=0)
            tables[r] = q
            np.testing.assert_allclose(rows[r]["q_true_probability_mean"], q.diagonal().mean(), atol=2e-7)
            assert rows[r]["q_rejected_classes"] == np.flatnonzero(q.argmax(1) != np.arange(16)).tolist()
            if r > 1:
                np.testing.assert_allclose(rows[r]["q_change_l1"], np.abs(q - tables[r - 1]).sum(1).mean(), atol=2e-7)
        if hp.get("q_collect") == "postepoch":
            assert 0 < rows[r]["q_collect_seconds"] <= rows[r]["t_train"]
    probes = []
    for rank in (0, 1):
        records = list(map(json.loads, (d / f"diagnostics_rank{rank}.jsonl").read_text().splitlines()))
        assert [x["round"] for x in records] == list(PROBE_ROUNDS)
        for record in records:
            assert record["teacher_round"] == (None if no_q or record["round"] == 1 else record["round"] - 1)
            assert record["probe_mode"] == "shadow_postepoch_fp32_trainBN_dropout0"
        probes.extend(records)
    for r in PROBE_ROUNDS:
        with np.load(d / f"train_cms_r{r:03}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343))
            assert np.array_equal(z["private"].sum(2), counts)
            check_summary(z["private"], rows[r], "train_private")
        clients = [c for record in probes if record["round"] == r for c in record["clients"]]
        assert sorted(c["client_id"] for c in clients) == sorted(hp["diagnostic_ids"])
        for c in clients:
            cid = c["client_id"]
            assert c["train_rows"] == counts[cid].sum() and c["probe_rows"] == 64 and c["used_weight"] == 1
            assert np.isfinite([v for v in c.values() if isinstance(v, (float, int))]).all()
            if r == 1 or no_q:
                assert c["ce_teacher"] is None and c["grad_kd_real_norm"] == 0
                continue
            q = tables[r - 1]
            weight = np.ones(16) if hp.get("kd_support", "off") == "off" else (
                np.full(16, .5) if hp["kd_support"] == "uniform" else 64 / (64 + counts[cid]))
            expected_rho, expected_ce = 0., 0.
            for x in c["classes"]:
                label, n = x["class"], x["rows"]
                np.testing.assert_allclose(x["teacher_true_probability"], q[label, label], atol=1e-8)
                keep = hp.get("real_kd", True) and (not hp.get("real_kd_gate", False) or q[label].argmax() == label)
                assert x["real_gate_keeps_class"] == keep
                expected_rho += n / 64 * weight[label]
                expected_ce -= n / 64 * np.log(max(q[label, label], 1e-6))
            np.testing.assert_allclose(c["ce_teacher"], expected_ce, atol=3e-6, rtol=1e-5)
            np.testing.assert_allclose(c["real_support_weight_mean"], expected_rho, atol=1e-7)
            if hp.get("synth_target") == "fallback":
                expected = (weight * (q.argmax(1) == np.arange(16))).mean()
                np.testing.assert_allclose(c["synthetic_teacher_weight_mean"], expected, atol=1e-7)
    external = []
    for r in (1, 10, 20):
        current = []
        for rank in (0, 1):
            with np.load(d / f"external_cms_rank{rank}_r{r:03}.npz") as z:
                assert z["private"].shape == (len(z["ids"]), 16, 16) and (z["private"] >= 0).all()
                assert (z["private"].sum(2) <= 256).all()
                current.extend(z["ids"].tolist())
                external.extend({"round": r, "client_id": int(cid), "selection_sha256": str(h),
                                 "support": cm.sum(1).tolist()} for cid, h, cm in
                                zip(z["ids"], z["selection_sha256"], z["private"]))
        assert sorted(current) == sorted(hp["diagnostic_ids"])
    log = json.loads((folder / (spec["slug"] + ".log")).read_text())
    text = "".join(x.get("data", "") for x in log)
    assert "['Tesla T4', 'Tesla T4']" in text
    assert all(f"compile gate rank{rank}: PASS" in text for rank in (0, 1))
    assert "Traceback (most recent call last)" not in text
    return {"arm": spec["arm"], "snapshots": snapshots, "external": external,
            "scope": "20-round CM/source/table/probe audit; no raw prediction replay or formal privacy proof"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("artifacts/m20"))
    ap.add_argument("--out", type=Path, default=Path("runs/audit-m20"))
    ap.add_argument("--batch", type=Path, default=Path("evidence/m20-summary/batch.json"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts = np.load("evidence/research-data/train_counts.npy")
    support = np.load("evidence/research-data/test_support.npy")
    common, extras, selections = [], [], {}
    for spec in json.loads(args.batch.read_text()):
        common.append(audit(args.root, spec, support, endpoint=20))
        detail = extra(args.root, spec, counts)
        for rec in detail["external"]:
            key = (rec["round"], rec["client_id"])
            value = (rec["selection_sha256"], rec["support"])
            assert selections.setdefault(key, value) == value, "External test pool changed across methods/seeds"
        extras.append(detail)
    (args.out / "audit.json").write_text(json.dumps(common, indent=2))
    (args.out / "diagnostic-audit.json").write_text(json.dumps(extras, indent=2))
    print("M20_ARTIFACT_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
