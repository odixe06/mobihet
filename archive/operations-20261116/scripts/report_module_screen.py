"""Describe audited module-screen artifacts, including ten metrics and actual wire costs."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN  # noqa: E402
from scripts.sim_fold_val import pick  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json  # noqa: E402


def main():
    b = sys.argv[1]
    out, art = ROOT / f"runs/sim-kaggle-{b}", ROOT / f"artifacts/sim-kaggle-{b}"
    assert json.loads((out / "audit.json").read_text())["pass"]
    result = json.loads((out / "verdict-r20.json").read_text())
    batch = json.loads((out / "batch.json").read_text())
    lines = [f"# {b} — audited r20, fold val", "", "Control paired by seed; no official test. Two seeds are not significance.", "",
             "| arm | Δmacro | Δp10 | Δlow256 | Δbenign | ΔtimeDelay | gates failed |",
             "|---|---:|---:|---:|---:|---:|---|"]
    details = {}
    if b == "b17":
        lines[3:3] = ["Primary: macro-F1 gain >= +.002 per seed. Secondary utility metrics are report-only (owner11/11).", ""]
    for spec in batch:
        arm = spec["arm"]
        if arm in result["verdict"]:
            v = result["verdict"][arm]
            d = v["delta"]
            lines.append("| " + arm + " | " + " | ".join(f"{d[k]:+.5f}" for k in (
                "f1_macro", "p10", "low256", "benign", "timeDelay")) + " | " +
                ", ".join(k for k, passed in v["gates"].items() if not passed) + " |")
        top = art / spec["slug"]
        log = next(top.glob("*.log"))
        text = "".join(e.get("data", "") for e in json.loads(log.read_text()))
        assert all(f"compile gate rank{r}: PASS" in text for r in (0, 1))
        assert "['Tesla T4', 'Tesla T4']" in text
        decoder, wire = json.JSONDecoder(), []
        for part in text.split("KM_CODEC_BYTES ")[1:]:
            wire.append(decoder.raw_decode(part)[0])
        assert wire and all(w == wire[0] for w in wire)
        run = next(top.glob("**/config.json")).parent
        rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()]
        with np.load(run / "cms_r020.npz") as z:
            cm = z["private"].sum(0)
        tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
        pr, rec = tp / np.maximum(pred, 1), tp / np.maximum(sup, 1)
        detail = {"bootstrap": wire[0], "compile_both_ranks": True,
                  "up_per_round": float(np.mean([r["bytes_up"] for r in rows])),
                  "down_per_round": float(np.mean([r["bytes_down"] for r in rows])),
                  "train_seconds": sum(r["t_train"] for r in rows), "test_rows": int(cm.sum()),
                  "max_skipped_frac": max(r["skipped_frac"] for r in rows),
                  "precision": pr.tolist(), "recall": rec.tolist(), "class_f1": result["stats"][arm]["endpoint"]["f1_class"],
                  "benign_to_attack_not_td": int(cm[0, 1:13].sum() + cm[0, 14:].sum()),
                  "benign_recall": float(rec[0])}
        detail["round_bytes_total"] = 20 * (detail["up_per_round"] + detail["down_per_round"])
        detail["total_with_bootstrap"] = wire[0]["total_unicast_mean"] + detail["round_bytes_total"]
        details[arm] = detail
    lines += ["", "## All ten endpoint metrics", "", "| arm | " + " | ".join(TEN) + " |",
              "|---|" + "---:|" * len(TEN)]
    for spec in batch:
        ten = result["stats"][spec["arm"]]["endpoint"]["ten"]
        lines.append("| " + spec["arm"] + " | " + " | ".join(f"{ten[k]:.6f}" for k in TEN) + " |")
    lines += ["", "## Communication and elapsed training", "",
              "| arm | bootstrap B/client | up B/round | down B/round | total B/client/20r | training seconds/20r |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, d in details.items():
        lines.append(f"| {arm} | {d['bootstrap']['total_unicast_mean']:.2f} | {d['up_per_round']:.2f} | "
                     f"{d['down_per_round']:.2f} | {d['total_with_bootstrap']:.2f} | {d['train_seconds']:.1f} |")
    counts = np.load(ROOT / "cache/fold_counts.npy")
    concentration = {}
    for key, n in (("sim200", counts[pick(200, cnt=counts), 0]),
                   ("full5343", np.load(ROOT / "evidence/research-data/train_counts.npy"))):
        concentration[key] = {"max_client_share_per_class": (n / n.sum(0)).max(0).tolist(),
                              "effective_clients_per_class": (n.sum(0) ** 2 / (n ** 2).sum(0)).tolist()}
    atomic_json(out / "detail-r20.json", {"arms": details, "class_count_concentration": concentration})
    (out / "report-r20.md").write_text("\n".join(lines) + "\n")
    print(f"{b}: stored ten metrics, both-rank compile, cost, precision/recall and concentration verified.")


if __name__ == "__main__":
    main()
