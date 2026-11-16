"""Post-hoc module-screen CM/wire diagnostics (default b14). No training or tuning."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_uplink import raw_body  # noqa: E402
from scripts.research_metrics import client_f1  # noqa: E402


def pooled(cm):
    c = cm.sum(0)
    p = np.diag(c) / np.maximum(c.sum(0), 1)
    r = np.diag(c) / np.maximum(c.sum(1), 1)
    f = 2 * np.diag(c) / np.maximum(c.sum(0) + c.sum(1), 1)
    return c, p, r, f


def teacher(run, round_no, mode):
    arithmetic, count = np.zeros((16, 16)), np.zeros(16)
    for rank in (0, 1):
        with np.load(run / "wire" / f"rank{rank}_r{round_no:03d}.npz") as z:
            q = z["Q"].astype(float)
            data, offsets = z["frames"].tobytes(), z["offsets"]
            for j, n in enumerate(z["counts"]):
                large, small = n >= 8, (n > 0) & (n < 8)
                raw = raw_body(data[offsets[j]:offsets[j + 1]], 36)
                end = int(large.sum()) * 32
                values = np.zeros((16, 16))
                v = np.frombuffer(raw[:end], "<f2").astype(float).reshape(-1, 16)
                if mode == "lg16":
                    v = np.exp(v - v.max(1, keepdims=True))
                values[large] = v / v.sum(1, keepdims=True)
                v = np.frombuffer(raw[end:], np.uint8).reshape(-1, 16).astype(float)
                values[small] = v / v.sum(1, keepdims=True)
                arithmetic += values * n[:, None]
                count += n
    arithmetic /= count[:, None]
    entropy = -(q * np.log(np.maximum(q, 1e-30))).sum(1)
    # LG arithmetic here averages per-client geometric distributions, NOT raw E[p].
    return {"entropy_per_class": entropy.tolist(), "diagonal": np.diag(q).tolist(),
            "zero_codes": int((q == 0).sum()), "mean_entropy": float(entropy.mean()),
            "mean_diagonal": float(np.diag(q).mean()),
            "mean_tv_to_arithmetic_client_distributions": float((.5 * abs(q - arithmetic).sum(1)).mean())}


def main():
    batch_name = sys.argv[1] if len(sys.argv) > 1 else "b14"
    out = ROOT / f"runs/sim-kaggle-{batch_name}"
    assert json.loads((out / "audit.json").read_text())["pass"]
    if batch_name == "b14":
        assert json.loads((out / "wire-audit.json").read_text())["pass"]
    verdict = json.loads((out / "verdict-r20.json").read_text())
    names = json.loads((ROOT / "knowledge/label_mapping.json").read_text())["id_to_label"]
    counts = np.load(ROOT / "cache/fold_counts.npy")
    batch = json.loads((out / "batch.json").read_text())
    histories, ids, paths = {}, None, {}
    for spec in batch:
        arm = spec["arm"]
        run = next((ROOT / f"artifacts/sim-kaggle-{batch_name}" / spec["slug"]).glob("**/config.json")).parent
        paths[arm] = run
        histories[arm] = []
        for r in range(1, 21):
            with np.load(run / f"cms_r{r:03d}.npz") as z:
                if ids is None:
                    ids = z["ids"].copy()
                assert np.array_equal(ids, z["ids"])
                histories[arm].append(z["private"].copy())
    n = counts[ids, 0].sum(1)
    cohorts = {"lt256": n < 256, "256_1023": (n >= 256) & (n < 1024),
               "1024_4095": (n >= 1024) & (n < 4096), "ge4096": n >= 4096}
    result = {"cohort_clients": {k: int(v.sum()) for k, v in cohorts.items()}, "arms": {}}
    lines = [f"# {batch_name.upper()} — phân tích CM và teacher", "", "Post-hoc; không đổi gate, không chọn best round. Fold val, hai seed; chưa xác nhận full-scale.", "",
             "| arm | Δmacro r20 | Δmacro TB16–20 | ΔFP benign→attack ngoài TD | lớp mất F1 nhiều nhất |", "|---|---:|---:|---:|---|"]
    for spec in batch:
        arm = spec["arm"]
        base = "C" if spec["seed"] == 42 else "CS43"
        if arm == base:
            continue
        c, p, r, f = pooled(histories[arm][-1])
        b, bp, br, bf = pooled(histories[base][-1])
        d = c - b
        cf, cb = client_f1(histories[arm][-1]), client_f1(histories[base][-1])
        trace = [float(pooled(a)[3].mean() - pooled(bc)[3].mean())
                 for a, bc in zip(histories[arm], histories[base])]
        assert abs(trace[-1] - verdict["verdict"][arm]["delta"]["f1_macro"]) < 1e-12
        changed = d.copy()
        np.fill_diagonal(changed, 0)
        worst_cells = np.dstack(np.unravel_index(np.argsort(changed.ravel())[-8:][::-1], (16, 16)))[0]
        detail = {"delta_macro_trace": trace,
                  "delta_macro_mean16_20": float(np.mean(trace[-5:])),
                  "cohorts": {k: {"mean_f1": float(cf[v].mean()), "delta": float((cf - cb)[v].mean())} for k, v in cohorts.items() if v.any()},
                  "clients_improved": int((cf > cb).sum()), "clients_worse": int((cf < cb).sum()),
                  "benign_to_attack_not_td": int(c[0, 1:13].sum() + c[0, 14:].sum()),
                  "delta_benign_to_attack_not_td": int(d[0, 1:13].sum() + d[0, 14:].sum()),
                  "class": [{"name": names[str(i)], "support": int(c[i].sum()), "precision": float(p[i]),
                             "recall": float(r[i]), "f1": float(f[i]), "delta_precision": float(p[i] - bp[i]),
                             "delta_recall": float(r[i] - br[i]), "delta_f1": float(f[i] - bf[i])} for i in range(16)],
                  "largest_added_errors": [{"true": names[str(i)], "pred": names[str(j)], "delta_rows": int(d[i, j])} for i, j in worst_cells]}
        mode = dict(x.split("=", 1) for x in spec["hp"]).get("uplink_mode", "prob")
        if mode != "prob":
            detail["teacher"] = {str(rr): teacher(paths[arm], rr, mode) for rr in (1, 5, 10, 19, 20)}
        result["arms"][arm] = detail
        worst = np.argsort(f - bf)[:3]
        cells = ", ".join(f"{names[str(i)]} {f[i] - bf[i]:+.5f}" for i in worst)
        lines.append(f"| {arm} | {trace[-1]:+.6f} | {np.mean(trace[-5:]):+.6f} | {detail['delta_benign_to_attack_not_td']:+d} | {cells} |")
    lines += ["", "## Nhóm xe (mean client F1; chỉ lớp có y_true)", "", "| arm | " + " | ".join(f"{k} (n={v.sum()})" for k, v in cohorts.items()) + " |",
              "|---|" + "---:|" * len(cohorts)]
    for arm, d in result["arms"].items():
        lines.append("| " + arm + " | " + " | ".join(f"{d['cohorts'][k]['delta']:+.5f}" if k in d["cohorts"] else "—" for k in cohorts) + " |")
    if any("teacher" in d for d in result["arms"].values()):
        lines += ["", "## Teacher thực, mean 16 lớp", "", "r19 mới là teacher tiêu thụ tại r20; r20 chỉ sản xuất. Entropy thấp không chứng minh correctness.",
                  "LG arithmetic reference = mean của softmax(mean centered log p) từng xe; không phục hồi E[p] từ E[log p].", "",
                  "| arm | round | mean entropy (nat) | mean diagonal | zero codes /256 | TV vs arithmetic reference |", "|---|---:|---:|---:|---:|---:|"]
    for arm, d in result["arms"].items():
        for rr, q in d.get("teacher", {}).items():
            lines.append(f"| {arm} | {rr} | {q['mean_entropy']:.5f} | {q['mean_diagonal']:.5f} | {q['zero_codes']} | {q['mean_tv_to_arithmetic_client_distributions']:.5f} |")
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print(f"{batch_name.upper()} CM/wire diagnostics written; original gates unchanged.")


if __name__ == "__main__":
    main()
