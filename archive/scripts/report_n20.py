"""Generate N20 metric appendix and standalone curves from analyse_n20.py output."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metrics import TEN


def main():
    result = json.loads(Path("runs/n-study/analysis/analysis.json").read_text())
    arms = result["arms"]
    audit = json.loads(Path("runs/n-study/collection/audit.json").read_text())
    assert len(audit) == 8 and all(len(x["rounds"]) == 20 for x in audit)
    lines = ["# N20 — đủ10metric mọi round và16lớp cuối", "",
             "Nguồn CM đã audit và đối chiếu lại:160round, endpoint20; không best-round. [Kết luận](results-n20.md).",
             "Accuracy=precision_micro=recall_micro=recall_weighted=f1_micro trong single-labelclassification."]
    labels = json.loads(Path("knowledge/label_mapping.json").read_text())["id_to_label"]
    for rec in audit:
        name = rec["arm"]
        lines += ["", f"## {name} — cell{arms[name]['cell']}, seed{arms[name]['seed']}", "",
                  "| round | " + " | ".join(TEN) + " |", "|---:|" + "---:|" * len(TEN)]
        for row in rec["rounds"]:
            lines.append("| " + str(row["round"]) + " | " + " | ".join(f"{row[k]:.6f}" for k in TEN) + " |")
        lines += ["", "| class | test support | F1r20 |", "|---|---:|---:|"]
        final = arms[name]["final"]
        for c, (support, f1) in enumerate(zip(final["support"], final["f1_per_class"])):
            lines.append(f"| {c}: {labels[str(c)]} | {support} | {f1:.6f} |")
    Path("docs/research/metrics-n20.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    names = ["A: no real KD / soft", "B: real gate / soft", "C: no real KD / fallback", "D: real gate / fallback"]
    for col, seed in enumerate((42, 43)):
        for j, label in enumerate(names):
            curve = arms[f"N{2 * j + col}"]["curve"]
            for ax, key in zip(axes[:, col], ("f1", "timeDelay_f1", "low256_mean")):
                ax.plot([x["round"] for x in curve], [x[key] for x in curve], label=label, lw=1.8)
        for row, title in enumerate(("Pooled macro-F1 (zoom)", "timeDelay F1", "Mean client F1, train rows <256")):
            ax = axes[row, col]
            ax.set_title(f"{title} — seed {seed}")
            ax.grid(alpha=.25)
            ax.set_ylabel("F1")
            ax.set_xlim(1, 20)
            ax.set_xticks((1, 5, 10, 15, 20))
        axes[0, col].set_ylim(.88, .897)
        axes[2, col].set_xlabel("Round (cosine schedule: 50)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .07, 1, 1))
    out = Path("docs/research/figures/n20-factorial.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    # Report exact values alongside rounded prose, including p10 functional vs paired delta.
    print("N20 report:160round tables,128class rows,figure", out)
    for name, a in arms.items():
        print(name, "matched_delay", a["timeDelay_matched_seen_train_test"],
              "gradient_ratio", a["probes"]["20"]["all"]["synth_to_real_grad_ratio"])


if __name__ == "__main__":
    main()
