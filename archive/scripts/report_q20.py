"""Render all320 Q20 metric rows, endpoint/cost tables and standalone figures."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN

BASE = ROOT / "runs/q-study/analysis"
DOCS = ROOT / "docs/research"
LABELS = ["stream", "query", "mixed class", "mixed input", "input sqrt", "real KD", "margin", "strength"]


def main():
    result = json.loads((BASE / "analysis.json").read_text())
    assert hashlib.sha256((ROOT / "runs/q-study/closure.json").read_bytes()).hexdigest() == result["closure_sha256"]
    arms = result["arms"]
    rows = list(csv.DictReader((BASE / "metrics-all-rounds.csv").open()))
    assert len(rows) == 320
    lines = ["# Q20 — đủ10metric own-test mỗi round", "",
             "16run PASS_BOUNDED ×20round,5343xe/W1/cosine50; headline r20 đã khóa. Không chọn besttest.",
             "Mỗi xe chấm own-test, pooled đủ9347679row/r. Số dạng0–1, không phần trăm.",
             "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro.", "",
             "[Kết quả](results-q20.md), [byte/time](cost-q20.md); fullprecision `runs/q-study/analysis/metrics-all-rounds.csv`."]
    for i in range(16):
        name = f"Q{i}"; arm = arms[name]
        selected = [r for r in rows if r["arm"] == name]
        assert [int(r["round"]) for r in selected] == list(range(1, 21))
        lines += ["", f"## {name} — {arm['mode']}, seed{arm['seed']}", "",
                  "| round | " + " | ".join(TEN) + " |", "|---|" + "---|" * 10]
        for row in selected:
            for key in TEN:
                assert np.isfinite(float(row[key])) and 0 <= float(row[key]) <= 1
            lines.append(f"| {row['round']} | " + " | ".join(f"{float(row[k]):.6f}" for k in TEN) + " |")
    lines += ["", "## Phân phối F1 từngxe tại endpoint20", "",
              "MacroF1/xe tính trên lớp có trong y_true; mean/p10/std không cân theo số testrow.", "",
              "| arm | mean | p10 | std | median | min | mean xe<256 | mean xe>=4096 |",
              "|---|---|---|---|---|---|---|---|"]
    for i in range(16):
        name = f"Q{i}"; final = arms[name]["final"]
        lines.append("| " + name + " | " + " | ".join(f"{final[k]:.6f}" for k in ["mean", "p10", "std", "median", "min", "low256_mean", "ge4096_mean"]) + " |")
    (DOCS / "metrics-q20.md").write_text("\n".join(lines) + "\n")
    endpoint = ["| cell | pooled F1 s42/s43 | mean xe<256 s42/s43 | p10 s42/s43 | total MB/xe20r |", "|---|---|---|---|---:|"]
    for cell, label in enumerate(LABELS):
        pair = [arms[f"Q{2 * cell + offset}"] for offset in (0, 1)]
        fields = ["/".join(f"{a['final'][k]:.6f}" for a in pair) for k in ["f1", "low256_mean", "p10"]]
        endpoint.append(f"| {label} | " + " | ".join(fields) + f" | {pair[0]['bytes']['total'] / 1e6:.6f} |")
    (BASE / "endpoint-table.md").write_text("\n".join(endpoint) + "\n")
    contrasts = ["| contrast | seed | delta pooled | delta mean<256 | delta p10 | delta timeDelay | xe giảm | xe<256 tăng |",
                 "|---|---|---|---|---|---|---|---|"]
    for e in result["effects"]:
        contrasts.append(f"| {e['label']} | {e['seed']} | " + " | ".join(f"{e[k]:+.6f}" for k in ["f1", "low256_mean", "p10", "timeDelay_f1"])
                         + f" | {e['worse_clients']} | {e['low256_better_fraction']:.3%} |")
    (BASE / "contrast-table.md").write_text("\n".join(contrasts) + "\n")
    costs = [{"arm": f"Q{i}", "mode": arms[f"Q{i}"]["mode"], "seed": arms[f"Q{i}"]["seed"],
              **arms[f"Q{i}"]["bytes"], **arms[f"Q{i}"]["timing"]} for i in range(16)]
    with (BASE / "costs.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(costs[0])); writer.writeheader(); writer.writerows(costs)
    lines = ["# Q20 — byte và thời gian", "",
        "Payload mỗi xe, gồm bootstrap + upload/download20round. Chưa transport/ACK/crypto/retransmission/contactproof.",
        "Thời gian toàn run mô phỏng5343xe/2T4, không latency từngxe hoặc measured quota/GPUkernelhour.", "",
        "| arm | bootstrap up | bootstrap down | round up tổng | round down tổng | tổng B/xe |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in costs:
        lines.append("| " + row["arm"] + " | " + " | ".join(str(int(row[k])) for k in ["bootstrap_up", "bootstrap_down", "rounds_up", "rounds_down", "total"]) + " |")
    lines += ["", "Scoring/server đã nằm trong train, không cộng lần hai. Unattributed gồm setup/compile/I/O/finalization và lệch mốc log, chưa phân rã riêng.", "",
              "| arm | train s | eval s | diagnostic s | log elapsed s | unattributed s | scoring local rank0 s (trong train) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in costs:
        lines.append("| " + row["arm"] + " | " + " | ".join(f"{row[k]:.2f}" for k in ["t_train", "t_eval", "t_diagnostics", "notebook_log_elapsed_seconds", "unattributed_seconds", "scoring_seconds_local"]) + " |")
    lines += ["", "Q không fitproxyserver. Chưa có timer riêng cho class/inputaggregation hoặc vận chuyển thật; server_seconds không ghi riêng không có nghĩa server miễn phí.",
              "S0/S0query giữ classcodec cũ: bodyfp16 decode, headerbytes theo quy ước; literalCRCheader ở inputpacket và classpacket S1 trở đi.",
              "Không gọi toàntransport đã wireproof. [Kết quả](results-q20.md), [đủ10metric](metrics-q20.md)."]
    (DOCS / "cost-q20.md").write_text("\n".join(lines) + "\n")
    figures = DOCS / "figures"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(1, 21)
    for cell, label in enumerate(LABELS):
        pair = [arms[f"Q{2 * cell + i}"] for i in (0, 1)]
        for ax, metric in zip(axes[0], ["f1", "low256_mean"]):
            values = np.asarray([[p[metric] for p in arm["curve"]] for arm in pair])
            ax.plot(x, values.mean(0), label=label, color=plt.cm.tab10(cell))
            ax.fill_between(x, values.min(0), values.max(0), color=plt.cm.tab10(cell), alpha=.10)
            ax.set_xlabel("Round (cosine50)"); ax.grid(alpha=.2)
    axes[0, 0].set(title="Own-test pooled macro F1", ylabel="F1")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower right")
    axes[0, 1].set(title="848 clients with <256 train rows", ylabel="Mean client F1")
    contrast_labels = list(dict.fromkeys(e["label"] for e in result["effects"]))
    for i, label in enumerate(contrast_labels):
        selected = [e for e in result["effects"] if e["label"] == label]
        axes[1, 0].scatter([e["f1"] for e in selected], [e["low256_mean"] for e in selected], label=label, color=plt.cm.tab10(i))
    axes[1, 0].axhline(0, c="grey", lw=.7); axes[1, 0].axvline(0, c="grey", lw=.7)
    axes[1, 0].set(title="Paired contrasts: each dot = one seed", xlabel="Delta pooled F1", ylabel="Delta mean F1, clients<256")
    axes[1, 0].legend(fontsize=6, loc="best"); axes[1, 0].grid(alpha=.2)
    for offset, seed in [(0, 42), (1, 43)]:
        values = [arms[f"Q{2 * cell + offset}"]["final"]["cohorts"][0]["by_local_support"]["absent"]["accuracy"] for cell in range(8)]
        axes[1, 1].bar(np.arange(8) + (offset - .5) * .35, values, width=.35, label=f"seed{seed}")
    axes[1, 1].set_xticks(range(8), LABELS, rotation=30, ha="right")
    axes[1, 1].set(title="1555 absent-class test rows, clients<256", ylabel="Accuracy")
    axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.2)
    fig.suptitle("Q20: fixed endpoint20; bands = two-seed range, not confidence intervals")
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(figures / ("q20-summary." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    boot = np.array([arms[f"Q{2 * c}"]["bytes"]["bootstrap_up"] + arms[f"Q{2 * c}"]["bytes"]["bootstrap_down"] for c in range(8)]) / 1e6
    total = np.array([arms[f"Q{2 * c}"]["bytes"]["total"] for c in range(8)]) / 1e6
    axes[0].bar(range(8), boot, label="Bootstrap")
    axes[0].bar(range(8), total - boot, bottom=boot, label="Two-way20round")
    axes[0].set(ylabel="Payload MB/client", title="Communication includes setup"); axes[0].legend()
    for offset, seed in [(0, 42), (1, 43)]:
        values = [arms[f"Q{2 * c + offset}"]["timing"]["notebook_log_elapsed_seconds"] / 3600 for c in range(8)]
        axes[1].bar(np.arange(8) + (offset - .5) * .35, values, width=.35, label=f"seed{seed}")
    axes[1].set(ylabel="Notebook log elapsed hours", title="Full5343 clients / 2 T4, not per-car latency"); axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(8), LABELS, rotation=30, ha="right"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "q20-cost.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    print("Q20_TABLES_FIGURES_DONE", flush=True)


if __name__ == "__main__": main()
