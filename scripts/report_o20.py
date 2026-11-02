"""Render verified O20 subset/full results, all-round metrics and replay curves."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metrics import TEN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--early", action="store_true"); args = ap.parse_args()
    base = Path("runs/o-study/early-collect-20261101" if args.early else "runs/o-study")
    result = json.loads((base / "analysis/analysis.json").read_text()); arms = result["arms"]
    suffix = "-early" if args.early else ""
    scope = "6/8run đã audit; chưa đủ mọi cặp seed" if args.early else "8/8run đã audit đủ20round"
    files = sum(x["files"] for x in result["artifact_verification"])
    size = sum(x["bytes"] for x in result["artifact_verification"])
    lines = [f"# O20 — {'audit sớm và phân tích một phần batch' if args.early else 'kết quả teacher replay'}", "",
        f"**{scope}.** {files}file/{size:,}B; SHA20snapshot/run, mọi file source, scientificclosure so N và20CM→10metric/support đã đối chiếu độc lập.",
        "Fullaudit còn gồm finite/probe/externalhash/teacherlag/rawQ/target. Không privatebank/predictionreplay/fullcontentfingerprintdataset hoặc DPproof.",
        "[Thiết kế khóa](screening-o20.md), [cơ chế/reference đã thử](method.md). R0=fullQ/noReal, R1=fullQ/realgate, R2=uniform/noReal, R3=permuted/noReal.",
        "Own-test5343xe, endpoint20/cosine50, W1; Q nguồn train-only N0/N1, dùng lag1, không livefeedback từ Q_live.", "",
        "## Own-test tại round20", "", "| arm / cell / seed | pooledF1 | mean clientF1 | p10 | mean xe<256 | timeDelayF1 | ΔF1r15→20 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, a in arms.items():
        f = a["final"]
        lines.append(f'| {name}/{a["cell"]}/{a["seed"]} | {f["f1"]:.6f} | {f["mean"]:.6f} | {f["p10"]:.6f} | {f["low256_mean"]:.6f} | {f["timeDelay_f1"]:.6f} | {a["delta_f1_15_20"]:+.6f} |')
    lines += ["", "## Đối chiếu cùng teacher nguồn", "", "Chênh tuyệt đối endpoint20. Other15=mean pooledF1 của15lớp trừ timeDelay; không headline mới.", "",
        "| contrast / seed | ΔpooledF1 | Δmean client | Δp10 | Δmean xe<256 | ΔtimeDelay | Δother15 | xe tăng/giảm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in result["effects"]:
        lines.append(f'| {e["contrast"]} {e["label"]}/{e["seed"]} | ' + " | ".join(f'{e[k]:+.6f}' for k in ("f1", "mean", "p10", "low256_mean", "timeDelay_f1", "other15_f1")) + f' | {e["better"]}/{e["worse"]} |')
    if not args.early:
        lines += ["", "### Điều phép kiểm chứng đã làm rõ", "",
            "1. **Tiếp nhận realKD vẫn có hại cục bộ khi giữ cùng Q:** R1−R0 timeDelay=−.056310/−.060312,",
            "   other15=+.005260/+.005244; cả15lớp còn lại đều tăng pooledF1 ở cảhai seed. Pooled tăng+.001412/+.001147,",
            "   nhưng meanxe<256 vàp10 chỉ tăng seed42. Không tắt/tăng KD đồng loạt hoặc chốt winner cho xe yếu.",
            "2. **Teachertrajectory cũng quan trọng:** O2/O3 so livegateN2/N3 pooled tăng+.001890/+.001882,",
            "   timeDelay tăng+.045720/+.045302, other15 giảm khoảng.00103/.00101. Cùng clientloss nhưng lịch teacher nguồn noReal khác livefeedback.",
            "   Kết quả phù hợp feedback/teachertrajectory làm trầm trọng đánh đổi; không là phân rã trung gian nhân quả hoặc tỷ lệ lỗi do feedback.",
            "   Q19 nguồn13 có Qyy≈.707/.705, còn Q_live do chính O2/O3 collect≈.595/.593 dù được lịch teacher nguồn mạnh hơn.",
            "   Đây là train-onlyconfidence/động lực học, chưa phải ước lượng teacherquality trên tương lai.",
            "3. **Không nên bỏ toàn bộ offdiagonal:** uniform giảm pooled.001146/.000943, permutation giảm.001799/.001451.",
            "   Permutation giữ diagonal/entropy mà vẫn giảm: có bằng chứng utility của quan hệ tên lớp trong đúng phép replay này,",
            "   chưa mọi permutation/architecture. Uniform lại tăng meanxe<256+.004825/+.001843 và timeDelay≈+.011/+.011;",
            "   không thể suy fullQ hữu ích đồng đều cho mọi xe/lớp hoặc diagonal-only đủ thay thế.",
            "4. **Chưa plateau toàn hệ thống:** R1 vẫn tăng r15→20+.003770/+.003842; không hứa mỗi round tăng hoặc ngoài20round.",
            "   Gap train/test timeDelay còn lớn ở xe≥4096: R1train≈.749/.747 so test≈.177/.176.",
            "   Thiếu mẫu/lớp không giải thích hết, cần validation theo thời gian chưa fit trước khi dùng confidence làm serverqualityweight.", "",
            "Ưu tiên tiếp: giữ cấu trúc Q đã có utility, kiểm tiếp nhận KD có chọn lọc theo thông tin train-only và feedbackserver;",
            "gate liên tục vẫn là giả thuyết, chưa chứng minh giải quyết crossclasshại hoặc chỉ giảm lossstrength.",
            "Không tăng packet/đổi generator hoặc triển khai cacheteacher như liveFL từ kết quả này. [Registry/falsifier tiếp](diagnostic-next.md)."]
    lines += ["", "FullQ vs uniform thay cả entropy; fullQ vs fixedpermutation giữ diagonal/entropy nhưng phá tên lớp.",
        "UniformO giữ diagonal/chia đều offdiagonal, khác uniformM12 trộn hard/softρ=.5; không cộng gain hai can thiệp.",
        "Không realKD ở uniform/permuted nên argmax sau biến đổi không mở gate. Permutation chỉ một phép đã khóa, chưa kiểm nhiều permutation.",
        "Sameclient giữa seed có tương quan; hai seed chưa significance. Không gọi pooledgain là mọi client được lợi.", "",
        "## Tripwire replaycontrol so N0/N1", "", "| control | Δpooled | ΔtimeDelay | Δp10 | Δmean xe<256 | giới hạn khóa |", "|---|---:|---:|---:|---:|---|"]
    for name, a in arms.items():
        if "replay_control_tripwire" not in a: continue
        t = a["replay_control_tripwire"]; delta = t["delta_vs_N"]
        lines.append(f'| {name} | ' + " | ".join(f'{delta[k]:+.6f}' for k in ("f1", "timeDelay_f1", "p10", "low256_mean")) + f' | {"PASS" if t["pass"] else "FAIL; hạn chế kết luận"} |')
    lines += ["", "Giới hạn đã khóa trước O: |Δpooled|≤.001, ba metric còn lại≤.01. Không CI/noisebound/utilitythreshold; không bitexact giữa T4sessions.", "",
        "## Xe nhỏ và tổng quát hóa", "", "Accuracy điều kiện trên1555absent-localtestrows của848xe<256; không macroF1 mới.", "",
        "| arm | absentaccuracy xe<256 | matched trainrecall timeDelay | matched testrecall timeDelay | cross-client34 meanF1 (phụ) |",
        "|---|---:|---:|---:|---:|"]
    for name, a in arms.items():
        m = a["timeDelay_matched_seen_train_test"]["all"]
        absent = a["final"]["cohorts"][0]["by_local_support"]["absent"]["accuracy"]
        lines.append(f'| {name} | {absent:.6f} | {m["train_recall"]:.6f} | {m["test_recall"]:.6f} | {a["external"]["20"]["mean_f1"]:.6f} |')
    lines += ["", "Matchedrecall cân theo supporttest481868rows cùng xe/lớp đã thấy; train+val đãfit, không independentvalidation.",
        "Gap không tự tách overfit/temporalshift/featureoverlap. External34 chỉ evaluator, không vào train/Q/bootstrap và không dùng chọnHP.", "",
        "## Byte và thời gian", "", "Mọi run: packet-equivalent3333504B/xe/20round; offlinepreload3402795B gồm3311744bootstrap+10880uplink+80171Binputmodule.",
        "Không cộng lại cached downlink hoặc cộng cả decoded61440B và cùng module như hai lượt tải. Replay là counterfactual, chưa liveFL/contactcostproof.", "",
        "| arm | train s/r r2–20 | train tổng s | eval tổng s | diagnostic tổng s | notebooklog elapsed s |", "|---|---:|---:|---:|---:|---:|"]
    for name, a in arms.items():
        lines.append("| " + name + " | " + " | ".join(f'{a[k]:.1f}' for k in ("train_seconds_mean_r2_20", "train_seconds_total", "eval_seconds_total", "diagnostic_seconds_total", "notebook_log_elapsed_seconds")) + " |")
    lines += ["", "Giây walljob2×T4, không latency mỗi xe. Notebookelapsed gồm setup/compile/IO/conversion, chưa queue/collectlocal.",
        f'Đủ10metric mọi round và16lớp cuối: [phụ lục](metrics-o20{suffix}.md). Officialtest đã đọc nhiều batch; không freshconfirmatorytest/best-round/HPtuning.',
        "Chưa r>20/mixedarchitecture/DP proof; không chọn phương pháp tổng từ replay. Artifact mới chưa DVCbackup, giữ local.", "",
        f'![O20 replay own-test curves](figures/o20{suffix}-curves.png)', "",
        "Nguồn phân tích: `" + str(base / "analysis/analysis.json") + "`; tái tạo `python scripts/analyse_o20.py" + (" --early" if args.early else "") + "`, rồi `python scripts/report_o20.py" + (" --early" if args.early else "") + "`."]
    Path(f"docs/research/results-o20{suffix}.md").write_text("\n".join(lines) + "\n")
    appendix = ["# O20 — đủ10metric mọi round và16lớp cuối", "", "Nguồn actualCM đã kiểm độc lập, endpoint20.",
        "Accuracy=precision_micro=recall_micro=recall_weighted=f1_micro trong single-labelclassification."]
    batch = json.loads((base / "batch.json").read_text())
    for spec in batch:
        location = base / "audits" / spec["arm"] if args.early else base / "collection"
        audit = next(a for a in json.loads((location / "audit.json").read_text()) if a["arm"] == spec["arm"])
        appendix += ["", f'## {spec["arm"]} — seed{audit["hp"]["seed"]}', "", "| round | " + " | ".join(TEN) + " |", "|---:|" + "---:|" * len(TEN)]
        for r in audit["rounds"]: appendix.append("| " + str(r["round"]) + " | " + " | ".join(f"{r[k]:.6f}" for k in TEN) + " |")
        appendix += ["", "| class ID | test support | F1r20 |", "|---:|---:|---:|"]
        for c, (n, f) in enumerate(zip(r["support"], r["f1_per_class"])): appendix.append(f"| {c} | {n} | {f:.6f} |")
    Path(f"docs/research/metrics-o20{suffix}.md").write_text("\n".join(appendix) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, seed in enumerate((42, 43)):
        for cell, label in enumerate(("R0 full/noReal", "R1 full/realgate", "R2 uniform/noReal", "R3 permuted/noReal")):
            name = f"O{cell * 2 + col}"
            if name not in arms: continue
            curve = arms[name]["curve"]
            for row, key in enumerate(("f1", "timeDelay_f1")):
                axes[row, col].plot([p["round"] for p in curve], [p[key] for p in curve], label=label, color=f"C{cell}")
        for row, title in enumerate(("Pooled macro-F1 (zoom)", "timeDelay F1")):
            axes[row, col].set_title(f"{title}, seed {seed}"); axes[row, col].grid(alpha=.25)
            axes[row, col].set_ylabel("F1"); axes[row, col].set_xlim(1, 20)
        axes[0, col].set_ylim(.88, .9); axes[1, col].set_xlabel("Round (cosine50)")
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(f"docs/research/figures/o20{suffix}-curves.png", dpi=160); plt.close(fig)
    print("O20_REPORT_RENDERED", scope)


if __name__ == "__main__": main()
