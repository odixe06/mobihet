"""Report the locked generator pilot endpoint from audited artifacts; never launch training."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core.metrics import TEN, summary
from scripts import y50_common as Y

OUT = ROOT / "runs/generator-pilot20"
REPORT = ROOT / "archive/docs-20261109/research/results-generator-pilot20.md"  # study closed 09/11


def table(columns, rows):
    return "\n".join(["| " + " | ".join(map(str, columns)) + " |",
                      "|" + "---|" * len(columns),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    closure = json.loads((OUT / "closure.json").read_text())
    result = json.loads((OUT / "analysis.json").read_text())
    assert plan["endpoint"] == 20 and len(json.loads((OUT / "artifact-recheck.json").read_text())["arms"]) == 4
    labels = json.loads((ROOT / "knowledge/label_mapping.json").read_text())["id_to_label"]
    support = np.load(ROOT / "evidence/research-data/test_support.npy")
    records, paths = {}, {}
    for seed, path in Y.CONTROL.items():
        name = f"Core{seed}"
        paths[name] = path
        records[name] = [json.loads(v) for v in (path / "rounds.jsonl").read_text().splitlines()][:20]
        for row in records[name]:
            with np.load(path / f'cms_r{row["round"]:03}.npz') as z:
                assert np.array_equal(z["ids"], np.arange(5343)) and np.array_equal(z["private"].sum(2), support)
                metrics = summary(z["private"])
            for key in TEN:
                assert abs(metrics[key] - row["private/" + key]) < 1e-12
    diagnosis = {}
    for spec in plan["specs"]:
        arm, seed = spec["arm"], spec["seed"]
        assert closure[arm]["final"] and closure[arm]["round"] == 20
        path = Path(spec["artifact_root"]) / spec["slug"] / spec["slug"]
        paths[arm] = path
        records[arm] = [json.loads(v) for v in (path / "rounds.jsonl").read_text().splitlines()]
        audit_path = ROOT / "runs" / ("standard-t4" if spec["study"] == "standard" else "forest-t4") / "collection" / (arm + "-r20") / "audit.json"
        audit = json.loads(audit_path.read_text())
        for row, verified in zip(records[arm], audit["rounds"], strict=True):
            for key in TEN:
                assert abs(verified[key] - row["private/" + key]) < 1e-12
        row, control = records[arm][-1], records[f"Core{seed}"][-1]
        delta = np.array(row["private/f1_per_class"]) - control["private/f1_per_class"]
        cm = json.loads((Y.CONTROL[seed].parent / "provenance/source_manifest.json").read_text())["files"]
        common = {n: h for n, h in spec["shared_training_sha256"].items() if n.startswith("core/") or n in
                  {"scripts/run_fl.py", "scripts/run_q20.py", "scripts/run_p21_recovery.py"}}
        assert all(cm.get(n) == h for n, h in common.items())
        diagnosis[arm] = dict(seed=seed,per_class_delta=delta.tolist(),
            train_macro_delta=row["train_private/f1_macro"]-control["train_private/f1_macro"],
            test_macro_delta=row["private/f1_macro"]-control["private/f1_macro"],
            q_true_mean=row["q_true_probability_mean"],control_q_true_mean=control["q_true_probability_mean"],
            bootstrap_bytes=audit["bootstrap_bytes"],total_bytes_20=int(audit["total_bytes_20"]),
            train_seconds_mean_r3_20=float(np.mean([r["t_train"] for r in records[arm][2:]])),
            skipped_max=max(r["skipped_frac"] for r in records[arm]),shared_core_harness_matches=True)
    for rows in records.values():
        assert [r["round"] for r in rows] == list(range(1, 21))
    (OUT / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    names = ["Core42", "Core43", "Z0", "Z1", "Z2", "Z3"]
    lines = ["# Generator pilot20 — kết quả 07/11/2026", "",
        "**Đủ20round, audit PASS cả4run; standard và forest đều FAIL gate pilot. Dừng tại20, giữ Core.**",
        "[Plan đã khóa](screening-generator-pilot20.md), [runtime r2](results-generator-runtime.md), [catalog](generators.md).",
        "State `generator-pilot20-0700-20261107`: exit0/marker, kết thúc11:22:41UTC+7. Đã chạy lại auditor trên artifact thực.", "",
        "## Phạm vi và tái lập", "",
        "VeReMi NextGen v4: train+val37.361.273, own-test9.347.679 dòng/5.343xe/77inputs/16lớp; mỗi xe giữ model riêng.",
        "Temporal split; benign construction, Sybil/session leakage và các dòng ambiguous theo [dataset](../../knowledge/dataset.md).",
        "DAGSNet W1/395.408params, batch64, một epoch thật/round, synthetic64/step, floor16, Q online/count/LA; không bank/realKD.",
        "Cấu hình lấy từ config đã audit: compiled2×T4sm75, cosine50; parent r2→r3…20, không đổi lịch thành20.",
        "Core F2/F3 so cùng r20; shared core/driver source SHA khớp. Chỉ generator khác; không phải code optimization tương đương RNG.",
        "Quota tiêu hao thực N/A: chưa đối chiếu final account inventory; timing dưới chỉ đo phần train, không gồm toàn bộ startup/upload.", "",
        table(["arm", "seed", "exact kernel / version", "round"], [[s["arm"], s["seed"], closure[s["arm"]]["sessions"][-1]["state"]["ref"], "20/20"] for s in plan["specs"]]), "",
        "Artifacts: `artifacts/{standard-t4,forest-t4}/<original-run>/<original-run>`; source/manifest và log session ở thư mục cha.",
        "Config, CM từng round, teacher snapshots và recovery giữ nguyên. Full private checkpoint nằm trong output Kaggle, không coi snapshot nhỏ là state resume.", "",
        "## Endpoint20 và gate", "",
        "Gate đã khóa: macroΔ≥+.003 cả2seed, guard p10≥−.006/low256≥−.003/timeDelay≥−.0035/benign≥−.0005/absent≥−.01 từng seed.",
        table(["metric r20", *names], [[k, *[f'{records[n][-1]["private/"+k]:.9f}' for n in names]] for k in TEN]), "",
        "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro về mặt định nghĩa; có thể lệch rounding float64.", "",
        table(["arm", "macroΔ", "p10Δ", "low256Δ", "timeDelayΔ", "benignΔ", "absentΔ", "gate vi phạm"],
              [[a["arm"], *[f'{a["delta"][k]:+.6f}' for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent")],
                ", ".join((["macro gain"] if a["delta"]["f1_macro"] < plan["macro_gain"] else []) + [k for k, v in plan["guards"].items() if a["delta"][k] < v])]
               for group in result.values() for a in group["arms"]]), "",
        "## Chẩn đoán thất bại", "",
        "Standard: suddenConstantSpeed giảm−.033132/−.029791; đóng góp Δ/16 của riêng lớp này lớn hơn toàn bộ macro giảm.",
        "Các lớp còn lại bù nhẹ; absent accuracy tăng+.005788/+.014148, chưa đủ đạt macro gate. Seed43 còn vi phạm guard timeDelay nhẹ.",
        "Forest: suddenConstantSpeed giảm≈.07, accelerationMultiplication≈.037, constantSpeedOffset≈.018 ở cả2seed; p10/low256/benign/absent cùng xấu.",
        "Train macro giảm chỉ≈.0005 ở cả hai generator, so own-test giảm≈.002 standard/≈.0097 forest. Đây là dấu hiệu sai lệch generalization,",
        "chưa chứng minh nguyên nhân do marginal, joint ngoài forest hay temporal shift. Q true-class mean vẫn≈.958–.959; không bằng chứng teacher sập.",
        "PairTV/TSTR local tốt không bảo đảm utility personalized. Không tăng thời lượng, đổi KD weight hoặc chọn cặp/HP theo official test để cứu gate.", "",
        table(["lớp", "test support", "Z0 ΔF1", "Z1 ΔF1", "Z2 ΔF1", "Z3 ΔF1"],
              [[labels[str(i)], int(support.sum(0)[i]), *[f'{diagnosis[n]["per_class_delta"][i]:+.6f}' for n in names[2:]]] for i in range(16)]), "",
        "## Byte, timing và bằng chứng", "",
        table(["arm", "bootstrap B/xe", "tổng20 B/xe", "train mean r3–20 (s)", "max skip"],
              [[n, diagnosis[n]["bootstrap_bytes"], diagnosis[n]["total_bytes_20"], f'{diagnosis[n]["train_seconds_mean_r3_20"]:.2f}', diagnosis[n]["skipped_max"]] for n in names[2:]]), "",
        "Q1088B/xe/round hai chiều; Core/standard bootstrap1000439B, forest1704951B (+704512B/xe). Timing quan sát khác session, không là benchmark causal speedup.",
        "Recheck: `runs/generator-pilot20/artifact-recheck.json`; `analysis.json`, `diagnosis.json`; method collection `Z*-r20/{audit,diagnostic-audit}.json`.",
        "Đủ IDs/support/CM→10metric/LR/source/config/teacher-chain/snapshot finite/round guards; resume2rank đúngr3, npy verified, Inductor tar restore exit0.",
        "Không full private bank reconstruction, prediction→CM replay, T4 straight-vs-resume bitexact hoặc full-content dataset fingerprint.",
        "Official test đã dùng lặp, hai seed không là significance/fresh confirmation; kết luận chỉ cho endpoint20/cấu hình này, không kết luận mọi copula/forest thất bại.", "",
        "## Đồ thị", "",
        "![Macro theo round và lớp r20](../../runs/generator-pilot20/pilot20.png)", "",
        "Trái: toàn bộ r1–20; phải: ΔF1 lớp tại endpoint20, giữ riêng từng seed. Không chọn best round.", "",
        "![CM r20 chuẩn hóa theo true class](../../runs/generator-pilot20/confusion20.png)", "",
        "CM gộp đủ own-test, chuẩn hóa mỗi hàng; ID lớp tra bảng trên. Đây không phải CM trên mẫu synthetic.", "",
        "## Bước tiếp", "",
        "Giữ Core. Trước cơ chế tiếp, kiểm synthetic-vs-real train/val theo lớp11/1/3, quan hệ ngoài cạnh và các chế độ điều kiện;",
        "chọn thống kê/cơ chế từ train/val, ghi problem→mechanism→dấu hiệu bác bỏ trước pilot mới. Chưa xác định nguyên nhân đủ để launch ứng viên khác.",
        "Pilot hiện đã dừng đúng20; không tự nối50.", "", "## Đủ10metric mỗi round", ""]
    for name in names:
        lines += [f"### {name}", "", table(["round", *TEN], [[r["round"], *[f'{r["private/"+k]:.9f}' for k in TEN]] for r in records[name]]), ""]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for name in names:
        axes[0].plot(range(1, 21), [r["private/f1_macro"] for r in records[name]], label=name)
    axes[0].set(xlabel="Round (cosine50)", ylabel="Own-test pooled macro-F1", title="Pilot endpoint20")
    axes[0].legend(); axes[0].grid(alpha=.25)
    for name in names[2:]: axes[1].plot(range(16), diagnosis[name]["per_class_delta"], marker=".", label=name)
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_xticks(range(16))
    axes[1].set(xlabel="Class ID", ylabel="F1 minus same-seed Core at r20", title="Endpoint per-class deltas")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "pilot20.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for name, ax in zip(names[2:], axes.flat):
        with np.load(paths[name] / "cms_r020.npz") as z:
            cm = z["private"].sum(0)
        image = ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        ax.set(title=name, xlabel="Predicted class", ylabel="True class")
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
    fig.colorbar(image, ax=axes, label="Share of true class", shrink=.8)
    fig.savefig(OUT / "confusion20.png", dpi=140); plt.close(fig)
    REPORT.write_text("\n".join(lines))
    assert all(k in REPORT.read_text() for k in TEN)
    print("PILOT20_REPORT_PASS", len(records) * 20, "round metric rows", flush=True)


if __name__ == "__main__":
    main()
