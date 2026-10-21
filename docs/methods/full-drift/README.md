# FullDrift — Gaussian/proxy distillation cho private IDS

**Hiện tại20/10:** q3_zl + gkd/la/kdpx PASS hai seed; 2-bit loại. Đang triển khai final seed42:
hai arm×20/50/100client,50round, train/chấm tách. Source r25 sửa startup r24; r24 chưa có round final đã commit.

- [Handoff hiện tại](../../operations/continue.md): exact kernels/quota/checkpoint và bước phiên sau.
- [Pipeline cuối](../../operations/final-pipeline.md): contract50round, session4/2/2, evaluator từng round,
  runtime/cost/tests và live health.
- [Thiết kế/cổng](design.md): cơ chế, đối chứng, giả thuyết và cổng đã khóa.
- [Kết quả đã audit](results.md): bảng theo vòng; [report](results/report.md), [10metrics](results/metrics-all.csv),
  [gates](results/gates.json), [per-class](results/per-class-f1.csv).
- [Build/source/scripts/notebooks](../../../methods/full-drift/builds/fd-probe).
- [Lịch sử build/probe](../../archive/operations/full-drift-build-history-before-final-2026-10-20.md),
  [chẩn đoán gốc](../../archive/research/plateau-diagnosis-2026-10-15.md).

Pilot dùng holdout nội bộ để chọn phương pháp; protocol cuối train toàntrain80%, chấm toàntest20% mọiround.
Không dùng test chọn HP/checkpoint; headline r50. Freeze source đã push, resume từ round committed đã audit.
