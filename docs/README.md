# Tài liệu dự án

Bắt đầu [CONTEXT](../CONTEXT.md) → [LESSONS kỹ thuật](../LESSONS.md) + [bài học vận hành](operations/lessons.md) → [continue](operations/continue.md).
[Index file](file-index.md) · [Archive](archive/README.md).

## Hiện hành — 23/10/2026 ~03:55Z

- [Bàn giao sống — continue](operations/continue.md): đang chạy gì, bước tiếp, lệnh, việc tay (file duy nhất; snapshot cũ ở archive).
- [Report final đủ 6 chuỗi × 50 round](methods/full-drift/results/final/report.md): đủ 10 metric, per-class, đồ thị, cost;
  [VERIFIED](methods/full-drift/results/final/VERIFIED.json), [provenance](methods/full-drift/results/final/provenance.json).
- Synthetic mới nhất: [P37 synthetic đổi trọng số full model + tài liệu](research/methods/synthetic-weight-dynamics-2026-10-23.md) →
  [P36 Project48 + copula hai seed → vòng 26 FAIL](research/methods/teacher-support-two-seed-2026-10-22.md) →
  [P35 trần năng lực bộ sinh](research/methods/support-capacity-oracle-2026-10-22.md) →
  [P34 bộ sinh giữ cấu trúc (FAIL)](research/methods/structured-support-generator-2026-10-22.md) →
  [P33 client học gì từ support](research/methods/client-learning-diagnosis-2026-10-22.md) →
  [P32 support realism](research/methods/synthetic-support-realism-2026-10-22.md) →
  [P29–31](research/methods/synthetic-gradient-diagnosis-2026-10-22.md) → [P23–28](research/methods/synthetic-learning-recovery-2026-10-22.md).
- [Pipeline train/chấm tách](operations/final-pipeline.md); [FullDrift results đã audit](methods/full-drift/results.md),
  [design](methods/full-drift/design.md); [sổ lỗi kernel](operations/notebook-errors.md).

## Phương pháp và bằng chứng

- [Danh mục đề xuất](proposals.md): trạng thái/căn cứ/cổng; [mỗi phương pháp theo tên](research/methods/README.md).
- [Hồ sơ method/build/source](methods/README.md); [bài học dùng tiếp](lessons/README.md).
- [Baseline H1/H2](methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md),
  [Hybrid full](methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md),
  [AntiMem/QProxy](methods/anti-memorization/results.md).
- Adaptation FAIL không rerun y nguyên; source/codec PASS không thay utility/R1/R2 proof.

## Protocol và kiến thức nền

- [Protocol](research/protocol.md), [luồng thông tin FL](research/federated-information-flow.md).
- [So sánh fl_dagsnet_veremi](research/comparison-fl-dagsnet-veremi.md), [computational cost](research/computational-cost.md).
- [Dataset/architecture/runtime](../knowledge/README.md); dữ kiện cố định không trộn với hyperparameter của method.

## Vận hành (`operations/`) — mỗi file một chủ đề

| câu hỏi | file |
|---|---|
| Đang chạy gì, bước tiếp theo, lệnh, việc tay? | [continue](operations/continue.md) |
| Run cuối: contract train/chấm tách, timing, source, test, cost? | [final-pipeline](operations/final-pipeline.md) |
| Account Kaggle, quota, quy tắc push/W&B/stop? | [kaggle](operations/kaggle.md) |
| Bài học vận hành (Kaggle/pipeline, collector/máy local, Git/DVC/cloud)? | [lessons](operations/lessons.md) |
| Lỗi kernel đã gặp và cách tránh (đọc trước khi sinh notebook)? | [notebook-errors](operations/notebook-errors.md) |
| File/code nằm đâu, tên cũ `mh/*`, nhãn V2–V10, `.compat`? | [project-layout](operations/project-layout.md) |
| Git/DVC: đích, lần backup gần nhất, cách đẩy/khôi phục/dọn? | [backup](operations/backup.md) |
| Mỗi manifest DVC: md5, số file remote/local (sinh tự động)? | [artifact-index](operations/artifact-index.md) |
| Output Kaggle của archive Build7–21 còn tải được không? | [kaggle-archive-output-index](operations/kaggle-archive-output-index.md) |
| Chuyển Google Drive → OneDrive 22/10 (đang chạy, phiên Codex)? | [backup](operations/backup.md) |

Máy đọc: [registry/](operations/registry/) (map layout/alias — script dùng, không sửa tay). Bằng chứng run cuối 20/10:
[archive/operations/evidence](archive/operations/evidence/final-2026-10-20/). Script: [scripts](../scripts/README.md).
Skill (bản chính `.claude`, mirror `.agents`): [đọc/ghi context](../.claude/skills/project-context/SKILL.md),
[workflow](../.claude/skills/federated-experiment-workflow/SKILL.md), [Kaggle](../.claude/skills/kaggle-training-notebook/SKILL.md),
[speed](../.claude/skills/federated-training-speed/SKILL.md), [W&B](../.claude/skills/wandb-training-monitor/SKILL.md).

## Lịch sử

[Bàn giao 22/10 report final + P29–31](archive/operations/continuation-2026-10-22.md),
[synthetic trước compact 22/10](archive/operations/continuation-synthetic-2026-10-22.md),
[bàn giao 17–20/10](archive/operations/continuation-2026-10-17.md), [nhật ký run cuối 20–21/10](archive/operations/continuation-2026-10-20-21.md),
[mục lục trước gọn tài liệu](archive/operations/docs-index-before-final-recovery-2026-10-20.md),
[index archive](archive/file-index.md). Snapshot cũ không thay live handoff hiện tại.
