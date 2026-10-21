# Tài liệu dự án

Bắt đầu [CONTEXT](../CONTEXT.md) → [LESSONS](../LESSONS.md) → [continue](operations/continue.md).
[Index file](file-index.md) · [Archive](archive/README.md).

## Đang làm — final seed42, 20/10/2026

- [Handoff phiên sau](operations/continue.md): kernel/source/checkpoint, bước tiếp, lệnh và manual action.
- [Pipeline train/chấm tách](operations/final-pipeline.md): protocol, timing, costs, kiểm chứng.
- [FullDrift results đã audit](methods/full-drift/results.md): q3_zl PASS hai seed, 2-bit FAIL;
  [design](methods/full-drift/design.md), [10metric/cổng](methods/full-drift/results/report.md).
- [Lỗi r24 và recovery](archive/operations/final-recovery-2026-10-20.md): final-only fixture, subprocess exit;
  [sổ lỗi dùng tiếp](operations/notebook-errors.md).

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
| Kernel nào đang chạy, bước tiếp theo, lệnh, việc tay? | [continue](operations/continue.md) |
| Run cuối: contract train/chấm tách, timing, source, test, cost? | [final-pipeline](operations/final-pipeline.md) |
| Account Kaggle, quota, quy tắc push/W&B/stop? | [kaggle](operations/kaggle.md) |
| Lỗi kernel đã gặp và cách tránh (đọc trước khi sinh notebook)? | [notebook-errors](operations/notebook-errors.md) |
| File/code nằm đâu, tên cũ `mh/*`, nhãn V2–V10, `.compat`? | [project-layout](operations/project-layout.md) |
| Git/DVC: đích, lần backup gần nhất, cách đẩy/khôi phục/dọn? | [backup](operations/backup.md) |
| Mỗi manifest DVC: md5, số file remote/local (sinh tự động)? | [artifact-index](operations/artifact-index.md) |

Máy đọc: [registry/](operations/registry/) (map layout/alias, kiểm package 14/10 — script dùng, không sửa tay);
bằng chứng run cuối: [evidence/](operations/evidence/). Script: [scripts](../scripts/README.md). Skill (bản chính `.claude`):
[workflow](../.claude/skills/federated-experiment-workflow/SKILL.md), [Kaggle](../.claude/skills/kaggle-training-notebook/SKILL.md),
[speed](../.claude/skills/federated-training-speed/SKILL.md), [W&B](../.claude/skills/wandb-training-monitor/SKILL.md).

## Lịch sử

[Bàn giao17–20/10](archive/operations/continuation-2026-10-17.md),
[mục lục trước gọn tài liệu](archive/operations/docs-index-before-final-recovery-2026-10-20.md),
[index archive](archive/file-index.md). Snapshot cũ không thay live health/handoff hiện tại.
