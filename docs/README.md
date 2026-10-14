# Mục lục tổng quan dự án

Đây là điểm đọc chính: [CONTEXT](../CONTEXT.md) giữ trạng thái/lưu ý vận hành; [LESSONS](../LESSONS.md) giữ bài học chung.
[Index đang làm](file-index.md) · [Index lịch sử](archive/file-index.md) · [Cấu trúc và alias](operations/project-layout.md).

## 1. Toàn bộ phương án

- [Một danh mục đề xuất](proposals.md): 26 nhóm +recipe EMA; đã kiểm/chưa kiểm, PASS/FAIL có phạm vi, link thiết kế/build/kết quả.
- [Hồ sơ phương pháp](methods/README.md): README ngắn → căn cứ/cơ chế → các build/source/artifact.
- [Bài học dùng tiếp](lessons/README.md); [hồ sơ lịch sử](archive/README.md) giữ báo cáo và nguồn gốc số đo.
- [Dự báo performance](research/performance-feasibility-2026-10-13.md): căn cứ trước mở pilot.

## 2. Những hướng đang được tiếp tục

| phương pháp | thiết kế / build / kết quả |
|---|---|
| Gaussian class moments | [Hồ sơ](methods/gaussian-class-moments/README.md), [full H1/H2](methods/shared-baselines/builds/federated-baselines-full/README.md) |
| DAGSNet mutual distillation | [Hồ sơ](methods/proxy-mutual-distillation/README.md), [10metrics cuối](methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md) |
| Gaussian + proxy Hybrid | [Hồ sơ](methods/gaussian-proxy-hybrid/README.md), [full build](methods/gaussian-proxy-hybrid/builds/hybrid-full/README.md), [10metrics cuối](methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md) |
| BoundaryCells counts | [Hồ sơ](methods/boundary-cell-counts/README.md), [build/calibration/pilot](methods/boundary-cell-counts/builds/boundary-cell-pilot/README.md) |
| ResponseRisk augmentation | [Thiết kế](methods/receiver-response-consistency/design.md), [build local probe](methods/receiver-response-consistency/builds/receiver-augmentation/README.md) |
| EMA deployment recipe | [Thiết kế/cổng](methods/ema-deployment-recipe/design.md); chưa source/pilot utility |

Các adaptation FAIL khác đều có thiết kế/report/source qua [danh mục đề xuất](proposals.md); không rerun y nguyên.

## 3. Vận hành, lỗi và khả năng tái lập

- [Bàn giao hiện tại](operations/reorganization-2026-10-14.md): exact kernel/version, checkpoint, quota, việc còn thiếu.
- [Sổ lỗi kernel](operations/notebook-errors.md): nguyên nhân, output dùng được, preflight ngăn lặp.
- [Launch plan/quota](operations/launch-batch-2026-10-14.json), [status snapshot](operations/kernel-status-current.json).
- [Layout hiện tại](operations/workspace-layout.json), [tên các đợt](operations/experiment-names.md), [manifest gốc](operations/layout-migration-2026-10-14.json), [Git/DVC verification](operations/layout-verification-2026-10-14.json).
- [Backup/restore](operations/backup.md), [lịch sử collect trước đợt này](operations/collection-2026-10-14.md).
- [Scripts dùng chung](../scripts/README.md); logs đúng build, không trộn với kết quả khoa học.
- [Package chung và module riêng build](operations/package-ownership.md): ownership, source canonical, alias và fingerprint.

## 4. Protocol và kiến thức nền

- [Protocol](research/protocol.md), [dataset](../knowledge/dataset.md), [architecture](../knowledge/architecture.md), [knowledge hub](../knowledge/README.md).
- [Kiến thức kế thừa](archive/legacy/kien-thuc-ke-thua.md), [LESSONS trước tổ chức](archive/legacy/lessons-before-layout-20261014.md).
- Source/codec/resume PASS không thay utility/R1/R2/DP proof; mọi kết quả headline cần terminal artifact audit.

## 5. Quy trình cho các phiên sau

- [Skill Codex](../.agents/skills/federated-experiment-workflow/SKILL.md).
- [Skill Claude Code](../.claude/skills/federated-experiment-workflow/SKILL.md).
- [Thông tin chuẩn skill đã đối chiếu](operations/workflow-skill.md).
- [Index toàn dự án](file-index.md): code, docs, notebooks, helper, skill; artifact bị ignore có route/DVC.
