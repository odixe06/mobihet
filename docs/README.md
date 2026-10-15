# Mục lục tổng quan dự án

Đây là điểm đọc chính: [CONTEXT](../CONTEXT.md) giữ trạng thái/lưu ý vận hành; [LESSONS](../LESSONS.md) giữ bài học chung.
[Index đang làm](file-index.md) · [Index lịch sử](archive/file-index.md) · [Cấu trúc và alias](operations/project-layout.md).

## 1. Toàn bộ phương án

- [Một danh mục đề xuất](proposals.md): 28 nhóm +recipe EMA; đã kiểm/chưa kiểm, PASS/FAIL có phạm vi, link thiết kế/build/kết quả.
- [Hồ sơ phương pháp](methods/README.md): README ngắn → căn cứ/cơ chế → các build/source/artifact.
- [Bài học dùng tiếp](lessons/README.md); [hồ sơ lịch sử](archive/README.md) giữ báo cáo và nguồn gốc số đo.
- [Dự báo performance](research/performance-feasibility-2026-10-13.md): căn cứ trước mở pilot.

## 2. Hồ sơ làm việc và trạng thái

| phương pháp | trạng thái chốt15/10 | thiết kế / build / kết quả |
|---|---|---|
| Gaussian class moments | Full q2 cùng controls; diag/lr4 riêng tạm hoãn | [Hồ sơ](methods/gaussian-class-moments/README.md), [full H1/H2](methods/shared-baselines/builds/federated-baselines-full/README.md) |
| DAGSNet mutual distillation | Full q2 đang chạy, chưa r50 | [Hồ sơ](methods/proxy-mutual-distillation/README.md), [10metrics cuối](methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md) |
| Gaussian + proxy Hybrid | Full q2 đang chạy; own pair chưa fresh heartbeat | [Hồ sơ](methods/gaussian-proxy-hybrid/README.md), [full build](methods/gaussian-proxy-hybrid/builds/hybrid-full/README.md) |
| BoundaryCells counts | Bốn adaptation hai seed FAIL; dừng | [Hồ sơ](methods/boundary-cell-counts/README.md), [fixed-r20 report](methods/boundary-cell-counts/builds/boundary-cell-pilot/results/utility-report.md) |
| ClassTail risk | Pairedr20 FAIL; dừng | [Hồ sơ](methods/class-tail-risk/README.md), [gates/forensics](methods/class-tail-risk/builds/class-tail-pilot/results/report.md) |
| CountSketch EF sửa overshoot | Smallr20 FAIL; medium partial kém own; dừng adaptation | [Hồ sơ](methods/proxy-countsketch-repair/README.md), [trial](methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results/report.md) |
| CrossClientRank | Probe/diagnostic auditPASS; utility42 recoveryr4,43 chờ startup; chưa utilityPASS | [Nguồn/cơ chế](methods/cross-client-ranking/README.md), [utility/recovery](methods/cross-client-ranking/builds/rank-utility/README.md) |
| ResponseRisk augmentation | Fidelity probe PASS; chưa train augmentation | [Thiết kế](methods/receiver-response-consistency/design.md), [build](methods/receiver-response-consistency/builds/receiver-augmentation/README.md) |
| EMA deployment recipe | Thiết kế đã chốt; chưa source/pilot utility | [Thiết kế/cổng](methods/ema-deployment-recipe/design.md) |

Các adaptation FAIL khác đều có thiết kế/report/source qua [danh mục đề xuất](proposals.md); không rerun y nguyên.

## 3. Vận hành, lỗi và khả năng tái lập

- [Bàn giao hiện tại và checklist phiên sau](operations/continuation-2026-10-15-afternoon.md): exact kernel/version, checkpoint, budget, lệnh one-shot. Owner đã dừng poll khi kết phiên.
- [Sổ lỗi kernel](operations/notebook-errors.md): nguyên nhân, output dùng được, preflight ngăn lặp.
- [Launch plan/budget](operations/continuation-plan-2026-10-15-q2.json); [global snapshot09:01UTC](operations/kernel-status-current.json) cũ hơn CCR error/recovery, không trạng thái live.
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
