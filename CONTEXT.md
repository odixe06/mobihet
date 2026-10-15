# CONTEXT — MobiHet / federated IDS V2X

Bắt đầu: CONTEXT → [LESSONS](LESSONS.md) → [bàn giao hiện tại](docs/operations/continuation-2026-10-15-afternoon.md).
[Mục lục](docs/README.md) · [Danh mục phương pháp](docs/proposals.md) · [Index file](docs/file-index.md).

## Bài toán và ràng buộc

Xe học IDS cộng tác qua RSU với giao tiếp hữu hạn. Giữ66features, partition/scaler/protocol gốc;
không raw/perturbed record hoặc per-record activation/gradient rời xe. Mọi learned model hiện DAGSNet;
heterogeneous R2 và contact/deadline R1 chưa được chứng minh.
Train/probe production nặng trên Kaggle, local chỉ fixture/algebra/collect/audit/report.
Official test chỉ báo cáo, không chọn HP/best round/continuation; group/old pilot nội bộ có protocol riêng.
[Protocol](docs/research/protocol.md) · [Dữ liệu/kiến trúc](knowledge/README.md).

## Quyết định còn hiệu lực

- Owner15/10: ưu tiên DML/Hybrid +controls và pilot mới; không dùng quota để nối adaptation đã FAIL.
  Ba account mới được owner xác nhận quota; **không query lại quota**. Full giữ lịch50 từr1,
  seed42,20clients,100%train80%, không internal holdout. [Plan/budget](docs/operations/continuation-2026-10-15-afternoon.md).
- Freeze source/notebook đã PUSHED và toàn40module `mh/*.py`; source build mới riêng.
  Exact-resume từ actual committed checkpoint, không W&B summary hoặc r1. Đổi vận hành dùng tag mới.
  [Lỗi và recovery](docs/operations/notebook-errors.md).
- Giữ scientific settings/gp/gates đã chốt; nghi vấn khoa học hoặc budget vượt scope thì hỏi owner trước.
  CrossClientRank primary fixed true-class-prior gp mọi arm/reference, head diagnostic.
  [Contract utility](docs/methods/cross-client-ranking/builds/rank-probe/utility-proposal.md).
- 15/10 tối owner yêu cầu backup+cleanup: đã push Git/DVC, xóa local ba build đóng ([backup](docs/operations/backup.md)).
  Ngoài yêu cầu mới, không Git/DVC add/commit/push/cleanup; selective restore được phép. Giữ owner edits credential reference.
  Private inline W&B key chỉ ignored private_push0600; private platform history vẫn giữ key.
  [Layout](docs/operations/project-layout.md) · [Account paths](docs/operations/kaggle-accounts.md).

## Trạng thái chốt15/10/2026 — đã dừng poll theo owner

Snapshot bằng chứng trước yêu cầu bàn giao; không là trạng thái live. Không poll/launch thêm trong phiên này.
[PUSHED/RUNNING không thay terminal audit](docs/operations/continuation-2026-10-15-afternoon.md).

- Batch cũ đã collect/audit toàn bộ output có thể lấy. H1 q1 maintenance mất output;
  W&B round cao hơn không checkpoint, recovery từ audited parent.
- Six q2 full đã PUSHED, import/controller PASS; chín identities có initial fresh heartbeat.
  Hybrid-own pair chưa heartbeat mới; owner thấy còn chạy nhưng Console không load. Giữ session, không rerun/cancel.
- ClassTail pairedr20 FAIL; CountSketch repair smallr20 FAIL, medium partial kém own;
  BoundaryCells cả bốn adaptation direct/teacher FAIL hai seed. **Không nối các adaptation này**.
  Nguyên nhân và giới hạn kết luận tại [LESSONS](LESSONS.md) và [danh mục](docs/proposals.md).
- CrossClientRank probe/diagnostic hai seed đã audit, locked probePASS; chưa utilityPASS.
  Utility hai seed đã push. Seed42 lỗi loader emptybytes, r3 hợp lệ/costPASS;
  recoveryv2 giữ source/fingerprint/weights, CPU audit PASS và đã tiếnr4.
  Seed43 owner thấy còn chạy, chưa fresh heartbeat/artifact; giữ session.
  [Build và recovery](docs/methods/cross-client-ranking/builds/rank-utility/README.md).
- ResponseRisk/EMA/SAM chỉ thiết kế/nghiên cứu, chưa mở trial mới.
  Fullr50 và paired utilityr20 chưa hoàn tất. Không có yêu cầu thao tác tay pending.

## Phiên sau

1. Đọc [checklist và lệnh bàn giao](docs/operations/continuation-2026-10-15-afternoon.md#việc-còn-thiếu--phiên-sau).
   Collector terminal chạy một lượt; không mở polling loop hoặc poll lại run đã stable.
2. Collect/audit q2 và CrossClientRank khi terminal. Seed43 chỉ recovery sau actual artifact/error/cost audit;
   đủ mounted SHA/state proof trước GPU, không push trùng hoặc lặp SGD đã commit.
3. Regenerate đủ10metrics/per-class/gates; giữ FAIL theo scope và ghi việc còn thiếu.
   Cân budget còn lại trước continuation; không mặc định quota được bổ sung hoặc gate đã PASS.

## Muốn biết → mở

| cần biết | tài liệu |
|---|---|
| Kernel, checkpoint, budget, việc chưa làm, lệnh tiếp tục | [Bàn giao hiện tại](docs/operations/continuation-2026-10-15-afternoon.md) |
| Toàn bộ phương pháp, PASS/FAIL/chưa kiểm, research mới | [Danh mục](docs/proposals.md), [hồ sơ](docs/methods/README.md) |
| Mọi con số/full metrics đã audit | [H1/H2](docs/methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md), [Hybrid](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md) |
| Lỗi runtime/recovery và nguyên tắc không lặp | [Sổ lỗi](docs/operations/notebook-errors.md), [LESSONS](LESSONS.md) |
| Dữ liệu, kiến trúc, protocol | [Knowledge](knowledge/README.md), [protocol](docs/research/protocol.md) |
| Script, layout, alias, restore/backup | [Scripts](scripts/README.md), [layout](docs/operations/project-layout.md), [backup](docs/operations/backup.md) |

CONTEXT giữ trạng thái/quyết định và đường dẫn; số đo/commands/contracts ở docs của build hoặc operations.
Tri thức dữ liệu/kiến trúc ở knowledge; quy trình dùng lại ở skills. Bài học mới cập nhật LESSONS ngay.
Sau thêm/chuyển file: cập nhật index, kiểm link/hash/ignore; không tự backup hoặc dọn artifact.
