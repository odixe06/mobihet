> **Snapshot 19/11/2026 trước khi rút gọn hub — lịch sử.** Hub sống: [CONTEXT](../../CONTEXT.md).

# MobiHet — điểm vào dự án

Đọc CONTEXT → [LESSONS](../../LESSONS.md) → [bài học vận hành](../../docs/operations/lessons.md) → [bàn giao sống](../../docs/operations/continue.md).
Không nạp `archive/` mặc định.

Federated IDS V2X trên VeReMi NextGen v4:5.343xe nhận,77input,16lớp; mỗi xe giữ model riêng.
Cơ chế trao thống kê/xác suất, hỗ trợ khác kiến trúc; không realrows/model/gradient/BN.

## Trạng thái — 19/11/2026 (cập nhật compact); 18/11/2026, 11:00 +07

**Mới nhất:** pilot FB r20 xong (DAGSNet N2/N4 PASS, BERT N2/N4 FAIL gate pilot); r50 hoãn. Đang dựng [FB-L](../../docs/research/screening-fbl.md)
(chọn dòng thật theo loss cũ, sim DAGSNet + BERT, chưa push). **Compute FB thực thi ≈ 2,07× FedAvg với engine hiện tại**
(1,0× chỉ là useful-work). **19/11: `compact` xong, 19/19 test, đo engine thật ⇒ executed 0,997× (N4) / 1,001× (N2) FedAvg**
([harness](../../docs/research/compute-harness.md)). **Sim FB-L compact xong: 6/6 arm PASS DAGSNet + BERT, chọn P4**
([FB-L](../../docs/research/screening-fbl.md)); pilot full-scale chờ owner. Đang đo tổng compute toàn roster (job: continue).

Dưới đây là trạng thái 17/11 (giữ để tham chiếu):

MobiHet hiện tại=T3 (KM16+lite6+Q8, liều synthetic theo cỡ xe/pha lịch).
DAGSNet và FeatureBERT đủr50 cả3seed41/42/43; mixed-model full-scale chưa chạy.
Kết quả/giới hạn audit: [comparison](../../docs/research/comparison.md). Job/Notion/push hiện tại chỉ tại [continue](../../docs/operations/continue.md).

Owner đang xây lại phương pháp: seed42, đơnmodel DAGSNet/BERT, sim→pilot→full;
performance/communication/compute xếp riêng top5, Macro-F1 cách bestbaseline≤.015; compute chấp nhận≤1,01×FedAvg (chốt sau cùng17/11).
[Task brief](../../task_brief.md) giữ yêu cầu mới. Task hiện tại = [FB — Fixed-row-budget](../../docs/research/screening-budget.md):
8 sim **DAGSNet** seed42 đã hoàn tất pull & audit 100% PASS; N4 được chọn (compute 1,000×, macro TB .9070, ΔTB −.0064 vs S0); pilot full-scale N4 + N2 (PASS sim) DAGSNet + FeatureBERT r20 đang chạy 17/11 (job: continue).
[Lite2](../../docs/research/screening-lite2.md) sim và pilot hai model PASS; DAGSNet r50 audit PASS theo gate cũ (.905877; 0,354116 MB; estimator1,364×).
**Hoãn push lại Lite2 BERT r50**, ưu tiên FB; không giữ quota riêng cho Lite2.
Gate Lite2≤1,37× là luật cũ, không chứng nhận đích compute mới; xem [brief](../../task_brief.md) §14–15.

## Quy tắc đang dùng

- [Protocol](../../docs/research/protocol.md) đồng nhất; headline round50, không best-round/HP qua test; audit đủ10metric và guard phụ.
- Core/CBB/KM50 là đối chứng lịch sử, không thay số T3. Baseline SCAFFOLD loại.
- Giữ verdict cũ; không đổi tên nhánh FAIL để chạy lại. Mixed-model/SecAgg/DP chưa được chứng minh.
- COMPLETE/exit0 không thay artifact audit; không sửa source/ledger worker đang sở hữu; không xóa artifact/cache.

| Cần biết | Mở |
|---|---|
| Job/owner/bước tiếp/việc tay | [Continue duy nhất](../../docs/operations/continue.md) |
| Xây lại phương pháp và những gì đã thử | [Task brief](../../task_brief.md), [FB](../../docs/research/screening-budget.md), [Lite2](../../docs/research/screening-lite2.md), [tried index](../../docs/research/tried-methods-index.md) |
| T3/cấu hình/kết quả | [Method](../../docs/research/method.md), [comparison](../../docs/research/comparison.md) |
| Dataset/model/máy | [Knowledge](../../knowledge/README.md) |
| Chạy/audit/backup | [Operations](../../docs/operations/README.md), [scripts](../../scripts/README.md) |

Kết quả→docs/research; job→continue; dữ kiện bất biến→knowledge; bài học→hai lessons.
[Bản đồ docs](../../docs/README.md). Lịch sử vận hành ra khỏi tuyến đọc; chỉ mở archive khi cần evidence cụ thể.
