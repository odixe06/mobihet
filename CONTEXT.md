# MobiHet — điểm vào dự án

Đọc CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).
Không nạp `archive/` mặc định.

Federated IDS V2X trên VeReMi NextGen v4:5.343xe nhận,77input,16lớp; mỗi xe giữ model riêng.
Cơ chế trao thống kê/xác suất, hỗ trợ khác kiến trúc; không realrows/model/gradient/BN.

## Trạng thái — 16/11/2026

MobiHet hiện tại=T3 (KM16+lite6+Q8, liều synthetic theo cỡ xe/pha lịch).
DAGSNet và FeatureBERT đủr50 cả3seed41/42/43; mixed-model full-scale chưa chạy.
Kết quả/giới hạn audit: [comparison](docs/research/comparison.md). Job/Notion/push hiện tại chỉ tại [continue](docs/operations/continue.md).

Owner đang xây lại phương pháp: seed42, đơnmodel DAGSNet/BERT, sim→pilot→full;
performance/communication xếp riêng top5, Macro-F1 cách bestbaseline≤.015, compute giảm tối đa.
[Task brief](task_brief.md) giữ yêu cầu mới; chưa launch training mới. Gate study cũ không tự áp cho nhiệm vụ mới.

## Quy tắc đang dùng

- [Protocol](docs/research/protocol.md) đồng nhất; headline round50, không best-round/HP qua test; audit đủ10metric và guard phụ.
- Core/CBB/KM50 là đối chứng lịch sử, không thay số T3. Baseline SCAFFOLD loại.
- Giữ verdict cũ; không đổi tên nhánh FAIL để chạy lại. Mixed-model/SecAgg/DP chưa được chứng minh.
- COMPLETE/exit0 không thay artifact audit; không sửa source/ledger worker đang sở hữu; không xóa artifact/cache.

| Cần biết | Mở |
|---|---|
| Job/owner/bước tiếp/việc tay | [Continue duy nhất](docs/operations/continue.md) |
| Xây lại phương pháp và những gì đã thử | [Task brief](task_brief.md), [tried index](docs/research/tried-methods-index.md) |
| T3/cấu hình/kết quả | [Method](docs/research/method.md), [comparison](docs/research/comparison.md) |
| Dataset/model/máy | [Knowledge](knowledge/README.md) |
| Chạy/audit/backup | [Operations](docs/operations/README.md), [scripts](scripts/README.md) |

Kết quả→docs/research; job→continue; dữ kiện bất biến→knowledge; bài học→hai lessons.
[Bản đồ docs](docs/README.md). Lịch sử vận hành ra khỏi tuyến đọc; chỉ mở archive khi cần evidence cụ thể.
