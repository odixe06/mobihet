# MobiHet — điểm vào dự án

Đọc CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).
Không nạp `archive/` mặc định.

Federated IDS V2X trên VeReMi NextGen v4:5.343xe nhận,77input,16lớp; mỗi xe giữ model riêng.
Cơ chế trao thống kê/xác suất, hỗ trợ khác kiến trúc; không realrows/model/gradient/BN.

## Trạng thái — 20/11/2026 ≈ 03:00 +07

**Bàn giao [continue](docs/operations/continue.md) đủ để tiếp tục mọi việc** (job, lệnh khởi động lại orchestrator, việc tay).
Phương pháp = [FB-L](docs/research/screening-fbl.md) cấu hình **L4** (sim r20 6/6 PASS, pilot r20 12/12 audit PASS);
**full r50 3 seed × 2 model đang chạy trên Kaggle**, orchestrator tự nối phiên (dừng khi tắt máy, chạy lại theo continue).
Compute tổng xong: L4 0,920× / 0,917× FedAvg FLOP ([harness](docs/research/compute-harness.md)).
Nền đã công bố = T3 (r50, 3 seed, [comparison](docs/research/comparison.md)); đích owner: [task brief](task_brief.md).
Baseline 14 phương pháp × kịch bản: repo `~/project/experiment`.

## Quy tắc đang dùng

- [Protocol](docs/research/protocol.md) đồng nhất; headline round50, không best-round/HP qua test; audit đủ10metric và guard phụ.
- Core/CBB/KM50 là đối chứng lịch sử, không thay số T3. Baseline SCAFFOLD loại.
- Giữ verdict cũ; không đổi tên nhánh FAIL để chạy lại. Mixed-model/SecAgg/DP chưa được chứng minh.
- COMPLETE/exit0 không thay artifact audit; không sửa source/ledger worker đang sở hữu; không xóa artifact/cache.

| Cần biết | Mở |
|---|---|
| Job/owner/bước tiếp/việc tay | [Continue duy nhất](docs/operations/continue.md) |
| Xây lại phương pháp và những gì đã thử | [Task brief](task_brief.md), [FB-L](docs/research/screening-fbl.md), [FB](docs/research/screening-budget.md), [Lite2](docs/research/screening-lite2.md), [tried index](docs/research/tried-methods-index.md) |
| Compute đo thế nào (useful/executed/tổng) | [Compute harness](docs/research/compute-harness.md) |
| T3/cấu hình/kết quả | [Method](docs/research/method.md), [comparison](docs/research/comparison.md) |
| Dataset/model/máy | [Knowledge](knowledge/README.md) |
| Chạy/audit/backup | [Operations](docs/operations/README.md), [scripts](scripts/README.md) |

Kết quả→docs/research; job→continue; dữ kiện bất biến→knowledge; bài học→hai lessons.
[Bản đồ docs](docs/README.md). Lịch sử vận hành ra khỏi tuyến đọc; chỉ mở archive khi cần evidence cụ thể.
