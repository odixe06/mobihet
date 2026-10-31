# MobiHet

Nghiên cứu federated IDS V2X trên VeReMi NextGen v4: 5.343 xe, 16 lớp. Mỗi xe giữ model riêng;
nhánh hiện tại trao đổi thống kê gộp và bảng xác suất theo lớp, không dùng proxy neural hoặc chia sẻ mẫu local.
Mục tiêu: cải thiện từng xe, đặc biệt xe ít mẫu/non-IID, trong giới hạn giao tiếp và hỗ trợ khác kiến trúc.

Bắt đầu tại [CONTEXT.md](CONTEXT.md). Mọi trạng thái job và bước tiếp chỉ ghi tại [bàn giao sống](docs/operations/continue.md).
Dùng env conda `nckh`; [protocol](docs/research/protocol.md), [dataset/model/runtime](knowledge/README.md).

| thư mục | trách nhiệm |
|---|---|
| `core/`, `methods/`, `tests/` | train/eval dùng chung, cơ chế hiện tại, kiểm chứng |
| `scripts/` | công cụ đang dùng; xem [danh mục](scripts/README.md) |
| `docs/` | kết quả, chẩn đoán và vận hành hiện tại |
| `knowledge/` | schema dữ liệu, kiến trúc, runtime |
| `runs/` | bằng chứng làm việc và artifact; dữ liệu nặng theo DVC |
| `archive/` | code/tài liệu đã nghỉ; loại khỏi đọc/tìm kiếm mặc định |

Git chỉ chứa code, tài liệu, DVC metadata. [Backup/khôi phục](docs/operations/backup.md).
