# Archive vận hành — 16/11/2026

Hồ sơ lịch sử, không phải bàn giao sống. Bắt đầu phiên tại [CONTEXT](../../CONTEXT.md),
job hiện hành tại [continue](../../docs/operations/continue.md).

- `before/`: snapshot hub/bàn giao/Kaggle/backup/lessons/sổ lỗi trước rút gọn; quyền, PID, quota và status bên trong có thể hết hiệu lực.
- `receipts/`: inventory backup/recovery/launch cũ; chỉ mở khi restore hoặc kiểm một sự kiện cụ thể.
- `scripts/`: launcher/watcher/recovery/analysis của study đã đóng; không còn entrypoint vận hành hiện tại.
- [migration.json](migration.json): đầy đủ đường dẫn cũ→mới, SHA256 từng script, danh sách giữ lại và hash source/DVC được bảo vệ.

Script được dời nguyên byte. Các script dùng `__file__.parents[...]`, import `scripts.*` hoặc đường dẫn root cũ
**không bảo đảm chạy trực tiếp tại archive**. Muốn tái lập: xem Git/source SHA của study, dựng checkout biệt lập đúng layout cũ;
không sửa source frozen hoặc tự launch study từ archive. Link trong snapshot đã chỉnh để tra cứu được.

Các helper tên study cũ còn trong `scripts/` vẫn là dependency của runtime/test; xem [danh mục hiện hành](../../scripts/README.md).
Không dời `core/`, `methods/`, `tests/`, `runs/`, `artifacts/`, notebook, cache hoặc metadata DVC.
