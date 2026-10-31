# Archive — không thuộc tuyến làm việc hiện tại

Lưu bằng chứng và code đã nghỉ, không xóa lịch sử. **Không đọc, tìm kiếm hoặc nạp thư mục này trong phiên thông thường.**
Chỉ mở khi owner yêu cầu tra lịch sử/khôi phục cụ thể. Tri thức còn dùng đã tổng hợp trong `LESSONS.md` và tài liệu active.

- `legacy-v1/`: toàn bộ lịch sử MobiHet v1, giữ nguyên byte.
- `docs/research/`: kết quả/spec trước batch M, kế hoạch đã được thay thế.
- `handoffs/`: snapshot bàn giao cũ, không phải trạng thái hiện tại.
- `scripts/`: helper riêng batch đã đóng; không dùng trực tiếp từ archive.
- `move-manifest.json`: đường dẫn cũ → mới và hash kiểm di chuyển. Khôi phục đúng đường dẫn cũ nếu cần chạy code lịch sử.

Tài liệu và link nội bộ trong archive được đóng băng; kiểm link active không duyệt archive. Artifact nặng quản lý qua DVC,
không chứa trong Git. Khôi phục artifact xem tài liệu backup hiện tại.
