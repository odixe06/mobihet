# Residual hàm từ nhãn local trên anchor

[Danh mục](../README.md) · [Bài học](lessons.md)

Nhãn lịch sử: K2. Thiết kế chưa sẵn sàng; đồng bộ probe/teacher vẫn có chi phí lớn. K2b kernel-anchor là hướng khác đã pilot.

## Cơ chế FL

Tính residual/gradient hàm trên anchor synthetic chung, gộp thống kê để cập nhật teacher DAGSNet.

## Tài liệu liên quan

- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)

Mọi learned model giữ DAGSNet; không raw record rời xe. Đây là thiết kế, chưa có bằng chứng F1/R1/R2 mới.
