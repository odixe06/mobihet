# Prototype trung bình lớp

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | H4/P1 |
| trạng thái | Đối chứng tham chiếu; không ưu tiên thêm biến thể mean. |

## Cơ chế FL

Gộp vector trung bình theo lớp, dùng regularization/prototype loss ở xe.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [de-xuat-huong-moi.md](../../docs/research/de-xuat-huong-moi.md)
- [pilot-p1.md](../../docs/research/probes/pilot-p1.md)
- [pilot.py](../../mh/pilot.py)
- [report.md](../../papers/pilot/report.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
