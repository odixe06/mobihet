# Proxy DAGSNet chưng cất hai chiều

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | H1/P1/P2 |
| trạng thái | P1 có lợi; P2 byte/prior còn giới hạn. |

## Cơ chế FL

Xe giữ private DAGSNet và proxy DAGSNet; proxy trao đổi tham số gộp, private/proxy học lẫn nhau tại xe.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [de-xuat-huong-moi.md](../../docs/research/de-xuat-huong-moi.md)
- [pilot-p1.md](../../docs/research/probes/pilot-p1.md)
- [pilot-p2.md](../../docs/research/probes/pilot-p2.md)
- [pilot.py](../../mh/pilot.py)
- [p2.py](../../mh/p2.py)
- [report.md](../../papers/pilot/report.md)
- [report.md](../../papers/p2/report.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
