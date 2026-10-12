# Gaussian từ moment lớp gộp

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | H2/P1/P2 |
| trạng thái | P1 có lợi; deadline/bootstrap P2 chưa chứng minh đủ. |

## Cơ chế FL

Xe gửi tổng theo lớp; RSU gộp moment, phát Gaussian lớp; xe train CE real + CE synthetic. Moment là thống kê, không learned model.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [de-xuat-huong-moi.md](../../docs/research/de-xuat-huong-moi.md)
- [h2-input-stats.md](../../docs/research/probes/h2-input-stats.md)
- [pilot-p1.md](../../docs/research/probes/pilot-p1.md)
- [pilot-p2.md](../../docs/research/probes/pilot-p2.md)
- [pilot.py](../../mh/pilot.py)
- [report.md](../../papers/pilot/report.md)
- [report.md](../../papers/p2/report.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
