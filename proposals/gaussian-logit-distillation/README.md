# Chưng cất logit trên điểm Gaussian

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | H5/P1 |
| trạng thái | DS-FL/FedGen adaptation trượt ưu tiên. |

## Cơ chế FL

DAGSNet local dự đoán trên điểm proxy Gaussian; RSU gộp soft target, xe distill.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [de-xuat-huong-moi.md](../../docs/research/de-xuat-huong-moi.md)
- [pilot-p1.md](../../docs/research/probes/pilot-p1.md)
- [pilot.py](../../mh/pilot.py)
- [report.md](../../papers/pilot/report.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
