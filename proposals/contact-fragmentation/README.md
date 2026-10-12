# Truyền snapshot qua contact hữu hạn

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | H6/P2 |
| trạng thái | Hạ tầng R1; P2 có giới hạn đã audit. |

## Cơ chế FL

Phân mảnh object, trả phí frame/ACK/retry; chỉ kích hoạt snapshot coherent đúng version trước deadline.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [de-xuat-huong-moi.md](../../docs/research/de-xuat-huong-moi.md)
- [pilot-plan.md](../../docs/research/pilot-plan.md)
- [pilot-p2.md](../../docs/research/probes/pilot-p2.md)
- [channel.py](../../mh/channel.py)
- [report.md](../../papers/p2/report.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
