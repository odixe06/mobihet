# Proxy trong cơ sở sinh từ seed

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V2-B |
| trạng thái | K512 numerical r1; K1024 teacher FAIL; không KD/ZO/contact. |

## Cơ chế FL

DAGSNet proxy theta0+Za; trao đổi hệ số FO K512/K1024, không trao đổi record. Teacher qua mới mở private KD/ZO.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [b-proxy-seed.md](../../docs/research/de-xuat-v2/b-proxy-seed.md)
- [pilot-v2.md](../../docs/research/pilot-v2.md)
- [pilot-v2.md](../../docs/research/probes/pilot-v2.md)
- [v2_methods.py](../../mh/v2_methods.py)
- [report.md](../../papers/v2/report.md)
- [rebuild.md](../../papers/v2/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
