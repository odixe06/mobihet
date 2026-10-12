# Nén update proxy bằng CountSketch

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V2-A |
| trạng thái | Hai seed FAIL; EF partial numerical. |

## Cơ chế FL

Sketch update proxy, gộp rồi decode patch; so kích thước nhỏ/vừa, có/không error feedback. Private DAGSNet nhận KD.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [a-proxy-sketch.md](../../docs/research/de-xuat-v2/a-proxy-sketch.md)
- [pilot-v2.md](../../docs/research/pilot-v2.md)
- [pilot-v2.md](../../docs/research/probes/pilot-v2.md)
- [v2_math.py](../../mh/v2_math.py)
- [v2_methods.py](../../mh/v2_methods.py)
- [report.md](../../papers/v2/report.md)
- [rebuild.md](../../papers/v2/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
