# Proxy: projection trực giao đổi seed

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V3-O |
| trạng thái | Hai seed đãaudit; small/medium pairedFAIL. |

## Cơ chế FL

Update dạng ma trận được chiếu lên cơ sở trực giao fresh rank1/rank32; EF parameter-space; control rank8 so PowerSGD.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [v3-carriers-sources.md](../../docs/research/v3-carriers-sources.md)
- [pilot-v3.md](../../docs/research/pilot-v3.md)
- [v3_math.py](../../mh/v3_math.py)
- [v3_methods.py](../../mh/v3_methods.py)
- [report.md](../../papers/v3/report.md)
- [rebuild.md](../../papers/v3/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
