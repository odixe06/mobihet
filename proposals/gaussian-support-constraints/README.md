# Gaussian giữ atom, profile và first-session

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V4-S |
| trạng thái | 10 arm đủ20round đã audit; paired FAIL. |

## Cơ chế FL

Moment gộp + atom probabilities; sampler analytic giữ binary/profile categorical và first-session zero-imputation. DAGSNet vẫn nhận66feature.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [pilot-v4-support.md](../../docs/research/pilot-v4-support.md)
- [pilot-v4.md](../../docs/research/probes/pilot-v4.md)
- [v4_support.py](../../mh/v4_support.py)
- [v4_schema.py](../../mh/v4_schema.py)
- [report.md](../../papers/v4/report.md)
- [rebuild.md](../../papers/v4/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
