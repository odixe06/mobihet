# Moment primitive + tái dựng feature hình học

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | K5/G51/G53 |
| trạng thái | V6đủ4kernel/30arm audit; G59numericalFAIL, G61quality/compressionpairedFAIL. |

## Cơ chế FL

Gộp moment trên primitive, sinh synthetic rồi tái dựng analytic66features theo extractor. So marginal mask/joint32 và full66 controls.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [v5-algebra-sources.md](../../docs/research/v5-algebra-sources.md)
- [probe_v5_algebra.py](../../scripts/probe_v5_algebra.py)
- [algebra-probe.json](../../papers/v5/algebra-probe.json)
- [decisions-2026-10-12.md](../../docs/operations/decisions-2026-10-12.md)

[Thiết kế đang hoàn thiện](design.md) · [Các quyết định đã chốt](decisions.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.

[Manifest V6](../../docs/research/pilot-v6-geometry.md) · [Ledger/code/proof](../../papers/v6/rebuild.md).
