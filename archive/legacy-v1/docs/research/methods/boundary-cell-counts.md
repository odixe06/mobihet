# BoundaryCells — đếm nhãn theo ô input

**Trạng thái (17/10):** FAIL bốn adaptation (dừng).

## Cơ chế
Đếm nhãn theo ô trên input, dạy soft target tại record local (direct hoặc qua teacher).

## Bằng chứng (audited)
P2 phủ 95,65% group probe nhưng target entropy 0,915, table F1 ≈0,129; P4 direct/P2–P4 teacher r20 hai seed FAIL so Gaussian.

## Bài học
Coverage không thay discrimination; target tĩnh không nhận tri thức mới khi thêm round.

## Nguồn
Nội bộ: [đề xuất](../../archive/research/new-fl-primary-sources-2026-10-13.md), [dự báo](../../archive/research/performance-feasibility-2026-10-13.md), [hồ sơ](../../methods/boundary-cell-counts/README.md).
