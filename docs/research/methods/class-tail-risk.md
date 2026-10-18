# ClassTail — CVaR theo lớp trên loss hard-label local

**Trạng thái (17/10):** FAIL (không nối).

## Cơ chế
Trao đổi histogram CE theo lớp để điều phối hard-example training trên record thật.

## Bằng chứng (audited)
Calibration r2 PASS; utility r20 hai seed FAIL: pooled không hơn local/own/shuffle/class-mass, kém DML 0,1967/0,1917.

## Bài học
Kênh thống kê vô hướng ≈ local; probe chỉ loại tail rỗng/calibration hỏng, không chứng minh hard-example knowledge hữu ích.

## Nguồn
Nội bộ: [đề xuất](../../archive/research/new-directions-2026-10-15.md), [build](../../methods/class-tail-risk/builds/class-tail-pilot/README.md).
