# ProxyFuse — trộn private gp với proxy khi triển khai

**Trạng thái (17/10):** FAIL (−0,032/−0,034 so proxy).

## Cơ chế
Trộn logit private-gp và proxy khi dự đoán (hậu kiểm, không train).

## Bằng chứng (audited, pilot r20 group)
Fused kém proxy −0,032/−0,034 nhưng hơn private gp +0,083; private đúng mà proxy sai chỉ ≈2% dòng.

## Bài học
Tri thức quần thể nằm trong proxy; trộn không thêm tín hiệu khi chấm trên test toàn cục. Nhánh personalized
(α_k học theo client) thuộc [personalized-branch-b](personalized-branch-b.md).

## Nguồn
Nội bộ: [kết quả](../../methods/proxy-fusion/results.md), [thiết kế](../../methods/proxy-fusion/design.md).
