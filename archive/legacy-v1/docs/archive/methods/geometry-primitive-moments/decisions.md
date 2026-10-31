# K5 — phạm vi đã chốt 2026-10-12

- G51 và G53; hoãn session/rate. Giữ derived không xác định nhánh missing làm primitive; tên/số chiều thực công bố trước train.
- Marginal bốn nhóm như V4 và joint32 = first×receiver-profile4×sender-profile4.
- 13 atom: thống kê có điều kiện trong mỗi ô joint; không mở joint131.088.
- Moment ô chỉ phát nếu ≥5 client đóng góp và đủ mẫu cho covariance. Ô hiếm fallback moment lớp gộp + mask ô; không bỏ record/lớp.
- Hai cổng riêng: F1 tăng≥0,015 so full66; carrier F1≥−0,005/byte≤0,65 control. Khóa trước chạy.
- Fresh full66plain/full66support/own/shuffle; cùng BN/sampler/compute; fidelitytoy missing/coincident trước pilot.

[Quyết định chung](../../operations/decisions-2026-10-12.md). Thiết kế/code/pilot chưa hoàn tất; không gọi approval là feasibility.

## Eligibility đã kiểm và quyết định tiếp theo

Toy extractor47case/94row gồm missing/nonfinite/coincident cho thấynominalG51/G53FAIL.
Giữ8derived bịảnhhưởng⇒cảhaicùng59primitive; khôngchạyhaicasegiốngnhau.
Chủ dự án chọn **kiểm mộtG59 chochấtlượng**, marginal/joint32; cổngcarrier65%FAILtrướctrain
chofullcovariance59. Không hạ ngưỡng byte để cứu cổng.
[Fidelityproof](fidelity-missing.json), [script](../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/scripts/probe_k5_missing.py).

## Bearing: so trực tiếp G59/G61

Sau cảnhbáo divisionnear0, chủdựán chọn **thử cảhai**: G59 dùnginverse(dist>0) đúngextractor,
G61giữbearing2/derivevelocity4+headingdot. Invalidsupport làdiagnostic; nonfinite dừng, khôngclampcứurun.
15arm/seed =3freshfull66controls+6G59FL/own/shuffle×M/J+6G61 tươngứng.
