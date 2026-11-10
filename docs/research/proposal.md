# Phương pháp được chọn — CBB (owner chốt 10/11/2026)

**Phương pháp chính = CBB = Core + generator KM16 w0 .5 + codec lite6 (Lloyd 4 lượt) + Q 8-bit (QQ8).**
Core = copula augmentation + class Q + synthetic step floor + compact bootstrap. Không input-bank, không realKD.
Cấu hình: hp KM50 + `cop_kmeans_q=int8 cop_kmeans_lite=6 q_bits=8 km_bytes=true` (`runs/codec-combo50/batch.json`, arm CBB42/CBB43).
Chi tiết thực thi ở [method](method.md) (mô tả Core; KM16/codec ở [km-lite](screening-km-lite.md), [screening-codec](screening-codec.md)),
cấu hình chung ở [protocol](protocol.md).

Số r50 (TB seed 42/43, audit PASS): F1 macro **.9078** (hạng 3 sau APFed .9119, CANPerFL .9088), recall macro .9097, p10 .7761, low256 .8604;
**265 B/round** (13,3 KB/50 round không tính bootstrap < pFedES 0,14 MB), bootstrap unicast **1,15 MB**/xe. So KM50 codec gốc cùng seed:
Δmacro TB r46–50 −.0003 cả hai seed. Đánh đổi đã biết, owner chấp nhận: benign .8403 thấp hơn Core+QQ8 .8410
(KM16 trượt riêng guard benign vs Core −.0015 ở KM50). Đối chứng cùng codec: CQ8 = Core+QQ8 F1 .8991, 259 B/round, bootstrap ≈1,00 MB.

## Giữ và bỏ theo bằng chứng

| thành phần | quyết định | căn cứ |
|---|---|---|
| Copula theo lớp, sinh tại xe | giữ | tri thức toàn mạng cho xe ít mẫu/lớp vắng; bỏ synthetic ở xe đủ dòng hại đơn điệu (sim b1) |
| Q theo true class, đo trên dòng thật | giữ | **ablation r50 `needed`** (bỏ Q: accuracy −.0038 cả hai seed, benign −.010) — [Y50](results-y50.md); Q từ synthetic FAIL (b10) |
| Synthetic step floor | giữ | **ablation r50 `needed`** (bỏ floor: p10 −.016/−.018, low256 −.028, absent −.09/−.11) — [Y50](results-y50.md) |
| Compact bootstrap + lossless CDF | giữ | [F50](results-f50.md), [L50](results-l50.md) |
| Generator KM16 w0 .5 | **giữ** (owner 10/11) | [km50](screening-km50.md): r50 +.0085/+.0086 macro vs Core; FAIL riêng guard benign (owner chấp nhận) |
| Codec KM lite6 (Lloyd 4 lượt) | **giữ** | LT6S43 PASS r20; combo CBB PASS r50 cả hai seed — [screening-codec](screening-codec.md) |
| Q 8-bit (QQ8) | **giữ** | PASS r20 + r50 (CBB, CQ8 cả hai seed); Q 4-bit FAIL (low256 −.0104) |
| Bỏ Lloyd (LY0) trong combo | bỏ | CBA r20 FAIL cả hai seed (benign) |
| Input-bank | bỏ | lợi r20 mất ở r50, trượt luật cả hai seed — [F50](results-f50.md) |
| realKD (cả NCKD/gate) | bỏ | đánh đổi timeDelay/lớp vắng |
| Prior shift lúc train | bỏ | [Y50](results-y50.md) FAIL r20 cả hai seed |
| Oracle | chỉ chẩn đoán | dùng pool dòng thật |
| Generator khác (standard, forest, pair, EM, K32…) | bỏ, đóng băng | [catalog](generators.md) |

## Giao tiếp và ràng buộc

Chỉ thống kê gộp và xác suất 16 lớp, không model/gradient/BN/dòng dữ liệu. Server dùng tổng trên mọi xe; client tự sinh dữ liệu
và giữ model riêng. Schema cho phép khác kiến trúc (sim b4 trộn W1/W0.25 không sụp) nhưng **chưa chạy model trộn ở full-scale**.
Không tuyên bố đã triển khai secure aggregation hoặc DP; moment theo ô nhỏ của từng xe không riêng tư nếu thiếu SecAgg
(owner 09/11 hoãn, phân tích ở [catalog đề xuất](improvement-proposals.md)). Byte phải tính cả bootstrap và hai chiều.
Số canonical: [F50/Core](results-f50.md), [L50/lossless](results-l50.md), [km-lite](screening-km-lite.md).

## Phần chưa được chứng minh

- Chỉ 2 seed (42/43), không phải significance; backlog owner: seed 41.
- Cải tiến ngoài generator: PH, EMA, QK, QT2 **FAIL sim b11**, floor/rows/gộp phiên **FAIL sim b12** (2 seed) — [catalog đề xuất](improvement-proposals.md); chưa có ứng viên mới.
- Mô hình trộn full-scale, dropout/contact, SecAgg/DP, đầu độc: chưa kiểm.

Trạng thái công việc chỉ ở [continue](../operations/continue.md).
