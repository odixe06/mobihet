# Phương pháp được chọn — Core (kiểm 09/11/2026)

**Phương pháp chính = Core = copula augmentation + class Q + synthetic step floor + compact bootstrap.** Không input-bank, không realKD.
Chi tiết thực thi ở [method](method.md), cấu hình chung ở [protocol](protocol.md).
**Nền nghiên cứu hiện tại = Core + generator KM16 w0 .5 (codec gốc)** (KM50 r50 +.0085/+.0086 macro nhưng FAIL guard benign → chưa thay Core;
codec lite6 FAIL pilot full-scale r20 09/11). Ứng viên PASS sẽ lên scale theo thang tự động ở [CONTEXT](../../CONTEXT.md).
**Owner 09/11 chưa chốt** Core hay Core + KM16 (KM16 r50 F1 .9077/.9085 = hạng 3): giảm băng thông trước — [screening-codec](screening-codec.md).

## Giữ và bỏ theo bằng chứng

| thành phần | quyết định | căn cứ |
|---|---|---|
| Copula theo lớp, sinh tại xe | giữ | tri thức toàn mạng cho xe ít mẫu/lớp vắng; bỏ synthetic ở xe đủ dòng hại đơn điệu (sim b1) |
| Q theo true class, đo trên dòng thật | giữ | **ablation r50 `needed`** (bỏ Q: accuracy −.0038 cả hai seed, benign −.010) — [Y50](results-y50.md); Q từ synthetic FAIL (b10) |
| Synthetic step floor | giữ | **ablation r50 `needed`** (bỏ floor: p10 −.016/−.018, low256 −.028, absent −.09/−.11) — [Y50](results-y50.md) |
| Compact bootstrap + lossless CDF | giữ | [F50](results-f50.md), [L50](results-l50.md) |
| Generator KM16 w0 .5 | nền nghiên cứu, chưa vào Core | [km50](screening-km50.md): r50 FAIL riêng guard benign |
| Codec KM lite6 | bỏ | sim PASS nhưng pilot full-scale r20 FAIL low256/absent — [km-lite](screening-km-lite.md) |
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

- Codec rẻ hơn bản gốc: lite6 FAIL r20 (trong nhiễu); codec study 10 pilot full-scale đang chạy ([screening-codec](screening-codec.md)).
- Cải tiến ngoài generator: PH, EMA, QK, QT2 **FAIL sim b11**, floor/rows/gộp phiên **FAIL sim b12** (2 seed) — [catalog đề xuất](improvement-proposals.md); chưa có ứng viên mới.
- Mô hình trộn full-scale, dropout/contact, SecAgg/DP, đầu độc: chưa kiểm.

Trạng thái công việc chỉ ở [continue](../operations/continue.md).
