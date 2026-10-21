# Hiệu chỉnh teacher bằng đặc trưng ảo theo lớp (`cv`) — sinh mẫu có chủ đích cho nguồn tri thức

**Trạng thái (21/10 ~16Z): owner đặt ngân sách ≤ 1,5× byte phương pháp cũ ⇒ chỉ chạy `cvl` (×1,45 ở 15 epoch); `cvf` ×6,8 loại.
Test local PASS `--part r25` + `--part par`; cổng khóa ([design §Vòng 25](../../methods/full-drift/design.md)); push tag r30 khanhmay0304.** Rút ra từ [phân tích vì sao sinh mẫu không ăn](../../methods/full-drift/results.md)
sau vòng 24 (xo FAIL). Bằng chứng offline: [probe 21](../../methods/full-drift/results/probe21-xa-s42-e15.json),
[probe 22](../../methods/full-drift/results/probe22-xa-s42-e15.json).

## Vấn đề (bằng chứng)

- Mọi mẫu tổng hợp của gkd/xo học KL từ snapshot ⇒ chỉ sửa được lỗi private mà teacher đúng. Headline xa ep15: lỗi private 0,136 =
  dạy được 0,031 + **chung 0,105 (77 %)**; phần chung không đổi qua mọi arm từ gkd tới xo. Trần nếu sửa hết phần dạy được: +0,036.
- Lỗi chung tập trung ở lớp teacher còn yếu: timeDelay (teacher 0,277), posMirror (0,476), dataReplay, benign.
- Feature của backbone teacher đủ thông tin: refit head trên dòng thật có nhãn gộp 0,842 → **0,919** (timeDelay 0,50, posMirror 0,84).
  Thiếu là tri thức có nhãn ở tầng quyết định — đúng loại tri thức mà moment theo lớp của client mang được.

## Cơ chế

Mẫu tổng hợp được sinh **trong không gian đặc trưng của teacher** và dùng để sửa teacher, không phải để thêm dòng KL cho private.

1. Đầu round r, client nhận snapshot S_r, tính `h256 = head[:4](combined(x))` trên 65.536 dòng own và gửi tổng theo lớp
   (n_c, Σh, Σhhᵀ) — cộng được dưới SecureSum (như moment Gaussian của gkd).
2. Server có μ_c, Σ_c gộp ⇒ rút đặc trưng ảo N(μ_c, Σ_c) theo tỷ lệ lớp gộp, fit lại Linear cuối (256 → 16) của S_r từ trọng số đã train
   (3.000 bước Adam, batch 4.096). Gửi head đã hiệu chỉnh xuống (16,4 kB fp32).
3. Teacher của private (KL trên mẫu Gaussian gkd, phần `la`/`kdpx` giữ nguyên) = S_r với head đã hiệu chỉnh. Proxy vẫn train như cũ
   (head hiệu chỉnh chỉ dùng làm teacher; báo cả F1 proxy gốc và hiệu chỉnh).

Vì sao có thể ăn: teacher hiệu chỉnh đúng trên **47 %** lỗi của private (teacher gốc 21 %) ⇒ trần qua kênh teacher 0,861 → 0,926;
la/kdpx từng biến lợi của teacher thành lợi private (la_kdpx: proxy +0,0105, private +0,0195).

| biến thể | teacher F1 group | lỗi private teacher đúng | byte lên/client/round (4 B/giá trị, nửa tam giác) | tổng byte/client so xa: 15 ep · 50 round |
|---|---:|---:|---:|---:|
| `cvf` covariance riêng 16 lớp | 0,893 | 47 % | 2,12 MB | ×6,8 · ×9,1 |
| `cv6` riêng 6 lớp teacher yếu nhất (theo confusion own gộp), chung cho phần còn lại | 0,852 | 44 % | 938 kB (chưa cài) | ×3,6 · ×4,6 |
| `cvl` một covariance chung (LDA) | 0,848 | 38 % | 148 kB | ×1,45 · ×1,62 |
| `diag` phương sai riêng từng lớp | 0,816 | 43 % | 33 kB (chưa cài) | ×1,13 · ×1,19 |

xa mỗi client: 5,15 MB/15 epoch (bootstrap 1,81 MB một lần + q3 223 kB/round), 12,95 MB/50 round; cộng thêm head xuống 16,4 kB/round.
Bản trước so byte cv **một client** với kênh q3 **cộng 20 client** (4,45 MB/epoch) nên ghi +48 %/+3,3 % — sai ~20×.

Byte: Σhhᵀ trên 65.536 dòng vượt tầm fp16 ⇒ gửi 4 B/giá trị (như mã fixed-point của SecAgg); probe 22 ghi số fp16 (1,06 MB / 74 kB) là sai.

**R2:** chỉ sửa head của snapshot (model chung); private kiến trúc tùy ý, vẫn chỉ nhận logit trên mẫu input. **Riêng tư:** moment đặc trưng
theo lớp; ô client×lớp nhỏ (4–10 dòng) ⇒ cần SecAgg như bootstrap Gaussian ([luồng thông tin](../federated-information-flow.md)).
**Khác giao thức hiện tại:** thêm một lượt trao đổi thống kê đầu round (không phải aggregation trọng số) — owner cần xác nhận hợp lệ với
quy tắc một aggregation/epoch.

## Kiểm (đề xuất, chưa khóa)

Screening s42 ep15 `--fast` vs xa (0,8246), cổng gain+worst như vòng 24 (Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất ≥ −0,010).
Ghi mỗi round: F1 teacher gốc vs hiệu chỉnh trên dòng own (nhãn own, không dùng group/test), phần lỗi own private mà teacher đúng.

**Dấu hiệu sai:** (a) teacher hiệu chỉnh không hơn gốc ≥ +0,02 ở các round ⇒ moment đặc trưng không đủ khi backbone còn trôi ⇒ dừng;
(b) teacher lên nhưng private < +0,010 ⇒ nút thắt là kênh truyền (KL trên mẫu Gaussian lệch đa tạp, probe 15–17), không phải tri thức
teacher ⇒ hướng tiếp là phủ input (mục dưới), không quét biến thể cv; (c) benign < −0,005 như pce ⇒ teacher hiệu chỉnh đổi tính chất prior.

## Hướng phụ (chỉ khi cv cho thấy teacher lên mà private không theo)

- **Đảo ngược theo moment đặc trưng lớp:** sinh dòng input sao cho đặc trưng snapshot khớp N(μ_c, Σ_c) gộp (DeepInversion khớp moment
  theo lớp thay vì BN toàn cục; probe 17 di_hi chỉ +0,012…+0,021). Dùng lại thống kê của cv (0 byte thêm), đích là phủ đa tạp client khác —
  đòn bẩy lớn nhất từng đo (probe 15: dòng thật client khác 0,842 vs Gaussian 0,792). Cổng offline trước: refit Linear (giao thức probe 17)
  ≥ di_hi + 0,02.
- **Ưu tiên có độ tin cậy** (trọng số cặp × max(0, 2ρ_own − 1), ρ từ nhãn own, 0 byte): cặp teacher đúng trên group 0,506 → 0,665 (probe 22 A).
  Không chạy riêng (sửa một adaptation đã FAIL); chỉ ghép với teacher hiệu chỉnh nếu cv PASS.

## Khác bản đã thử

- gkd / g512 / gk2 / cop / xo: đổi mẫu input, giữ teacher ⇒ chạm tối đa 23 % lỗi. cv đổi teacher ⇒ chạm phần 77 %.
- pce (proxy chỉ CE, +0,059 proxy) không giúp private và hại benign: đổi cả cách train proxy. cv giữ proxy, chỉ hiệu chỉnh head làm teacher.
- probe 19 (bias 16 lớp): bias proxy chuyển sang private không lợi; cv không chuyển bias, đổi teacher mà private học KL theo.
- Nguồn: CCVR (Luo et al., NeurIPS 2021) hiệu chỉnh classifier global bằng đặc trưng ảo từ moment lớp; ở đây dùng làm teacher cho private
  khác kiến trúc trong FD, và chọn biến thể theo byte.
