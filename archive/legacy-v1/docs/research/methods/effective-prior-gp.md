# C — Luật gp theo prior huấn luyện thực (`gpe`) và nguyên tắc chung về deployment rule

**Trạng thái (17/10 ~17:10Z): FAIL hai seed** — gpe −0,094 (s42) / −0,095 (s43) so gp, client tệ nhất −0,14/−0,16 ([kết quả probe](../../methods/full-drift/results.md)); private đã học prior gần π_k, gp giữ làm luật chính. Owner duyệt chạy 17/10 ~16Z. Không cần train: kernel probe vòng 13 chấm lại weights ep6 đã audit
(gpool s42/s43, kd2, kdpx); arm vòng 13 có `gpe` sẵn trong eval. Luật chính vẫn `gp` cho mọi cổng khác.
[Design §Vòng 13](../../methods/full-drift/design.md).

## Vấn đề
gp: z + log π_g − log π_k — đúng dưới label shift khi head phản ánh prior local. gpool train CE real (prior π_k) + CE Gaussian
(prior q_k, đều) với trọng số 1:1, nên prior huấn luyện tới không phải π_k; frozen teacher đã cho ví dụ gp sửa quá tay
(head 0,714/0,698, gp 0,587/0,577).

## Cơ chế
gpe: z + log π_g − log π_e,k, π_e,k = (π_k + w q_k)/(1 + w); q_k phân bố nhãn Gaussian của client (gpool: đều; fill: water-fill),
w trọng số CE Gaussian (1). Không tham số, khóa trước; bỏ qua KD (xấp xỉ). Cổng: gpe vs gp cùng weights, gain + worst, phải qua
trên cả gpool s42 và s43.

## Giới hạn (đề xuất owner §6)
Ngay cả CE real + λ CE synthetic, optimum lý tưởng p*(y|x) ∝ π_k(y)p_k(x|y) + λ q(y)Q(x|y); khi Q(x|y) ≠ p_k(x|y) không tồn tại
một prior hiệu dụng duy nhất bằng phép cộng log. KD làm phức tạp hơn. Không suy từ tỷ lệ batch 512:256 hay λ một phép cộng
log-prior chắc đúng — gpe là phép thử thực nghiệm khóa trước, không phải đẳng thức.

## Nguyên tắc chung (từ §6 đề xuất owner)
- P1–P3: báo head và gp; cổng mặc định private gp. gp fail dù head tăng thì ghi fail của arm; đổi rule là thiết kế mới, khóa
  trước, cùng calibration recipe cho controls.
- P4–P6: teacher đã mang prior quần thể, không mặc định áp gp local; khóa deployment map bằng calibration split riêng, báo kèm gp.
- Calibration cần nhãn toàn cục thì dùng thống kê/CM gộp có tính byte; tách theo receiver/session khỏi tập chấm (`old` khác dòng
  chưa bảo đảm khác receiver). Calibration là recipe triển khai, không gọi lợi FL nếu local-only đạt tương tự.
- Không đưa FedLC/prior weighting thành ưu tiên riêng ([prior-consistent DML](prior-consistent-dml.md)).

## Tiền lệ
Build3: gp có τ `argmax(z − (1−τ) log π_i + log π_g)` +0,025 so head ([kế thừa](../../archive/legacy/kien-thuc-ke-thua.md)).
