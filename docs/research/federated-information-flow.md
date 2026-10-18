# Kiểm toán luồng thông tin — các phương pháp FullDrift có đúng FL không (18/10/2026)

Owner 18/10: kiểm lại thật kỹ phương pháp hiện dùng có phải federated learning, đặc biệt việc chia sẻ thông tin từ dữ liệu local.
Phạm vi: headline `fd_dml_m4b_gpool` và họ vòng 15–17 (`frz`, `gkd`, `la`, `kdpx`, q2/q4), source `fd.py` r17 + `mh/` (đọc code, đo
trên smoke và trên data manifest đã audit). Không xét probe 14–19 như phương pháp (chúng dùng oracle có chủ đích, xem §5).

## 1. Cái gì rời client (đo bằng byte log của chính source)

| lúc | chiều | nội dung | byte/client | sinh từ dữ liệu local thế nào |
|---|---|---|---:|---|
| một lần (bootstrap) | lên | `prior`: đếm nhãn 16 lớp | 104 | đếm dòng train theo lớp |
| một lần | lên | `moments`: mỗi lớp n_c, Σx (66), Σxxᵀ tam giác trên (2.211), bit có lớp | 145.968 | thống kê đủ bậc 1–2 theo lớp của input |
| một lần | xuống | prior gộp; factor Gaussian gộp (μ_c, Cholesky Σ_c) | 96 + 72.912 | chỉ tổng 20 client |
| một lần | xuống | proxy khởi tạo | 1.593.440 | không từ dữ liệu |
| mỗi epoch | lên | trạng thái proxy (weights + BN running stats) sau 1 epoch local | 1.593.440 (q4 208.268, q2 109.512) | train trên dòng local + KL từ private |
| mỗi epoch | xuống | proxy gộp (= teacher `frz`/`gkd` epoch sau) | như trên | trung bình 20 client |

**Không bao giờ rời client:** dòng dữ liệu, nhãn từng dòng, activation/gradient theo dòng, model private (R2), EMA private, logits.
`m4b` (che 13 feature receiver), `gkd` (mẫu Gaussian vẽ tại chỗ từ factor gộp), `la` (dịch logit bằng log π_k − log π_g), `kdpx`
(trọng số loss proxy) đều tính tại client. Server: tổng/trung bình đồng đều (`Sum.finish` chia K, không lộ n_k).

⇒ Về cấu trúc: **FL thật** — họ federated mutual learning / FedKD (proxy chung FedAvg + private cá nhân hóa không gửi đi) cộng một lần
**chia sẻ thống kê lớp gộp** (cùng họ CCVR/FedMix: tri thức dạng thống kê, không phải record).

## 2. Điểm yếu phải công bố hoặc sửa

1. **Secure aggregation chỉ được giả lập.** `SecureSum` (`mh/v2_math.py`) là "trusted sum emulator… NOT cryptography/DP": người gọi chỉ
   nhận tổng, cổng ≥ 5 client. Triển khai thật cần SecAgg (vd Bonawitz et al. 2017). **Nếu không có SecAgg**, server thấy moment theo lớp
   của từng client; manifest train 20 client: mọi ô client×lớp > 0 nhưng có ô **n = 4 và n = 10 dòng** (12 ô ≤ 100, 34 ô ≤ 1.000) ⇒ Σx và Σxxᵀ
   của 4 dòng gần như lộ chính các dòng đó (vi phạm tinh thần "không record rời xe"). **Có SecAgg**: chỉ tổng gộp ≥ 215.429 dòng/lớp; client
   nhận tổng gộp trừ phần mình = tổng 19 client còn lại, một client khác chiếm tối đa 42 % ⇒ không cô lập được client nào.
   Đề xuất: SecAgg thật + ngưỡng tối thiểu theo ô (client bỏ lớp có n_c < ngưỡng khỏi bootstrap; ảnh hưởng ≤ 34 ô rất nhỏ so tổng) và/hoặc
   nhiễu DP trên Σx, Σxxᵀ (dữ liệu đã z-score, cần clip). Chưa đo utility.
2. **Proxy gửi rõ như FedAvg thường.** `Sum.add` giải mã từng client ⇒ server thấy proxy (và BN running stats) của từng client — rủi ro
   gradient/model inversion chuẩn của FL. Dense là tổng ⇒ tương thích SecAgg; **q4/q2 (`QPacket`) có scale theo hàng riêng từng client**,
   server giải mã từng gói ⇒ không cộng được dưới SecAgg cộng tính như hiện cài; muốn dùng R1 cùng SecAgg cần scale chung (vd scale do server
   phát trước round). Headline hiện dense.
3. **Scaler z-score fit tập trung trên 43 M dòng train gộp** (gồm cả holdout nội bộ) — do dữ liệu được giao đã chuẩn hóa
   ([dataset §7](../../knowledge/dataset.md)); giữ nguyên để đối chiếu `fl_dagsnet_veremi` (cùng scaler). Bản FL tương đương: SecureSum
   Σx, Σx² theo feature (132 số/client), trùng giá trị trừ phần holdout. Phải ghi trong báo cáo.
4. **Đánh giá/chọn lựa dùng holdout gộp ở server.** `group`/`old` (gộp mọi client) và test20% toàn cục dùng cho cổng và cho chọn luật
   head/gp — protocol nghiên cứu mô phỏng, không có trong triển khai. Luật triển khai cố định theo nguồn teacher (không tune theo client);
   bản FL của bước chọn: client chấm holdout riêng, gửi confusion 16×16 qua SecAgg.

## 3. Lỗi cài đặt nhỏ (cùng thông tin, sai đường code)

- `la_shift` tính log π_g trực tiếp từ nhãn mọi client trong tiến trình mô phỏng (`Run.__init__`) thay vì từ tổng SecureSum (`prior()`
  ghi đè `R.log_pg` sau đó). Đo smoke: lệch tối đa 2,4e-7 (làm tròn float), cùng một đại lượng gộp. Sửa ở tag source sau (lấy shift sau
  `m.prior()`); không sửa giữa chừng để ra/rc vòng 17 giữ bit-exact với source đã push. Luật gp dùng đúng giá trị SecureSum. **Đã sửa ở source r18.**

## 4. Đọc chéo client chỉ trong mô phỏng, không vào huấn luyện

- `compile_preflight`/`compile_extra` chạy forward trên 512 dòng client 0 để kiểm torch.compile, rồi khôi phục BN + RNG (test bit-exact PASS).
  Đã đổi ở source r18 sang dòng client của chính tiến trình.
- Mọi tiến trình nạp nhãn mọi client để đếm (`R.counts`) = đúng giá trị SecureSum; bootstrap Gaussian tính trong một tiến trình cho 20 client
  = từng client tính rồi cộng (code `upload()` theo từng client, cộng gói đã mã hóa).
- Chẩn đoán (`kd_probe`, `syn_accept`, drift) tính trên dữ liệu của chính client, chỉ ghi log.

## 5. Oracle trong probe (không phải phương pháp)

probe 14–19 dùng: dòng thật client khác (cov_real, refit gộp), nhãn toàn cục qua chính private (bias F1 oracle). Đó là phép đo trần; hướng
cần thông tin này bị loại đúng vì không làm federated được (probe 19 FAIL hợp đồng). Không phương pháp đang chạy nào dùng chúng.

## 6. Kết luận

Huấn luyện của headline và các arm vòng 15–17 là **federated**: dữ liệu và model private ở lại client; trao đổi = proxy (mỗi epoch) +
thống kê lớp gộp (một lần) + prior gộp. Điều kiện để nói "riêng tư mức FL": SecAgg thật cho bootstrap (bắt buộc vì có ô 4–10 dòng) và
nên cho proxy. Phải công bố: SecAgg giả lập, scaler tập trung, holdout gộp cho chọn lựa. (b) `la` lấy prior từ SecureSum và (c) preflight
không dùng dữ liệu client 0: đã sửa ở r18.

## 7. Biện pháp bảo mật để dành (owner 18/10 ~09:15Z: ghi lại, bổ sung khi cần; mục tiêu hiện tại là performance)

Chưa cài; không đổi kết quả đang báo cáo. Khi cần một bản "riêng tư mức FL" thì làm theo thứ tự:

| # | biện pháp | chặn | chi phí / ảnh hưởng dự kiến |
|---|---|---|---|
| a1 | ngưỡng ô tối thiểu: client bỏ lớp có n_c < ngưỡng (vd 100) khỏi bootstrap moment/histogram | lộ dòng ở ô 4–10 dòng khi không có SecAgg | ≤ 34 ô bị bỏ, khối lượng rất nhỏ so tổng ≥ 215k/lớp; phải chạy lại đối chứng |
| a2 | nhiễu DP Gaussian trên Σx, Σxxᵀ (và histogram `cop`), clip dòng đã z-score | suy ngược thống kê bootstrap | cần chọn ε; moment bậc hai nhạy nhiễu, phải đo utility |
| d | scale chung cho q4/q2 (server phát scale trước round) để gói lượng tử cộng được dưới SecAgg | server thấy proxy từng client khi dùng R1 | chỉ khi headline dùng q4/q2; headline hiện dense |
| s | SecAgg thật (Bonawitz et al. 2017) cho bootstrap và proxy | mọi điểm trên ở mức giao thức | chi phí giao tiếp thêm, không đổi phép toán |
| f | scaler fit bằng SecureSum Σx, Σx² theo feature | scaler tập trung | trùng giá trị trừ phần holdout; giữ scaler hiện tại để đối chiếu `fl_dagsnet_veremi` |

