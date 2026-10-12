# V2-A — proxy DAGSNet, sketch cộng được và bù sai số tại RSU

Ngày 2026-10-11. **Thiết kế chưa chạy. Ưu tiên đầu tiên.**
[Index](README.md) · [hợp đồng chung](thiet-ke-chung.md) · [nguồn S1–S2](nguon-nghien-cuu.md).

## 1. Giả thuyết và khác biệt với build trước

P1 cho thấy tham số model chung mang được ranh giới mà Gaussian/prototype bỏ sót.
P2 cho thấy chuyển nguyên proxy qua B0 quá chậm; tăng private F1 khi chưa nhận proxy không chứng minh FL.
Giả thuyết: **proxy giữ toàn bộ sức chứa DAGSNet nhưng mỗi version chỉ truyền phần cập nhật quan trọng;
phần chưa truyền được tích lũy tại RSU, đủ để private nhận tri thức sớm hơn full snapshot**.

Đây là hướng truyền thông cho proxy FL, không phải mô hình IDS mới. H6 cũ chia frame của object toàn mạng;
V2-A thay cả nội dung object. Khác local top-k: mọi client gửi sketch có cùng tọa độ, server giải nén sau khi gộp.

[FetchSGD](https://proceedings.mlr.press/v119/rothchild20a/rothchild20a.pdf) hỗ trợ cơ chế sketch tuyến tính;
[error feedback](https://proceedings.mlr.press/v97/karimireddy19a/karimireddy19a.pdf) giải thích giữ sai số nén.
Thiết kế dưới dùng **delta sau local epoch**, không nguyên stochastic-gradient FetchSGD; không kế thừa định lý hội tụ.

## 2. Model và objective được đề nghị

Mỗi xe có private `f_i` và proxy `g(θ,b_i)`, đều DAGSNet. `θ` là mọi tham số học được,
`b_i` là BN running buffers ở xe. BN affine thuộc θ; running buffers không nằm trong sketch.
Chọn BN riêng theo [hợp đồng chung](thiet-ke-chung.md); dense control phải dùng y hệt.

Để cô lập chất lượng carrier, vòng đầu dùng proxy CE và KD **một chiều**:

\[
L_{proxy,i}=E_{D_i} CE(g(\theta,b_i;x),y),
\]
\[
L_{private,i}=CE(f_i(x),y)+\lambda T^2
KL(\operatorname{sg}[\operatorname{softmax}(g(\theta_v,b_i;x)/T)]\Vert
\operatorname{softmax}(f_i(x)/T)).
\]

Đề nghị `T=2, λ=1`; bật KD sau khi xe nhận aggregate đầu tiên, teacher eval/frozen khi cập nhật private.
KD trên dữ liệu thật **của chính xe**, không upload x; gradient cập nhật toàn private backbone.
Không reverse-KD từ private sang proxy ở pilot đầu. Vì vậy baseline mới là dense proxy-CE+KD;
**không gọi đây là tái lập chính xác H1 DML**, cũng không thay số H1 cũ bằng baseline mới.

Proxy dùng bản clone của θ_v, train một epoch CE trên local train, AdamW/lịch đề nghị như private,
tạo `Δ_i=θ_i^local−θ_v`. Mỗi client chỉ tạo một contribution cho version v.
Khi contribution đang truyền/chờ cohort, giữ nó bất biến; private vẫn học với teacher snapshot đã kích hoạt.
Không tạo hàng đợi vô hạn delta sinh trên cùng base. Proxy compute/version và private compute/round báo riêng.

## 3. Thuật toán trao đổi

### 3.1 Tại client

1. Xác nhận đã có θ_v đầy đủ, đúng schema tham số, seed hash và phiên bản.
2. Recalibrate BN theo lịch trên local train; lưu checkpoint để audit. Không dùng group/old/test.
3. Train proxy clone, tạo Δ_i. Vector hóa theo thứ tự tham số cố định, tính sketch `S(n_i Δ_i)`.
4. Upload sketch và count `n_i` trong cùng phép tổng bảo vệ. Client không gửi chỉ số top-k riêng hoặc delta riêng.
5. Tiếp tục nhận patch còn thiếu; chỉ sau khi áp đủ patch chain mới được đóng góp ở base mới.

### 3.2 Tại RSU

Với cohort C_v đủ ≥5 xe cùng base, nhận **chỉ tổng** sketch và tổng count:

\[
\bar S_v={\sum_{i\in C_v} S(n_i\Delta_i)\over\sum_{i\in C_v}n_i},\qquad
A_v=E_v+\bar S_v.
\]

Decode ước lượng các tọa độ từ A_v; chọn k giá trị lớn nhất theo trị tuyệt đối, serialize patch,
giải mã lại thành vector thưa \(\widehat\Delta_v\) dùng đúng giá trị client sẽ nhận:

\[
\theta_{v+1}=\theta_v+\widehat\Delta_v,\qquad
E_{v+1}=A_v-S(\widehat\Delta_v).
\]

Không momentum ở cấu hình đầu để cô lập residual. Đây là residual **trong sketch**, không phải lưu vector
sai số thật đầy đủ; collision có thể làm decode sai. Không coi error feedback là bảo đảm không mất thông tin.

Residual trừ patch đã decode khi RSU commit canonical version, không trừ float trước fp16.
ACK từng xe chỉ ghi việc nhận version; không trừ residual lần nữa khi nhiều xe ACK cùng patch.
Commit canonical chưa có xe nhận không được tính thành lợi FL đã phục vụ.

Sketch seed/hash cố định suốt run để cộng residual. Trọng số n_i được gộp, không để RSU đọc count cá nhân.
Mỗi object có cohort/version ID; client/dropout khác version không đưa vào cùng tổng.

## 4. Byte và trạng thái chậm

Thông số dưới là **hai cấu hình ứng viên**, không phải kết quả hoặc sweep được tự phép chạy.
Counter sketch fp32, index patch uint32, value fp16; count aggregate uint64. DAGSNet có 395.024 tọa độ,
uint16 index toàn mạng không đủ. FP16 overflow/nonfinite là fail codec, không tự saturate âm thầm.

| ứng viên | sketch | k | upload ứng dụng | download ứng dụng |
|---|---|---:|---:|---:|
| A-small, nhắm B0 | 3×256 |128|3.072+8+32 = **3.112 B**|128×6+32 = **800 B**|
| A-medium, nhắm B2 | 5×2.048 |2.048|40.960+8+32 = **41.000 B**|2.048×6+32 = **12.320 B**|

Với công thức frame ở hợp đồng chung, A-small cần tối thiểu **3.208 B lên +824 B xuống =4.032 B**,
chưa bootstrap/prior/secure-aggregation/control/retransmit. A-medium: **41.984 B lên +12.632 B xuống**.
Hai chiều nhỏ không có nghĩa nhận được trong cùng contact: phải đợi đủ cohort rồi RSU commit.

Ban đầu θ₀ từ seed chung, source/initializer được pin và kiểm hash. Bootstrap descriptor có phí;
nếu không tái lập được model thì full state phải truyền thật. Seed không thay cho pretrained model miễn phí.
Một full parameter tensor fp16 đã **790.048 B**, chưa buffers/header; không dùng snapshot vô hình để cứu B0.

Xe v−j cần j patch cùng chuỗi, tổng byte tỷ lệ độ dài chuỗi. Đề nghị tối đa giữ 20 patch trong pilot;
nếu client vượt lịch sử lưu thì phải tải checkpoint có trả phí hoặc ghi chưa phục vụ được.
Không cộng gộp patch từ các base không liên tục rồi áp lên θ khác. Checkpoint resume giữ E_v, θ_v,
patch log, cohort, cursor và ACK; thiếu E_v không được gọi resume tương đương.

Small sketch là nén rất mạnh so 395k tham số. B0 có thể thất bại vì collision/ít tọa độ hữu ích dù đủ byte.
Nếu chỉ medium đạt cổng thì kết luận hướng phù hợp B2; không dùng con số B2 chứng minh B0.

## 5. Vì sao vẫn là FL và hỗ trợ R2 ở thiết kế

Mọi CE proxy được tính tại xe trên raw local; RSU chỉ tổng hợp sketch, không train với record.
Private không gửi weights/latent. Chỉ proxy cần cùng tọa độ DAGSNet; private tương lai nhận logits
trên x của chính mình, nên có thể khác kiến trúc và số chiều hidden. Hiện chưa chạy đa kiến trúc.

Không DP, không Byzantine robustness. Sketch không mã hóa bảo mật; ranh giới tổng cohort là yêu cầu riêng.
Lệch client do contact có thể làm proxy thiên về xe kết nối tốt; residual không sửa selection bias này.

## 6. Đối chứng và cổng bác bỏ

**Trước carrier:** dense proxy-CE+KD cùng BN policy phải có lợi riêng so local và counterpart không trao đổi.
Nếu dense teacher yếu, dừng V2-A; sửa teacher/BN là nghiên cứu khác, không tăng k để che nguyên nhân.

Đề nghị các arm U tối thiểu cho một kích thước sketch đã chọn:

| arm | thay đổi duy nhất cần đo |
|---|---|
| local | mốc IDS không auxiliary |
| proxy-local KD | proxy chỉ học từ chính xe; budget compute/lịch KD khớp arm trao đổi |
| dense aggregate KD | trần carrier mới, cùng optimizer/BN/KD |
| sketch-no-EF KD | sketch, không giữ residual |
| sketch-EF KD | phương án đầy đủ |

Ở proxy-local KD, lịch bật teacher khớp lịch nhận version của arm FL nhưng teacher vẫn chỉ là local;
nhờ vậy đối chứng không thắng/thua chỉ vì được bật KD sớm hơn. Lịch này dùng cho ablation, không là protocol triển khai.

Cổng riêng đề nghị:

- Dense proxy phải hơn private local ít nhất0,05 group F1; private dense-KD qua cổng chung.
- Sketch-EF giữ ≥80% gain private của dense-KD so proxy-local KD ở U; contrast với proxy-local KD ≥0,01.
- Nếu EF không hơn no-EF ≥0,005 cả hai seed: không tuyên bố EF tạo lợi; vẫn có thể giữ compression
  nếu qua các cổng còn lại, nhưng chọn thiết kế đơn giản theo kết quả, không gộp tên đóng góp.
- Cổng C B0/B2 và activation theo hợp đồng chung. Đếm số version đóng góp/nhận thực, không chỉ bytes_used.

Math probe cần đo dot/cosine giữa delta pooled và patch decode, norm residual theo version, loss của proxy
trước/sau update trên local diagnostic split; dot âm lặp lại là tín hiệu compressor sai hướng.
Các diagnostic raw record vẫn ở client, chỉ tổng scalar cohort phục vụ báo cáo; không dùng chúng tự tune k trên test.

## 7. Điểm cần chốt khi được yêu cầu triển khai

Chọn small hoặc medium làm cấu hình đầu; chốt exact BN recalibration budget, số cohort/version và lịch KD.
Tất cả đều là lựa chọn thuật toán cần ghi manifest, không phải việc thủ công bắt buộc hiện tại.
Rủi ro chính: proxy BN-local không giữ sức mạnh H1, local-delta bias, collision, patch catch-up và phiên bản ít
trong 20 round. Kết quả âm có giá trị: biết giới hạn độ nén có thể dùng của DAGSNet trên contact V2X.
