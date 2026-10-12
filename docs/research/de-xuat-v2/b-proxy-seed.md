# V2-B — học proxy DAGSNet bằng hệ số trong cơ sở tham số sinh từ seed

Ngày 2026-10-11. **Thiết kế rủi ro cao, chưa chạy.**
[Index](README.md) · [hợp đồng chung](thiet-ke-chung.md) · [nguồn S3–S4](nguon-nghien-cuu.md).

## 1. Ý tưởng và điều phải đánh đổi

Thay vì gửi hàng trăm nghìn weight, mọi xe tái dựng proxy từ cùng một khởi tạo và K hướng tham số cố định:

\[
\theta(a)=\theta_0+Za,\qquad
Z=[z_1,\ldots,z_K]\in\mathbb R^{395024\times K}.
\]

Chỉ `a∈R^K` được học và trao đổi. `θ(a)` vẫn có đúng kiến trúc, shape và số tham số DAGSNet;
không thêm head/projection học được. Mọi tầng có thể thay đổi, nhưng **bậc tự do của proxy chỉ ≤K**.
Private DAGSNet vẫn học toàn bộ tham số, nhận KD trên input local từ proxy.

Khác V2-A: V2-A cho proxy học trong toàn không gian rồi nén update; V2-B giới hạn ngay không gian học.
Ưu điểm dự kiến: kích thước trạng thái bắt kịp luôn O(K), không dài dần theo số patch/version.
Rủi ro lớn nhất: proxy nhỏ về bậc tự do có thể lặp lại bài học proxy thiếu sức chứa, dù tên kiến trúc vẫn DAGSNet.

[FedKSeed](https://proceedings.mlr.press/v235/qin24a.html) dùng seed và scalar accumulator trong fine-tune LLM.
[FedKRSO](https://arxiv.org/html/2602.03019v1) hỗ trợ ý tưởng tối ưu first-order trong subspace,
nhưng dùng matrix accumulator và thay cơ sở theo round. Thiết kế fixed scalar basis dưới đây là adaptation riêng;
không nhận bảo đảm từ hai paper cho IDS DAGSNet khởi tạo ngẫu nhiên.

## 2. Cơ sở chung và trạng thái

Đề nghị hai cỡ để review: K=512 nhắm B0, K=1.024 làm đối chứng sức chứa. Chưa tự mở sweep.
`θ₀` từ khởi tạo DAGSNet có seed/hash, `a₀=0`. Basis lấy từ seed master và thứ tự tham số khóa trước.

Để tránh tầng lớn lấn át hoàn toàn, đề nghị với block tham số l:

`Z_l[j] = s_l × ξ_lj / sqrt(K)`, với `ξ_lj` Rademacher ±1,
`s_l = max(RMS(θ₀,l), 10^-3)`.

Scaling lấy từ khởi tạo public, không phải thống kê raw client. Đây là lựa chọn thiết kế, không phải tối ưu đã chứng minh.
Tất cả vector một chiều, conv, BN affine đều có mapping; không bỏ bias hoặc chỉ chiếu head cho tiện.
Running mean/variance BN là local buffers theo hợp đồng chung, không thể tái dựng từ a.

Không đổi seed pool giữa run. Đổi pool làm mất tính chất catch-up O(K) từ θ₀, cần định nghĩa lại base/checkpoint
và trả byte. Nếu sau này dùng basis đổi mỗi round thì đó là thiết kế khác, không được giữ bảng byte hiện tại.

## 3. Học first-order là nhánh chính

Ở version v, xe có a_v, tạo bản local a_i và tối ưu CE proxy:

\[
\nabla_a L_i=Z^\top\nabla_\theta L_i(\theta_0+Za).
\]

Train một epoch local, gửi vector có cùng tọa độ `n_i(a_i−a_v)` và count trong secure sum.
RSU khi đủ cohort ≥5 tính:

\[
a_{v+1}=a_v+{\sum_i n_i(a_i-a_v)\over\sum_i n_i}.
\]

Đây là FedAvg trong không gian hệ số. Chỉ nếu cùng θ₀ và Z thì trung bình hệ số tương đương trung bình
trainable weights đã tái dựng; không đúng với BN riêng hoặc seed khác. Local epoch vẫn có client drift.

Private dùng proxy snapshot để KD một chiều như V2-A, λ/T và lịch bật KD khớp đối chứng.
Trước epoch proxy, calibrate BN local rồi **đóng băng BN buffers và tắt dropout trong epoch proxy** ở mọi arm B,
để định nghĩa cùng objective cho probe FO/ZO. Private giữ train mode bình thường.
Dense control của B phải dùng chính policy này; không lấy dense control A hoặc H1 cũ làm thay thế.

Optimizer a cần lịch riêng do đơn vị a khác θ. Đề nghị khởi điểm AdamW wd0, LR cosine0,1→0,001,
clip norm hệ số1, reset theo version; đây **chưa được xác nhận**. Cần preflight kiểm độ lớn `Za`,
finite loss và bước giảm loss trước khi khóa pilot. Không bê LR1e-3 của weight sang a rồi kết luận subspace không học được.
Private giữ lịch protocol chung. Mọi thay đổi HP sau preflight được ghi trước run so sánh hai seed.

## 4. Zeroth-order chỉ là đối chứng cơ chế

Với hệ số j và cùng minibatch B, ước lượng:

\[
\widehat g_j={L_B(\theta(a)+\epsilon z_j)-L_B(\theta(a)-\epsilon z_j)\over2\epsilon}.
\]

Nếu chọn j đều trong K tọa độ, `K·ĝ_j e_j` là ước lượng coordinate gradient của loss làm trơn thích hợp,
không tự thành unbiased full-parameter gradient. Hai vế phải có cùng batch/RNG, không cập nhật BN giữa hai lần đo.
Đề nghị ε=10^-2 trong đơn vị hệ số, tính loss fp32; kiểm cancellation/finite difference trước dùng.

ZO cần hai forward cho mỗi direction; nhiều direction có thể đắt hơn một backward FO.
So FO/ZO bằng cả số batch thật đã đọc, loss evaluations và thời gian, không chỉ số round.
Nếu FO cùng K còn kém thì không có lý do kỳ vọng ZO giải được giới hạn sức chứa đó.
Nếu FO tốt nhưng ZO kém, nguyên nhân khả dĩ là ước lượng/tối ưu; chưa bác bỏ representation bằng seed.
Không bắt buộc chạy ZO trong pilot đầu; giữ nó như phép chẩn đoán có điều kiện, không tạo thêm arm mặc định.

## 5. Payload, catch-up, compute

Đề nghị coefficient fp32 hai chiều để giữ trạng thái cộng ổn định. Upload vector delta weighted K×4 B,
count uint64 và header32 B. Download a mới K×4 B +header32 B.

| K | upload ứng dụng | download ứng dụng | tổng wire tối thiểu theo frame24 B |
|---:|---:|---:|---:|
|512|2.088 B|2.080 B|**4.312 B/version/client**|
|1.024|4.136 B|4.128 B|**8.504 B/version/client**|

Chưa gồm prior, seed/schema/bootstrap, secure protocol, control và retransmission.
K512 chỉ gần trung vị B0; cohort delay và hai chiều không đảm bảo xong một contact.
K1024 cần nhiều contact B0 hoặc B2. Không dùng chiều xuống một mình để gọi “2 kB FL”.

Master seed + thuật toán RNG + schema/version + θ₀ hash đều phải phân phối có phí.
Nếu truyền danh sách K seed uint64 thay master seed thì thêm8K B; phải ghi lựa chọn thực trong codec.
Hash coefficient/basis chứng minh cùng representation; tái dựng float cần thứ tự arithmetic xác định.
Preflight kiểm weights tái dựng và logits với cùng BN trên thiết bị mục tiêu; không dựa vào seed PyTorch đơn lẻ
để hứa bitwise giữa mọi GPU/library version.

Xe bị trễ có thể bỏ vector a cũ chưa kích hoạt, tải a mới đầy đủ vì a là absolute accumulator.
Không cần replay toàn bộ chuỗi delta. Frame tải dở vẫn trả phí. BN phải được calibrate lại ở xe sau kích hoạt.
Đóng góp trên a_v cũ không được cộng trực tiếp vào cohort a_v+1; hủy/recompute với phí đã phát được ghi nhận.

**Compute không miễn phí:** một basis dense fp32 K512 cần `395024×512×4 =809.009.152 B`, khoảng772 MiB;
K1024 gấp đôi. `Zᵀg` và `Za` cần nhân/đọc basis, sinh lại từ seed cũng tốn thời gian.
Có thể sinh theo block thay lưu cả basis, nhưng chưa có benchmark. Không gọi phương án “nhẹ bộ nhớ”
chỉ từ kích thước payload; phải đo resident memory, FLOP/latency và năng lượng nếu sau này đánh giá thiết bị xe.

## 6. Cổng kiểm chứng

Trước huấn luyện private nhiều arm, kiểm chất lượng proxy ở U:

1. Dense proxy cùng CE/BN policy.
2. FO-seed K512; nếu đã chọn trước, thêm K1024 như đối chứng sức chứa.
3. Proxy chỉ học từ chính xe, cùng K/compute; không nhận aggregate.

Đề nghị proxy FO phải hơn private local thuần **≥0,05 group F1** cả hai seed, đồng thời giữ ≥80% gain
của dense proxy so proxy-local cùng policy. Trượt ⇒ không cần mở KD/contact cho cấu hình K đó.
Đây là cổng teacher đề nghị, không lấy nguyên mốc proxy P1 do BN và optimizer đã khác.

Sau teacher gate, private-KD phải qua cổng chung và hơn counterpart local-proxy-KD ≥0,01.
Ở U giữ ≥80% gain private của dense-KD so counterpart không trao đổi; ở C giữ≥70% theo hợp đồng chung.
Nếu chỉ K1024 thắng U, phải dùng byte K1024 để đánh giá C; không quảng cáo byte K512 kèm F1 K1024.

Math checks tương lai:

- Chain rule `Zᵀg` khớp finite difference trên fixture nhỏ, không sai shape/seed.
- Aggregate a khớp aggregate θ trong sai số số học đã khóa, với BN tách riêng.
- Absolute a snapshot tái dựng đúng sau bỏ nhiều version; resume giữ đúng optimizer/config/seed/state.
- Ghi norm update theo từng nhánh DAGSNet; không để lượng cập nhật tập trung vào head rồi gọi toàn model học hữu ích.

## 7. Giới hạn và quyết định tiếp theo

Không có pretrained DAGSNet cộng tác miễn phí. Nếu muốn warm-start từ model FL đã học trước,
phải có provenance hợp R3, trả toàn chi phí training/phân phối và baseline cùng warm-start.
Pilot đầu đề nghị khởi tạo ngẫu nhiên để bộc lộ giới hạn này.

Seed compression không phải bảo mật. Chỉ mở vector aggregate cohort; không công khai local scalar loss/directional gradient.
R2 qua KD giống A, hiện chỉ thiết kế. Không có thao tác tay cần làm trong phiên docs.
Kết quả cần tìm là “cần bao nhiêu bậc tự do để DAGSNet proxy còn truyền được ranh giới”, không phải chạy đến khi ZO thắng.
