# MobiHet — phương pháp T3 được chọn

Owner chọn T3 ngày15/11/2026; trên bảng so sánh gọi **MobiHet**. T3 giữ nền CBB
(copula KM16 + codec lite6 + Q8), đổi liều synthetic theo cỡ xe và pha lịch.
[Protocol](protocol.md) giữ nguyên; [thiết kế và gate B21](screening-dose-b21.md) là hồ sơ lựa chọn,
[proposal](proposal.md) ghi quyết định, kết quả hiện hành theo [bàn giao](../operations/continue.md).

## Bootstrap một lần

Mỗi xe dùng fold train+val của chính nó; server chỉ nhận thống kê và phát thống kê gộp.

1. Histogram256bin theo lớp/feature cho77đầu vào → CDF chung. Biến đổi normal-score dùng mid-rank tại atom.
2. Moment lớp, sau đó k-means liên kết16cụm/lớp với4lượt Lloyd; xe gửi count/tổng/moment cụm.
   Covariance cụm co về covariance lớp với `cop_kmeans_w0=0.5`, loại cụm<20dòng.
3. Codec lite6: moment/tâm int8, sparse+zlib; Cholesky cụm phần dưới6bit, đường chéo/scale fp16;
   CDF lossless. Mỗi xe decode và tự sinh mẫu từ mixture trong latent copula, ánh xạ qua CDF ngược.

Không gửi dòng thật, model, gradient hoặc BN. Thống kê ô nhỏ vẫn có thể tiết lộ thông tin;
implementation chưa có SecAgg/DP. Schema cho phép kiến trúc khác nhau, chưa chứng minh full-scale mixed-model.

## Liều T3 — không dùng cố định64dòng mọi xe

$n_k$ là số dòng train+val của xe, $s_k=\lceil n_k/64\rceil$.

| Tầng | Điều kiện | Synthetic/bước r1–25 | Synthetic/bước r26–50 |
|---|---|---:|---:|
| S | n_k<4096 |64|64|
| M |4096≤n_k<16384|64|32|
| L |n_k≥16384|48|16|

Mỗi round đúng1epoch dòng thật, batch64. Nếu $s_k<16$, thêm bước chỉ synthetic tới floor16;
không lặp thêm epoch thật. Synthetic labels cân bằng16lớp. Các dòng vượt liều của xe bị mask khỏi loss/BN.
Đây là giảm công việc hữu ích; engine grouped vẫn có thể thực thi dòng đệm nên FLOPs lý tưởng không suy ra wall time.

Cấu hình khóa: `synth_tier_min=[4096,16384]`, `synth_tier_rows=[[64,64],[64,32],[48,16]]`,
`synth_late_from=25`, `synth_step_floor=16`; method `mobihet_compute`.

## Loss và bảng teacher Q

Prior được làm trơn Laplace: $\pi_{k,c}=(n_{k,c}+1)/(n_k+16)$;
$\pi_{g,c}=(\sum_k n_{k,c}+1)/(\sum_k n_k+16)$.
Loss thật là CE trên logits $z+\log\pi_k-\log\pi_g$; synthetic dùng logits thô.

Round1 chưa có Q: synthetic hard-label CE. Từ round2: KL$(Q_{r-1}[y]\parallel\mathrm{softmax}(f_k(\tilde x)))$,
trọng số1. Loss thật và synthetic đều lấy mean riêng trên dòng hợp lệ rồi cộng;
không nhân loss synthetic với tỉ lệ số dòng.

Trong lượt train dòng thật, tích lũy xác suất dự đoán theo true class (online, không thêm forward thu Q).
Server gộp tổng xác suất/count theo lớp; Q của round r chỉ dùng từ r+1.
QQ8 lượng tử hóa xác suất8bit, uplink chỉ lớp có mặt; downlink bảng Q chung.
Q là teacher có điều kiện theo lớp, không phải classifier theo input.
Byte từng round phải đọc `bytes_up`/`bytes_down`; không chép số CBB thành số đo T3.

## Thực thi, resume và audit

Nguồn: [T3/loss tầng](../../methods/mobihet_compute.py), [copula/LA](../../methods/mobihet.py),
[KM16/codec](../../methods/mobihet_sim.py), [floor/compact bootstrap](../../methods/mobihet_wstudy.py).
BERT port ở repo `../fl_veremi_bert`, dùng cùng HP/T3 với engine FeatureBERT; model không buffer.
Baseline dùng group_rows8192; T3 production4096 như frozen build. Không đổi protocol để tối ưu riêng method.

Checkpoint đủ resume phải có private bank, teacher, method/driver RNG và CPU/CUDA RNG cả2rank cùng round;
optimizer local khởi tạo lại mỗi round. Snapshot nhỏ và W&B resume không thay checkpoint.
Headline r50; model riêng chấm own-test, gộp đủ5343xe/support để báo10metric. Không chọn best round qua test.
Audit CM/source/config/finite/state có phạm vi riêng; chưa per-row prediction→CM replay toàn mạng,
chưa dataset full-content fingerprint. Ba seed không tự chứng minh significance.
