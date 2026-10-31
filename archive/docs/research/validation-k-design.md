# K — kiểm tương tác, bỏ teacher và conditional cache (31/10/2026)

**Đề nghị 8 run mới, full 5.343 xe × 10 round.** Ưu tiên hoàn thiện factorial H3×J4 ở seed42/43;
thêm augmentation-only và conditional cache ở cả hai seed. Không mở lại GMM/MADE/NTD/hidden-KD/BN/head.
Đây là thiết kế nghiên cứu, chưa có utility K; nguồn thực thi/HP/manifest phải khóa riêng trước push.
Giữ [protocol](protocol.md), W1/batch64/cosine50 dừng r10, hai T4, một epoch/mọi xe/round; không chọn best-round.
Seed43 là replication bổ sung được owner duyệt, không sửa protocol headline seed42.

## Bằng chứng và mức kết luận hiện tại

Nguồn: [H đã audit](screening-h10.md), [thiết kế J](screening-j10.md), artifact local `runs/j-screening/audit.json`,
[code](../../methods/mobihet.py), [chẩn đoán teacher](synthetic-kd-next.md).
Số J dưới là kết quả audit CM/source/config hiện có; báo cáo cuối vẫn phải ghi rõ phạm vi pull/snapshot/state của parent.

| arm, seed42 | F1 pooled r10 | p10 | điều suy ra được |
|---|---:|---:|---|
| G4: 32 dòng, KL real bật | 0,873187515 | 0,573761905 | nền khóa |
| H3: 64 dòng, KL real bật | 0,876895 | 0,608332 | dose có lợi mô tả, chưa qua cổng |
| J4: 32 dòng, KL real tắt | 0,876816745 | 0,566416781 | pooled tăng nhưng p10 giảm; chưa qua cổng |
| J1: anchor4096, teacher từng mẫu | 0,871238493 | 0,578149 | chưa hỗ trợ ưu tiên anchor ensemble |
| J3: J1 + dose64 | 0,872993046 | 0,616190 | p10 tốt hơn nhưng pooled chưa qua cổng |

J0/J2 cũng chưa qua cổng. Anchor label-agreement J1≈0,8828/J3≈0,9033 vẫn không bảo đảm pooled gain.
Không kết luận “mọi teacher theo input vô ích”: J chỉ kiểm teacher đánh giá trên bank synthetic và support hữu hạn.
Q từ dòng thật là kênh khác; nhưng dữ liệu hiện tại chưa chứng minh nó sẽ có ích.

H3 và J4 mỗi arm tăng khoảng0,0036–0,0037. Nếu cộng thuần, combo≈**0,880524**, vẫn dưới cổng
**0,883187515** khoảng0,002663. Combo cần tương tác dương mới PASS; không dự đoán chắc vượt cổng.
J4 timeDelay F1≈0,239879 cho thấy ablation có thay đổi lớp mục tiêu; phải so đủ CM/per-class với G4,
không quy mọi gain cho timeDelay hoặc coi nhãn timeDelay khó là bằng chứng duy nhất.

## Batch 8 run và factorial tái dùng

| ID đề nghị | seed | dose | KL real | synthetic teacher | mục đích |
|---|---:|---:|---|---|---|
| K0 | 42 | 64 | tắt | Q[y] | combo H3×J4 |
| K1 | 43 | 64 | tắt | Q[y] | lặp combo |
| K2 | 43 | 64 | bật | Q[y] | ô H3 còn thiếu seed43 |
| K3 | 43 | 32 | tắt | Q[y] | ô J4 còn thiếu seed43 |
| K4 | 42 | 64 | tắt | CE nhãn synthetic, không teacher học được | đối chứng augmentation-only |
| K5 | 43 | 64 | tắt | như K4 | lặp đối chứng |
| K6 | 42 | 64 | tắt | Q[y,h(x)] từ dòng thật | cache conditional exploratory |
| K7 | 43 | 64 | tắt | như K6 | lặp cache |

Tái dùng seed42 G4/H3/J4 đã audit; seed43 G4 là **H0b**, F1≈0,874508/p10≈0,568757.
Không chạy lại ô đã có chỉ để tạo cùng batch. Trước tái dùng, kiểm source đường mặc định tương đương, HP, topology hai GPU,
RNG/grouping và support. Run mới độc lập r1, không giả resume từ snapshot thiếu trạng thái private.

Với mỗi seed, gọi F(d,r) là F1 r10, d∈{32,64}, r∈{KL bật,tắt}. Báo cả bốn ô và:

\[
I=F(64,tắt)-F(64,bật)-F(32,tắt)+F(32,bật).
\]

I>0 chỉ là tương tác mô tả của hai thao tác tại cấu hình này; không đủ chứng minh cơ chế sinh học/thống kê hay tối ưu.
Báo cùng phép tương phản với p10, mean/xe và từng lớp. H3 tăng số dòng, vẫn lấy trung bình loss synthetic;
không mô tả thành nhân đôi hệ số KD. J4 chỉ bỏ KL real: Q vẫn ảnh hưởng synthetic và adaptive weight.

## Đối chứng không có teacher học được

K4/K5 dùng cùng copula streaming, dose64, prior global/local và LA trên real như K0/K1:

\[
L=\operatorname{CE}(z_r+\log\pi_k-\log\pi_g,y_r)
 +\operatorname{mean}\operatorname{CE}(z_g,y_g).
\]

Hệ số synthetic **1 mọi round**, khớp công thức round1 hiện hành. Không KL, không Q trong loss hay adaptive weight;
không thu/gửi/nhận bảng xác suất private theo round. Giữ forward real+synthetic chung và đường BN/dropout như combo.
Không thêm LA cho synthetic. Tính loss riêng để kiểm đúng công thức.

Đây là **đối chứng bỏ toàn bộ kênh teacher học được, gồm chuẩn hóa adaptive**, không cô lập riêng soft-target vs hard-label.
Nếu K4≈K0, chưa có bằng chứng kênh Q+adaptive mang giá trị thêm tại r10; nếu K4 tốt hơn, không tự suy KL là nguyên nhân duy nhất.
Muốn tách soft-target và normalization cần batch sau với hệ số khớp, khóa trước kết quả; không âm thầm thêm run trong K.

**Vẫn chia sẻ thống kê copula và prior toàn cục.** Không gọi local-only, không gọi không giao tiếp, không dùng nó chứng minh
không cần thông tin client khác. Bootstrap copula vẫn tính đủ; payload teacher steady-state bằng0.
Đối chứng G0 cũ tắt synthetic còn giữ KD real, nên không thay được K4/K5.

## Cache conditional: cơ chế và giới hạn khoa học

Q[y] xóa khác biệt dự đoán trong lớp. Thu conditional table trên **dòng thật đã forward trong epoch** giữ một phần thông tin đó,
không cần private đánh giá thêm anchor copula ngoài support. Dùng ba vector sign cố định R∈{−1,+1}^{3×77}:

\[
h(x)=\sum_{j=0}^{2}2^j\,1[(Rx)_j\ge0],\quad
N_{kyb}=\#\{i:y_i=y,h(x_i)=b\},\quad
Q_{ybc}=\frac{\sum_kN_{kyb}\widehat m_{kybc}}{\sum_kN_{kyb}}.
\]

m là mean xác suất private trên các dòng thật tương ứng trong epoch, không phải private eval cố định cuối epoch.
Counts dùng toàn train+val theo protocol, không đọc test; teacher roundr chỉ dùng ở roundr+1.
Mỗi client upload mean fp16; server **decode wire trước gộp**, gộp theo N, downlink mean fp16 rồi decode/chuẩn hóa.
Không gửi sum fp16 vì count lớn có thể overflow. Bucket rỗng toàn cục fallback Q[y] có trọng số count từ cùng bảng;
không uniform giả, không phát sinh0/0. Label y chỉ dùng lúc training; inference private không cần label/router/server.

**Khuyến nghị khóa raw theo marginal Q[y]**, suy từ cùng table/count, chỉ đổi target synthetic thành Q[y,h(x_g)].
KL real giữ tắt như combo. Như vậy K6−K0 hỏi tác động conditional target rõ hơn.
Nếu implementation đổi cả CE_teacher trong raw sang conditional, phải ghi deviation trước push: phép so đồng thời đổi
target và strength; cần log raw để thấy khác biệt. Normalization/fp16 wire của cache khác đường Q hiện hành cũng là giới hạn suy luận.

Ba bit/B8 là lựa chọn dung lượng nhỏ để sàng lọc, **chưa có bằng chứng B8 hoặc router ngẫu nhiên hợp ranh giới VeReMi**.
FedCache dùng encoder pretrained, cache từng mẫu và hàng xóm cùng lớp; table8bucket trên input thô là adaptation của dự án,
không phải tái hiện FedCache, không thừa kế kết quả paper. [FedCache, §III-B/C](https://arxiv.org/html/2308.07816v1).

Random hyperplane chuẩn cho angular LSH dùng phân bố đối xứng cầu và có xác suất collision theo góc;
Rademacher±1 không thỏa đối xứng cầu tổng quát. Không viện định lý đó để bảo đảm semantic similarity của router này.
[Laarhoven, §2, Eq.3–4](https://arxiv.org/html/1702.05760v1).
Ngay cả Gaussian đúng định lý cũng chỉ giữ angular neighborhood, không chứng minh bucket cùng decision boundary.
Feature scalar đã scale không làm 77 chiều có cùng ý nghĩa; ba phép chiếu có thể bị nhóm feature tương quan/atom/offset chi phối.
Đặc biệt, router qua gốc không phân biệt độ lớn trên cùng tia; synthetic cùng bucket thật vẫn có thể ngoài manifold.

Ưu tiên một router cố định chung hai seed train; hash source/matrix trong provenance. Nếu generator dùng seed42/43 tạo router khác,
ghi rõ replication gồm **SGD+router**, không diễn giải chênh như SGD-only. Không thử nhiều router rồi chọn theo test.
Router train-only không khẳng định bảo đảm privacy; thống kê chia sẻ không có DP/secure aggregation mặc định.

## Bằng chứng → can thiệp → falsifier

| câu hỏi | bằng chứng hiện có | cơ chế K kiểm | dấu hiệu bác bỏ |
|---|---|---|---|
| dose và bỏ KL real cộng lợi? | H3/J4 tăng pooled riêng lẻ dưới cổng | thêm phủ synthetic trong khi bỏ ép real về Q[y] | combo không qua cổng, hoặc tương tác âm/đổi dấu giữa hai seed |
| cần teacher học được? | G0 không phải control không-teacher; H4 còn KL real/raw | giữ shared support/prior, bỏ cả Q+adaptive | augmentation-only không kém combo: chưa chứng minh Q+adaptive cần thiết |
| Q[y] gộp mất thông tin hữu ích? | Q timeDelay argmax benign; anchor từng mẫu pooled FAIL | conditional teacher từ real, không thêm OOD probe | buckets sụp/conditional≈marginal, hoặc F1 không tăng trên combo và không qua cổng |
| cache sửa nhầm timeDelay? | lỗi lớn timeDelay→benign, Q sai đích | bucket giữ miền có dự đoán khác nhau | Qconditional không phục hồi vùng nào, hoặc sửa recall nhưng phá precision/benign/p10 |

Không dùng entropy/fit/teacher agreement/probe PASS thay online utility. v1 P30 conditional cross-moment FAIL;
P35 GMM/MADE FAIL; P36 frozen-head PASS nhưng online FAIL; P37 BN/head không là nút thắt.
[Bài học v1](../../history/v1/LESSONS.md). Conditional cache khác P30 vì truyền target trực tiếp, không hồi quy input/gradient;
khác biệt cơ chế đủ cho một exploratory screen, chưa đủ lý do mở quét bucket/router hoặc bộ sinh mới.

## Chẩn đoán khóa trước kết quả

- Mọi run: CEreal, synthetic CE/KL, KLreal, raw mean/quantile, weighted synthetic loss, applied/skipped steps,
  train/eval/setup thời gian; runtime hai T4 thật, không suy từ FLOPs.
- Factorial: đủ10metric r10, per-class, benign, mean/p10/median/min theo xe; timeDelay↔benign và rare-class precision/recall.
  Đếm client được/sai thêm theo cặp và phân tầng train-size; không coi848xe nhỏ là nhóm duy nhất cần sửa.
- Cache: N[y,b], số client góp mỗi bucket, entropy occupancy và effective bucket count exp(H(B|y)), max bucket fraction,
  empty/fallback mass; cùng histogram bucket của synthetic, tỷ lệ synthetic rơi bucket ít support.
- Teacher: weighted trong lớp \(\sum_b(N_{yb}/N_y)\|Q_{yb}-Q_y\|_2^2\), entropy, true-class probability,
  argmax từng bucket/lớp; marginal từ bảng khớp estimator count-weighted. Teacher biến thiên không tự chứng minh teacher đúng.
- Liveness/source: seed, full IDs/support, source/overlay SHA, hash router, counts, packet decoded, finite loss/grad;
  snapshot hữu hạn không thay predictions→CM hoặc exact-resume proof.

Không tune từ test diagnostics. Nếu router sụp, báo hypothesis chưa được can thiệp hiệu quả và arm FAIL/degenerate;
không đổi router giữa batch. Cohort bootstrap paired CM theo xe có thể mô tả bất định lấy mẫu client,
không thay uncertainty từ training seed; hai seed chưa đủ claim statistical significance tổng quát.

## Chi phí dự tính, trước số đo

Payload/xe, hai chiều; header32B/packet như harness; chưa transport/secure aggregation. Counts int32 đủ cho37,36M dòng.

| biến thể | bootstrap B/xe | steady-state B/round | tổng10r B/xe |
|---|---:|---:|---:|
| combo/H3/J4 marginal Q | 3.311.744 | 1.088 | 3.322.624 |
| augmentation-only | 3.311.744 | 0 | 3.311.744 |
| cacheB8, thay Q cũ | 3.312.768 | 8.256 | 3.395.328 |

Cache mỗi chiều32+16×8×16×2=4.128B; counts thêm16×8×4×2=1.024B bootstrap nếu ghép packet hiện có.
Marginal Q phải suy từ bảng; gửi thêm packet Q riêng thì cộng đúng1.088B/round, không dùng số8.256 ở trên.
Nếu seed/schema/router gửi thêm hoặc packet-count khác, dùng byte wire thật. Toàn cohort nhân5.343; downlink tính từng xe.
Augmentation-only chỉ tiết kiệm10.880B/xe trong10r vì bootstrap chi phối; không mô tả như giảm chi phí tổng vô hạn.

Cache không thêm private forward/backward; thêm projection/scatter và communication/table reduction. Có thể cache bucket thật một lần:
37.361.273 byte cho toàn train nếu uint8, ≈18,7MB/rank chia đều; counts K×16×8 int32≈2,74MB toàn cohort.
Router ba vector tạo231 MAC/dòng; một lượt toàn train≈8,63tỷ MAC, chưa overhead tải/scatter.
Bucket synthetic vẫn tính mỗi bước; đo bằng profiler vì host-sync/shape guard có thể tốn hơn số MAC gợi ý.
Dose64 có thời gian H3≈631s/train-round r2–10, nền H0b≈552s; chưa dự báo combo/cached runtime hoặc quota từ tỷ lệ này.
8run × cap4h =32account-hour reserved tối đa, khác GPU-hour32×2; đo quota/account ngay trước push.

## Cổng và quyết định sau K

Headline seed42 giữ **F1≥0,883187514625593 và p10≥0,5637619047619047**.
Seed43 báo gain cùng seed so H0b; replication mạnh yêu cầu gain≥0,010 và Δp10≥−0,010 riêng seed43,
không dùng baseline42 cho seed43 hay lấy trung bình seed để cứu arm FAIL.
Ngoài cổng gốc, báo K6−K0/K7−K1 và K4−K0/K5−K1 để xét giá trị tăng thêm của cơ chế; không thay ngưỡng sau kết quả.

Nếu combo PASS và lặp dấu/gates tốt, cân nhắc validation horizon50 sau owner duyệt; screen10 không xác nhận final50.
Nếu augmentation-only ngang/tốt hơn, ưu tiên kết luận thiếu bằng chứng cần teacher hiện tại, rồi thiết kế control normalization khi cần.
Nếu cache teacher đa dạng mà utility FAIL, dừng B8 này; không mở router/bucket/generator sweep từ diagnostic đẹp.
Nếu mọi arm FAIL, giữ kết luận chưa có cải tiến vượt cổng, không đổi cổng và không gọi best-so-far là phương pháp đã PASS.
