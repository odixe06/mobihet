# Synthetic + KD tiếp theo — nghiên cứu, 30/10/2026

**Đề xuất ưu tiên: teacher theo từng dòng synthetic, truyền thẳng private → ensemble → private.**
Đây là đề xuất chưa train/chưa chứng minh tăng F1. Không gọi tối ưu; không mở lại bộ sinh GMM/MADE/cvp/cop v1.
Bản ghi này chỉ nghiên cứu; không launch, không sửa code. Nền và cổng vẫn theo [H screening](screening-h10.md),
[protocol](protocol.md), [improvement plan](improvement-plan.md). F1 ưu tiên; byte và thời gian phải báo thật.

## Bằng chứng và khoảng trống

- `none` hiện tại ([code](../../methods/mobihet.py)) tích lũy xác suất private trên **dòng thật trong epoch**, gộp thành
  \(Q_{yc}=\sum_{k,i:y_i=y}p_{k,c}(x_i)/N_y\). Cả real và synthetic dùng cùng \(Q_y\), bất kể \(x\).
  Đây là teacher phụ thuộc nhãn, chưa phải hàm quyết định theo mẫu. Đổi bộ sinh không phục hồi thông tin đã bị gộp mất.
- G4 copula r10 F1=0,87319, S4 Gaussian=0,86790; +0,00529 chưa qua cổng +0,010. Oracle sampler=0,8802:
  khoảng cách tham khảo, **không phải cận trên toán học**. Xem [kết quả S/G](screening-r10.md).
- S1 `srv` FAIL: phải học proxy từ ensemble trên synthetic rồi private học proxy. Chưa cô lập đường **ensemble từng mẫu → private trực tiếp**.
- v1: synthetic nhìn thật hơn không bảo đảm utility, online có thể bác bỏ frozen-head; NTD/hidden-KD và GMM/MADE đã FAIL.
  Không dùng acceptance, KL fit, teacher agreement hoặc probe làm bằng chứng tăng F1. Xem [LESSONS v1](../../history/v1/LESSONS.md).
- Số r10 **đã audit** provenance/metric, xem [kết quả H](screening-h10.md): H1=0,87394, H2=0,86982,
  H3=0,87690, H4=0,86870; mọi arm FAIL cổng gain +0,010 vs G4. H3 p10=0,60833 cho dấu hiệu liều có thể hữu ích,
  chưa chứng minh tương tác với teacher mới. H0 paired copula−gauss seed43=+0,00546, cùng dấu seed42;
  hai seed không đủ kết luận ý nghĩa thống kê. Không diễn giải H4 yếu hơn thành bằng chứng q_i chắc chắn tốt.

## 1. Shared-anchor KD trực tiếp — ưu tiên đầu

**Nguồn.** FedMD cho phép kiến trúc private khác nhau bằng cách trao đổi điểm lớp trên tập chung, gộp theo từng mẫu,
client học consensus trực tiếp. Bài dùng public data và pretraining, nên ứng dụng seeded synthetic ở đây là **adaptation**,
không thừa kế kết quả bài. [FedMD, Algorithm 1 và §2.2](https://arxiv.org/html/1910.03581v1).
CFD nghiên cứu giảm byte bằng lượng tử soft label và delta coding; không chứng minh int8 giữ F1 trên VeReMi.
[CFD, tác giả/Fraunhofer](https://iphome.hhi.de/samek/pdf/SatTNSE22.pdf).

**Thiết kế tối thiểu.** Giữ `arm=none`, real CE + LA, real KD Q[y], liều synthetic 32/bước và hệ số adaptive hiện hành.
Không thêm proxy. Sinh bank copula cân bằng \(A_r=\{(x_i,y_i)\}_{i=1}^{M}\) bằng seed chung;
đề xuất F1-first \(M=4096\), bằng kích thước transfer đã khóa trước trong S1. Cuối round r, mỗi private đã train
forward ở eval mode trên \(A_r\), upload xác suất uint8. Class count \(n_{k,y}\) đã bootstrap:

\[
\widehat p_{k,i}=\operatorname{decode8}(p_k(x_i)),\qquad
q_i^r=\operatorname{decode8}\!\left(\frac{\sum_k n_{k,y_i}\widehat p_{k,i}}{\sum_k n_{k,y_i}}\right).
\]

`decode8` nghĩa round(255p), chia lại tổng mỗi hàng; server phải gộp **giá trị đã decode wire**, downlink cũng decode thật.
Không giả lập byte uint8 nhưng train bằng q float32 chưa nén. Trong r+1, lấy 32 hàng của **đúng bank A_r**:

\[
L_g=\frac1{32}\sum_i D_{KL}(q_i^r\Vert p_k(x_i)).
\]

Bank có thể đổi mỗi round nhưng seed/index/teacher phải đi cùng nhau. R1 giữ CE synthetic như nền;
round r dùng bank đã được teacher đánh giá ở cuối r−1. Không KD target cũ trên dòng synthetic mới.
Mọi client vẫn tham gia; model/representation private không truyền. Trọng số class count dùng nhãn sinh, không phải độ tin cậy mẫu:
client có lớp vẫn có thể sai ngoài support. Không gọi trọng số này tối ưu.

**Bằng chứng → cơ chế → bác bỏ.** Q[y] xóa khác biệt trong lớp → q_i bảo toàn khác biệt dự đoán theo x,
bỏ bước xấp xỉ bởi proxy S1 → bác bỏ nếu r10 không qua cổng hoặc q_i gần hằng theo lớp và không hơn control cùng bank.
Đo \(V_y=M_y^{-1}\sum_{i:y_i=y}\|q_i-\bar q_y\|_2^2\), entropy, nhãn argmax, coverage từng lớp; chỉ là chẩn đoán.
Nếu teacher đã khác nhau nhưng F1 không tăng, không tiếp tục chỉ vì V lớn.

**Control cần thiết.** Chuyển infinite fresh sampler sang bank hữu hạn cũng đổi support. Chạy cùng bank nhưng teacher Q[y]
để cô lập sample target. Nếu quota chỉ đủ hai run: shared-bank Q[y] vs shared-bank q_i; vẫn báo cả hai so G4.
Không kết luận q_i là nguyên nhân nếu chỉ so q_i bank với G4 fresh.

## 2. KD có chọn lọc theo mẫu — arm tiếp sau 1

**Nguồn.** Selective-FD phân biệt tri thức sai ngoài miền client và ensemble mơ hồ; dùng client density-ratio selector và
server confidence selector. Bài cảnh báo lọc quá mạnh làm thiếu mẫu và tăng phương sai. Không thể thay bằng confidence tùy ý rồi
gọi là tái hiện nguyên bài. [Selective-FD, Nature Communications 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10774276/).

**Adaptation rẻ, tách riêng.** Giữ anchor ensemble arm 1, không train discriminator/model thứ hai.
Từ các prediction client đã upload, tính mức bất đồng Jensen–Shannon theo từng anchor:

\[
J_i=H(q_i)-\sum_k w_{k,i}H(\widehat p_{k,i}),\quad
s_i=\max(0,1-J_i/\log 16),\quad
L_g=\frac1{32}\sum_i s_i D_{KL}(q_i\Vert p_k(x_i)).
\]

Công thức này cần q trước downlink rounding để J không âm do lượng tử; clamp cuối vì số học.
Đây là trọng số mềm do dự án đề xuất, **không phải selector của bài**. Giữ chia 32, không chia tổng s để tránh
vô tình khuếch đại mẫu còn lại; vì vậy phải báo effective KD mass để giải thích giảm liều.
Gửi thêm s_i uint8, không đổi số dòng synthetic hay optimizer. Client không có lớp y_i có w=0 như arm1.

**Bằng chứng → cơ chế → bác bỏ.** Ensemble trên synthetic có thể trung bình các private không biết mẫu đó → giảm KD khi private
bất đồng → bác bỏ nếu không hơn arm1, rare-class effective mass sụp, hoặc hại p10. Agreement cao vẫn có thể đồng loạt sai;
đây không sửa support giả một cách bảo đảm. Chỉ dùng nếu arm1 cho thấy khác biệt theo mẫu và vấn đề teacher disagreement.

## 3. Cache Q[y, bucket(x)] từ dòng thật — phương án song song rẻ hơn

**Nguồn.** FedCache dùng hash chung, quan hệ hàng xóm cùng lớp và cache logits theo sample để cấp teacher theo mẫu;
nguyên bài cần encoder pretrained, hash/index từng mẫu và giao tiếp từng mẫu.
[FedCache §III-B/C](https://arxiv.org/html/2308.07816v1). Không bê nguyên thiết kế này sang 37 triệu dòng.

**Adaptation tối thiểu.** Router chung không train: 3 random hyperplane Rademacher cố định trên 77 input đã scale,
\(h(x)\in\{0,\ldots,7\}\). Thay mỗi Q[y] bằng bảng 8 bucket/class, tích lũy ngay từ forward real đã có:

\[
Q_{ybc}=\frac{\sum_{k,i}1[y_i=y,h(x_i)=b]p_{k,c}(x_i)}{N_{yb}},\qquad
T(x,y)=Q_{y,h(x)}.
\]

Counts N_yb bootstrap một lần vì router/data cố định. Gộp mean có trọng số count; upload **mean** fp16 để tránh overflow
khi sum xác suất >65504. Bucket rỗng dùng Q[y] suy từ cùng bảng/count. Real và synthetic đều dùng T(x,y),
không thêm forward và không buộc feature/hidden architecture giống nhau. Label y chỉ dùng lúc train KD; inference không cần router/nhãn.

**Bằng chứng → cơ chế → bác bỏ.** Real conditional probabilities có support thật nhưng bị Q[y] gộp mất → giữ một ít cấu trúc theo input,
không cần teacher đoán trên copula → bác bỏ nếu bucket entropy quá thấp, conditional Q không khác Q[y], hoặc F1 không qua cổng.
Random hyperplane có thể chia sai yếu tố quyết định; nếu FAIL không suy ra mọi cache có điều kiện đều vô ích.
Khác v1 P30: không hồi quy cross-moment thành synthetic gradient/input; dùng trực tiếp bảng conditional teacher.
Tuy nhiên P30 FAIL nhắc rằng thêm conditional statistic vẫn chưa bảo đảm utility. Không claim secure aggregation/DP chỉ vì additive sums.

## Chi phí dự tính — chưa phải số đo runtime

Mọi số dưới là payload/xe, hai chiều, header 32 B mỗi packet như harness; chưa transport/secure aggregation.
Bootstrap copula nền giữ **3.311.744 B**, Q nền **1.088 B/round**, theo [cost](cost.md).
Bank sinh bằng seed; không gửi input M×77. Downlink tính **mỗi xe**, không chia K vì multicast.

| đề xuất | steady-state hai chiều/round | thêm bootstrap | công việc thêm |
|---|---:|---:|---|
| 1, M4096 uint8 soft | 132.224 B gồm Q cũ + anchor | seed/schema nhỏ, phải đo | 4.096 eval-forward/xe/round |
| 1, M1024 uint8 soft (chưa chọn) | 33.920 B gồm Q cũ + anchor | như trên | 1.024 eval-forward/xe/round |
| 2 trên M4096 | arm1 +4.096 B downlink nếu chung packet | không | entropy/JS server; s byte |
| 3, B8, mean fp16 | 8.256 B, thay Q cũ | +1.024 B counts128 int32 lên+xuống, nếu chung packet | 3×77 projection/row + scatter |

Arm1 M4096: anchor packet hai chiều = 2×(32+4096×16)=131.136 B; thêm Q=132.224 B.
Tổng 10 round copula+arm1 = **4.633.984 B/xe** nếu cả 10 round đều gửi/nhận đủ hai packet;
r1 chưa có anchor downlink nên phải dùng log packet thật khi báo cuối. Count/seed/header phát sinh cũng cộng đúng.
Ở 5.343 xe, probe4096 =21.884.928 forward-row/round, khoảng 58,6% số real train-row; đây không phải +58,6% thời gian train:
train còn backward và synthetic. Không dự báo GPU-giây bằng tỷ lệ row; đo probe/train/eval/serialization riêng.

## Batch sàng lọc đề nghị

Parent đang chuẩn bị batch dưới; đây là spec định hướng, bản khóa chính thức/source manifest do parent ghi trước push:

| arm | support/target | câu hỏi |
|---|---|---|
| J0 | copula anchors4096, Q[y], 32 dòng/bước | control finite bank vs G4 fresh |
| J1 | cùng anchors J0, ensemble q_i, 32 dòng/bước | teacher theo mẫu có hơn Q[y]? |
| J2 | Gaussian anchors4096, ensemble q_i, 32 dòng/bước | lợi có cần copula, bootstrap có thể giảm? |
| J3 | J1, 64 dòng/bước | liều cao có cộng lợi với teacher mới? |

J1−J0 là so sánh cơ chế chính; J2−J1 đổi support; J3−J1 đổi liều. J3 kết hợp hai yếu tố so G4 nên không dùng nó
để quy mọi lợi cho teacher mới. Conditional bucket là phương án dự phòng, chưa nằm trong J batch.
Selective arm chỉ khi có bằng chứng cần lọc. Nền/copula và kết luận H cần được parent audit trước.
Mỗi run full5.343, seed42, r1–10 trên cosine50, protocol giữ nguyên. Khóa M/bucket/gating trước push, không chọn theo test.
Primary r10 F1 pooled; cổng kế thừa gain≥0,010 và p10≥nền−0,010, báo đủ 10 metric, per-class, mean/p10/min, benign,
bootstrap+up+down, wall/GPU-time, skip. Những cơ chế ở đây đều **UNTESTED — đề xuất**, chưa có utility PASS.

Không chọn FedGen/FedFTG/BN inversion cho screen này: cần generator/model/head server hoặc gradient access vượt thiết kế none hiện tại.
FedGen gửi generator và dựa vào predicted knowledge để học phân bố; đó không phải thay loss nhỏ.
[FedGen, ICML 2021](https://proceedings.mlr.press/v139/zhu21b/zhu21b.pdf).
Chưa có căn cứ cần mở lại bộ sinh lớn trước khi kiểm teacher theo mẫu bằng kênh trực tiếp.

Triển khai batch J dùng **một bank cố định suốt run** (không refresh theo round), khóa ở [screening-j10](screening-j10.md). J4 bổ sung ablation KL real từ chẩn đoán Q/timeDelay.

31/10: [J đãaudit](screening-j10.md) bác bỏutilitygaincủaanchor trongscreen này; hướngKconditioning từrealdata ở [thiếtkế](federated-design.md). Không tiếp tụcselectoranchor chỉvì agreementcao.
