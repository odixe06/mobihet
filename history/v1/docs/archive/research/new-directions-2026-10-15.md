# Hướng mới từ hard labels local — nghiên cứu 15/10/2026

**Trạng thái:** đề xuất khoa học; chưa chứng minh utility, chưa triển khai trong ghi chú này.
Chủ dự án đã cho phép triển khai thí nghiệm sau đề xuất; ClassTail đã chốt và chạy calibration hai seed.
Chi tiết đã frozen trong [contract/source thực thi](../../methods/class-tail-risk/builds/class-tail-pilot/README.md)
ưu tiên hơn các giá trị còn ở dạng đề nghị bên dưới: fixed-tail bắt đầu r3 sau hai round CE;
class-shuffle xoay lớp một bước; local CE chỉ scan diagnostic ở r2, không scan mỗi round.
Paired probe r2 đã audit PASS; utility r20 chưa có kết quả. Không sửa source sau PUSHED.
Không đọc trạng thái RUNNING/COMPLETE thành bằng chứng, không kéo dài adaptation FAIL.

**Ưu tiên:** ClassTail — trao đổi phân bố loss để điều phối hard-example training trên record thật.
Hai hướng dự phòng: phạt covariance giữa margin và receiver annotation; điều phối xử lý xung đột
gradient theo cặp lớp. Ba hướng đổi **objective/optimizer**, không đổi tên carrier đã thử.

Đối chiếu [26 nhóm](../../proposals.md), [LESSONS](../../../LESSONS.md),
[nguồn và đề xuất 13/10](new-fl-primary-sources-2026-10-13.md), [protocol](../../research/protocol.md).
InputSensitivity và ErrorCosts đã xuất hiện 13/10, không tính là đề xuất mới ở đây.
Mọi learned model vẫn DAGSNet; 66 feature/partition giữ nguyên; record, perturbation, activation
và gradient từng record không rời client. Official test chỉ báo, không chọn phương pháp/HP/round.

## 1. Vấn đề thực tế và giới hạn dự báo

- [K2](../../lessons/kernel-anchor-distillation.md), [V9](../../lessons/anchor-function-residual.md)
  và [gradient books](../../lessons/gradient-synthetic-books.md) cho thấy teacher/synthesis fit,
  coverage hoặc cost PASS vẫn có thể đi cùng utility FAIL. Hướng mới dùng hard labels tại x thật,
  không đi qua sampler hoặc teacher yếu. Đây là lựa chọn thiết kế, chưa là dự báo tăng F1.
- [Bài học chung](../../../LESSONS.md): private dễ học thuộc receiver, covariance quần thể hữu ích,
  timeDelay có giới hạn thông tin. Do đó không coi loss cao là bằng chứng sample học thêm được.
- Dữ liệu 20 client có đủ lớp trước fixture; label skew không đồng nghĩa thiếu lớp. Fixture/holdout
  có thể tạo lớp rất ít row. [Thống kê dữ liệu đã audit](../../../knowledge/dataset.md).
- Carrier risk nhỏ truyền **cách phân bổ cập nhật**, không mang feature pattern/ranh giới từ peer.
  Nếu pooled không hơn own/shuffle, lợi local hard mining không được gọi là lợi FL.

Không nguồn dưới đây chứng minh tăng metrics trên VeReMi/DAGSNet. Công thức adaptation,
codec, threshold và gates là đề xuất của dự án, không kế thừa guarantee của paper.

## 2. ClassTail — CVaR theo lớp trên loss hardlabel local

### Nguồn primary và phạm vi khác paper

CVaR có biểu diễn qua một threshold và phần loss vượt threshold; công thức này cho phép
subgradient tập trung vào tail. [Rockafellar–Uryasev, bản tác giả, §2/Theorem 1](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).
Với empirical loss có atom/tie, phần tail cần định nghĩa cẩn thận; không mặc định conditional
mean của các điểm vượt threshold bằng CVaR chính xác.
[Rockafellar–Uryasev, general loss distributions](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf).

Superquantile FL của Pillutla et al. chọn **tail client** bằng risk của **cùng model w**, rồi
aggregate model updates. ClassTail bên dưới giữ các private θ_i khác nhau, xét **tail record
trong mỗi lớp**, chỉ gửi histogram. Không tái hiện Δ-FL, không kế thừa bound convergence,
generalization hoặc DP của paper. [Paper gốc, §3–4](https://arxiv.org/html/2112.09429v2).

Group DRO có thể thất bại khi neural model fit gần hết train: training worst-group loss thấp
vẫn đi cùng generalization kém. Đây là cảnh báo trực tiếp cho tail-training trong dự án,
không là bằng chứng ClassTail sẽ fail. [Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731).

### Objective và tri thức FL thực sự

Đặt `ell_i(x,y)=CE(y,private_i(x))`, K=16. Pooled histogram dùng **row-count weighting**
trong từng lớp; không gọi đây là client-equal aggregation. Với `pi_y=N_y/N_total`:

```text
F({theta_i}, zeta) = (1-lambda) * mean_over_all_train_rows(ell_i)
                  + lambda * sum_y pi_y * [zeta_y
                      + (1/alpha) * mean_over_train_rows_with_y((ell_i-zeta_y)_+)]

alpha = 0.25; lambda = 0.25
zeta_y ~= pooled 75th-percentile of current TRAIN CE for class y
L_i(x,y) = 0.75*ell_i + 0.25*[zeta_y + 4*(ell_i-zeta_y)_+]
gradient weight = 0.75 + 1[ell_i > zeta_y]  # 0.75 or 1.75
```

Threshold stop-gradient; không differentiate qua histogram/rank. Đây là objective hợp lệ của
**joint collection private models** nếu mọi loss/threshold lấy đúng thời điểm và cùng empirical
mixture. Histogram quantization và frozen threshold trong một round tạo approximation/staleness;
không gọi optimizer là exact CVaR descent trên toàn run.
`grad_theta_j ell_i=0` khi i≠j: shared threshold chỉ điều phối độ ưu tiên cập nhật tại client j,
không cho gradient của private j trên peer x. Không giải receiver boundary transfer bằng theorem.

So với ErrorCosts 13/10: ErrorCosts sửa penalty theo **true/predicted class pair**; ClassTail
chọn **record khó trong cùng lớp** từ hardlabel CE. So với BoundaryCells: không feature-cell,
soft target hoặc lookup classifier. Không sampler, teacher, book, common anchor hoặc proxy carrier.

### Schedule tối thiểu đề nghị

1. Dùng đúng fixture pilot hiện hành ~1M train rows, 20 client, group/old holdout từ train80%.
   Giữ row memberships/sampling/class map/scaler. Không thêm partition.
2. r1 train CE chung; chưa shared-tail update. Từ đầu r2, scan **T train** bằng model ở cuối
   round trước, BN eval/no-grad, CE FP32. Pooled/own threshold cố định cho round kế tiếp.
   Training giữ BN/optimizer/LR/steps giống fresh baseline. Scan lại mỗi round.
3. Toàn bộ arm cùng một scan policy để codec/cost so sánh được; local CE reference báo thêm
   thời gian chuẩn **không scan** để tránh denominator bị inflated. Scan/communication đều trả cost.
4. Class không đủ cohort hỗ trợ thì tail term không activate, dùng CE cho class đó. Không dùng
   donor duy nhất. Lưu row counts, support, skipped classes và activation round.

Scan T được dùng **optimization**, không gọi T là holdout. Group/old chỉ dùng diagnostic và
locked gate; không cập nhật zeta từ đó. Không lấy official test chọn bin/alpha/lambda/warmup.

### Codec và guard trước implementation

**Đề nghị log bins**, vì linear bin width0,25 có thể gom gần hết train CE đã nhỏ vào bin đầu:
threshold sai làm weight gần constant, biến thí nghiệm thành LR shift.

```text
e_0=0
e_j=2**(-12 + (j-1)*15/31), j=1..32  # e_1=2^-12, e_32=8
bins b=0..31: [e_b,e_(b+1)); bin32: [8,+inf)
H_i[y,b] = count TRAIN CE in bin b
N_i[y] = sum_b H_i[y,b]
H[y,b] = sum_i H_i[y,b]
b_y = first b whose cumulative count >= ceil(0.75*N[y])
zeta_y = upper finite edge of bin b_y
```

Các edges serialize FP32, dùng chính byte đã deserialize ở producer/consumer. Khi `ell==zeta`
tail derivative=0; guard `ell>=0`, finite. Không clip CE vào8 rồi gọi overflow biến mất.
Quantile trong overflow bin không xác định được hữu hạn từ count: đánh dấu class invalid,
CE fallback, báo cost/quality preflight FAIL nếu không đạt coverage gate. Không tự đặt zeta=8
và gọi exact quantile. `N[y]==0` không chia; client class thiếu vẫn gửi zero count thật.
Producer count uint32 overflow phải fail; RSU aggregate uint64. Guard cohort contributors≥5
cho class activate; threshold eligibility mask phát cùng snapshot.

| payload | kích thước đề nghị |
|---|---:|
| up: header32 +16×33 uint32 counts +16 uint64 N | 2.272B |
| down: header32 +16 FP32 zeta +16 FP32 paid prior +16bit eligibility | 162B |
| bootstrap: 33 FP32 finite edges +hash32 | 164B |
| hai chiều mỗi refresh, trước channel overhead | 2.434B/client |

Giả định header32 đủ version/schema/cohort/CRC phải kiểm bằng serializer; nếu cần hash dài hơn
hoặc contributor array ngoài header, cộng byte thật. Scaler nếu client chưa có: 132 FP32=528B;
class map/schema/ACK/retransmission/bootstrap cũng trả phí. 19 refresh r2–r20 tối thiểu46.246B/client,
20 client924.920B, trước bootstrap/channel. Đây accounting, chưa contact proof.
Tại cap4.500B/contact, snapshot hai chiều có thể vừa **trước** overhead; chưa chứng minh deadline
hoặc đủ cả bootstrap trong contact đó. Chỉ activate coherent cohort snapshot đủ/hash đúng.
Tail histogram/count không là secure aggregation/DP; cohort5 không tạo privacy theorem.

### Diagnostic falsifiable trước train nặng

Áp codec/objective lên checkpoint local DAGSNet đã audit nếu restore được. Không SGD, không
quét official test. Nếu chưa có checkpoint tương thích, r1 CE chung trên Kaggle là diagnostic
production; random-init mọi model có CE gầnlog16 nên không được lấy pooled≈own ở init để bác bỏ.

**Threshold đề nghị phải khóa trước diagnostic**, không thay sau kết quả:

- Tổng class activation≥12/16; trong mỗi class active, ≥5 contributing clients và pooled scan N≥128.
  Mỗi own threshold chỉ dùng nếu N_i,y≥32; own thiếu dùng CE fallback và báo riêng.
- Với ≥12/16 lớp, actual pooled tail fraction trên scan T nằm [0,10;0,40]. Quantile phải finite.
  Khoảng này kiểm quantization/mass; không là utility proof. Class timeDelay vẫn báo, không tự bỏ.
- Macro theo lớp, fraction record mà pooled-tail membership khác own-tail membership≥0,05
  trên class/client đủ own support. Nếu không đạt, shared decision quá giống own; không mở pilot
  standalone để tìm win. Đồng thời báo size-weighted và client-equal diagnostic, không đổi pooling.
- Lưu tail theo lớp/receiver/scenario, train/group CE gap, effective sample mass, CE gradient-norm
  theo class khi có bounded probe. Nếu pool tail chỉ phản ánh local calibration, hoặc hard tail
  không tập trung trên cohort transfer khó, ghi đó là dấu hiệu chống giả thuyết.

Không chọn thêm alpha/lambda/bin chỉ vì FAIL; codec bug có thể sửa với bằng chứng lỗi, scientific
tail/mass FAIL cần một proposal mới. Không loại sample/class từ official test để làm gate đẹp hơn.

### Pilot 2 seed ×20 round, 7–8 arm

| arm | mục đích |
|---|---|
| local CE | mức utility và cost chuẩn |
| fixed-tail | zeta_y=log16 từ r2; tách dynamic peer threshold khỏi objective hard mining |
| own-tail | quantile riêng client, cùng codec/scan/steps |
| pooled-tail | candidate, pooled count/class quantile |
| pooled-class-shuffle | cùng16 zeta nhưng class derangement cố định seed1729; phá label alignment |
| pooled-class-mass CE | weight CE theo class bằng mean của tail weight trên T; tách sample-hardness khỏi class/LR mass |
| fresh DML DAGSNet | incumbent có utility, đúng fixture/seed/round/eval hiện hành |
| fresh Gaussian mạnh nhất | thêm nếu chưa có exact matching fresh reference; không chọn theo test |

Mỗi arm sở hữu model/hist riêng. Class derangement được khóa trước run, cấm identity; không shuffle
receiver/raw rows giữa client. **Class shuffle không giữ effective per-class mass**, vì zeta đổi
class nên actual tail fraction đổi. Báo hạn chế này, giữ class-mass arm; không gọi shuffle là
perfect matched control. Có thể kiểm thêm within-class local weight permutation trong diagnostic,
nhưng không tự mở arm/HP mới sau FAIL.

Target-mass gp: tích lũy coefficient CE theo true class trên **mọi step thật**, không chỉ scan:
`mass_i,y += sum_{samples:y} w_sample`, `pi_eff=(mass+eps)/(sum_mass+K*eps)`; khóa smoothing
theo baseline trước train. Dùng `logits-log(pi_eff)+log(pi_global)` cho gp đã chốt; head báo phụ.
Vì w phụ thuộc x, đây không phải pure label-shift correction; đúng mass không bảo đảm gp calibrated.
Không đổi gp/head để cứu cổng. Class-mass CE cần own actual weighted mass tương tự.

**Cổng đề nghị cho mỗi seed42/43, fixed r20, group/private/gp F1 macro:**

- pooled−own≥0,010; pooled−class-shuffle≥0,010; pooled−local≥0,015.
- Benign F1 giảm không quá0,005 so local và incumbent mạnh nhất; old holdout/per-class/đủ10metrics
  báo toàn bộ. Gate không chọn best round.
- Chỉ xét nối dài nếu pooled−incumbent mạnh hơn giữa fresh Gaussian/DML≥0,015 **cả hai seed**,
  đồng thời mechanism gates bên trên PASS. Hơn local chưa đủ.
- Class-mass CE ngang pooled trong≤0,005 cả hai seed: chưa chứng minh hard-example mechanism;
  không dùng tên ClassTail để gọi thắng FL. Pooled thua own/shuffle: dừng hướng shared threshold.
- Cost arm candidate≤2×fresh local CE thật; tất cả scan/import/eval/I/O/budget trả phí.
  Dự toán whole batch từ slowest-arm measured pairmax trên2T4, spare≥15%; target≤10h/session.
  Nếu không vừa, split notebook/batch/chuyển phiên có resume; không âm thầm giảm controls hoặc seed.

Lưu confusion/prediction đủ kiểm lại10metrics, perclass/benign, histogram/zeta/actualmass/eligibility,
source/data/config hashes, checkpoint exact-resume, W&B online. Pilot unlimited communication
chưa là R1; DAGSNet-only chưa là R2. Không tăng round nếu locked paired FAIL.

## 3. NuisanceCov — giảm covariance margin–receiver, không tạo perturbation

Nguồn HSIC cho phép dùng penalty phụ thuộc giữa residual và input để học predictor robust
dưới covariate shift, với giả định conditional label relation ổn định.
[Greenfeld–Shalit, ICML 2020](https://proceedings.mlr.press/v119/greenfeld20a.html).
Adaptation này chỉ dùng **linear covariance**, không kernel HSIC đầy đủ, không có independence/causal
guarantee và không kế thừa theorem paper. Chọn nuisance block phải dựa generator/evidence hiện có.

```text
u(x) = tanh(public receiver-annotation coordinates), G=4
r_i(x,y) = tanh((logit_y-logsumexp_other_logits)/tau), fixed tau
mu_u[y] = mean u | y; mu_r[y] = mean r_i | y
C[y] = mean r_i*u | y - mu_r[y]*mu_u[y]
F = CE + lambda/K * sum_y ||C[y]||^2
```

Client gửi sum(r), sum(u), sum(r*u), N_y; RSU gửi C và mu_u. Với row-weighted aggregation,
derivative local contribution tỷ lệ `2*C[y]·(u-mu_u[y])*grad_theta r_i`; hệ số N_i/N và class
normalization phải match objective. Snapshot C stale trong một round là surrogate, không exact
gradient của full joint F. All computations first-order từ logits thật; không teacher/sampler hoặc
input-gradient mixed derivative. Khác ResponseRisk: không T(x), paired forward, JS hoặc invariance
dưới can thiệp. Khác InputSensitivity: không match grad_x moments.

G4 scalar margin codec: up header32 +16×(1+2G) FP32 +16 uint64=736B;
down header32 +16×2G FP32 +16 FP32 prior=608B; total1.344B/refresh trước support/ACK/bootstrap.
u semantic dictionary/hash/transform scale trả phí. C có thể cancel giữa clients; thêm own covariance
diagnostic, đừng coi C≈0 là mọi client independent. Learned θ khác nhau vẫn chung class/u coordinates;
calibration/private offsets có thể góp covariance mà không là receiver shortcut.

Diagnostic: conditional covariance từ checkpoint trên T và group, cosine/norm của penalty gradient
với CE, class/receiver support; constant margins/collapse có thể giảm penalty. Nếu covariance train≈0
nhưng group transfer vẫn kém, objective không có signal đúng bottleneck; dừng trước train.
Pilot tương lai cần CE /local penalty/own/pooled/shuffled nuisance-coordinates/random matched penalty,
utility hơn own/shuffle/local; chưa khóa tau/lambda/nuisance quyền và chưa ưu tiên quota.

## 4. ConflictGraph — chia sẻ lớp nào thường xung đột khi cập nhật

PCGrad xử lý task gradients có dot product âm bằng projection local; lợi phụ thuộc geometry
optimization, không phải cứ cosine âm là xử lý sẽ cải thiện generalization.
[Yu et al., NeurIPS 2020, §2–3](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf).
Adaptation xem class-conditional CE như các task; chưa có guarantee cho 16-class IDS.

Client tính `g_i,y=grad_theta mean CE | class y` tại private model, lưu `cos(g_i,y,g_i,k)`;
chỉ gửi số đếm conflict, mean cosine và support theo **class pair**. RSU trả conflict frequency.
Client chỉ project local `g_y` khỏi local `g_k` khi local dot<0 và shared frequency vượt cổng:
`g_y <- g_y-(g_y·g_k)/(||g_k||^2+eps)*g_k`. Không truyền gradient/parameter dimension,
không common proxy, không sketch/EF. Scalar chung điều phối khi sửa local conflict, không cung cấp
peer gradient direction. R2 design dùng class-pair semantics chung dù local θ khác dimension.

120 unordered pairs: up32 +120×(FP32 mean +uint32 conflict count +uint32 support)=1.472B;
down32 +120FP32 +64prior=576B; total2.048B/refresh trước contributor/bootstrap/channel overhead.
16 perclass backwards có thể vượt cost gate rất xa. Không giả sử payload nhỏ nghĩa runtime nhỏ.
Giảm xuống2 supertasks là adaptation khác với task definition phải khóa; không cứu sau FAIL bằng
thay group classes. PCGrad stationary point có thể khác optimum CE; classwise objectives vốn không
độc lập, loại conflict indiscriminately có thể phá decision tradeoff.

Diagnostic trước train: pair-gradient signs reproducible qua minibatches/checkpoint, norm imbalance,
bounded finite-difference direction effects trên T; không nhận diện negative transfer chỉ từ cosine.
Pilot tương lai cần standardCE/localPCGrad/own-gated/pooled-gated/shuffledpair/norm-rescaledCE,
same optimization steps và cost thật. Chỉ mở nếu measurable pooled gate khác own, derivative/code
đúng, cost≤2×localCE. Chưa ưu tiên trong budget hiện tại.

## 5. Quyết định còn phải khóa và điểm dừng

ClassTail: alpha/lambda/warmup, log-bin codec/tie/overflow, row-count pooling/cohort/mincount,
refresh/BN policy, seven/eight-arm roster, target-mass gp smoothing, preflight thresholds,
incumbent reference và gates phải nằm trong frozen config trước launch. Các lựa chọn này thay
objective/evaluation; quyền launch chung không tự chứng minh cấu hình khoa học đã chốt.

Ưu tiên ClassTail vì gradient tới private backbone từ hard labels thật, không cần auxiliary model,
sampler, synthetic fit hoặc sixteen-backward optimizer. Cơ hội tăng metric là giả thuyết:
tail training có thể dành cập nhật cho biên chưa học; có thể chỉ học noise/memorize nhanh hơn.
Muốn tăng round phải có paired utility và incumbent gates, không chỉ tail-loss giảm.

Không mở cả ba hướng cùng lúc để dùng hết quota. Nếu ClassTail preflight/paired FAIL, ghi rõ nguyên
nhân quan sát và phạm vi kết luận, cập nhật proposal/build lessons; secondary mechanism chỉ triển khai
sau quyết định riêng. Không thêm learned non-DAGSNet, không tune official test, không sửa source
frozen đang chạy, không Git/DVC commit/push trong công việc nghiên cứu này.
