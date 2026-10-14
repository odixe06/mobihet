# Hướng FL khác carrier/model/synthetic hiện có — nguồn và thiết kế 2026-10-13

**Trạng thái:** ghi chú nghiên cứu và cơ chế đề nghị; chưa triển khai, chưa pilot, chưa khóa HP/cổng với chủ dự án.
Đối chiếu [24 nhóm hiện hành](../../proposals/README.md), [bằng chứng](../../proposals/evidence-summary.md),
[LESSONS](../../LESSONS.md) và [protocol](protocol.md). Không kéo dài adaptation đã FAIL để tìm điểm thắng.

Ưu tiên nghiên cứu: **bảng ranh giới ở input bằng đếm có giám sát**, **rủi ro dưới can thiệp local**,
rồi **thống kê sensitivity ở input**. Một hướng bổ sung nhẹ là **chi phí lỗi theo cặp lớp**;
hướng này hỗ trợ H2/H1 tốt hơn làm phương pháp đứng riêng. Không gọi mỗi bảng/size/seed là một họ mới.

Mọi phương án dưới đây là adaptation của dự án, không phải tái hiện nguyên thuật toán paper.
Không có paper nào trong ghi chú này chứng minh tăng IDS metrics trên VeReMi/DAGSNet hoặc giải được R1+R2.

## 1. Vì sao đổi loại tri thức

- P1: toàn bộ model đủ sức chứa và covariance input gộp có ích; prototype/head-only không đủ.
- V3: carrier nhỏ có contribution nhưng không giữ đủ utility dense; tăng thời gian chưa được phép thay utility gate.
- V5/V9: coverage/fit teacher tốt có thể đi kèm target rất mềm và không hơn nhãn Gaussian miễn phí.
- V4/V6: support/fidelity hợp lệ không tự tạo ranh giới IDS tốt hơn.
- V10: mật độ normalizable chưa đủ sampling/cost; tránh thêm sampler nặng khi chưa có bằng chứng lợi.
- Receiver shortcut và local memorization đã có bằng chứng trong [LESSONS](../../LESSONS.md).
  Đổi objective hoặc trao đổi cấu trúc ranh giới có thể tác động toàn private model mà không bắt buộc chuyển tham số,
  latent embedding hay synthetic records. Đây là giả thuyết cần bác bỏ bằng own/shuffle/no-FL.

Các số và trạng thái trên thuộc report của dự án, không suy từ paper bên ngoài.

## 2. Nguồn primary đã đọc và giới hạn áp dụng

| nguồn | nguyên lý được hỗ trợ | điều không được suy |
|---|---|---|
| [Fishr, ICML 2022](https://proceedings.mlr.press/v162/rame22a.html), [PDF §3, §4.2.2](https://proceedings.mlr.press/v162/rame22a/rame22a.pdf) | Match variance của gradient theo **tham số** giữa domain; implementation tiết kiệm dùng classifier subset | Match input-gradient giữa private khác kiến trúc không phải Fishr nguyên bản; không kế thừa Fisher/Hessian theorem |
| [Right for the Right Reasons, IJCAI 2017](https://www.ijcai.org/proceedings/2017/371), [PDF §2.1](https://asross.github.io/publications/RossHughesDoshiVelez2017.pdf) | Input-gradient regularization tác động decision behavior của differentiable model; irrelevant-coordinate annotation là giả định | Gradient consensus không tự xác định causal feature hoặc sửa shortcut nếu mọi client cùng dùng shortcut |
| [Risk Extrapolation, ICML 2021](https://proceedings.mlr.press/v139/krueger21a.html), [PDF §2–3](https://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf) | REx xử lý biến thiên risk giữa domain của cùng predictor; V-REx dùng variance penalty | Gửi scalar CE của nhiều private model không cho gradient của mỗi private trên domain khác |
| [Agnostic Federated Learning, ICML 2019](https://proceedings.mlr.press/v97/mohri19a.html), [PDF algorithm](https://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf) | Minimax theo mixture client distributions; primal update vẫn cần model-gradient information | Chỉ gửi client weights/scalar risk không giữ cơ chế AFL; convex guarantees không áp thẳng DAGSNet |
| [Counterfactual Invariance, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html), [PDF](https://proceedings.neurips.cc/paper/2021/file/8710ef761bbb29a6f9d12e4ef8e4379c-Paper.pdf) | Dùng can thiệp phần không liên quan để stress-test invariance; ý nghĩa phụ thuộc causal structure | Shuffle noise/profile không tự là counterfactual đúng vật lý hoặc bảo toàn nhãn IDS |
| [RFL, ICML 2026](https://proceedings.mlr.press/v306/qin26g.html), [primary PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/qin26g/qin26g.pdf) | Carrier dạng class-discriminative rules; paper dùng prototype evidence và **learned rule network** | Nguyên bản không hợp mọi learned model đều DAGSNet; bảng đếm input dưới đây là adaptation khác, không tái hiện RFL |
| [Cotter et al., ICML 2019](https://proceedings.mlr.press/v97/cotter19b.html), [PDF](https://proceedings.mlr.press/v97/cotter19b/cotter19b.pdf) | Tách data cho train model và data cho constraint player để cải thiện generalization của rate constraints | Không được dùng official test cập nhật multipliers; feasibility trên local domains không chứng minh transfer sang receiver mới |

RFL full PDF đã đọc phần rule induction: binarization với learnable thresholds, logical layers và classifier;
không nhập rule network đó vào phương án DAGSNet-only. Nguồn chỉ hỗ trợ chọn **carrier quyết định theo lớp**.

## 3. F-BoundaryCells: đếm ranh giới trên các ô input, dạy ngay tại record local

### 3.1. Bằng chứng vấn đề → cơ chế

Mean prototype không giữ ranh giới. H2 giữ covariance nhưng giả định Gaussian; anchor teacher mất utility khi
bandwidth phải lớn để phủ train. Thay khoảng cách kernel/gaussian synthesis bằng ô input cố định và số đếm nhãn.
RSU chỉ tổng hợp thống kê; private DAGSNet học từ soft targets tra tại **chính x local**. Không query teacher yếu,
không sampling, không lưu book, không đưa record hoặc activation lên RSU.

Chọn P cặp feature từ schema gốc và B bin mỗi trục; pilot đề nghị P=2/4, B=4. Đây là hai cỡ cùng cơ chế.
Ví dụ dictionary cần khóa trước pilot: `(f_snd_spd, f_sess_implied_spd)`,
`(f_dist, f_sess_pos_pred_err)`, `(f_sender_rate_5s, f_sender_share_5s)`,
`(f_snd_acl, f_sess_acl_residual)`. Không dùng noise/profile để cố tình mang receiver shortcut.
Chưa chứng minh bốn cặp này đủ tách attack; probe train-only phải kiểm trước lựa chọn cuối.

Cắt bin trên z-score bằng public `{-1,0,1}`, giữ hai outer bins không hữu hạn về chiều rộng;
không clamp feature thật, không đổi 66 input, không chia lại partition. Tie convention phải khóa trong codec.
Các cutoff này là đề nghị cần chủ dự án chốt; không chọn theo official test hay theo seed thắng.

Cho `b_r(x)` là ô của cặp r. Client i tính trên subset train chưa dùng cập nhật model:

```text
n[i,r,b,y] = count{x in D_i : b_r(x)=b, label=y}
a[i,r,b,y] = 1[n[i,r,b,y] > 0]
N[i,y] = count{label=y}

RSU: n[r,b,y] = sum_i n[i,r,b,y]
     c[r,b,y] = sum_i a[i,r,b,y]
     N[y]     = sum_i N[i,y]
```

Đổi likelihood theo lớp thành balanced target, để không âm thầm biến class prior thành tri thức ranh giới:

```text
ell[r,b,y] = (n[r,b,y] + alpha) / (N[y] + alpha*B^2)
t[r,b,y]   = ell[r,b,y] / sum_k ell[r,b,k]
t(x)       = average_r t[r,b_r(x)] over eligible cells
L_i        = CE(y, p_i(x)) + lambda * KL(stopgrad(t(x)) || p_i(x))
```

`alpha`, `lambda`, refresh schedule chưa khóa. Ngoài class prior trả byte, có thể giữ nguyên snapshot counts một lần
vì input partition không đổi; targets không bị stale bởi backbone. Chỉ private DAGSNet là mô hình học được.
**Cần chốt phạm vi quyền:** lookup table là thống kê soft-target trung gian giống ngoại lệ K2b đã được phép,
không classifier triển khai; nếu chủ dự án xem nó là model phụ không-DAGSNet thì không mở train.

Khác H2: không tạo x mới hoặc fit moment, không Gaussian. Khác K2/V9: không public anchors, không kernel,
không common probe weights/BN, không public teacher fitting. Khác prototype: điều kiện theo vùng input,
không mean latent. Tri thức có thể phân biệt các lớp cùng tâm nhưng khác vùng/disjunction đơn giản.

### 3.2. Cổng trước pilot

Tại từng `(r,b,y)` cần cohort ít nhất 5 client nếu count khác 0. Nếu một cell có nhãn rare chưa đạt,
không broadcast vector target của cell đó; đánh dấu unusable và giữ CE thật. Không dùng một donor client fallback.
Rule này có thể giảm coverage đáng kể: báo coverage theo lớp, tỷ lệ skipped, entropy và margin targets trên
train-only probe. Không lấy coverage/entropy PASS làm IDS PASS.

Probe phải đo **transfer của targets** trên group holdout nội bộ bằng rule stats từ train thôi, so own và
label-permutation/cell-permutation. Nếu targets gần uniform, độ tin cậy không hơn own hoặc rule support
không phủ các lớp cần giải thì dừng hướng trước Kaggle. Threshold cuối cần khóa; không tự tăng P/B sau FAIL.

Pilot phải có: local; shared balanced prior-only; own-cell; pooled-cell; cell-shuffle;
class-mean target cùng aggregate priors; Gaussian-diag/full fresh reference. Own dùng cùng codec/refresh/compute.
Shuffle hoán các **cell target** trong từng dictionary, không shuffle raw records giữa client;
giữ target histogram/entropy, tổng class mass và số eligible cell.
Thêm single-axis marginals để kiểm contribution của conjunction; không gọi tăng do marginal là lợi rule interaction.

Đề nghị gate pilot hai seed, group/private/gp r20: pooled hơn own và cell-shuffle ít nhất 0,010 mỗi seed,
hơn Gaussian reference mạnh hơn ít nhất 0,015; benign F1 giảm không quá 0,005. Các số này chưa khóa.
Head là diagnostic vì soft-target balanced population có thể đổi effective prior; không đổi headline sau kết quả.

### 3.3. R1/R2, byte và failure

Với 16 lớp, B=4, số cell-label `M=256P`. Plain codec đề nghị:
count uint32, contributor indicator uint8, class N uint64, header32B;
down target FP32, eligibility bitset và paid prior FP32. Counter overflow phải guard; aggregate client≤255
mới dùng uint8 contributor sum. Tổng train hiện nhỏ hơn uint32 limit nhưng không bỏ guard.

| P | lên / snapshot | xuống / snapshot | tổng hai chiều / client |
|---:|---:|---:|---:|
| 2 | `32+5*512+8*16 = 2.720B` | `32+4*512+512/8+4*16 = 2.208B` | 4.928B |
| 4 | `32+5*1024+8*16 = 5.280B` | `32+4*1024+1024/8+4*16 = 4.320B` | 9.600B |

Down eligibility ở đây bit-per-cell-label bảo thủ; unusable cell mask có thể ít hơn nhưng chưa claim byte tiết kiệm.
Public dictionary metadata có 2P feature IDs uint16, 3 edges/trục FP32 và header/hash32B:
`4P+24P+32 = 88/144B`, chưa scaler/class-map/bootstrap. Codec/layout phải đo bằng serialized bytes thật.
Scalers nếu chưa có tại client: 66 mean+66 sd FP32 thêm528B, class order/schema/hash vẫn có phí.

Ở cap4.500B/contact, bỏ overhead và chia hai chiều riêng: P2 cần ít nhất 1 upload+1 download contact;
P4 ít nhất 2 upload+1 download. Đó là lower bound; header, ACK, retransmission, cohort setup/recovery,
truyền schema và deadline có thể tăng số contact. Chỉ apply target sau snapshot đủ/hash đúng/cohort commit;
partial counts/target không dùng để train. Atomic snapshot không đồng nghĩa cả payload phải nằm trong một contact.

R2 không align weights/latent: cùng66feature/class semantics đủ; DAGSNet-only pilot chưa chứng minh đa kiến trúc.
Risks chính: ô coarse chỉ mang marginals nghèo; input dependence≥3 feature bị mất; rare suppression làm
coverage thấp; targets sai hướng gây negative transfer; class-balanced target tác động `gp` prior;
counts trong cohort nhỏ lộ thông tin. Cohort≥5/trusted sum không là DP hoặc secure aggregation crypto.

## 4. F-ResponseRisk: cộng tác bằng phản ứng dưới can thiệp local

### 4.1. Cơ chế khác gradient/teacher

Mục tiêu là bớt dựa vào receiver cues đã biết, **không cung cấp nhãn cho điểm từ noise**.
Client tự tạo `T_b(x)` tại local cho G nhóm feature được chốt là nuisance hợp lệ.
Không x/T_b(x)/activation rời client. Đánh giá paired outputs của cùng private DAGSNet:

```text
e[i,y,b] = E_{x in local-probe,y} JS(p_i(x), p_i(T_b(x)))
u[y,b]   = client-mean(e[i,y,b])
v[y,b]   = client-var(e[i,y,b])
w[y,b]   = bounded normalization of v/(u^2+epsilon)
L_i      = CE(y,p_i(x)) + lambda * sum_b w[y,b] * JS(p_i(x),p_i(T_b(x)))
```

`w` từ cohort cập nhật theo snapshot; stop-gradient. G cao không chạy G transformed pass mọi batch:
chọn một transform bằng lịch public seeded, importance correction nếu sampling probabilities không đều.
Chu kỳ/quy tắc normalization cần khóa trước pilot. Backward là first-order qua hai forwards;
khác input-gradient penalty cần mixed derivative. Có thể dùng class-conditional CE risk-gap thay JS,
nhưng đó là objective khác; không dùng hai tên để tăng số họ.

JS invariant với constant logit shifts nhưng không invariant với arbitrary temperature/calibration.
Client-size/count weighting vs client-equal weighting phải chốt: nếu lớn receiver dominated, pooled mean
có thể chỉ tái hiện shortcut chung. Variance response cao không chứng minh feature không liên quan.

### 4.2. Điểm chưa chắc phải hỏi trước train

Không có phép đổi noise/profile mặc định bảo toàn nhãn và tất cả geometry/session semantics.
`f_rcv_noise_mag` phụ thuộc noise_x/y; đổi profile có thể đổi measurement-noise regime;
66feature đã impute0 không xác định được nhánh missing. Không gọi block shuffle là causal intervention
chỉ vì được làm tại local. Có thể thử **feature stress** nhưng phải gọi đúng tên và đo support/fidelity.

Phạm vi đề nghị nhỏ: noise annotation block receiver; không sửa raw claimed motion, sender discrepancy,
first-session/session/rate hoặc vị trí attack. Kiểm generator/extractor cho label preservation và derived closure.
Nếu chỉ có imputed features không tái dựng faithful, ưu tiên local masked-feature stress với explicit approximation,
hoặc dừng hướng; cần chủ dự án chốt trước thực nghiệm. Không bỏ feature khỏi input66.

### 4.3. Đối chứng/cổng và byte

Local CE; uniform local JS cùng transforms; own-response weights; shared-response weights;
class/transform-shuffled response weights; random weights cùng entropy/norm. Tất cả arms train cùng
paired forwards, BN/dropout policy, augmentation, batch size, optimizer steps. Giữ constant/uniform
counterfactual objective để tách lợi augmentation khỏi lợi FL.

Gate mechanism: shared hơn own/shuffle/uniform trên group holdout, không chỉ JS giảm.
Diagnostics: response mean/variance theo lớp, prediction entropy, train real CE, group F1/per-class,
benign F1, số transforms valid/skipped, weight effective support. Collapse constant logits có JS thấp
nhưng không được qua quality gate. Fail support, cost hoặc own/shuffle thì không tăng lambda/refresh để cứu.

Up plain: mỗi class/group có sum(e),sum(e²) FP32; N_y uint64; header32B.
Down: w FP32 + prior16 FP32 + header32B. Vì snapshot tính cohort variance của **mean mỗi client**,
sum(e) và sum(e²) ở đây phải là mean_i và mean_i², không sums per-record rồi đọc thành client variance.
G=4: up `32+2*16*4*4+16*8 = 672B`; down `32+16*4*4+16*4 = 352B`;
tổng1.024B/refresh, trước ACK/recovery/bootstrap/transform-schema. Nếu thêm client-contributor masks,
cost phải cộng thật; không dùng zero đối với missing class và gọi đó là mean.

R2 chỉ dùng class×transform identity, không parameter/latent alignment. R1 object nhỏ nhưng tốn scan/2forward,
contact skew có thể bóp méo nuisance weights. Chỉ publish coherent cohort snapshot và version/transform hash;
stale version policy trước pilot. Có nguy cơ tất cả client cùng sai/shortcut nên consensus củng cố lỗi chung.

## 5. F-InputSensitivity: định hướng explanation tại input66

### 5.1. Adaptation cần giữ giới hạn theorem

Fishr gốc dùng parameter-gradient; parameter dimensions/private architecture khác nhau không chung tọa độ.
Đề nghị input class-margin có semantics chung:

```text
m_i(x,y) = f_i(x)_y - logsumexp_{k!=y} f_i(x)_k
g_i(x,y) = grad_x m_i(x,y)
h_i(x,y) = g_i(x,y)/(||g_i(x,y)||_2+epsilon)
mu_i[y]  = E h_i ; q_i[y] = E h_i^2
mu[y], q[y] = aggregate class-conditional sums/counts
L_i = CE + lambda_mu*||mu_i - stopgrad(mu)||^2
           + lambda_q*||q_i - stopgrad(q)||^2
```

Margin-gradient giảm vấn đề CE-gradient gần0 khi train thuộc; normalization giảm khác logit scale,
không giải điểm gradient0. Cần finite/degenerate-gradient guards và báo norm trước normalization.
Class-direction histogram không giữ vị trí x của ranh giới; hai model khác boundary location vẫn có mean
và variance giống nhau. Matching moments chỉ là hypothesis cho shared feature usage, không bằng chứng causal.

Receiver cues chung có thể đạt consensus cao. Matching toàn66 không tự bỏ các cues đó;
có thể cần nuisance-specific constraint đã có label-preservation evidence, nhưng đó là thành phần phải ablate.
Không đổi nghĩa `q-mu²` thành Fisher/Hessian theo theorem Fishr: input derivative là toán khác.

### 5.2. Kiểm nghiệm ngắn trước train

Mixed derivative `d/dtheta grad_x margin` với DAGSNet FP32/eval cần probe riêng;
K3 mixed derivative PASS không bảo đảm memory/performance của per-example input gradients production.
BN phải eval khi đo per-example statistics để không trộn batch-dependence;
train BN policy giữ đúng protocol, không gọi `grad(loss.sum(), batch_x)` trong train-BN là independent per-sample proof.
Không tính covariance/variance trên row đã dùng train model mà không báo rõ contamination của statistic probe.

Đối chứng local CE; generic local Jacobian-norm penalty; own moments cùng refresh;
pooled moments; feature-permuted moments trong class; class-permuted moments;
mean-only và second-moment-only nếu cost cho phép. So cùng augmentation/compute/prior/codec.
Khóa subset/epoch/refresh và monitor degenerate gradients; invariant-loss nhỏ không phải quality gate.

Up mean+second sums FP32: `32+2*16*66*4+16*8 = 8.608B`;
down mean+second moments FP32 plus prior: `32+2*16*66*4+16*4 = 8.544B`.
Tổng17.152B/refresh, trước cohort contributors/crypto/channel costs. Nếu chỉ second moment thì
up4.384B/down4.320B; hai versions không có cùng thông tin, phải khóa cổng riêng.
Không quảng cáo “4KB giải R1” khi hai chiều/full mean+second cần fragmentation và schema/bootstrap.

R2 input coordinates/class semantics chung; private model khác cấu trúc vẫn tính h được nếu differentiable.
Model không differentiable về input cần finite-difference route tốn forwards; đó là thiết kế khác chưa kiểm.
All DAGSNet hiện dùng differentiable ops nhưng mixed-derivative/runtime phải test đúng build.
Privacy: class aggregates của input gradients vẫn có thể lộ model/data dependence; không xuất gradient từng record,
clip norm local và cohort≥5 không tạo DP guarantee. Cost gate T4 trước pilot; local chỉ bounded probe.

## 6. F-ErrorCosts: trao đổi chi phí lỗi theo cặp lớp, ưu tiên adjunct

Thay mọi client/RSU đều biết scalar CE bằng pair-specific error information trên local **internal probe**:
`C_i[y,k]=E[p_i(k|x) | true y]` hoặc normalized argmax confusion.
RSU tính class-balanced mean, cập nhật cost multipliers rồi phát cùng ma trận cost:

```text
nu[y,k] <- clip_nonnegative(nu[y,k] + eta*(C[y,k]-tau[y,k]))
L_i = CE(y,p_i(x)) + sum_{k!=y} nu[y,k]*p_i(k|x)
```

Primal train và constraint probe tách trong train80%; nếu không đủ data cho probe riêng thì phải ghi là
cross-fitting design/pending, không dùng group holdout dùng cho quyết định làm optimization data rồi báo cùng holdout.
Official test không cập nhật `nu`, không chọn `tau`, không chọn round.

Ý tưởng rate-constraint/tách constraint data từ [Cotter ICML2019](https://proceedings.mlr.press/v97/cotter19b.html).
Primal-dual clipped update trên DAGSNet không được gán guarantee nonconvex stochastic classifier của
[proxy-Lagrangian paper](https://jmlr.org/papers/v20/18-616.html). Soft confusion giảm không tự đảm bảo argmax F1 tăng.

Carrier240off-diagonal FP32 +N_y uint64 +header32B: up1.120B; down240FP32+prior64+header32=1.056B,
tổng2.176B/refresh, trước ACK/cohort/schema. Khi cần normalized class counts/client contributors thêm phí thật.
R2 label-pair shared, không feature/parameter alignment; không gửi record, teacher hoặc logits từng x.

**Giới hạn quyết định:** biết global error pair không cho input pattern của lỗi đó. Local memorization có thể
đã đạt mọi constraint tại local mà vẫn sai receiver mới. Hướng này không được ưu tiên thay H1/H2;
kiểm như correction objective khi đã có carrier hữu ích: H2+uniform costs / H2+own costs /
H2+pooled costs / H2+shuffled costs, cùng no-cost H2. Thêm class-weight CE control để biết cặp lỗi có hơn scalar.
F1/benign/group transfer phải hơn own/shuffle; ν chạm cap/oscillation hoặc constraint giảm nhưng transfer không tăng
thì dừng. TimeDelay irreducible không được ép constraint vô nghiệm rồi tune để tìm thắng.

## 7. Những tên paper hấp dẫn nhưng không mở như phương án chính

1. **Scalar-only V-REx/AFL:** đặt `R_i(theta_i)` từ các private khác nhau không phải `R_i(theta)` của cùng model.
   `grad_theta_j R_i(theta_i)=0` với i khác j; variance risk penalty chủ yếu reweight local gradient của j,
   không cho gradient của private j trên peer domain. Cần common learner/model gradient hoặc transferable
   functional structure; nếu có full proxy thì là sửa H1 objective, không một carrier mới hoàn toàn.
2. **Split FL “không activation/raw”:** cut-layer gradient phụ thuộc activation và Jacobian của receiver;
   nếu không truyền activation/compatible intermediate state hoặc functional approximation thì server không
   thực hiện được joint forward/backward. Đổi activation thành compressed embedding vẫn mang record-conditioned
   signal và không tự phù hợp DAGSNet/byte/privacy. Không gọi input stats exchange là split learning.
3. **RFL nguyên bản, cây/learned-rule network/GMM:** không phù hợp quyết định mọi learned model đều DAGSNet.
   Fixed predicate counts/soft-target table phải được chốt như **thống kê**; cấm lén thêm classifier rule triển khai.
4. **Copula/marginal/quantile Gaussian replacement:** có thể cải tiến H2; không đếm là hướng hoàn toàn khác.
5. **Common anchor Jacobians/adaptive bandwidth:** vẫn gần K2/V9, cần novelty/contribution gate rõ;
   không quảng cáo là khỏi anchor coverage failure chỉ vì thêm đạo hàm.

## 8. Trình tự và quyết định cần chốt

1. **BoundaryCells:** owner chốt quyền bảng count/soft-target; dictionary/cutoffs/P2/P4;
   rare-cell rule; train-only probe gates; objective/prior/refresh. Probe coverage/targets/byte trước train.
2. **ResponseRisk:** generator/extractor proof cho nuisance intervention; nếu không faithful thì owner chốt
   phạm vi feature stress rõ. Không train trước khi quyết định này được trả lời.
3. **InputSensitivity:** FP32/eval derivative + statistics streaming + finite/degenerate + codec smoke,
   exact resume/source snapshot; rồi mới T4 cost/pilot. Mean+second là cỡ lớn, second-only cổng riêng.
4. **ErrorCosts:** chỉ adjunct của H2/H1, không dùng scalar objective để tuyên bố đã truyền ranh giới.

Sau khóa: pilot ngắn same20client/fixture2,5%/20round/seed42/43, gp/group chính;
cost so fresh baseline T4, own/shuffle/no-FL, đủ10metric/per-class/benign, serialized bytes và coherent commit.
Mốc20/2seed/cost2× là đề nghị kế thừa protocol, không thay yêu cầu chủ dự án cần chốt.
Không mở full/contact từ local derivative/codec PASS hoặc invariance/coverage/fit PASS.
Không sửa notebook/source đang train; source mới ngoài `mh/` và frozen full-validation/hybrid để không đổi fingerprint.
Mọi notebook train mới W&B online trước train; local chỉ bounded smoke/fixture/report.

