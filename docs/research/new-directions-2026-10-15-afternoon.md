# Hướng bổ sung 15/10/2026 — xếp hạng giữa private models, SAM dự phòng

**Cập nhật15/10:** CrossClientRank train-only probe/diagnostic hai seed đã terminal/audit;
owner đã chốt và push [utility r20](../methods/cross-client-ranking/builds/rank-utility/README.md).
[Contract probe](../methods/cross-client-ranking/builds/rank-probe/README.md) và
[utility đã duyệt](../methods/cross-client-ranking/builds/rank-probe/utility-proposal.md) thay các lựa chọn đề nghị dưới đây.
Primary owner chọn fixed true-class-prior gp mọi arm/reference; head diagnostic. Chưa terminal utility.
SAM giữ recipe dự phòng, chưa được chọn triển khai. Không thêm phương pháp chỉ để dùng quota.
Đối chiếu [28 họ](../proposals.md), [bài học](../../LESSONS.md),
[ClassTail/NuisanceCov/ConflictGraph](new-directions-2026-10-15.md) và [protocol](protocol.md).

Raw/perturbed records, per-record activations/gradients ở xe; RSU chỉ dùng tổng cohort.
Mọi learned model vẫn DAGSNet. Statistics bên dưới không là secure aggregation/DP thật.
Không đổi source đã PUSHED; mọi SGD/probe production trên Kaggle, local chỉ fixture/algebra/report.
Official test report-only; group/old nội bộ và T có vai trò riêng, không gọi T là holdout.

## 1. Nguồn primary và loại trừ

- [Guo et al., ICML2020](https://proceedings.mlr.press/v119/guo20f.html): AUC là mục tiêu pairwise,
  khác CE; thuật toán paper vẫn truyền primal/dual của model chung.
- [Yuan et al., ICML2021, §3 eq1–2](https://proceedings.mlr.press/v139/yuan21a/yuan21a.pdf),
  [code tác giả](https://github.com/Optimization-AI/ICML2021_FedDeepAUC_CODASCA): squared-AUC có
  reformulation bằng moments/minimax. Paper dùng **cùng w** và giả định PL/smoothness cho bounds.
  Không chuyển communication complexity hoặc guarantee sang private θ khác nhau dưới đây.
- [Foret et al., ICLR2021, eq2–3/Algorithm1](https://arxiv.org/pdf/2010.01412): SAM dùng gradient ở
  weight perturbation để giảm loss quanh weights; hai backward mỗi step. Không bảo đảm VeReMi F1.
- [Kwon et al., ICML2021, §3–4](https://proceedings.mlr.press/v139/kwon21b/kwon21b.pdf): sharpness
  phụ thuộc parameter scaling; ASAM đổi miền perturbation. Không coi rho SAM/ASAM hoán đổi được.

Không chọn FedFomo/peer-expert ensemble: upload model riêng trái yêu cầu RSU chỉ thấy tổng cohort.
FedEM không được tự thêm: [ràng buộc trước](../archive/research/de-xuat-v2/README.md) đã loại mixture
học bằng EM ngoài phạm vi DAGSNet. SAM/FedNova/SCAFFOLD là tối ưu/recipe cho proxy có sẵn,
không được đổi tên thành phát hiện họ tri thức mới. Copula/cell-size mới không tự thoát họ moments/cells.

## 2. CrossClientRank — trao đổi score moments cho pairwise ranking

### Căn cứ, khác biệt và rủi ro chính

Teacher/synthesis fit và coverage từng PASS nhưng utility FAIL; V9 correction chủ yếu class-constant,
BoundaryCells P2 quá mềm. Hướng này dùng hard labels tại **real x local**, không teacher/sampler/book.
Imbalance41:1 đã đo trong [dataset](../../knowledge/dataset.md); ranking theo từng lớp là giả thuyết
hợp lý cần kiểm, chưa dự báo macro-F1 tăng. Không bổ sung feature evidence từ peer.

Tri thức mới là **mức score positive/negative của từng class trên quần thể private models**.
ClassTail chỉ dùng true-label CE tail/threshold; ErrorCosts dùng cặp true/predicted class rời rạc.
Ở đây mỗi record nhận gradient ở cả true class và 15 negative scores, phụ thuộc liên tục vào score.
Không match nuisance covariance, input derivatives hoặc class-gradient conflict; không đổi carrier proxy.
Khác biệt này đủ đề xuất objective mới trong MobiHet, chưa là novelty công bố hoặc lợi FL.

### Objective khóa đề nghị: bounded scores, hai phía phải có gradient

Đặt `s_i,c(x)=softmax(z_i(x))_c`, T=1; không temperature fit/clip/margin normalization sau khi xem probe.
Score có cùng đơn vị [0,1]/class semantics; **không bảo đảm calibration tương đương** giữa clients.
`N=sum_i N_i`, `N_c+=#(y=c)`, `N_c-=N-N_c+`; tất cả chỉ trên T train, row-count pooling.

```text
mu_c+ = E[s_i,c(x) | y=c];  q_c+ = E[s_i,c(x)^2 | y=c]
mu_c- = E[s_i,c(x) | y!=c]; q_c- = E[s_i,c(x)^2 | y!=c]
R_c = E_[(i,x)+,(j,x')-][(m - s_i,c(x) + s_j,c(x'))^2], m=1
    = m^2 + q_c+ + q_c- - 2*m*mu_c+ + 2*m*mu_c- - 2*mu_c+*mu_c-
F = sum_i (N_i/N)*CE_i + lambda/K * sum_c R_c; K=16, lambda=0.25
```

Pairs gồm i=j và i≠j, không gọi toàn bộ pairs là peer-only. Không bỏ own contribution khỏi pooled sums;
không cộng thêm own-pair loss lần nữa. Gradient block chính xác tại snapshot hiện tại:

```text
dR_c/dtheta_i = 2/N_c+ * sum_(x in T_i,y=c) (s_i,c-m-mu_c-) * grad s_i,c
             + 2/N_c- * sum_(x in T_i,y!=c) (s_i,c+m-mu_c+) * grad s_i,c

L_i = mean_(T_i) CE
    + lambda*N/K * mean_(T_i) sum_c [1[y=c]/N_c+ * (s_i,c-m-stopgrad(mu_c-))^2
                                  +1[y!=c]/N_c- * (s_i,c+m-stopgrad(mu_c+))^2]
```

`grad L_i=(N/N_i)*grad_i F` tại cùng snapshot: block scaling dương, không exact global-gradient SGD.
Frozen means từ cuối round trước tạo **surrogate stale** khi θ thay đổi; không gọi L_i value bằng F.
q dùng tính R/variance và audit, **không là SGD coefficient**. Nếu chỉ dùng positive side, hoặc
differentiate thêm qua mean đã frozen, không còn gradient trên. Own arm thay N/means bằng của chính xe.
Fixture đại số local15/10:9 logits/3classes, block sizes2/3/4, FP64 explicit all-pairs so surrogate
cho max gradient error3,47e−18; F=1,75693 khác surrogate value2,02752 đúng dự kiến. Chỉ một fixture,
chưa đủ missing-support/wire/resume/production gate, chưa có SGD hoặc bằng chứng utility.

Mục tiêu là squared-ranking surrogate của **joint collection private models**. Cross-model ranking
không bằng AUROC/F1 của private model triển khai. Pooled moments có thể chỉ phản ánh local prior/
calibration; perfect classifier cũng có thể đã làm means gần0/1 khiến peer information vô ích.
Macro one-v-rest ranking không bảo đảm multiclass argmax F1 hoặc benign precision.

### Cohort, wire và cost đề nghị

Warm-up CE r1–r2; scan T cuối r2 BN-eval/FP32 logits, sums FP64; từ r3 dùng snapshot trước round.
Mỗi class cần positives/negatives ≥128 rows, ≥5 contributing clients **mỗi phía**; own cần ≥32 rows
mỗi phía. Thiếu support: bỏ class ranking term, không impute pseudo-count; CE giữ nguyên.
Loss normalization vẫn K=16 để không âm thầm đổi lambda khi mask đổi. Báo counts, eligibility và own
fractions trong pooled mass; cohort trừ own chỉ diagnostic, không secretly đổi objective thành leave-own-out.

| payload đề nghị | bytes/refresh/client |
|---|---:|
| up: header32 +16×[2 uint64 counts +4 FP64 sums/sumsq +2 uint32 contributor flags] | 928 |
| down: header32 +16×[2 FP64 means +2 uint64 counts +2 uint32 contributors] +eligibility2 +prior64 +N8 | 746 |
| tổng trước bootstrap/channel overhead | 1.674 |

Counts aggregate uint64 với overflow guards; producer flags0/1; check `0≤sum(s²)≤sum(s)≤N` và
variance không âm ngoài tolerance. Header/hash/schema, support mask, scaler/prior, ACK, recovery,
retransmission/catch-up đều trả byte thật. Down prior là accounting; không tự hợp thức hóa gp.
Coherent snapshot version/hash mới activate. 1.674B trước overhead chưa là contact/deadline R1 proof.

Một model forward/backward train như CE; rank tính trên16 scores, thêm eval scan mỗi round.
Không tự gọi overhead nhỏ: phải đo full epoch T4, scan/eval/I/O/import; candidate≤2×plain CE thật.
Không scan/dummy-inflate local denominator. Không đổi batch/steps/fixture/model để làm vừa cost.

### Probe falsifiable, rồi mới utility

**Đề nghị chốt toàn bộ gate trước chạy**, seeds42/43, cùng fixture/BN/CE warmup; không HP sweep.

1. Algebra: toy multi-client/classes/unequal counts, explicit all-pairs autograd so block surrogate
   ở đúng snapshot (FP64 relative error≤1e−8, finite-difference≤1e−5); kiểm hai phía, self-pairs,
   missing class/zero denominator, eligibility, wire roundtrip. Exact-resume deterministic.
2. Production train-only r2: ≥12/16 classes eligible; ≥90% valid scanned rows trong cohort đủ support;
   mọi moments/gradients finite. Báo per-client/receiver/class và relative age snapshot.
3. Peer signal: trên T bounded seeded probe, RMS logit-gradient `delta=g_pooled_rank-g_own_rank`
   ≥5% RMS own-rank gradient ở ≥12/16 lớp, cả hai seed. Denominator gần0 thì đánh dấu degenerate,
   không epsilon-amplify để PASS. So thêm pooled-v-own prior/calibration diagnostics.
4. Screen class scaling/calibration đơn giản: trên một nửa probe T, least-squares **diagnostic**
   chiếu delta lên span {true-class weighted CE gradients, constant class-logit biases, temperature
   logit direction}, hệ số **riêng mỗi client**; trên nửa T còn lại cần residual-energy fraction≥20%,
   macro theo class, cả hai seed. Đây phép phân tích gradient, không train/deploy model phụ;
   residual lớn không bác bỏ mọi dạng calibration artifact. Phải khóa exact basis/normalization/support
   trước implementation. Không đạt => không mở r20 hoặc đổi scores để tìm thắng.

Probe chỉ chứng minh gradient tồn tại/khác own và không dễ giải thích bằng calibration ở checkpoint đó;
không chứng minh hướng update hữu ích, truyền feature boundary hoặc improve unseen receiver.

Nếu probe PASS: fixed r20, two seeds, CE /own-rank /pooled-rank /class-permuted pooled means /
pooled stats frozen r2 /nonnegative class-weight CE projection /fresh Gaussian /fresh DML.
Mỗi arm có model/stats riêng; shuffle giữ counts và đổi score-class association theo derangement khóa.
Class-weight control dựa projection train-only, phải khóa map/clipping trước run; không gọi matched
full gradient. Actual cost/steps khác phải báo, không giảm controls để scale candidate.

Gate utility đề nghị mỗi seed: pooled−own/shuffle≥0,010; pooled−CE≥0,015; benign giảm≤0,005;
muốn mở dài phải pooled−max(Gaussian,DML)≥0,015 và cost≤2×CE. Nếu class-weight control ngang pooled
trong0,005 cả hai seed: chưa chứng minh pairwise mechanism; không gọi thắng FL từ riêng loss/AUC giảm.
Đủ10metrics/per-class/old/group, source/config/data hashes, predictions/CM/resume cần audit terminal.

**Owner cần chốt:** objective/score/lambda/margin; counts/refresh/gates; basis/control projection;
và **primary head hay gp cho toàn bộ arms/references**. Ranking gradient không là nonnegative
CE target mass; không tự dùng ClassTail target-mass gp. Đề nghị head primary cho fresh pilot này,
gp cùng fixed true-class prior chỉ diagnostic/heuristic; nếu giữ gp primary phải khóa rule trước trial.

## 3. SAM — recipe dự phòng; không thêm tri thức FL

Giả thuyết: private memorization/late metric decline có thể hưởng optimizer regularization.
Chưa đo sharpness gây failure; SAM không sửa target quá mềm, thiếu feature information hay decoder overshoot.
Đề nghị **SAM chuẩn rho0,05**, CE-only/private trước; không ASAM/rho sweep. Cấu hình là adaptation,
không kế thừa kết quả image benchmarks hoặc exact min-max inner solution của paper.

```text
g0 = grad_theta CE_B(theta)
epsilon = rho * g0 / ||g0||_2    # norm0 => epsilon0; stop-gradient
g1 = grad_theta CE_B(theta + epsilon)
restore exact theta; base_optimizer.step(g1)  # ONE optimizer step
```

[Code tác giả, get_sam_gradient](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py)
giữ state của forward đầu, bỏ state forward perturbed; gradient cuối mới clip/update. Adaptation PyTorch cần
khóa BN/dropout/AMP riêng: cùng batch, cùng dropout RNG; hai pass dùng batch stats; chỉ BN state pass1 commit.
Không đổi pass2 sang eval (khác objective), không để num_batches_tracked/EMA update hai lần.
AMP cần FP32 unscaled g0 trước norm; overflow skip **toàn step**, restore weights/BN/RNG theo policy;
không optimizer.step hai lần hoặc sửa Adam state ở ascent. Restore bằng saved weights, không add/sub FP32
rồi giả exact khi cancellation làm sai bytes. Finite-difference/zero-grad/skip/resume fixtures trước T4.

Train-only engineering gate: rho0 đúng CE update/BN state; perturb norm rho khi g≠0; no nonfinite;
đo T4 full epoch chi phí≤2×CE thật, không gọi hai backward mặc định walltime đúng2×.
Đó là trainability/cost gate, không utility. Fresh r20 two-seed CE vs SAM cần hơn CE≥0,015,
benign giảm≤0,005; head/gp cố định, đủmetrics. Giữ thêm true-compute CE reference với nhiều steps
chỉ khi owner chốt schedule riêng trước trial, không kéo dài baseline sau khi xem kết quả.

Nếu thử trên dense DML: phải CE/SAM ×DML-off/on cùng own controls; giữ proxy original, chỉ SAM private
loss theo contract owner chốt (CE-only hay CE+KD). SAM private không mặc định nhân đôi whole DML cost;
teacher/proxy forward, KD targets/BN/dropout phải khóa và đo riêng. Dense snapshot bytes vẫn trả đủ;
SAM không giải R1/R2 hoặc chứng minh hiệu quả cộng tác nếu local/own hưởng lợi tương đương.

Chỉ CrossClientRank train-only probe đã được chốt và push; utility và SAM chưa launch.
Nếu gate FAIL, ghi phạm vi quan sát rồi dừng; không đổi score/rho/headline để cứu.
