# Synthetic transfer: thông tin nào xác định được update private?

22/10/2026; nối lại nghiên cứu sau yêu cầu owner. Phạm vi: đọc source/artifact đã audit, nguồn primary,
suy luận đại số; **không train, launch, sửa production hoặc nhận phương pháp mới**.
[Bàn giao](../../archive/operations/continuation-synthetic-2026-10-22.md),
[kết quả P23–28](synthetic-learning-recovery-2026-10-22.md),
[veto/build cũ](synthetic-build-history-audit-2026-10-22.md).

**Kết luận:** objective hiện tại xác định input giống nhau theo cách nhìn của teacher; điều cần cho private
là update hữu ích theo representation/loss/optimizer của chính private. Hai điều kiện khác nhau.
Ngay cả khớp hoàn hảo teacher features/logits và full input mean/covariance theo lớp cũng không đủ trong mọi trường hợp.
Đây là thiếu thông tin có thể chứng minh; chưa chứng minh nó là nguyên nhân duy nhất của plateau final50.

## 1. Phạm vi chứng cứ hiện có

- P23 Project48 nâng teacher và qua teacher gate hai seed; P24 transfer/inversion trượt cổng private.
  Vì vậy không thể quy mọi FAIL cho teacher không đủ mạnh hoặc cho nén không giữ teacher.
- P27, cùng teacher mới và cùng recipe, real-support oracle đạt mean0,841418/worst0,818250;
  Gaussian0,828478/0,783014; inversion0,831036/0,792128. Support làm mất khả năng chuyển giao trong
  **frozen-backbone head diagnostic này**. Oracle raw pooled input không hợp lệ để deploy theo R3.
- P28 giảm weighted train CE nhưng không tăng teacher utility. Không mở lại gradient-only head adaptation.

Các số trên từ [hồ sơ audited](synthetic-learning-recovery-2026-10-22.md); không dùng số paper làm dự báo IDS.

## 2. Objective đang khớp gì, private cần gì?

[P24 `invert`](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe24.py) tối ưu từng input noise:

```text
z_T(x) = Pᵀ h_T(x), dim(z_T)=48
goal ~ Gaussian(class mean, class covariance)
L_inv = mean(((z_T(x)-goal)/feature_scale)^2)
        + .01 mean(((x-x_initial)/input_scale)^2)
```

13 feature mask giữ0. Teacher cố định; chưa có private feature, Jacobian hoặc private update trong objective.
Goal là virtual feature từ Gaussian, không phải feature của một record thật. Goal có thể ngoài image của encoder;
MSE thấp chưa chứng minh đúng joint distribution của input. Head CCVR được học và đánh giá trong cùng không gian
teacher, nên CCVR teacher PASS không chứng minh phép nghịch đảo input hoặc heterogeneous transfer PASS.
CCVR gốc cũng là post-hoc classifier calibration trên feature extractor chung, không đề xuất phép đổi latent teacher
sang input phù hợp mọi private. [Luo et al., NeurIPS2021, §4](https://proceedings.neurips.cc/paper/2021/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf).

Với private logits `s_i(x)`, soft target `q_T(x)` và KL ở temperature1:

```text
g_i(x) = J_{θ_i}s_i(x)ᵀ [p_i(x)-q_T(x)]
Head-only: g_Wi(x) = [p_i(x)-q_T(x)] h_i(x)ᵀ
```

Đây là công thức xác suất normalized chính xác. Khi audit fp32, dùng `p_i*sum(q_T)-q_T` theo target thực;
không mặc định `exp(log_softmax)` cộng thành1 bit-exact. Sai số số học phải tách khỏi mismatch phương pháp.

Khớp `z_T` hoặc `q_T` chưa khớp `h_i`, chưa khớp Jacobian, nên chưa khớp `E[g_i]`.
P24 còn CE own, logit adjustment, real KD và adaptive weight `1/(CE.detach()+proxy_CE)`;
full harness thêm backbone/BN và optimizer state đang học. Diagnostic update phải dùng đúng loss và trạng thái,
không coi cosine một gradient head CE là update AdamW cuối cùng.

Synthetic component chọn đều16 lớp **không** đồng nghĩa soft target prior đều. Muốn tách prior khỏi geometry,
phải đo/ghép `mean(q_T)` cùng norm và class schedule; reweight theo nhãn component chỉ là control khác.

## 3. Không thể nhận diện: phản ví dụ giữ cả input moments

Phản ví dụ đại số của ghi chú này; không phải số đo VeReMi. Binary `y∈{-1,+1}` đều, `s` đều trên `{-1,0,1}`.
Hai quần thể khác nhau bởi `ε=+1` hoặc `−1`:

```text
x = (y, s, ε y(s²-2/3))
h_T(x)=x₀;       q_T(y=+1|x)=sigmoid(κ x₀), κ>0
h_private(x)=x₂ x₁²; private binary logit θ h_private(x), θ=0
```

Trong **mỗi lớp**, cả hai quần thể có:

```text
E[x|y] = (y,0,0)
Cov[x|y] = diag(0,2/3,2/9)
Toàn bộ distribution h_T|y và logits teacher giống nhau.
```

Nhưng `E[h_private|y]=ε (2/9)y`; gradient KL private tại `θ=0` bằng:

```text
E[(1/2-q_T(x)) h_private(x)] = -ε (2/9) [sigmoid(κ)-1/2]
```

Hai dấu đối nhau. Teacher đúng cùng nhãn, class prior giống, teacher distribution giống, full input moments
theo lớp giống, nhưng update private cần hướng ngược nhau. Thêm logits teacher, class confusion hoặc
teacher-margin histogram vẫn không phân biệt được ví dụ này: chúng chỉ là hàm của teacher feature đã biết.
Muốn phân biệt cần thống kê chung có chứa `x₂x₁²`, hoặc thông tin đáp ứng của chính private.
Đã enumerate6 điểm/quần thể bằng Python standard library: class moments và công thức gradient PASS tới1e−14;
đây là kiểm đại số nhỏ, không train model. Giả định private giữ feature cố định, head scalar và KL normalized.

Nguồn inversion cũng nhấn mạnh một representation có nhiều inverse do invariance;
reconstruction chỉ giống input gốc theo representation ấy. Không suy nghĩa semantic từ feature fit đơn độc.
[Mahendran & Vedaldi, CVPR2015, §2](https://openaccess.thecvf.com/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf).

### Điều kiện đủ và giới hạn của finite carrier

Nếu biết chính xác joint distribution `(z_T,y)`, thì kỳ vọng gradient private được xác định cho **mọi** phân phối
có joint ấy khi `g_i(x,y)` là hàm của `(z_T(x),y)`; tương đương gradient không đổi trên mỗi fiber teacher.
Một factorization đủ là `h_i(x)=ψ_i(z_T(x))` cho head probe. R2 khác kiến trúc không bảo đảm factorization này.

Trên finite input domain, carrier chỉ giữ `E[Φ₁],…,E[Φ_m]`: kỳ vọng một thành phần gradient `g` được xác định
cho mọi phân phối khi và chỉ khi `g` thuộc span của `1,Φ₁,…,Φ_m`. Nếu không, tồn tại vector chênh phân phối
trong nullspace của constraints nhưng có tích vô hướng khác0 với `g`; thêm/bớt một lượng nhỏ từ distribution
có full support cho hai phân phối hợp lệ, cùng carrier nhưng khác gradient. Đây là suy luận linear algebra,
không tuyên bố finite carrier luôn vô dụng trên distribution thực tế.

Vì vậy **không có bảo đảm phổ quát** rằng một vector thống kê cố định nhỏ sẽ xác định update của mọi private
architecture/snapshot. Cần test function mới liên quan tới private, hoặc chấp nhận approximation rồi kiểm utility.

## 4. Gradient chung không tự thêm label-boundary information

Với common head `A` trên `u`, gradient gộp supervised:

```text
G(A)=E[p_A(u) uᵀ] - E[onehot(y) uᵀ]
G(A₁)-G(A₀)=E[(p_A₁(u)-p_A₀(u)) uᵀ]
```

Label term chỉ mang count/mean từng lớp; lấy hiệu nhiều head loại bỏ chính label term. Không đồng nghĩa mọi
gradient matching đều bằng mean matching: **giữ riêng từng true class** còn giữ `E[p_A(u)uᵀ|y=c]`,
một nonlinear labeled crossmoment. Với predictor gần uniform trong mỗi lớp, nó lại gần scaled class mean.
Zhao & Bilen diễn giải đúng last-layer gradient là probability-weighted feature mean và chỉ ra uniform-prediction
limit về mean matching. [DM, WACV2023, §2.4 và appendixE](https://www.pure.ed.ac.uk/ws/files/302628065/Dataset_Condensation_ZHAO_DOA14102022_AFV.pdf).

V5 đã thử producer-private gradientbooks; đổi tên thành gradient synthesis không tạo cơ chế mới.
Đối tượng gradient matching vốn là effect trên learner, không phải realism; gradient matching gốc xét learner
parameters, còn trajectory matching xét nhiều bước học. Chúng không cấp bảo đảm transfer cho arbitrary private,
không cho phép gửi expert/private model miễn phí theo R1.
[Zhao et al., ICLR2021](https://www.pure.ed.ac.uk/ws/files/202549394/main.pdf),
[Cazenavette et al., CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPR_2022_paper.html).

## 5. Chỉ hai cơ chế có điều kiện để kiểm nhỏ

### A. Thêm joint giữa true class, teacher boundary và input

Carrier giả thuyết, **chưa implement, chưa utility/packet PASS**:

```text
u(x)=[1;53 active standardized input features]
B[c,a,j]=Σ_record w_record 1[y=c] p_T,a(x) u_j(x)
```

Client dùng cùng frozen teacher; weighted sums cộng được trước normalize. `w` phải là inclusion weight đúng
từ full train counts, không counts fixture. Server/recipient chỉ xem cohort sum trong giả định SecAgg;
không gửi per-record activation. Carrier này giữ nonlinear true-label×boundary×input joint, không chỉ các
marginal teacher latent moments/input moments riêng. Nó tương đương class-conditioned gradient tại0 của
**input residual readout** `logits_T(x)+V u(x)`, nên thuộc gradient-matching family; không claim novelty thuật toán.

Khác P24: objective synthesize thêm joint `B`, không chỉ target latent point. Khác V5: common teacher/input
coordinates và aggregate carrier, không broadcast bank khớp gradient của producer private khác recipient.
Khác BoundaryCells: không dùng static input-cell counts làm teacher mềm. Tuy vậy interface mới không là utility proof.

**Giới hạn:** teacher probability gần constant trong lớp ⇒ B gần `p(c) E[u|c]`, không thêm nhiều thông tin.
Carrier cũng không phân biệt phản ví dụ §3. Chỉ dùng projected teacher latent thay `u(x)` không sửa fiber problem.

**Controls/falsifier:** cùng source teacher, Gaussian bank size/steps/seed/mask; input/latent moments-only;
confusion/soft-mass-only control; class-shuffle B giữ mọi marginal; own versus pooled B. Trước generation,
đo crossmoment residual `Cov(p_T,u|c)` và sự khác biệt directional private gradient ở các support khớp carrier.
Nếu B chỉ đổi prior/mean, hoặc loss B giảm mà private gradient/benign/worst không tiến bộ, loại. Không tăng IPC,
inversion steps hoặc nới gate để cứu cùng adaptation. Không dùng oracle real input như deployable arm.

**Byte:** tensor fp32 `16×16×54×4=55.296 B`;16 uint64 class counts128 B; một lượt raw55.424 B lên và55.424 B
rebroadcast xuống =110.848 B/client. Có thể loại output row dư vì tổng probability1 và class means đã biết,
nhưng chưa serialize/decode nên chưa ghi số nén thành wire proof. Số trên chưa header/CRC/SecAgg/ACK/model.
Generation làm local từ noise và aggregate target; không truyền ngân hàng4.096×66 float32 hơn1MB/contact.

14/49 refresh chỉ carrier này thêm1,552/5,432MB, nominal ratio~1,301/1,419 so xa5,153/12,95MB ở15/50round.
**Stack Project48 thêm81.608 B mỗi refresh ⇒ nominal ratio~1,523/1,728, FAIL cap1,5×.**
Không coi refresh/contact miễn phí. Xa peak228.658 B để lại2.342 B/contact; bootstrap và peak final50 còn vượt231.000 B.
Carrier standalone dưới cap không làm toàn phương pháp R1 PASS. Cần replacement/sparse schedule có căn cứ,
wire codec và contact timeline riêng trước online; không chốt lịch từ dự toán này.

### B. Sinh/refine tại recipient bằng actual private update, có local safety

Chỉ xét nếu diagnostic hiện tại xác nhận mismatch update. Giữ shared carrier/source đã chứng minh có ích;
private locally refine input noise bằng đáp ứng của **chính recipient**, tại snapshot hiện tại, với loss thật
và optimizer state thật. Reference labeled minibatch/heldout và derivative ở lại client; synthetic không rời client.
Điểm thay đổi so V5: không đòi recipient hấp thụ gradient đã khớp producer khác; không dùng CE books broadcast.
Đây là giả thuyết local adaptation, chưa có algorithm/HP/gate mới được nhận.

Đại lượng kiểm trước: nếu proposed full update là `Δθ`, Taylor change trên reference local risk là
`∇L_refᵀ Δθ`; dùng finite-step loss thực để kiểm curvature/AdamW, ghi riêng benign và class có đủ support.
Scalar directional derivative có ích **để kiểm một update đã xác định**, không đủ làm kênh FL dạy ranh giới.

**Controls/falsifier:** source/support/soft-mass/norm matched; teacher-only refinement; recipient-aware refinement;
safety gate không refinement (để tách lợi do bỏ synthetic khỏi lợi do sinh tốt). Reference batch tách minibatch
đang optimize; report tất cả20 client, benign/worst. Nếu safety làm mọi synthetic update≈0 hoặc lợi chỉ do
bỏ KD, không nhận generator. Nếu chỉ local loss giảm mà global heldout xấu, không claim transfer PASS.

**Giới hạn:** own reference không có class/receiver peer ⇒ không nhận diện global beneficial direction.
Local safety có thể chặn đúng update cần cho global classes; không giải quyết thông tin peer thiếu. Đưa peer raw
pool vào reference chỉ là oracle chẩn đoán, không deploy. Không có extra payload tự động; compute tăng cần đo.
Không coi cosine âm một mình là nguyên nhân: conflict, magnitude và curvature phải đo đồng thời.
[Yu et al., NeurIPS2020, §2](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf).

## 6. Cách xác nhận căn nguyên trước phương pháp mới

1. **Cùng teacher feature/logits, khác inverse:** nhiều noise starts cho cùng target, ghép tolerance teacher;
   đo private feature/actual-update variance. Teacher fit cố định mà update đổi dấu ⇒ fiber ambiguity thực nghiệm.
2. **Đúng target mass:** ghép teacher soft-label mass và step/norm; so Gaussian/inversion/real-support oracle
   ở actual private gradients và finite optimizer steps. Tách support khỏi class prior/KD dose.
3. **Cùng update direction:** norm-matched và source-matched controls; report benign false positives,
   attack precision/recall, min client; đừng để macro che đánh đổi đã thấy ở client18.
4. **Qua diagnostic rồi mới generator:** carrier phải thêm thông tin có thể phân biệt update, generator phải
   thực hiện được thông tin ấy, private phải hấp thụ được. Ba cổng khác nhau; từng PASS không thay online proof.

CMI ghi nhận inversion constraints có nhiều nghiệm và diversity là yếu tố riêng; Adaptive DeepInversion thêm
teacher–student disagreement. Đây là nguồn động cơ đo inverse variability, **không** là lý do lặp DI/khai thác
bất đồng `xo` đã FAIL. [Fang et al., IJCAI2021, §3](https://www.ijcai.org/proceedings/2021/0327.pdf),
[Yin et al., CVPR2020](https://research.nvidia.com/index.php/publication/2020-06_dreaming-distill-data-free-knowledge-transfer-deepinversion).

Nghiên cứu KD cũng có teacher–student fidelity gap dù student đủ capacity; nguồn transfer và optimization đều cần
kiểm, không chốt capacity hoặc LR chỉ từ gap0,03. [Stanton et al., NeurIPS2021](https://research.google/pubs/does-knowledge-distillation-really-work/).
Giữ nguyên cổng mean/benign/worst hai seed, horizon/R1/R2 và veto lịch sử; chưa có phương án mới đủ căn cứ launch online.

## 7. Sau P30/P31: chỉ giữ đề xuất conditional input density để kiểm capacity

**Trạng thái đóng phiên:** owner yêu cầu tổng hợp/compact trong lúc ghi mục này. Chỉ lưu phân tích;
không viết generator code, train, launch hoặc chốt yêu cầu khoa học mới. Chỉ mở lại khi owner tiếp tục.

### Bằng chứng mới giới hạn hướng tiếp theo

- [P30 audited](../../methods/full-drift/results/probe30-joint-input-s42.audit.json): joint mean0,832083;
  mean-only0,832315; zero-cross0,832284; Gaussian gốc0,828539. Joint gain+0,003544 <+0,005,
  không hơn các controls. **Đóng adaptation A (joint carrier); không chỉnh steps/weights/rank để mở lại.**
- [P31](../../methods/full-drift/results/probe31-private-observability-s42.json):16 train points×20 private,
  strictFP32; teacher48 Jacobian rank48,5 null directions. Private-Jacobian null-energy ratio mean0,004618,
  median0,001618,max0,048707. Ở ε0,001, normalized finite response teacher0,0002645/private0,04249.
  Có inverse ambiguity tại các điểm đo, nhưng phần tổng năng lượng Jacobian private trung bình nhỏ.
  **Không gọi projection blind space là căn nguyên chính**; sample16 điểm không định lượng F1 final50.
- P29 vẫn cho thấy nguồn synthetic thiếu joint input/target cần cho private update. Sau P30 FAIL,
  thêm một teacher-inversion constraint không còn đủ căn cứ. Hướng dưới thay nguồn thông tin học density.

### Cơ chế duy nhất còn đề nghị kiểm, chưa nhận làm phương pháp

Học `p_φ(x_active|y)` **trực tiếp từ true train records tại client**, chia sẻ aggregate update của một
generator chung; sinh từ noise vào **input53 chiều**, phục hồi13 chiều m4b=0 trước classifier66 input.
Private classifiers giữ tham số/kiến trúc riêng. Không upload record, per-record activation/gradient hoặc bank lấy
từ record. Generator dùng nhãn thật local để học joint input, thay vì chỉ rút density từ teacher features.

Family cụ thể để kiểm capacity: conditional RealNVP, **hai affine coupling blocks**, split26/27 đảo nhau,
class onehot16, mỗi block một hidden layer64 và output chung cho scale/shift. Với class cố định, phép coupling
triangular có inverse và log-Jacobian dễ tính, nên có thể fit conditional likelihood và sample từ Gaussian noise.
Đây là adaptation nhỏ, không phải kiến trúc/số layer đã có utility trong paper.
[Dinh et al., ICLR2017, §3.1–3.3 và appendixF](https://arxiv.org/pdf/1605.08803).

Khác FedGen latent: density được fit từ local true input, không chỉ làm generator hợp ensemble predictor;
khác MALA/density-ratio, DM/KME/books: không tối ưu finite bank theo teacher energy hoặc moment loss.
FedGen gốc truyền generator latent học từ prediction rules; không lấy kết quả FedGen làm utility proof cho đề xuất này.
[Zhu et al., ICML2021, §4](https://proceedings.mlr.press/v139/zhu21b/zhu21b.pdf).

Rà532 Markdown files trong `docs/` và `archive/methods/`: không thấy mô tả experiment RealNVP/conditional
normalizing flow/VAE/affine coupling. Đây là kết quả tìm trên tài liệu local, **không chứng minh mọi artifact cũ
chưa từng thử**. Hướng liên quan đã có: [V2 density-ratio/support warning](../../archive/research/de-xuat-v2/nguon-nghien-cuu.md);
thiết kế đó cấm tự thêm jitter/dequantization để né support mismatch. Không lặp Gaussian copula hoặc geometry
constraints đã FAIL dưới tên density learner.

### Atoms/profile: điều kiện chưa được phép bỏ qua

Continuous invertible flow với continuous Gaussian base có continuous density; không thể sinh đúng point mass
ở active-feature atoms hoặc exact deterministic relation trên tập thấp chiều. MobiHet đã đo nhiều atoms mạnh;
hai affine blocks còn có thể thiếu capacity cho đa mode/profile. Không mặc định likelihood cao là samples đúng
class/manifold hoặc update hữu ích. Fitting continuous density vào discrete observations có nguy cơ density
collapse; dequantization thay target distribution và cần thiết kế rõ.
[Ho et al., ICML2019, §3.1–3.2](https://proceedings.mlr.press/v97/ho19a/ho19a.pdf).

Capacity screen ban đầu nếu được mở: giữ nguyên input/scaler/partition, không tự jitter, round/clamp atom,
đổi feature hoặc thêm profile label. Khóa scale bound/numerical policy trước chạy để tránh density runaway;
không dùng NLL như exact mixed-measure likelihood. Nếu mục tiêu đòi atoms chính xác thì family continuous này
không đáp ứng: dừng hoặc hỏi owner chốt mixed-data representation; không tự mở family khác trong phiên compact.
Nếu muốn dequantization/atom decoder, phải chốt semantics, inverse, noise scale và toàn bộ byte mới trước thực hiện.

### Capacity oracle trước federated generator

**Bước tiếp theo có điều kiện, chưa chạy:** fit đúng family nhỏ trên train pool local theo class-conditional
population weights. Pool chỉ là centralized capacity oracle chẩn đoán, **R3-ineligible để deploy**;
không đọc old/group/test để optimize density. Fit budget/checkpoint rule khóa bằng train trước chạy, không chọn
checkpoint bằng private holdout F1.

Sinh bank4096 cùng class schedule/seed; P24 frozen private backbone/BN, own rows/512+256/1000steps như nhau,
đủ20 client. Controls: Gaussian gốc, full-Gaussian từ **cùng rows và full-count weights**, fitted flow,
real-support oracle. Teacher gốc/Project48 factorial giữ nguyên; targets mềm, không tự chuyển hard labels.
P27 real-support gain teacher gốc chỉ+0,004124, teacher Project48+0,012940; vì vậy không mặc định flow
với teacher gốc có đủ source ceiling, cũng không giấu chi phí Project48 nếu chỉ tổ hợp mới thắng.

**Falsifier/cổng đề nghị trước triển khai:** flow−weighted-Gaussian **và** flow−Gaussian gốc≥+0,005 mean-private
group, benign≥−0,005,worst≥−0,010; báo đủ predictions/CM/10metric và class precision/recall. NLL, teacher agreement,
atom/KS plots và gradient cosine chỉ diagnostic. Nếu capacity oracle không qua cổng thì đóng family/adaptation
hai block này, không federate hoặc thêm layers để tìm win. Nếu qua, mới khóa hai-seed T4 confirmation và
matched own-generator/cohort-generator control để chứng minh tri thức peer, rồi online/horizon/R2 riêng.

Federated fitting còn là giả thuyết riêng: common initialization/base/version, không average các flow train độc lập
từ đầu. Objective class-balanced pooled density là `L=(1/16)Σ_c E_{pooled real|c}[−log p_φ(x|c)]`.
Full-data row sampling + global class weight `N/(16 N_c)` + sample-weighted client aggregation thực hiện đúng
population objective tại một common snapshot; many local steps không còn bằng exact pooled gradient.
Class-balanced mỗi client rồi FedAvg theo n_k nói chung sai mixture; fixture subsampling phải dùng inclusion
weights `N_kc/m_kc` và full counts. Federated generator có thể FAIL dù pooled capacity PASS.
Model-update aggregation lấy động cơ từ [McMahan et al., AISTATS2017](https://proceedings.mlr.press/v54/mcmahan17a.html),
không kế thừa bảo đảm utility cho generator non-IID. SecAgg vẫn chỉ giả lập; không claim DP/mật mã.

### Param và ngân sách nominal, chưa codec proof

Chốt cấu trúc chỉ để tính kích thước: hai-layer MLP nghĩa **một** hidden64; không embedding trainable,
không BN/ActNorm/running statistics, không gửi optimizer state. Scale và shift dùng chung output MLP.
Fixed global mean/scale53 mỗi loại là106 fp32 buffers, cùng mọi client; masks/permutation tái dựng từ schema.
Không thêm normalization lần nữa vào dataset: đây là inverse-aware internal transform nếu cần và phải khóa.

| Thành phần | Phép tính | Số scalar |
|---|---|---:|
| Block1:26condition+16class→64→2×27 | `42×64+64+64×54+54` |6.262 params|
| Block2:27condition+16class→64→2×26 | `43×64+64+64×52+52` |6.196 params|
| Trainable tổng | `6262+6196` |**12.458 params**|
| Fixed mean/scale | `2×53` |106 buffers|
| Một model fp32 payload | `(12458+106)×4` |50.256 B|
| Hai chiều payload | `2×50256` |100.512 B|
| Nominal packet với header32B+count8B/chiều | `2×(50256+40)` |**100.592 B/exchange**|

Số là arithmetic đã kiểm, chưa serialize/decode/equivalence test. Không có class-count vector/per-client
metadata/crypto/ACK/retransmission trong model payload; chỉ coi counts cũ miễn truyền lại nếu protocol chứng minh
đã cache cùng version. Hai hidden64 thay vì một sẽ thêm8.320 params; không được lén dùng count nhỏ bên trên.

| Kịch bản conservative |15round, baseline xa≈5,153MB|50round, baseline xa≈12,95MB|
|---|---:|---:|
|15/50 exchange density, tính cả initialization trong số lượt này|+1,509MB;ratio≈1,293|+5,030MB;ratio≈1,388|
|Cộng Project48 mỗi14/49 refresh|ratio≈**1,515 FAIL**|ratio≈**1,697 FAIL**|

Input density không phụ thuộc private encoder; về cơ chế có thể học một lần rồi dùng frozen generator qua
các FL rounds. Đây là **amortization hypothesis**, chưa chọn J density exchanges hoặc chứng minh J đủ học.
Mọi pretraining contact đều trả phí. Dùng `extra≈J×100592 + J_CCVR×81608` kiểm cả15/50 horizon;
không dùng con số14/49 loại mất bootstrap hoặc final exchange. Với baseline dự toán, thêm CCVR14/49 chỉ còn
khoảng14 dense density exchanges ở cổng15 trước overhead; đây là ceiling xấp xỉ, không launch budget.

Model100,6k riêng dưới231k/contact nhưng cộng xa peak228.658 B thành329.250 B, **FAIL contact cap**;
bootstrap và q3 late peaks vốn còn vi phạm. Cần contact schedule hoặc replacement thực, cùng lịch baseline,
codec/commit/SecAgg costs; không gọi split frame hoặc warmup miễn phí. Giữ R1 chưa PASS.

**Việc chưa chốt khi đóng phiên:** atom/mixed-measure treatment; numerical scale policy; training budget/stop rule;
pooled capacity screen source/controls; J và contact schedule nếu utility qua. Không có yêu cầu tay hoặc launch
mới trong bước lưu tài liệu này. Không thay phương pháp chốt `gkd + la + kdpx`.
