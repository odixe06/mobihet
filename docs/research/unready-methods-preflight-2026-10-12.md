# K2/K4 — đưa thiết kế chưa sẵn sàng về thí nghiệm tối thiểu

Ngày 2026-10-12, UTC+7. **Nghiên cứu và đại số CPU; chưa triển khai/training/Kaggle.**
Các số bước, hệ số loss và cổng mới dưới đây là đề nghị của dự án, cần chủ dự án chốt trước code/push.
Không đổi source `mh/` đã chạy; không Git/DVC. Đọc [CONTEXT](../../CONTEXT.md),
[LESSONS](../../LESSONS.md), [thiết kế K2/K4 gốc](v3-knowledge-sources.md) trước tài liệu này.

## 1. Quyết định tối thiểu đề nghị

| hướng | thí nghiệm U đầu tiên | còn chưa chứng minh |
|---|---|---|
| [K2 residual hàm](../../proposals/anchor-function-residual/README.md) | common DAGSNet probe + BN **gửi snapshot có tính byte**; cập nhật centered-logit target từ residual trên real local; teacher-first | độ hữu ích của residual ngoài anchor; carrier hữu hạn; reconstruction trên thiết bị khác |
| [K4 joint-energy](../../proposals/dagsnet-joint-energy/README.md) | common balanced Gaussian-mixture base; CE raw + energy trên positive perturb local; independence-MH noise-init; teacher-first | mixing, chất lượng teacher/generator; faithful support; carrier hữu hạn |

Hai phương án là **Federated Learning**: chỉ aggregate thống kê/model update rời xe;
không raw/perturbed record, latent row hoặc gradient từng record. Mọi model học được là DAGSNet.
Gaussian mixture dưới đây là tổng hợp analytic từ moment, không EM/GMM/learned generator phụ.
Target table của K2 là thống kê supervision; classifier triển khai vẫn DAGSNet.
U kiểm sức chứa/chất lượng; chưa gọi giải được R1 hoặc R2. Chưa dùng official test chọn cấu hình.

## 2. K2 — residual có common probe, không lặp K2b

### 2.1 Bằng chứng vấn đề và giới hạn nguồn

K2b đã audit: h4 đạt coverage nhưng soft target gần phẳng; small teacher fit entropy floor mà F1 rất thấp.
Hard-label trên cùng Gaussian anchor cũng yếu: không quy hết failure cho target entropy.
[Diagnostic artifact thật](../../proposals/kernel-anchor-distillation/diagnostics.md).
K2 phải kiểm thông tin **residual của common function trên real local**, không đổi tên bảng kernel-count tĩnh.

[FFGB](https://arxiv.org/pdf/2103.06972) nghiên cứu tối ưu hàm với weak oracle, giả định heterogeneity
và residual của oracle. Finite anchor + kernel projection + DAGSNet ở đây **không triển khai FFGB nguyên bản**;
không kế thừa định lý convergence. Công thức/cổng sau là adaptation của dự án.

### 2.2 Bootstrap và common function

1. Dùng đúng fixture pilot hiện hành, 20 client, T 2,5%, seed42/43, 20 round.
2. SecureSum cohort≥5 thu prior và H2diag moment; decode packet FP16 trước mọi phép dùng.
3. Sinh anchor noise-only, **M64 và M512**, nhãn generating class cân bằng. Anchor FP16 serialize,
   dùng chính anchor decoded tại mọi xe; trả phí gửi anchor ban đầu, không mặc định RNG cross-device miễn phí.
4. Shared scale `D_j` là pooled SD từ decoded class moment, floor SD=0,01. Kernel
   `k_m(x)=exp(-||D^{-1}(x-z_m)||²/(2h²))`.
5. h∈{1,2,4,√66}; chọn nhỏ nhất đạt coverage≥90% tổng và≥80% từng lớp, chỉ scan T local/gộp scalar.
   Không đạt thì dừng cỡ M đó trước teacher training. Coverage không phải quality gate.
6. Cùng initial DAGSNet; `t⁰_m=Center(log((1-0,05)onehot(label_m)+0,05/16))`.
   Fit teacher vào t⁰ bằng đúng lịch public bên dưới. Sau đó RSU serialize **toàn state probe**, decode
   và gửi snapshot; RSU/xe đều tính residual với **snapshot đã decode**, không server FP32/xe FP16 khác nhau.

Public init/anchor/bandwidth/cohort/source/scaler/model-state hash đều lưu; thiếu hash/phiên bản đúng
thì không nhận contribution. `SecureSum` hiện tại là trusted emulator, không crypto/DP.
[Codec/source hiện có](../../mh/v2_math.py), [bootstrap H2](../../mh/v2_methods.py).

### 2.3 Residual, update, gauge

Với `pᵗ(y|x)=softmax(DAGSNet_{θᵗ}(x))`, tất cả client dùng **cùng θᵗ và BN**, eval/dropout-off,
forward residual FP32. Prior count `N_y` được gộp một lần; class-balanced weight `a_y=1/max(N_y,1)`:

\[
G_{imy}=\sum_{(x,l)\in T_i}a_l k_m(x)[p^t(y|x)-1(l=y)],\quad
B_{im}=\sum_{(x,l)\in T_i}a_l k_m(x).
\]

Chỉ gửi tổng G/B; **không** gửi k(x), p(x), residual/gradient từng row. Aggregate FP64 sau decode
FP32 client packet để hạn chế thứ tự cộng; không âm thầm gọi `SecureSum` FP32 hiện có là FP64.
Khóa `v_m=G_m/max(B_m,1e-12)`; nếu B_m=0 thì update=0 và ghi zero-mass anchor.
Denominator floor không biến anchor không phủ thành có tri thức.

Đề nghị **η=1; L2 trust radius=0,25/anchor/update**, rồi:

\[
\Delta_m=-\eta v_m\min(1,0.25/(\eta\|v_m\|_2+10^{-12})),\quad
t_m^{t+1}=Center(Clip_{[-8,8]}(t_m^t+\Delta_m)).
\]

Center lần cuối sau clip; `Σ_y G_my≈0` và centered-target mean≈0 là codec/math checks.
Center/clip/trust có thể gây bias; log tỷ lệ clipped và norm trước/sau, không coi chúng là theorem ổn định.
**t là centered logits**, không probability. Không lấy `t−ηG/B` trên probability rồi dùng MSE logits.
Gaussian label chỉ initialization/control; không sharp lại target sau nhìn kết quả.

Residual ở real x không phải CE gradient trên anchor z. Kernel projection là surrogate;
cần đối chứng để biết update mang địa chỉ ranh giới hay chỉ class bias.

### 2.4 Distill common teacher và BN

Đề nghị **20 AdamW applied steps/round**, lr cosine1e−3→1e−5, weight_decay0, grad clip1;
full-anchor batch; FP32 MSE `mean((Center(fθ(z_m))−t_m)²)`.
Optimizer reset mỗi round, teacher weights warm-start từ common snapshot decoded trước đó.
BN canonical recalibrate bằng4096 shared decoded-Gaussian point trước public fit, freeze trong fit/residual;
dropout off. Snapshot mới gồm parameters, BN buffers, integer counters; mọi xe nhận cùng state decoded.
Theo dõi fit gap và trôi target riêng; không suy teacher tốt từ MSE giảm.

Control own giữ **cùng global anchor/bootstrap**, nhưng chỉ residual local và teacher local;
đo phần lợi thêm của aggregate residual. Vì bootstrap vẫn global, own này **không là no-FL baseline**.
Muốn teacher own học hoàn toàn local phải thêm arm riêng; không tráo denominator sau chạy.

### 2.5 Arm teacher-first, cổng và compute

Mỗi M64/512 có: `dynamic_fl`, `dynamic_own`, `residual_shuffle`, `residual_classmean`,
`hardlabel_sameanchors`, `frozenprobe_residual`. Fresh `dense_CE` và private-local dùng cùng fixture/schedule.

- `residual_shuffle`: permutation cố định anchor-address của G/B theo seed, giữ norm/count/compute;
  t⁰ và anchor không đổi. Phá địa chỉ residual, không shuffle label real.
- `residual_classmean`: lặp mean_m(v_m) tại mọi anchor; cùng trust/fit; đo lượng thông tin chỉ bias lớp.
- `frozenprobe_residual`: residual luôn từ snapshot initial θ⁰; target/teacher vẫn update theo cùng công thức.
  Đây là ablation common-function refresh, không thay bản dynamic để giả thấp byte.
- `hardlabel_sameanchors`: fit t⁰ không cập nhật; cùng20steps/round/BN; scan real/residual làm dummy
  nhưng discard contribution, công bố wasted compute và **không trả phí upload G thật không dùng**.

Final r20 group/head teacher, từng seed độc lập: FL≥local private/gp+0,05;
FL≥hardlabel+0,01; FL−own/shuffle/classmean≥0,01. Dynamic−frozen≥0,005 là cổng **refresh contribution** riêng.
Không qua teacher gate thì không private KD/contact. Nếu qua: manifest private riêng,
GP chính, private FL−diag≥0,015; −own/shuffle≥0,01; common local/old/benign guards như pilot trước.
Không head/best-round/seed-averaging cứu cổng.

Preflight Kaggle đo forward toàn T cho residual và public20step; dự toán log thật+dư≥15% trước full pilot.
Đề nghị trần **≤2×fresh dense_CE active time** cho teacher arm; vượt thì COST_FAIL, giữ output,
không tự giảmsteps/M/client để giữ cổng. Cỡ M độc lập; M64 fail không che M512.

### 2.6 Byte thật: dense sync chưa biến mất

Đại số CPU ngày2026-10-12, build đúng CFG:395.024 learnable parameters +3.264 floating BN buffer
+31 integer `num_batches_tracked`. Giả định codec mới **68B header/hash**, chưa implement:

| packet | M64 | M512 |
|---|---:|---:|
| G/B FP32 + uint64 local count |4.428B|34.892B|
| centered target FP16 |2.116B|16.452B|
| anchor66 FP16 một lần |8.516B|67.652B|
| common probe: params+BN FP16, counters int64 |796.892B|796.892B|

Giá trị trên là application bytes trước ACK/retransmission/cohort/query/hash-response.
Header68B là đề nghị contract, khác legacy codec32B; ledger phải dùng packet serialized thực tế.
Prior/H2diag/coverage/bandwidth/target cold-start/catch-up và per-client fanout cộng riêng.
Dense probe+target down/round ít nhất799.008B/813.344B, chưa phí kênh.
Không gọi K2 này fit một B0/B2 contact; trả phí coherent multi-contact hoặc chưa mở contact.

**Reconstruction hoãn khỏi U đầu tiên.** Có thể học lại θ từ public target stream/init/optimizer/BN,
nhưng chỉ miễn snapshot sau strict weight+BN+optimizer+resume hash trên haiT4/account và loại thiết bị
đích đã PASS. Hash mismatch ⇒ bỏ contribution hoặc resync dense có phí; tolerance logits không thay hash.
[PyTorch2.10 reproducibility](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/docs/source/notes/randomness.rst)
không bảo đảm cùng seed cho bitwise giữa CPU/GPU/platform. Codec/resume PASS nội bộ chưa là proof ấy.

## 3. K4 — JEM có common base, pilot tối thiểu bằng independence-MH

### 3.1 Sửa normalizability và giữ classifier

Bản `exp(f_y(x))` trên LebesgueR66 đã FAIL trước train: DAGSNet head có bounded logits khi weights
hữu hạn, integral không hữu hạn. [Đại số/checkpoint proof](../../papers/v3/energy-probe.json),
[head/source](../../mh/dagsnet.py). Không rerun bản thiếu base.

[JEM](https://arxiv.org/html/1912.03263v3) ghép CE với generative objective từ logsumexp;
paper thừa nhận finite-chain bias/instability. [Official training code](https://raw.githubusercontent.com/wgrathwohl/JEM/master/train_wrn_ebm.py)
tính real/negative energy rồi CE; đó là nguồn cơ chế, không chứng minh tabular FL/contact.

Đề nghị base chung, độc lập target label:

\[
q(x)=\frac1{16}\sum_{c=1}^{16}\mathcal N(x;\mu_c,diag(\sigma_c^2)),\quad
p_\theta(x,y)=q(x)e^{f_\theta(x)_y}/Z_\theta.
\]

μ/σ từ H2diag đãdecode; variance floor1e−4 như bootstrap hiện tại. q là normalized analytic mixture,
σ>0, không mạng/GMM học được. Bounded head cho finite Bθ: `e^{-Bθ}≤Z_y≤e^{Bθ}`;
`16e^{-Bθ}≤Zθ=Σ_yZ_y≤16e^{Bθ}`. Đây proof tồn tại normalizer, **không** bảo đảm estimate/mixing tốt.

Common q triệt tiêu khi điều kiện theo x: `pθ(y|x)=softmax(fθ(x))` đúng classifier hiện tại.
Nếu thay q bằng q_y riêng label thì posterior thành `softmax(f_y+log q_y)` đối với common joint Z;
nếu chuẩn hóa từng class còn cần `−log Z_y+log π_y`. Không tự dùng raw logits rồi gọi cùng posterior.
Đây suy luận đại số của dự án từ formulation, không một kết quả IDS đã chạy.

### 3.2 Support và positive local

Gaussian q liên tục; dữ liệu có binary/zero atoms. Đề nghị định nghĩa **positive density được làm trơn**:
`x̃=x+0,05*σ_y*ε` trên **mọi66tọa độ**, ε iidGaussian, σ_y đãdecode/floor; tính tại local,
không export x̃. CE vẫn trên x raw. Full-rank Gaussian convolution cho positive có density liên tục
và q>0 trênR66; không cần khẳng định density của raw atoms so vớiLebesgue.
Nhãn giữ nguyên; smoothing có thể không label-preserving/physically-valid. Cần owner chốt;
không dùng approval τ trước của V2-C để suy đã chấp thuận mục tiêu K4 này.

Do generator continuous, generated binary/profile/zero support sẽ không exact. Báo invalid-support
riêng theo schemaV4; **không hứa fidelity0**, không đem cổng0invalid củaK1 sang rồi gọi K4 khả thi.
Nếu owner bắt synthetic giữ exact support thì chưa launch minimum này; dùng alternative§3.6.

### 3.3 Objective và negative sampler

Đặt `Aθ(x)=logsumexp_y fθ(x)_y`. Theo factorization CE+marginal-density, loss đề nghị:

\[
\mathcal L_i=CE(f_\theta(x),y)
+0.1[-\mathbb E_{\tilde x\sim P_i^{\tau}}A_\theta(\tilde x)
+\mathbb E_{s\sim p_\theta(x)}A_\theta(s)].
\]

Base q không phụ thuộc θ; các term logq(positive) không có gradientθ nên không cần density evaluation.
Negative finite chain cho **biased gradient estimator**; không gọi exactML. λE=0,1 là đề nghị cố định,
không tuning theo F1. Không clip logits thành floor như V2-C.
CE trên raw và energy trên positive được làm trơn là hai mục tiêu trên hai distribution;
không gọi tổng này exact joint likelihood của raw data, dù family joint có normalizer hữu hạn.

Independence-MH marginal: proposal `s'∼q` từ noise; logaccept
`Aθ(s')−Aθ(s)` vì q cancels. Uniform logu<min(0,logaccept) thì nhận.
Class-conditional synthetic sau teacherPASS: logaccept `fθ(s')_y−fθ(s)_y`;
**proposal vẫn common q**, không q_y rồi bỏ proposal-density ratio.

Đề nghị256negative chains/client; đầu mỗi round init noise-only từq,20burn transitions,
sau đó1transition/batch, giữchain giữa batch trong round. Không raw replay/persistent real-init.
**Sau optimizer update, tính lại current old-state energy**, không tái dùng cachedA của θ cũ trongMH.
BN/dropout fixed trong energy/sampling: canonical4096 q draw/round, eval mode trong toàn teacher fit,
parameters/BN affine vẫn học; tất cả đối chứng dùng cùng BN protocol.
Grad clip1, AdamW cosine/reset theo pilot; density/sampler logits FP32; finite chains/state hard guard.
Mọi chain/sample chỉ tại xe; RSU chỉ nhận aggregate dense parameter update.

Independence-MH có correct invariant target khi q/proposal/full current state đúng;
20burn/1transition **không bảo đảm mixing**. Base support rộng và bounded tilt chỉ cho existence;
tilt quá mạnh vẫn acceptance gần0. Đo probability movement, acceptance, jump distance và dwell.

### 3.4 Arm, teacher gate, cost và W&B

Teacher-only pilot42/43,20client/T2,5%/20round:

| arm | vai trò |
|---|---|
| `ce_raw` | fresh dense-FL CE, cùng fixedBN/dropout/compute schedule |
| `ce_aug` | CEraw+0,1 CE(x̃,y), cô lập smoothing; cùngτ/negative forward dummy |
| `ce_energy_fl` | objectiveCE+energy + dense aggregation |
| `ce_energy_own` | cùng objective/base/init/sampler nhưng không aggregate modelupdate |
| `ce_energy_shuffle` | energy-input fixedcoordinate permutation sauperturb/proposal; CEraw nguyêninput; phá địa chỉ density |

Shuffle thực sự dùng target `q(x)exp Aθ(Px)` và acceptance theo `Aθ(Ps')−Aθ(Ps)`;
không đổi coordinate tại bướcenergytrain nhưng dùng unpermuted sampler. Cùng permutation/schema/sourcehash.
Permutation có thể phá physical semantics; đó là mechanism control, không deployment candidate.
Ở control này energy dùng `f(Px)` còn CE dùng `f(x)`: hai nhánh không là cùng joint posterior.
Không lấy posterior của shuffle để thay classifier raw hoặc báo như một JEM deployment hợp lệ.
Own vẫn share analytic base/prior nên đo lợi **aggregate energy/model**, không noFL.
CEraw/CEaug dummy negative forwards discard output; time/report nêu cost gồm sampler dummy riêng;
**trần compute so vớiCEraw thật không dummy** để không làm denominator vô nghĩa.

Teacher cuối r20 group/head: energyFL≥max(CEraw,CEaug)−0,01;
energyFL−own/shuffle≥0,01; common benign≥CEraw−0,03/old no worse CEraw−0,01.
Teacher quality retention chỉ là điều kiện cần; private stage riêng sauPASS phải so
`Gaussian diag`, `CEproxyKD`, `energyown`, `energyshuffle` bằng cùng aux/samplecost;
energyFLprivate−max(diag,CEproxyKD)≥0,015,−own/shuffle≥0,01 +commonGP/local/old/benign gates.

FiniteFP32/FP16wire/aux, source/codec/resume/cohort valid bắt buộc. Đề nghị **cost≤2×CEraw thật**;
preflight budget từT4 trướcpush. Vượtcost/numerical/teacherFAIL thì dừng, không đổiλ/τ/burn để cứu.
Không mở contact từteacher-onlyPASS; dense modelcarriage vẫn chưa phù hợp B0/B2.

W&B online/key fail-before-train. Round1–5 log riêng CEraw/energypositive/energynegative,
λE, lr, applied/skipped, parameter/gradnorm, chainacceptance/movement/maxdwell,
time/byte/version/BNhash. Train finite nhưnglossâm không tựlà lỗi: energycontrastloss không CE≥0.
Nonfinite/samplerstuck hoặc applied=0 phải báo owner dừng run đang hoạt động;
không tự stop/retry/push bảnthayHP.

Đề nghị samplerdiagnostic scalar: mỗi r1/5/10/20,64freshchain×100transition,
hai noise-init sets; eval thêm marginalenergy và classconditional logits. Log trajectory chỉsynthetic/scalar,
lag1/10/20 autocorrelation, split-chain estimates vàESS; không coi acceptance cao làmixingPASS.
Hai set meanA difference≤0,1 pooledSD; movementfraction≥5%; không chain nào frozen toàn100step.
Đây finite diagnostic guard, không theorem mixing; FAIL ⇒ giữ diagnostic/khôngprivate/contact.
Costdiagnostic cộngvào budget; không dùngdiagnostic để chọn10/20/100step theoholdout.

### 3.5 Byte và checkpoint

H2diag/prior bootstrap tính riêng. Aggregate modelupdate FP16 params có giá790.048B payload;
snapshot param+BN+counters/header tương ứng796.892B theo toy§2.6; sendupdate/schema thực tế
cần codec mới và hash/buffer policy khóa trướccode. Cả hai chiều,20clients/round +ACK/retransmission.
Không gửi book thaymodel rồi quênuploaddense. Không tính localnegative/syntheticdraw nhưcommunication.

Resume phải giữ teacher/privateweights, BN, optimizer/resetstate, chains, RNG, publicbasepacket,
sampler-counter/versions + roundmarker; runtimewalltime chỉdiagnostic, không bitwisealgorithmstate.
Chỉ reuse old helper/codec trongsource mới riêng; không sửaV2–V6 snapshot đã chạy.

### 3.6 Alternative chưa chọn: MALA và mixed support

Common continuous q cũng cho score `∇logq+∇Aθ`; MALA cần forward/reverse proposal correction.
[Roberts–Tweedie primarypaper](https://projecteuclid.org/journals/bernoulli/volume-2/issue-4/Exponential-convergence-of-Langevin-distributions-and-their-discrete-approximations/bj/1178291835.pdf)
phân tích diffusion/discrete/Metropolis và điều kiện convergence; không cho DAGSNet20steps mixing.
V4 ratioMALA đã FAIL ở cấu hình khác; không đổi tên ratioMALA thành K4 đã kiểm.
[SourceV4](../../mh/v4_mala.py), [cổngV4](../../papers/v4/gates.json).

Nếu cần exact first/profile/zero atoms: define common mixed probabilitymeasure
`Q(ds,dv)=w_s q_s(v)dv`, energy tilt`Q exp f`. Local smoothing chỉcontinuous/freecoordinate,
MALA giữmask và independence-MH proposal**toànQ** để đổi mask; tỷ lệQ cancels cho independence step.
Chỉ samplemask theoQ rồi freeze mãi là sai marginalmask củaenergytilt: weights phải tỷlệ
`w_s E_{q_s}exp Aθ`, chưaQweight. Áp dụng được without enumeratingallmask nhưng đòi schema/density,
cohort smoothing/maskrate/client fallback mới; **hoãn khỏi minimum**.

## 4. Đại số/toy đã kiểm, không phải pilot

CPU Python/torch, khôngtrain/GPU/record/test, ngày2026-10-12:

- Build DAGSNet đúngCFG kiểm parameter/buffer counts và packet arithmetic§2.6.
- Hai mask cóQweights(0,75;0,25), energies(0;log9): targettilt=(0,25;0,75).
  Freeze maskbase giữ(0,75;0,25) ⇒TVerror0,5; independenceMH2state cóstationaryerror0.
- logits(0,3;−0,4): rawposterior=(0,66819;0,33181);
  thêmclassbases(0,2;0,8) ⇒posterior=(0,33486;0,66514), đảoargmax.
- Centeredlogitupdate invariant với cộng123 vào mọilogit; CEresidual sumzeroPASS.

Các toy là counterexample/proofmechanics, không chứng minh Gaussianmixture là datafaithful,
cost2×/reconstruction/teacherF1/contactPASS. Không phát sinh trainingartifact hoặc filedata.

## 5. Những câu cần chủ dự án chốt trước triển khai

1. **K2:** cảM64/512 teacher-first, dense commonprobe+BN có tínhbyte; classbalancedresidual;
   η1/trust0,25/targetclip±8;20publicAdamWsteps/canonicalBN4096; cácarm/cổng/cost2×§2?
2. **K4:** cho commonH2diagGaussianmixture + smoothingmọi66tọa độτ0,05;
   CEraw+energyλ0,1; independenceMH20burn/1transition,256chain, fixedBN; exactsupport chỉdiagnostic?
3. **K4:** giữteacher-first controls/cổng/cost2×§3.4; chỉ mở privateKD/synthetic/contact nếu qua
   từngcổng, không runK4unbased hoặc tuneresume cứu scientificFAIL?

Không thao tác tay bắt buộc trong bước research. Root hỏi/chốt cácscientificchoices trên;
sau đó code/source mới, bounded localchecks, Kagglepreflight/onlineW&B, sourceimmutable,
pull/audit/prediction10metric/perclass/bytes/cost như cácpilot trước. Không push kernel trùng đang hoạt động.
