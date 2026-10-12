# V3 — carrier proxy DAGSNet sau lỗi sketch-EF

Ngày nghiên cứu: 2026-10-11. **Thiết kế mới, chưa code/chạy/chốt cổng với chủ dự án.**
Nguồn trạng thái: [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md),
[manifest V2](pilot-v2.md), [chẩn đoán sketch](../../papers/v2/sketch-diagnostics.json).
Không thay source/cổng kernel V2 đang chạy. Kết quả V2 terminal phải pull/verify trước kết luận.

## 1. Bằng chứng → vấn đề cần sửa

V2-A EF ở sketch space đã khuếch đại collision: residual/target small khoảng16,75–17,92;
medium khoảng2,93–3,97, rồi proxy/BN không hữu hạn. Đây là lỗi adaptation;
không bác bỏ mọi thuật toán error-feedback. Exact codec/resume đúng cũng không chứng minh compressor co.
P1 cho thấy proxy DAGSNet đủ sức chứa có ích; P2 cho thấy B0 truyền dần không kích hoạt proxy trong20 round.
V2-B cơ sở cố định≤K DOF còn đang kiểm; carrier mới nên giữ khả năng tích lũy thay đổi trên toàn395.024 parameter.

Đề xuất bốn cơ chế khác nhau dưới đây. Hai hướng đầu dùng coordinate thật, hai hướng sau dùng
ma trận của **update vector**, không model phụ. Mọi learned model vẫn DAGSNet; projection/mask/QR là toán codec,
không learned encoder. Private IDS lấy tri thức qua KD như A; chỉ proxy chung đòi kiến trúc đồng nhất.
R2 khác kiến trúc private vẫn là thiết kế, chưa bằng chứng thực nghiệm.

Nguồn nền: *Sparsified SGD with Memory* phân tích sparsifier với điều kiện contraction;
*Error Feedback Fixes SignSGD* nghiên cứu sửa sai lệch compressor bằng residual.
Không chuyển định lý SGD của nguồn thành bảo đảm cho local AdamW/non-IID/DAGSNet pilot này.
[Nguồn Stich 2018](https://proceedings.neurips.cc/paper/2018/file/b440509a0106086a67bc2ea9df0a1dab-Paper.pdf),
[Karimireddy 2019](https://proceedings.mlr.press/v97/karimireddy19a.html).

## 2. Hợp đồng chung và byte thật

- D=395.024 learned parameter, gồm BN affine; BN buffers không upload, xe calibrate local.
- Xe i giữ proxy snapshot coherent θ_t và private DAGSNet. Chỉ local train data vào loss;
  chỉ message model-update được cohort ≥5 trusted-sum mở tổng. Không latent/record/per-record gradient.
- Seed/model-layout/version/mask là public; cohort cùng base version và codec schema. Không cộng update khác base.
- α_i=n_i/Σn_j là trọng số public từ counts đã trả phí; message chứa α_i nhân update trước aggregation.
  Upload counts nếu thay cohort phải tính riêng. Trusted simulator không phải cryptographic secure aggregation/DP.
- Bộ nén U dùng **fp32 upload và download** để residual trừ đúng update đã decode, tránh thêm biến lượng tử.
  fp16/int8 chỉ mở adaptation riêng có cổng riêng, không gọi định lý contraction lý tưởng áp trực tiếp.
- Byte ví dụ dưới đây: header32 B/message; ACK16 B/message thành công; tổng contact là **upload+download**.
  Đây là đề nghị schema để triển khai; phải khóa theo channel codec thật, không lấy bảng thay ledger.
- B0 dùng cap bảo thủ4.500 B/contact; B2 dùng231.000 B/contact. Header/ACK/fragment/retry đều trả phí;
  cắt frame không commit. Không tính broadcast chung là download miễn phí cho từng xe.
- Prior hiện hành upload104 B/down96 B/xe một lần, tổng200 B. Public init seed/layout recipe cũng phải
  nằm trong bootstrap/header trả phí; không miễn phí vì có seed. DAGSNet θ₀ tái dựng từ seed chuẩn;
  phiên bản runtime/hash là hợp đồng đã cấp. Nếu phải gửi full θ₀ thì thêm1.580.096 B fp32 và nhiều contact.
- Byte cryptographic mask/key exchange chưa gồm; chỉ pilot trusted aggregate được tuyên bố.

## 3. V3-R: random-k chung, EF tại parameter space

**Khác A:** bỏ CountSketch/collision/heavy-hitter decode; chọn coordinate trực tiếp cùng mọi client.
Mỗi round public seed chọn uniform subset S_t, |S_t|=k; seed mới độc lập với update/data.
Xe train proxy đầy đủ một epoch từ θ_t, có Δ_i. Giữ residual E_i tại xe:

```
a_i = α_i Δ_i + E_i
u_i = a_i[S_t]                        # fp32 k entries, không indices
u = cohort_sum(u_i)
θ_(t+1) = θ_t + scatter(S_t, u)
E_i,new = a_i - scatter(S_t, decode(u_i))
```

Residual **local contribution**, không trừ global u khỏi từng E_i. Chỉ commit E_i,new khi cohort update
thực sự commit; cohort fail không xóa residual. α_i đổi theo cohort cần khóa bookkeeping: E_i ở đơn vị
weighted update đã được tạo, không nhân lại E_i với α_i mới. Timeout/xe rời cohort cần ledger ownership rõ.

Ở số học lý tưởng, C_S là orthogonal coordinate projection:
E||a−C_S(a)||²=(1−k/D)||a||². Không nhânD/k; phép nhân đó đổi operator/variance.
Đây là tính chất có thể kiểm math, **không hứa residual bounded khi Δ_i vẫn thay đổi**.
Nguồn có random-k/memory; việc dùng cohort mask chung, delta local epoch và KD là adaptation của dự án.
[Nguồn Stich 2018](https://arxiv.org/abs/1809.07599).

| cỡ | k | upload | download | header+ACK hai message | tổng/contact |
|---|---:|---:|---:|---:|---:|
| B0 | 512 | 2.048B | 2.048B | 96 B | 4.192B |
| B2 | 24.576 | 98.304B | 98.304B | 96 B | 196.704B |

B0 contact đầu thêm prior200 B →4.392B, chỉ còn108 B trước fragment/bootstrap khác; **chưa bảo đảm fit**.
Seed/layout cần tái sử dụng trường header hoặc contact bootstrap riêng được trả phí. B2 còn34.296 B.
Cohort cộng k-vector tuyến tính; coordinate không mang nhãn cá nhân. Secure-sum float emulator hợp lệ;
crypto integer encoding cần cổng fixed-point/overflow riêng.

Rủi ro: B0 k/D≈0,001296,20 round kỳ vọng chỉ≈2,56%coordinate từng được chọn;
residual lớn dù compressor co. Đây là stress test có quyền FAIL, không cứu bằng kéo dài sau xem điểm.
Đối chứng: random-k no-EF, random-k EF, dense proxy, own proxy, local.
Chẩn đoán: residual norm/target, từng tensor coverage, decoded update norm, cohort mismatch/loss/KD/F1.

## 4. V3-T: rolling-block train đúng coordinate được trả phí

**Khác R:** không train full proxy rồi nén bỏ phần lớn; train chỉ coordinate sẽ truyền.
S_t từ public permutation tạo trước run, chia block k, quay vòng deterministic.
DAGSNet forward vẫn đầy đủ; parameter chưa chọn gradient mask0 và optimizer không được đổi chúng
(momentum/state reset mỗi round; wd0). Không rút submodel nhỏ, không đổi architecture/head.

```
Δ_i = local_epoch(θ_t, allowed_coordinates=S_t) - θ_t
assert Δ_i[outside S_t] == 0
θ_(t+1)[S_t] = θ_t[S_t] + Σ α_i Δ_i[S_t]
```

Không residual, không sketch. Cohort message k-vector cùng schema. Đây là block-coordinate FL adaptation;
không phải FedRolex nguyên bản. FedRolex dùng rolling sub-model extraction để phủ model; chỉ mượn
ý tưởng lịch phủ đều, không mượn kết quả heterogeneity làm bằng chứng R2 của ta.
[Nguồn tác giả FedRolex](https://research.google/pubs/fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction/).

Byte: cùng bảng R, k512/B0 và24.576/B2 fp32; metadata block-id nằm header. Full cycle lần lượt772round
và17round, vòng cuối ngắn/padding phải ledger đúng. B0 chỉ20block trong20 round, khả năng học backbone rất yếu.
Để tránh chọn mask theo F1, permutation seed cố định trước run; block assignment phân tầng theo tensor
để mỗi round cập nhật được stem/backbone/head, tỷ lệ theo tensor size và min1/tensor nếu k đủ.
Điều này đổi cycle chính xác; code phải báo coverage thực thay dùng772 như số đã đo.

Đối chứng quan trọng: **rolling train-only**, **rolling communication-only no-EF** cùng mask/budget,
plus own train-only cùng mask. So sánh tách “local optimization khớp carrier” khỏi “lịch phủ đều”.
Không claim hơn R do nhiều compute: batch/update count/epoch giống nhau, log wall-time/steps.
Cổng thêm: train-only−communication-only≥0,005 cả seed để nhận đóng góp riêng tối ưu hạn chế;
carrier có thể được nhận dù contrast này thiếu, nếu cổng private chung qua.

## 5. V3-O: một vòng projection trực giao ma trận, EF local

**Khác B:** không cố định θ=θ₀+Za trongK DOF. Proxy dense được train tự do, mỗi round nén delta qua
subspace mới; θ tích lũy update ở nhiều subspace. **Khác A:** projection trực giao không decode collision.

Pad update vector thành M_i∈R^(512×772), thêm240zero (512×772=395.264).
Public Q_t∈R^(772×r), QᵀQ=I, QR từ Gaussian seed mỗi round; sign QR canonical, recipe/runtime hash.
Parameter residual pad rows giữ240zero trước projection; reconstructed padded coordinates discarded.
C=unpad(project(pad(a))) là compression trênD coordinate; padding không tạo learned parameter.

```
A_i = reshape(pad(α_i Δ_i + E_i))
P_i = A_i Q_t
P = cohort_sum(P_i)
θ_(t+1) = θ_t + unpad(vec(P Q_t.T))
E_i,new = α_i Δ_i + E_i - unpad(vec(decode(P_i) Q_t.T))
```

RSU chỉ mở tổng P; mỗi xe tái dựng Q từ seed, không gửi full Q. Sum-before-reconstruct tuyến tính.
Q fresh independent mỗi round: trong số học lý tưởng, expected residual squared≤(1−r/772)||a||²;
đây là suy luận projection của adaptation, **không phải định lý FedAdam/IDS từ PowerSGD**.
Khi finite float QR, kiểm||QᵀQ−I||₂≤1e−5 trước train; không đạt thì fail numeric trước GPU epoch.
Không dùng warm-start Q theo update vì làm mất lập luận independence này.

| cỡ | r | P entries | upload fp32 | download fp32 | header+ACK | tổng |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 1 | 512 | 2.048B | 2.048B | 96 B | 4.192B |
| B2 | 32 | 16.384 | 65.536B | 65.536B | 96 B | 131.168B |

B0 prior/bootstrap caveat giống R. Không theo layer nên phá structural low-rank giả thiết;
đổi public packing permutation trước run có thể giúp/không giúp, không tune sau điểm.
R1/B0 yếu tương tựR: contraction≈1/772, chỉ20 round thường thiếu thông tin. B2 rank32 có thể hợp lý hơn.
Đối chứng cùngpacking/rank: no-EF, EF, own dense proxy, dense proxy, local; thêm fixedQ no-EF
để phân biệt tích lũy nhiều subspace với bottleneck fixed subspace. Cổng riêng EF−noEF≥0,005 để claim EF.

PowerSGD chứng minh tính thực dụng của update low-rank trong distributed optimization;
thuật toán ở đây chỉ **một phép projection ngẫu nhiên**, không gọi PowerSGD và không mượn speedup CIFAR/LSTM.
[Nguồn PowerSGD](https://arxiv.org/html/1905.13727v2#S3).

## 6. V3-P: PowerSGD hai vòng cohort, dành B2

Mục tiêu: đổi rank ngẫu nhiên thành subspace bám update quần thể, trả đúng giá round-trip.
Cùngpack512×772, rank8 trước. Vòng1 cộng P_i=A_i Q₀, RSU QR tổng thành P̂ (512×r).
Gửi P̂ xuống cohort; vòng2 cộng Q_i=A_iᵀ P̂ (772×r), rồi gửi Q̄ xuống.
Model update P̂ Q̄ᵀ. P̂ và Q̄ là payload thật; không tính một factor như public seed.
PowerSGD nguồn dùng chính hai lượt aggregation này; adaptation ở đây là local-epoch delta,
trusted cohort và private KD. [Thuật toán nguồn §3](https://arxiv.org/html/1905.13727v2#S3).

Residual mỗi xe: E_i,new=α_iΔ_i+E_i−unpad(vec(P̂ decode(Q_i)ᵀ)),
với P̂ đúng bytes đã nhận vòng1. Tổng residual tương ứng projection của tổng update;
không lấy P_i local nhân Q_i local rồi cộng, vì cross-term khác update global.
Staging E_i/new và θ chỉ commit sau vòng2/ACK hợp lệ; fail/timeout giữ E_i cũ và base θ cũ.
Vòng1 đủcohort nhưng vòng2 mất xe: cohort phải cùng tập; abort, trả phí đã gửi; không âm thầm đổi mẫu số.

| message/xe | bytes rank8 fp32 |
|---|---:|
| upload P_i 512×8 | 16.384 |
| download P̂ 512×8 | 16.384 |
| upload Q_i 772×8 | 24.704 |
| download Q̄ 772×8 | 24.704 |
| 4header32 +4ACK16 | 192 |
| **tổng** | **82.368B** |

Rank16 tổng164.544B; rank1 đã10.464B, **không fit B0 một contact**.
B2 đủ byte chưa chắc đủ deadline hai lượt; contact scheduler phải mô phỏng thời gian/RTT thật.
Không giấu second round-trip trong một aggregate giả. Có thể dànhhai contact nhưng retention/coverage phải đo.
Với P̂ trực giao chính xác, phép A_i→P̂P̂ᵀA_i không tăng norm residual;
không có δ>0 uniform cho mọi update vì subspace có thể bỏ sót tín hiệu. Với codec/số học hữu hạn,
phải kiểm norm và orthogonality thực. Cần math adversarial/random tests,
norm diagnostics và fail guard. Không hứa “PowerSGD sửa mọi lỗi A”.
Đối chứng: randomO cùngrank; P no-EF/EF; dense/own/local. Nếu P vượt O≥0,005 cảseed mới nhận
đóng góp adaptive basis; nếu không, chọn carrier rẻ hơn theo byte/latency, không thêm warm-start để cứu cổng.

## 7. Cổng hai seed và kế hoạch kiểm nghiệm có thể bác bỏ

**Đề nghị khóa trước triển khai**, không dùng threshold chọn sau kết quả. Fixture/holdout/lịch20 round,
20 client, seed42/43, DAGSNet66 feature, KD T2λ1 từr2, BN-local, optimizer/numeric/10metric
giữ như A của [V2](pilot-v2.md). Chỉ thay carrier; own/private/dense controls recipe hoàn toàn khớp.
Không dùng official test, không proxy teacher-only score để nhận private IDS.

### M — nhỏ local, bắt buộc trước Kaggle

1. Sum/encode/decode tuyến tính ở message space; cohort<5/base/schema/mask mismatch bị từ chối.
2. R/O: residual identity và projection norm với random/adversarial tensors; decoded-value rounding
   được tính; không kiểm “residual phải nhỏ hơn update mới” như theorem vì E_i có lịch sử.
3. T: ngoàimask thay đổi đúng0, optimizer/BN-affine đúngpolicy; coverageperlayer không đóihead/backbone.
4. P: hai giai đoạn giống projection tổng, biết cross-term lỗi thì test từ chối; abort giữa2stage giữstatecũ.
5. Byte/header/ACK/prior/framecut đúng ledger; poisoned/tampered payload fail trước commit.
6. Driver20×256rows/3round, exactresume r1→r3 trùngstate/weights trên cùngbackend; không lấy F1 smoke.

### U — teacher quality, private utility, không contact-limit

Mọi proposed carrier vẫn dùngcodecbytes thật; U bỏcontactlimit để tách chất lượng khỏi scheduling.
Mở pilot các hướng khả thi, không phải full-scale. Mỗi cỡ báo riêng; không lấysmallbyte/mediumF1.

- Dense proxy group head−local private gp≥0,05; dense private đạtcổng chung, nếu thiếu thì pipeline teacher
  không còn upper-reference đủ tốt; dừng nghiên cứu carrier cho recipe này.
- Carrierprivate group gp−own private gp≥0,01, giữ≥80%positivegain dense private−ownprivate.
- Cổng chung: carrier private group gp−localgp≥0,03; old gp>localgp; benign F1≥local−0,03.
- Mọi arm hoàn thànhr20 với finite weights/BN; skipped optimizer step≤16/clientphase theoV2.
  Cơ chế scientificfail dừngarm, giữcheckpoints/log và chốtFAIL; khôngrerun y nguyên/kéo dài cứuđiểm.
- Claim EF/adaptivebasis/trainmask chỉ khi contrast riêng≥0,005 mỗi seed; thiếucontrast không xóa
  utilitycarrier đã qua nhưng hạclaim về thànhphần mới.
- Hai seed quađồngthời. Mean10metric/perclass/CMprediction/statehash/bytes/steps/time đầyđủ.

Teacher gate có thể dùng nhưstage0 nếu quota thiếu, nhưng vẫn phải mởKD/private stage đã khóa
trước nhận hướng; không đánhđồng proxy head cao với private gp hayR1 đạt.

### C — finite contact chỉ mở hướng U qua

Source/manifest attempt mới, cảB0/B2 khảthi theo bảng; P chỉB2. Prior/bootstrap paid,
1contact/xe/round, cohort ≥5, timeout5 round, códeadline/framecut/ACK; stale bản coherent,
16/20 xe phải kíchhoạt tri thức trong20 round. Retention≥70%U gain, group gp−own≥0,01,
benign guard nhưU. Byte không đủ hoặc0activation là **R1 fail**, không gọi mutual learning local là FLgain.

Catch-up: R/T/O phải tải patchlog mọiversion cònthiếu hoặc snapshot full model paid; P tảihaifactor
mỗi version. Không áp patcht lên θ_(t−3) rồi gọi coherent. Replay log growth/TTL/bootstrap phải ledger;
xe bị lỡ snapshot không dùngKD teacher mới cho đếncomplete catch-up. EF contribution chưacommit giữpending;
không đồngthời táisend kháccohort vàdoublecount. Đây là implementation khó cần lock kỹ trướcpilotC.

## 8. Ưu tiên và bài học dự kiến

1. **R B2**: carrier trực tiếp đơn giản, math contractivity rõ; đáng thử trước bất kỳ cứu sketch.
2. **O B2**: thay coordinate bottleneck bằng projection dense mộtvòng; rank/packing vẫn rủiro thực.
3. **T B2**: trả lời có nên train theo phầncóthểtruyền thaytrainrồibỏ; coverage17round là độngcơ test.
4. **P B2**: thêmhai round-trip đểkiểm lợiadaptivebasis cóđáng; không phảiđầu tiên.
5. **R/O/T B0**: stress riêng, cùngcodec/controls; expectedcoverage/contraction cho thấy rủiro cao.
   KếtquảFAIL vẫn tạo bàihọc sứcchứa/time-to-activation, không lýdo tự tănground.

Có thể dùngdense/local controls từA **chỉ khi exactsource/config/BN/KD/init/trainingbudget khớp và artifacts
đã verify**; khácsource/backend không bảođảmpairedcomparison. Phải giữcontrol cùngattempt nếu thayrecipe.
W&Bonline mandatory trướctrain; private keysource workflow; sourcehashimmutability; Kaggleonlytrainheavy.
KhôngGit/DVC. Không việc tay dự kiến ởthiết kế này; nếuquota/secret/runtime không khảdụng thìbáo đúngchỗthiếu.

## 9. Phạm vi nguồn và điều không nhận làm kết luận

Tấtcả papernguồn làdistributed/FL nền, chưapaper nào ở đây chứng minhIDS V2X trênfixture ta.
FedPAQ cho precedent periodicaveraging+quantization trongFL; **không đưa thêmarm quantize dense vội**:
395.024 parameter ngay4bit đã197.512B/mỗi chiều, tổng395.024B trướcheader vượtB2,
secureaggregation khácscale từngxe cũng không phảiphép cộnginteger trực tiếp.
[Nguồn FedPAQ](https://proceedings.mlr.press/v108/reisizadeh20a.html).
Randomrotation trướcquantization cónghiêncứu mean-estimation nhưngkhông xóa size toànvector;
đểthiếtkếsau khi carrier qua, không dùngclaimMSE làmclaimF1.
[Nguồn Distributed Mean Estimation](https://proceedings.mlr.press/v70/suresh17a.html).

Đây là bốn adaptation cóthểkiểm, khôngkếtquảđãđạt. Họ PowerSGD/EF/FedRolex chỉnguồn cơ chế;
contactdeadline, cohorthội tụ, sứcchứa20 round vàKD private đều phải kiểmnghiệm trongdựán.
