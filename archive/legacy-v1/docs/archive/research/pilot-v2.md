# Pilot V2 — cấu hình đã chốt, trước khi mở Kaggle

Khóa ngày2026-10-11, sau trả lời chủ dự án. Nền: [bộ V2](de-xuat-v2/README.md),
[hợp đồng](de-xuat-v2/thiet-ke-chung.md). Source/config hash cuối được ghi trong [ledger](../methods/shared-comparisons/builds/countsketch-subspace-ratio/results/rebuild.md) trước push.
Không sửa notebook đã PUSHED; đổi thuật toán phải là attempt/source mới và nói rõ nguyên nhân.

## Quyết định chủ dự án

- Triển khai kiểm khả thi cả ba hướng; mọi model học được DAGSNet, R2 vẫn ở thiết kế.
- A: cả small3×256,k128 và medium5×2048,k2048, có đối chứng no-EF cùng cỡ.
- B: first-order K512/K1024; zeroth-order chỉ có điều kiện sau teacher gate, không tự chạy mặc định.
- C: τ0,05 chỉ local, weight cap10, normalize theo lớp, BN calibration bằng q chung; khóa cổng dưới đây.
- Local chỉ math/driver smoke/fixture/report; nặng Kaggle. W&B online trước train, thiếu key/network thì fail.
- Không Git/DVC commit/push/backup. Quota pilot ưu tiên account ít còn đủ budget, không đổi active credential toàn cục.

## U — fixture và recipe

Fixture pilot giữ nguyên:20client α0,5,2,5% train; group/old65.536dòng/tập; không mở official test.
20round×1epoch, private batch512, AdamW wd0/reset, cosine1e-3→1e-5, clip1, AMPfp16/lossfp32.
Group/old r5/10/15/20; private headline gp, teacher B headline head. Không thay headline sau khi xem điểm.

| gia đình | arm | vai trò |
|---|---|---|
| A | local; a_own; a_dense; a_small_noef; a_small_ef; a_medium_noef; a_medium_ef | private IDS + proxy CE/KD1chiều T2,λ1 |
| B | b_dense; b_own512; b_seed512; b_own1024; b_seed1024 | **teacher-only trước**, không dùng private chưa học làm kết quả IDS |
| C | diag; c_own; c_fl; c_shuffle | private IDS, q cùng aggregate,256auxiliary/class-balanced |

A BN-local: mỗi round calibrate từ tối đa4.096train local, deterministic sample; proxy CE train-mode.
Teacher frozen eval trong private epoch; bật KD từr2, sau aggregate đầu tiên. Own control cũng bật r2 để khớp lịch.
B BN-local calibrate y hệt, frozen BN/dropout trong proxy epoch; coefficients AdamW cosine0,1→0,001, clip1/reset.
Basis fixed Rademacher seed1901, s_l=max(RMSθ₀,l,10^-3)/sqrt(K). Không pretrained/warm-start.
C positive x+0,05σ_yε ở local; one negative q_y/positive; loss weight1/(16π_y) từ prior train aggregate không smoothing.
q là moment fp32 cohort-total rồi μ/σfp16 thực, var floor10^-4; q version cố định suốt U.
Ratio BN:4.096q points cân bằng lớp, public seed/version recipe, freeze BN/dropout trong epoch.
Weights exp(clip r,−10,10) cap10; normalize theo class batch; shuffle chỉ trong class, không đổi synthetic draws.

Prior `gp` riêng có+1 smoothing theo protocol; counts/prior thực qua codec, charge mọi arm.
Every model snapshot là weights decoder đã nhận; BN-local chỉ ở xe, q-BN được tái dựng bằng recipe chung.
Server mở **tổng cohort20client** trong U; chưa cryptographic secure aggregation/DP. Không record/latent/per-record gradient gửi.

## Cổng U — cả hai seed42/43, headline r20

Chung cho private arm: groupgp−localgp≥0,03; oldgp>localgp; benignF1≥local−0,03.
Teacher B chỉ đánh giá cổng teacher; chưa cần chạy KD/contact cho cấu hình B trượt.

### A

- a_dense proxy group head−local private group gp≥0,05 và a_dense private qua cổng chung.
- sketch-EF−a_own private groupgp≥0,01; giữ≥80% gain của a_dense−a_own.
- EF−noEF≥0,005 để nhận đóng góp riêng EF. Nếu thiếu contrast EF nhưng carrier qua các cổng khác,
  chỉ kết luận carrier có ích; không gán lợi cho residual.
- Small và medium xét riêng; không gộp byte small với F1 medium.

### B

- seed proxy grouphead−local private groupgp≥0,05.
- seed−own giữ≥80% gain b_dense−own, với own đúng K; denominator phải dương.
- Qua teacher gate mới mở private KD với counterpart đúng K, cổng riêng≥0,01 và giữ≥80% gain dense KD.
  Source/manifest stageKD khóa riêng trước chạy; không tự coi teacher PASS là private/R1 PASS.

### C

- c_fl−diag private groupgp≥0,015.
- c_fl−c_own≥0,01; c_fl−c_shuffle≥0,01; qua cổng chung.
- Diagnostic4.096q draw độc lập/lớp: median ESS fraction≥0,25, từng lớp≥0,10.
  c_fl là một function canonical; c_own báo ESS từng model và mean; guard chính c_fl.
- Synthesis/logistic accuracy không thay gate F1. Không chỉnh τ/cap/kéo dài để cứu cổng.

## M và bằng chứng

Local math: codec real fp16/fp32, tamper/base rejection, cohort<5/duplicate; interrupted frame charged/no commit;
sketch additivity+decoded residual; seed chain rule/finite difference trên DAGSNet thật; weight normalization/shuffle.
Driver smoke3round:20×256train,256hold; B smokeK8 chỉ kiểm codepath, **không thay pilot K512/K1024**.
Production-basis probe riêng K512/K1024 trên batch32local, ít bước để kiểm shape/finite/compute, không lấy F1.
Resume r1→r2–3 so weights/state/prediction với uninterrupted, marker-last; tamper phải bị verifier từ chối.

Checkpoint mỗi round: private/proxy state_dict đầy đủ BN, algorithm state/residual/coeff đúng round, hash chain.
Aggregate payload giữ riêng; individual contributions không xuất ra artifact RSU. Prediction+confusion từng client,
10metric/CSV, CE real/aux/NCE tách; applied/skipped, time, bytes theo loại. Sampler/RNG từng phase xác định,
optim/scaler reset perclient/round nên không cần giữ state đã bỏ.
Numeric backend khóa deterministic algorithms, cuDNN deterministic/benchmark off, TF32off,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; local atomic scatter trước khóa đã làm residual lệch bit.
Backend này áp cả đối chứng, không sửa HP/đổi gate; không hứa bitwise giữa các account/runtime khác nhau.
Sketch median lấy giá trị giữa qua stable-sort, top-k tie theo index tăng; CUDA median có indices bị deterministic guard từ chối.
Verifier strict-load+savedpred→metrics; thêm sampled inference client0 ở r5 vàr20, decisive margin>0,02,
giữ cùng eval batch; ghi rõ phần ambiguous và giới hạn cross-runtime. Không tuyên bố tái-inference toàn bộ holdout.

## Kaggle và giai đoạn C sau U

Ba batch hai seed, một notebook/seed/batch, cùng source/fixture. Ước tính ban đầu A≤3h, B≤2h, C≤3h
mỗi seed; đây chưa đo T4 V2. Quota kế hoạch8h/account gồm dư, chia phiên thay chạy full-scale.
2×T4 assert sm75, pinned image;4jobslots [0,0,1,1], ngừng admit sau3h, budget mỗi driver4h.
Trước round mới driver dự phòng1,15×worst_round+60s; nếu dừng sớm, thu và resume từ artifact, không rerun r0.
Nguồn key theo workflow private đã được phép; secret-free notebook giữ để audit, bản nhúng key chỉ private_push ignored.

Chỉ hướng U qua cổng mới mở finite-contact B0/B2. Cần source/manifest C riêng: prior/bootstrap trả phí,
mincohort5, timeout5round, exact codec/commit,1contact/xe/round,16/20xe nhận tri thức, retention≥70% và gain≥0,01.
Không suy R1 từ bảng byte U hoặc size probe. Không full-scale/đa kiến trúc trong đợt pilot này.
