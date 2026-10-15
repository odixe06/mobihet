# CrossClientRank utility — protocol đã chốt sau diagnostic

**Owner đã chốt cả protocol và budget15/10; đang triển khai build riêng rank-utility.** [Probe và diagnostic](results/report.md) đã audit.
Locked gate PASS; cap-ratio diagnostic0,354/0,361 và bỏ class chi phối0,311/0,319 vẫn>0,20.
Đây chỉ rationale thử utility ngắn, không thay gate hoặc dự báo F1 thắng DML.

## Trial và đối chứng đã chốt

Pilot20clients gốc, 100%T pilot; giữ66features/partition/scaler/DAGSNet, seed42/43, fixedr20.
Batch512, AdamWwd0/reset mỗi round, cosine1e−3→1e−5 lịch20, clip1, AMP như các pilot đã audit.
Không official test. Group primary, old diagnostic; đủ10metrics/perclass/predictions/CM/weights/BN/separate resume.

8arms/seed: CE; own rank; pooled rank; class-shuffle means; frozen pooledr2; class-weight CE;
freshGaussian-full; freshDML. Sáu arm đầu fork cùng audited CE-r2 weights/RNG, không lặp warm-up SGD;
reference Gaussian/DML giữ recipe gốc từr1, source/provenance mới và gp rule chung dưới đây.
Đây fork có provenance, không giả fingerprint exact-resume từ ClassTail sang objective mới.

Rank từr3: CE+0,25ranking, softmaxT1/margin1, cả positive/negative side; N/K normalization như probe.
Scan toàn T cuối mỗi round BN-eval/FP32, FP64 moments; means stop-gradient từ snapshot trước round.
Counts và global support như probe; pooled/own cùng common support own≥32 mỗi phía.
Mỗi arm có models và moments riêng. Shuffle chỉ xoay means positive/negative một class (`roll(...,1)`),
giữ true-class counts/support; frozen arm giữ cả cohort/own support của r2.
Loss normalization vẫn K16; không tăng lambda khi thiếu class.

Class-weight CE control đã chốt riêng vì probe OLS có coefficients lớn:
NNLS16true-class CE directions để xấp xỉ **pooled rank gradient đã gồm lambda0,25**, riêng từng client;
fit trên train-only fit-half128/class tối đa của sample seed1729, refresh mỗi round từ snapshot trước.
Beta clip[0,4], không renormalize; SGD dùng `(1+beta[y])*CE`. Không bias/temperature directions trong control này.
Những giới hạn này làm control không exact-gradient-matched; không suy toàn bộ calibration explanations bị bác.

Primary tất cảarms/references: `argmax(logits - log(pi_local) + log(pi_global))`, true-label counts T cố định,
`pi=(count+1)/(N+16)`. Prior global trả actual downlink/bootstrap bytes; không target-mass gp.
Head chỉ diagnostic; không đổi sau FAIL.

## Gate và budget đề nghị

Fixedr20 mỗi seed: pooled−own/shuffle≥0,010; pooled−CE≥0,015;
benign≥max(CE,strongestGaussian/DML)−0,005. Class-weight CE ngang pooled trong0,005 cả hai seed:
chưa chứng minh riêng pairwise mechanism. Muốn mở dài: pooled−strongestGaussian/DML≥0,015 mỗi seed.
Candidate algorithm và full-round cost≤2×plain CE thực, cùng applied epochs/rows; không scan/dummy baseline.
T4 engineering/cost preflight trước utility; FAIL kỹ thuật thì sửa nguyên nhân, không đổi objective/gate để cứu.
Không chọn round/HP từ group hoặc official test; không kéo dài sau utilityFAIL.

Hai session2,5h, đồng thời cùng owner minhtriethihi:5h+15%=5,75h.
[Quota owner cũ](../../../../operations/quota-rank-utility-proposal-2026-10-15.json):6,4676h, reserved0 lúc08:24UTC.
Không dùng thêm envelope89,7/90h của ba owner mới, không query quota ba owner đó.
Nếu session thiếu horizon: audit committed rounds, không tự gọi utilityPASS hoặc thay LR horizon.

Điểm cần owner chốt: mở pilot này; common-support/refresh/fork/rules trên; NNLS beta clip[0,4] không renormalize.
