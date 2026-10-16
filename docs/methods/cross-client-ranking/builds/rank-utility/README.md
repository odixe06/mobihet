# CrossClientRank — paired utility r20

**Cập nhật15/10 tối.** Seed42 v2 terminal, collector audit PASS (7/8 arm r20; cr_dml dừng r16 do budget, owner chốt
không nối reference). Group gp r20: pooled 0,466075; −CE local **+0,001735**, −own −0,000261, −shuffle +0,001196,
−fixed +0,001099, −classweight +0,001142; Gaussian 0,571137; cr_dml r16 private gp 0,633531 / proxy 0,769833.
⇒ **utility FAIL seed42** (cần −CE≥0,015, −own/shuffle≥0,010), không phụ thuộc reference. Gate theo seed nên CCR FAIL;
seed43 recovery v2 (PUSHED10:49Z, CPU mounted r3 audit PASS, cost r3 1,2459/1,1991×CE) tiếp tục chỉ để mô tả.
Collector đã sửa miễn `metrics.csv` ghi lại khi resume (checker bug, không đổi dữ liệu). Không nối hay đổi λ/gp.

Kết phiên15/10: owner yêu cầu dừng poll; [checklist/lệnh phiên sau](../../../../operations/continuation-2026-10-15-afternoon.md#việc-còn-thiếu--phiên-sau).
Không poll/launch thêm trong phiên bàn giao.

Owner chốt15/10 sau [probe/diagnostic đã audit](../rank-probe/results/report.md).
[Contract đã duyệt](../rank-probe/utility-proposal.md) · [Nguồn/objective](../../../../research/new-directions-2026-10-15-afternoon.md).

GPU version1 đã push09:00UTC15/10:
[seed42](https://www.kaggle.com/code/minhtriethihi/mobihet-cr-u-s42-20261015-v1),
[seed43](https://www.kaggle.com/code/minhtriethihi/mobihet-cr-u-s43-20261015-v1).
Version1 seed42 ERROR sau validr3 do empty CE `wire=b''` bị weights-only loader từ chối.
Artifact strict state/wire/pred→10metrics audit PASS với scoped builtin-bytes allowlist;
r3 algorithm/full-round cost1,255634/1,217979×CE PASS. Không phải numerical/utilityFAIL.
Local fixture trước chỉ cover pooled, bỏ sót CE empty state; regression giờ tái hiện và kiểm subprocess.
[Recovery seed42 v2](https://www.kaggle.com/code/minhtriethihi/mobihet-cr-u-s42-20261015-v2)
PUSHED09:20UTC sau CPU mounted r3/compat/strict state/wire/80metric checks PASS.
Source/fingerprint/weights giữ nguyên, chỉ launcher scoped allow builtin bytes, vẫn weights_onlyTrue;
không SGD lại r3. Seed42v2 CE/pooled đã có fresh finite heartbeat r4; dừng health polling seed42.
Seed43 version1 RUNNING/chưa fresh heartbeat, chưa chốt cost.
PUSHED chưa là utility PASS. Mỗi session kiểm CE/pooled full-T r3 cost trước tiếp lịch20;
xem [startup và proof](../../../../../methods/cross-client-ranking/builds/rank-utility/probes).

## Cấu hình và lineage

Pilot20clients gốc từ train80%, giữ66features/partition/scaler/DAGSNet; **không full43M training**.
Group/old nội bộ, không official test. Một epoch/client/round, batch512, AdamWwd0/reset mỗi round,
cosine20 1e−3→1e−5, clip1, AMP như recipe pilot. Mọi client tham gia, guard applied>0/skipped≤16/finite.

8arms mỗi seed: `cr_local`, `cr_own`, `cr_fl`, `cr_shuffle`, `cr_fixed`, `cr_classweight`,
`cr_gaussian`, `cr_dml`. Sáu arm đầu fork cùng audited ClassTail-local CE-r2 weights;
không lặp warm-up SGD. Inherited2 ghi riêng; committed metrics/weights mới bắt đầur3.
Không gán metrics r1/r2 dưới gp mới. Checkpoint gốc/parent hash còn nguyên tại ClassTail calibration.
RNG là per-round SeedSequence/generator của recipe cũ, không optimizer/scaler persistent giữa round.
Gaussian/DML fresh từr1, recipe gốc trong frozen40mh; không tái dùng metrics dưới gp khác.

Rank: CE+0,25×squared-ranking block surrogate, softmaxT1, margin1, cả positive/negative side;
N/K normalization như probe. Means stop-gradient từ cuối round trước; full-T BN-eval/FP32 scan, FP64 sums.
Support global≥128rows/≥5contributors mỗi phía; pooled/own cùng own≥32 mỗi phía, K16 normalization cố định.
Shuffle xoay hai means1class, giữ counts/support; fixed giữ snapshotr2, không refresh mới.

Class-weight control: fit-half seeded1729 tối đa128row/class/client; NNLS16true-class CE directions
xấp xỉ pooled rank gradient đã gồmλ. Vì columns rời nhau theo true-class, dùng nghiệm NNLS orthogonal
đại số, đã so independent solver. Beta clip[0,4], CE weight1+beta, không renormalize, không exact-matched.
Mỗi arm có models/moments riêng; không score/gradient per-record rời xe hoặc ghi output.

Primary mọi arm/reference: private/group/gp theo true-label counts T cố định,
`pi=(count+1)/(N+16)`, `argmax(logits-log(pi_local)+log(pi_global))`.
Prior global qua actual paid codec; head diagnostic. Ranking không có CE-target-mass prior claim.
Không đổi gp/headline sau FAIL.

## Cost, byte và gate

Hai session2,5h trên minhtriethihi;5h+15%=5,75h, trong snapshot6,4676h owner cũ.
Recovery tính trong5h đã chốt: seed42 v2 budget2,25h; seed43 chờ terminal trước audit/recovery.
Snapshot09:13UTC owner cũ còn6,20075h; thời gian thực tế/checkpoint phải tính trước session tiếp.
Không dùng thêm envelope89,7/90h ba account mới; không query quota ba owner mới.
Budget gồm setup/hash/import/scan/compile/eval/commit, stop trước boundary với worst-round+margin.

Cost gate mỗi seed: actual full-T CE và pooled r3, cả algorithm_seconds/full-round sec≤2×CE;
strict checkpoint/state/wire/prediction audit của r3 trước tiếp20. CE một DAGSNet thật, không scan/dummy baseline.
R3 gate không thay full post-warm r3–20 cost gate hoặc utility.

Utility mỗi seed fixedr20: pooled−own/shuffle≥0,010; pooled−CE≥0,015;
benign≥max(CE,strongestGaussian/DML)−0,005. Class-weight ngang pooled≤0,005 cả hai seed:
chưa chứng minh riêng ranking mechanism. Chỉ mở dài nếu hơn strongest≥0,015 mỗi seed và cost≤2×CE.
Không test/group-based best round/HP tuning; không nối FAIL để tìm điểm thắng.

Score payload giữ inner codec928/746B, thêm32B refresh frame mang round/fingerprint/CRC:
actual up960B/down778B mỗi client/refresh. Bootstrap dictionary/prior và references trả actual bytes riêng.
Fixed snapshot chỉ trả refresh ban đầu; lưu payload lặp trong checkpoint không giả thêm downlink.
Không contact/deadline R1, heterogeneous R2, secure aggregation/DP proof.

## Kiểm chứng và tiếp tục

Local actual gradient error2,17e−19, independent NNLS, stale/CRC/identity refusal PASS.
Bounded synthetic DAGSNet SGD/state replay PASS; actual driver tiny20×512-row fixture kiểm fork,
next-round exact resume, orphan checkpoint ignored, tampered prediction refusal; W&B mock **chỉ trong fixture đó**.
8emitted-controller cases normal/costFAIL/workerFAIL/budget PASS. CPU mounted117names/sizes,
116inputSHA và20strict audited CE-r2 models PASS trước GPU.
Recovery thêm4emitted-controller cases và actual Kaggle CPU r3 audit PASS cho seed42.
Actual tiny CE driver qua đúng compat→W&B adapter→driver cũng full-vs-resumer3/r4 exact PASS;
pooled replay, orphan/tamper refusal giữ PASS. Không dùng production records cho SGD local.
Đây chưa terminal utility hoặc independent saved-weight inference.

[Source/scripts/notebooks](../../../../../methods/cross-client-ranking/builds/rank-utility/README.md).
Khi terminal: `scripts/collect_rank_utility.py`, rồi `scripts/report_rank_utility.py` từ build này.
Collector exact executed/source/fork/fixture, strict mọi SD/BN, rank wire và saved DML codec,
fixed priors, predictions→đủ10metrics/per-class. Không independent full-gradient/NNLS refit.
Incomplete session giữ actual committed round; không resetr1 hoặc tự gọi allr20.
Private key ignored0600; private version history vẫn giữ key. Artifact mới chưa DVC backup.
