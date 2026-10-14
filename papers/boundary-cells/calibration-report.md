# BoundaryCells — calibration đã audit

10arm-seed case ×2completedround của20roundschedule. Source/config/weights/BN/pred→per-clientCM/10metrics/targetmass auditPASS.
Đây là cost calibration; chưa r20utility, R1contact, R2 hay DP. Tất cả candidatecostPASS; HP/cổng giữ nguyên.

[CSV đầy đủ](calibration-metrics-all.csv) · [Contract](contract.md)

| arm | seed | ratio single-DAGS | costPASS | r2groupgpF1 | benignF1 |
|---|---:|---:|---|---:|---:|
| bc_single | 42 | 1.000000 | True | 0.446259 | 0.385205 |
| bc_direct2_fl | 42 | 1.588173 | True | 0.447947 | 0.514420 |
| bc_direct4_fl | 42 | 1.600705 | True | 0.452612 | 0.504559 |
| bc_teacher2_fl | 42 | 1.588146 | True | 0.440782 | 0.507189 |
| bc_teacher4_fl | 42 | 1.590978 | True | 0.439178 | 0.500691 |
| bc_single | 43 | 1.000000 | True | 0.452054 | 0.391776 |
| bc_direct2_fl | 43 | 1.600725 | True | 0.450017 | 0.516080 |
| bc_direct4_fl | 43 | 1.563785 | True | 0.456226 | 0.506940 |
| bc_teacher2_fl | 43 | 1.486603 | True | 0.439133 | 0.508345 |
| bc_teacher4_fl | 43 | 1.531497 | True | 0.444118 | 0.505551 |

## Đủ10metric r2 — mean20client/group/private/gp

| arm | seed | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc_single | 42 | 0.549883 | 0.504390 | 0.549883 | 0.618775 | 0.476676 | 0.549883 | 0.549883 | 0.446259 | 0.549883 | 0.542082 |
| bc_direct2_fl | 42 | 0.627686 | 0.602956 | 0.627686 | 0.646464 | 0.435583 | 0.627686 | 0.627686 | 0.447947 | 0.627686 | 0.593489 |
| bc_direct4_fl | 42 | 0.627956 | 0.607644 | 0.627956 | 0.645726 | 0.439506 | 0.627956 | 0.627956 | 0.452612 | 0.627956 | 0.593226 |
| bc_teacher2_fl | 42 | 0.624389 | 0.596504 | 0.624389 | 0.639470 | 0.430828 | 0.624389 | 0.624389 | 0.440782 | 0.624389 | 0.588645 |
| bc_teacher4_fl | 42 | 0.621063 | 0.598107 | 0.621063 | 0.639449 | 0.426565 | 0.621063 | 0.621063 | 0.439178 | 0.621063 | 0.586103 |
| bc_single | 43 | 0.553251 | 0.514112 | 0.553251 | 0.625789 | 0.481492 | 0.553251 | 0.553251 | 0.452054 | 0.553251 | 0.545469 |
| bc_direct2_fl | 43 | 0.629746 | 0.601133 | 0.629746 | 0.650217 | 0.441179 | 0.629746 | 0.629746 | 0.450017 | 0.629746 | 0.594197 |
| bc_direct4_fl | 43 | 0.628255 | 0.602254 | 0.628255 | 0.644469 | 0.445215 | 0.628255 | 0.628255 | 0.456226 | 0.628255 | 0.593638 |
| bc_teacher2_fl | 43 | 0.620864 | 0.586658 | 0.620864 | 0.633002 | 0.429266 | 0.620864 | 0.620864 | 0.439133 | 0.620864 | 0.585164 |
| bc_teacher4_fl | 43 | 0.621322 | 0.599193 | 0.621322 | 0.641243 | 0.432715 | 0.621322 | 0.621322 | 0.444118 | 0.621322 | 0.588004 |

## Dự báo cập nhật, giữ giới hạn

DirectP4−single r2≈+0,00635/+0,00417; directP2 đổi dấu; teacherP2/P4 thấp hơn single ở cảhai seed.
Có cơ sở mở pilot sau costPASS, ưu tiên tín hiệu directP4, nhưng vẫn kiểm cảP/mode đãchốt.
Không kếtluận teacherqualityFAIL từr2; không tăngweight/steps hoặc chọnseed/bestround đểcứucổng.
BenignF1 tăng không đồngnghĩa macroF1 tăng: tri thức/massprior cóthể phânphối lại lỗi giữa lớp.
Chi phí gồm diagnosticteacher; baseline denominator chỉmộtDAGS, khôngdummy-inflate.
Strictcheckpointrebuild có; sampled saved-prediction inference chưa thực hiện trongcollector calibration.
