# Generator — bằng chứng runtime

Catalog/quyết định: [generators](../../../docs/research/generators.md). Job sống chỉ ở [continue](../../../docs/operations/continue.md).
Runtime PASS không thay utility r50; official test đã được đọc, không chọn HP/best round từ r2.

## Standard latent — T4 r2, hai seed

Session1 dừng budget r1; session2 import verified state/RNG hai rank, resume đúng r2 vàrestore Inductor tar exit0.
Đủ5.343xe,2T4sm75/compiled hai rank, source/config/IDs/support/LR/CM→10metric/teacher/snapshot/finite/normalization auditPASS.
Cảhai r2 skip0. Không gọi cachehit/speedup từchỉtarrestore; chưaT4 bitexactresumevsstraight hoặc utility50.
Gate: [screening](screening-standard-t4.md). Evidence:
`runs/standard-t4/collection/Z{0,1}-r2/{audit,diagnostic-audit}.json`,
`runs/standard-t4-resume/{closure,artifact-recheck}.json`.

| metric r2 | seed42 | seed43 |
|---|---:|---:|
| accuracy | 0.894971682 | 0.893994220 |
| precision_macro | 0.886286719 | 0.888865477 |
| precision_micro | 0.894971682 | 0.893994220 |
| precision_weighted | 0.886281866 | 0.885887921 |
| recall_macro | 0.842883228 | 0.841983282 |
| recall_micro | 0.894971682 | 0.893994220 |
| recall_weighted | 0.894971682 | 0.893994220 |
| f1_macro | 0.853074615 | 0.853092993 |
| f1_micro | 0.894971682 | 0.893994220 |
| f1_weighted | 0.880014452 | 0.878759678 |

## Forest + quantile — preflight và sampler GPU

[Gate production](screening-forest-t4.md).8testsPASS; seed42/43 trênlocal sm86/world1/2xe613,2029:
4round straight vsstop2→resume3→restore2→4 khớpbit fullstate/history/CM/snapshot, sourceSHA khôngđổi.
Static samplerSHA khớpcheckpoint/recovery. Sampler dùngseparate RNG cho calibration/benchmark, khôngadvance trainingstream.

GPU sampler trênpaneltrain→val cũ16class: pairTV .199354639→.067298627; singletonmeanTV/class khôngtăng>.02.
KhôngfitHGB lại, khôngofficialtest; khôngsuyT4/FLutility từstatisticalPASS.
Evidence `runs/forest-production/{preflight,preflight-artifact-audit,sampler-gate}.json`.

## Forest + quantile — T4 r2, hai seed, audit PASS07/11

Đủ2round/5.343xe/2T4sm75/compiled cả2rank, source/config/CM/support/LR/teacher/snapshot/finite/byte vàcount consistencyPASS.
Static sampler SHA giống giữa hai rank; bootstrap1704951B/xe, Q1088B/r; r2 cảhai seed skip0.
Khôngfullprivatebank/predictionreplay hoặc T4bitexactresumeproof.
`runs/forest-t4/{closure,artifact-recheck}.json`, `collection/Z{2,3}-r2/`; worker đóng03:48:23.

| metric r2 | seed42 | seed43 |
|---|---:|---:|
| accuracy | 0.892538779 | 0.891458832 |
| precision_macro | 0.884106294 | 0.886692246 |
| precision_micro | 0.892538779 | 0.891458832 |
| precision_weighted | 0.884406415 | 0.884067170 |
| recall_macro | 0.836139139 | 0.835019164 |
| recall_micro | 0.892538779 | 0.891458832 |
| recall_weighted | 0.892538779 | 0.891458832 |
| f1_macro | 0.848309603 | 0.848388095 |
| f1_micro | 0.892538779 | 0.891458832 |
| f1_weighted | 0.877738739 | 0.876652595 |

SoCore cùngseed/r2: macro−.004786/−.005147, p10−.021925/−.007484, low256−.012449/−.006818.
Đây làđiểm sớm exploratory, khôngutilityFAIL/PASS hoặcđượcchọnHP từnó. Khônggọi TV/TSTRPASS làcảitiếnFL.

T4r2 sampler benchmark riêng: forestG1 khoảng3,90–3,96ms soCore .463–.511ms; G32 khoảng3,87–4,07ms soCore4,55–5,60ms.
Tácđộng phụthuộcbatch/G, khôngđồngnghĩa end-to-end speedup. R2train636,7/663,3s, eval17,1/19,0s;
khôngtraintruncatehoặcHPđổichođạtthroughput.

Pilot tiếp theo đã đóng: [kết quả20](results-generator-pilot20.md), standard/forest đủ2seed/r20/audit,
cả hai FAIL utility gate; dừng20, giữ Core. Runtime PASS ở tài liệu này vẫn đúng phạm vi kỹ thuật.

