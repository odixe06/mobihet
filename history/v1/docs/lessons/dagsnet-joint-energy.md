# Bài học — Joint-energy bằng DAGSNet

LayerNorm làm logits bị chặn, exp(logit) có sàn dương trênR66 nên integral∞. Gaussianbase/compactdomain phải định nghĩa trước; finiteSGLD không chứng minh density/mixing. MALA NCE đã thử là adaptation riêng.

[Thiết kế/nguồn](../archive/methods/dagsnet-joint-energy/README.md).

## V10 — paired cost gate

Đãpull/audit14arm hai seed:2freshCEraw đủ20round;12candidate dừngsau2roundbởilockedcost.
EnergyFL/freshCEraw=2,96867/3,10663; q-negative bỏMH=2,32326/2,24832 vẫn>2.
CE-matched~2,96/3,04 cho thấymatcheddummy overhead lớn; khôngdùngcontrol này làmcostdenominator.
ChưađoF1r20energy; không kếtluậnqualityFAIL. Source/codec/weights/BN/step/prediction auditPASS;
Sampler FL r1 **FAIL cả hai seed**: movement5,27%/6,31% nhưng2/1chain đứngyên cả100transition;
seed43 chênh mean-energy hai khởi tạo0,177 pooled-SD>0,1. ESS energy/chain4,20/4,04.
Shuffle r1 PASS seed42, FAIL seed43; không thay kết luận của FL. Ghi chú r1PASS trước đây đã sửa theo artifact.
Probe local nhỏ PASS không dự báo mixing production; finite/acceptance không làstationarity/generalizationproof.
KhôngsuyMHlàduy nhấtgâycost; FP32/positive-negative branches/chunking chưaablate riêng.
Khônggiảmburn/λ/τ/batch hoặckéodàirun cứu cổng. Compile/fusedforward cóthể làdesignmới,
chỉbàn lại vớinumerical/compute gates, khônggọi đãPASS.
[Artifactreport](../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/results/report.md), [gates](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/probes/gates.json).
