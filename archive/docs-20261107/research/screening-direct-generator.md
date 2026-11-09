# Direct conditional generator — khóa probe 06/11/2026

Owner cho tiếp tục thử→phân tíchFAIL→cơchếmới, tối ưu data generator; không cần xin lại quyền cho từng probe trong ràng buộc hiện tại. Không tự đổi protocol/HP/loss/Q/floor, không chọn bằng officialtest. Dừng một arm FAIL, ghi evidence→mechanism→falsifiable sign trước arm tiếp; không nới gate cũ.

## Problem và cơ chế

[Reweight FAIL](results-pair-generator.md): ESS1–8/16384 candidate trênheldout draw. [Thiết kế](generator-direct-conditional.md): forest12edge/19feature/7root sinhtrựctiếp từbảngcounts; remaining58features theo Gaussianconditional gốc. Loại dependencefinitecandidate; cóthể mấtdependence giữa cáccomponent S hoặc làmxấu marginalR.

CodeCPU `scripts/direct_generator.py`, runner `scripts/probe_direct_generator.py`; chỉ đọctrain/val từloaderprobeđầu, không đọcofficialtest. Datasetpanel đãkhámphá nên không là freshconfirmation. NumPy/SciPy CPU approximateCorecompact, chưa samplerCUDA/FL. Stats/counts tạiS dùng fullprecision count, CDFtruyền uint16 simulated; chưawirecodec/tablequantizationproof.

## Khóa trước số

- Danhsách12pairs giữ từprobeđầu; uniform16bin[-4,4], fine256bin. Root minindex, components independent.
- Copula fit cùngdata mỗiclass, covariancepopulation+1e-4I; mean/Choleskyfp16, CDFuint16; covarianceconditional derive từLdecoded@Ldecoded.T đểsamplingkernelnhấtquán.
-16384freshdraw/class cho mỗiarm, seed8100+c. KhôngESSgate cho direct vìkhôngimportanceweights.
- Cổng giữ: pairTVmeanmacro val giảm≥20%; mọiclass singletonTVmean không tăng>.02. Chỉ nếuqua, TSTR HGB40iter/31leaf/seed42,4096synthetic/class, gainmacro≥.003. Khônghạgate.
-64outsidepairs chọn bằngseed8110, khôngdata-driven; báoTV nhưdiagnosticmấtdependence, không chọnlạiedge bằngval.
- TSTR lưu16×16CM và10metric; đây làpooledprobe, khôngthay utilitypersonalizedFL.
- Plan/sourceSHA immutable `runs/direct-generator/plan.json`; selectedarray SHA trainX/Y,valX/Y trongresult (khôngfull-datasetfingerprint).

## Tests và bước tiếp

5testsPASS gồm2testsreweightcũ và3testsmới: aggregatecount identity; nonlinearforestpairsamplingTV<.02/100kdraw; conditionalGaussianjointmean/covariance; cycle rejection. KhôngchứngminhT4backward/2rank/resume.

NếuFAIL: đọcclasswise singletonS/R vàoutsidepair, táchmismatch doancestraltable/coarsebin khỏiconditionalGaussian hoặcindependentroot. Chỉ thayđúngcơchế códấuvết; khôngquét HPtrênval hoặc officialtest đểtìmwin.
NếuPASS: triểnkhai streamingGPU, gatesrealdata/resume/codec/count/throughput; sauđó FLcontrol/candidate2seed,50roundcosine. Cache npy, Inductor vàcompile giữ, tài khoản/quota chọnlive, đượccrossaccount qua checkpointprivateđãverify. KhôngsửaY50worker đangsống.
