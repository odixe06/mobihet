# Marginal-only copula — attribution06/11

[HybridPASS](results-marginal-direct-generator.md) nhưng80,7% gainmacro từclass15, class1/13giảm. Hypothesis: phầnlớngain do sửaunivariateCDFdistortion, chưa cầnpairtables.

GiữcopulaGaussian mean/cov vàsamplergốc; chỉtransport77marginal theoCDFhisttrain cósẵn, calibration32768synthetic/class seed9200/grid4097. Khôngforestconditional haypairreweight. Cặp12chỉđểevaluate, khôngcầnuploadpairtables trongproduction củaarm này. Prototypecófitobjectjoint chochunginterface nhưngsampleplain khôngdùngjoint; chưaproduction/wireproof.

Script `scripts/probe_marginal_copula.py`, module `scripts/marginal_copula_generator.py`. Data/samples/seedsHGB giốnghybrid. Đọcreference3CMđãverify+sampleSHA, khôngreruncontrol. Samplercontrolbitexact đượcassert; chưapredictionreplay.
Gates giữpairTV≥20%improvement/singletonincrease≤.02perclass/TSTRmacro+.003. RiêngTSTR vẫnchạy nếuTVFAIL **đểattribution**, khôngpromoteFAIL hoặc nớigate. Source/planimmutable trướcdata, corecondparams vàhistbinskhôngđổi; noofficialtest.

Kếtluận đọc: marginal-only thắnggần/to hơnhybrid→pairtable chưautilityđãchứngminh; hybridhơn→pairrelationcònphầnđónggóp nhưngphảixemclassloss. CảhaiFAILtừutilityclassfutureFL→chẩnđoánkhác, khôngtựnớigateđểgiữphươngán.

Đây làlocalpooledTSTR, chưaDAGSNet/FL2rank proof. TrướcFL cầnGPUstreamingsampler, finite/backward/resume, payloadidentity, timingT4 vàcontrol2seedendpoint50. Owner đãcho tiếptục, khônghỏi lại sauFAIL.
