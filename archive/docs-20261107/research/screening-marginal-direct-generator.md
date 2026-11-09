# Direct generator + marginal transport — khóa06/11 trước kết quả

Owner cho vòng phân tíchFAIL→thửcơchếmới. [Directtrial](results-direct-generator.md) đãsửa pair nhưng conditionalR làmxấu singleton class12/15. Cơchếmới: probabilityintegraltransform tạiR đểsửa marginal bằng histogramtrain sẵncó. Khôngnới gate, khôngđổi12pairs/16bin/77inputs hoặc model/LA/Q/floor.

Code `scripts/marginal_direct_generator.py`, runner `scripts/probe_marginal_direct_generator.py` táisửdụng immutable `probe_direct_generator.main` với sampler override được fingerprint rõ trong `variant-plan.json`; `plan.json` làconfig chung, phảiđọc cảvariantplan đểxácđịnhsamplerthật.

- Calibration32768synthetic/class, RNG9200riêng; quantilegrid4097, tiesmidCDF. Dữliệucalibration do chínhgeneratorsinh, khôngtrainrows hoặctestrows truyền qua.
- SinhxS giữnguyên; transport58dimensionsR. CDFtrain lấycompactCDFuint16cũ. Khôngmodelweights/gradient/neuralgenerator.
- Đây làCPUprototype; metadata0extrastatistic khôngcónghĩa0computecost hoặc0quantileapproxerror. TrongFLkhácxe sẽtựcalibrate từsameglobalstatistic; nếuđịnhbroadcastquantilegridthaycalibratelocal phảiđếm thêmbyte vàkhóagiới hạnmới, khôngâmthầmcoilà0.
- Data/seed/sizes/gates/TSTRharness giữdirecttrial: pairTV giảm≥20%, singletonmeanTVperclass khôngtăng>.02; sauđó TSTRmacro≥+.003. Outside64pair cùngseed8110, báođểđọctradeoff.
- Khôngofficialtest. Panelval đãdùngnhiềuprobe nênkhôngfreshconfirmation; thửFL2seedendpoint50 mớiđánhgiápersonalizedutility, khôngselectHP từofficialtest.
- 7testsPASS cảprior vànew: transportkhôngđổixS, khôi phụcmarginal trênindependentdraw nonlinearfixture; controlbitexact; counts/conditionalGaussian/cycle/reweightsupport.

Output `runs/marginal-direct-generator/{variant-plan,plan,statistical-result,result}.json`, log `probe.log`; nếuTSTRchạy lưu3CM16class vàđủ10metric.

FAIL: phântíchgain/loss ởS/R,outsidepairs,classprecision/recall; khôngnớigate. PASS: implementGPU/codec/resume/2rank +T4benchmark trướcFL. Quyềnowner tiếp tụcđãcó, khôngcầnhỏilại; reportblockerthật nếucontract/quota/accessngăn.
