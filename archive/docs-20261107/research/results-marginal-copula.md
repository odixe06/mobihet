# Marginal-only copula — attribution, 06/11

`runs/marginal-copula/result.json`: worker exit0/marker, đủ16class, same sampleSHA nhưhybrid. Scores từCM đãlưu, khôngFL.

- TSTRmacro copula 0.761808674, hybrid 0.831997139, marginalonly 0.853304370.
- PairTV 0.199355→0.199748, khôngqua20%gate. Singleton/TSTRPASS. **Overall feasibilityFAIL theo plan**, khôngđổiverdict saugainmacro.
- Attribution: hybridpairTV cải thiệnkhôngcầnthiếtđểcóTSTRgaintrongprobe này; marginalonly còncao hơnhybrid. Khôngnhậnpairtable vàoCore hoặcgángainchojointcorrection.
- Class1F1 marginalonly .899375 sohybrid .736356; class13 .239983 sohybrid .210829; class15 .934345 sohybrid .906469. KhôngchứngminhmodelFL cógain tươngứng.

[Plan](screening-marginal-copula.md); full10metric+classF1 trongartifact. Hypothesismới: [standardnormal latent trướcPhi](screening-standard-copula.md), nhắm đúngmarginaldistortion, cổngphùhợpmục tiêu khóa trước số mới. Khôngnới gatepairtable cũ.
