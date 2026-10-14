# V9 — common DAGSNet residual trên anchor

Chủ dự án đã chốt teacher-first M64/512, η1/trust0,25/clip±8,
20publicstep/round/canonicalBN4096, denseprobe weights+BN trảbyte; cost≤2×freshdenseCE T4.
Chi tiết cơ chế/cổng/primarysources: [research](unready-methods-preflight-2026-10-12.md)§2.

13arm/seed42/43: freshdenseCE; mỗiM×FL/own/shuffle/classmean/hardlabel/frozenprobe.
20client/fixturepilot2,5%/20round; teacher group/head r20; old/perclass/benign diagnostic.
Private modelkhôngtrain/KD trongđợt này; localprivategp referenceV2 cùngfixture已audit.
AdamWwd0/reset/cosine1e−3→1e−5,clip1; publicfitFP32/eval/dropoutoff.

PublicnoiseGaussiananchors FP16serialize/decode; labelsbalanced t0=center(log(.95onehot+.05/16)).
Bandwidthtrain-only h1/2/4/√66, kernelcoverage k≥1e−3,90%total/80%class;
khôngqua trướcpublictrain thìcoverageFAIL, khôngtunebandwidththeoF1.
Balancedresidual G=sum k(x,z)(pθ(x)−onehot_y)/N_y vàB=sum k/N_y;
G/B clippedL2step0,25, targetclip±8 rồicenter; FP16targetroundtrip/recenterdecoded.
FP32clientpacket→FP64cohortsum, currentprobe **SHA256 toànparams+BN+counters**
prefix32B mỗiresidualpacket; version/header/CRCphảimatchtrướcaggregate.
RSU chỉcohortsum; sourcecode trustedemulatorchưacrypto/DP. Khôngperrecordk/logits/residual exported.
Commonprobe FP16**whole-state** decode cùng tạiRSU/xe. Actualmodelheader40B,
floatparams+BN FP16/intcounterint64; bytekháctoyheader68B đượcbáo từserializedpacket.
Prior/H2diag/anchor/coverage/probe/target/bootstrap đều cóphí; source/architecture/scalerpreinstalled.

Residual dùngsnapshotđãphát trướcpublicfit. Trướcmỗipublicfit calibrateBN4096Gaussian,
freeze BN/dropout khi20fitstep/residual. Fitwarm-startcommonθ; saufit coherentFP16snapshot
đượcbroadcast cho roundtiếp, khônghiddenBNlocal/crossdeviceseedreconstructionfree.
Own giữsameglobalanchors/bootstrap/prior, residual/teacherlocal; đây khôngnoFL baseline.
Hardlabel scan dummyreal/residual, giữt0; khôngupload unusedG. Frozenprobe luônθ0 đãdecode.
Shuffle fixedanchorpermutation trênresidualaddress, classmean meanresidual theoanchor;
không đổi labels thật hoặcGaussiananchorinitialization.

## Cổng r20 từngseed

FL−localprivategp≥0,05; FL−hardlabel≥0,01; FL−own/shuffle/classmean≥0,01.
Dynamic−frozen≥0,005 làrefreshcontributionriêng. Cổng source/codec/cohort/BN/finite/
resume/pred→10metric/inference phảiPASS. CảhaiMđộc lập; khônghead/bestround/seedavgcứu.
TeacherPASS chưaprivate/R1/R2/contact. Denseprobe gần0,8MBdown/round, khôngfitB0/B2singlecontact.

Costpreflight2completedround vớischedule20, so measuredalgorithm_seconds freshdenseCE
trêncùngT4; phải≤2×, giữpartialcheckpoint/costreason khiFAIL. Startup/coverage/fit/eval/I/O
báoriêng vàtotalwall/quota. Khôngdummyinflatedbaseline; khônggiảmM/step/batchđểcứucost.
NotebookonlineW&B/logpublicteacherMSE/target/residualnorm/LR/publicapplied/clientstatrow/time/
10metric; clientresidualphaseapplied=0 vìkhôngSGD, publicteacher_applied=20 hoặc400 own.
First1–5roundhealth vàpulledterminalaudit theoquyđịnhchung.
