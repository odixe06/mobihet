# V10 — Gaussian-base DAGSNet joint energy + independence-MH

Chủ dự án chốt: commonH2diagmixtureq, localsmoothingτ0,05 trênmọi66feature,
energyweight0,1,256noise-init chain/client,20burn-in/round+1transition/batch;
teacher-first/cost≤2×freshCE T4/probe vàsamplergate. Khôngraw/perturbedrecordrời xe.
[Cơ chế/primarysources](unready-methods-preflight-2026-10-12.md)§3.

7arm/seed42/43: CEraw thật(costdenominator), CEmatched(dummyenergy/sampler),
CEaug(CEraw+.1CEsmooth), energyFL/own/coordinateshuffle/q-negativecontrol.
20client/fixture2,5%/20round; r20teacher group/head; old/benign/perclass diagnostic.
Denseunweighted20clientstateaverage cảparams/BN, countersmax; FP32up/FP16down,
prior/H2diag/coldstart paid. Own vẫnsharebase/prior, chỉmodelaggregationkhác.
BNcanonical4096Gaussian, frozenBN/dropout teacherfit; AdamWreset/cosine/wd0/clip1.
AMPCEraw, **FP32energy/MH**, finite/applied/skipguard≤16warm-up mỗi client/round.

q(x)=uniform16mixtureGaussiandiag decoded; positive x+.05σ_yε local.
pθ(x,y)∝q(x)expf_y(x), normalizerfinite từboundedDAGShead khiweightsfinite.
Commonq triệttiêu trongconditionalclassifier: softmaxf; khôngclass-dependentbaseposterior.
LossCEraw+.1[-meanA(smooth)+meanA(negative)], A=logsumexpf.
MHproposalnoiseq, logaccept=Aθ(new)−Aθ(old); **recomputeoldenergy currentθ sauoptimizer**.
20burn khôngmixingproof. Continuousq khôngsupport-faithful trênbinary/zeroatoms;
exact-support không làgateđợt này. Qnegativecontrol bỏMH, giữenergyobjective/noisebase.
Shuffleenergy dùngf(Px) chopositive/negative/MH; CEraw giữx. Khôngdeploymentjointposteriorclaim.
Chains noise-initmỗiround, giữtrongbatch; resumestate lưuchains/sampler, roundr+1 reinit nhưfullrun.

## Cổng

TeacherFL≥max(CEraw,CEaug)−0,01; −own/shuffle≥0,01;
benign≥CEraw−0,03; old≥CEraw−0,01; codec/cohort/finite/resume/pred→10metric/inferencePASS.
Trướcpilotfull:2completedroundalgorithmcost≤2×CEraw thật khôngdummy;
FAILgiữpartial/costreason, khôngđổiλ/τ/burn/batchđểcứu.
Tạir1/5/10/20:2noise-init×64chain×100transition, movement≥5%,0chainfrozen100step,
mean-energy difference≤0,1 pooledSD. Autocorr lag1/10/20 vàpositive-sequenceESS20lag báo riêng.
Finite-diagnostic gate, khôngtheoremconvergence; trượt thìchưaopenprivate/contact.
Khôngmởprivate/contact từteacherqualityPASSmộtmình; privateKD/conditional sampler cầnmanifest riêng.
W&Bonline trướcdata/train, CE/energypositive/negative/sampleracceptance/jump/dwell/time/
LR/applied/skipped/10metric. Lỗikhitrain đangchạy báochủdựándừng; khôngauto-cancel.
