# V6 — K5 geometry primitive moments: hợp đồng pilot

Khóa trước push,2026-10-12 UTC+7. Chủ dự án chốt cảG59/G61, marginal/joint32,
fidelity trước train; hai cổng quality/byte riêng. [Quyết định](../methods/geometry-primitive-moments/decisions.md),
[thiết kế](../methods/geometry-primitive-moments/design.md).

## Phạm vi và đối chứng

15arm/seed,seed42/43; một cơ chế K5, không đếm arm thành15hướng.

- FamilyA: `k66_plain_fl`, `k66_marginal_fl`, `k66_joint_fl`,
  G59×marginal/joint×FL/own/shuffle (9arm).
- FamilyB: G61×marginal/joint×FL/own/shuffle (6arm); dùng full66 control cùngseed củaA.
- V2 local đãaudit cùng fixture/recipe làm common guard; không dùng officialtest để chọn.

20privateDAGSNet,66input,20round×1epoch,fixture1.000.369trainrow/alpha.5;
batch512,aux256,CEreal+CEsyntheticλ1,AdamWreset mỗi client/round,cosine1e-3→1e-5,
clip1,AMPfp16/lossfp32. Evaluationold/group r5/10/15/20,privatehead/gp; headlinegroup/gp.
10metric meanclientconfusion,perclass,benign,checkpoint/resumestate/prediction mỗi round.

## Codec, cohort, thống kê thật

Source `mh/v6_geometry.py`, `v6_schema.py`, `v6_run.py`; driverV2 không sửa.
Scaler/meta/publicschema cóprovenance hash; source/fixture pin tronginput/notebook.

Applicationpacket68B overhead: header36B (magic/kind/precision/length/version/context16)
+SHA25632B; context bindarm/seed/fixture/scaler. Version0 bootstrap tĩnh.
Counts vàsum/uppertriangleQ **fp64 wire thật**; mean/lower-Cholesky **fp16 down thật**;
maskprob/eligibilityfp64 down. Khôngshadowhigherprecision chođối tượng đãdecode.
Byteestimate nghiên cứu fp32/countuint64 không phảiledger này; actualpacketlen lànguồn số.
Applicationbyte chưa gồmframe/ACK/retry/securitysetup/RTT; StageU khôngR1proof.
Prior inheritedcodec trả200B/xe hai chiều ngoàiknowledgebootstrap.

Covariance z-score; n/sum/Q tínhfloat64, servercov symmetrize+Cholesky(cov+1e-4I)float64,
factor castfp16 vàdecode rồi dùngfloat32 sampling, như numericalridgeP1.
Cholesky/fp16overflow ⇒FAIL, khôngtunejitter/clip đểcứu. Globalclass cầnN≥67,≥5contributors,
guard chung full66plain dimension66. Cohortproduction20; SecureSumemulator chỉ dùngtổng,
khôngcryptographic/DP. No raw/perturbed/latent/recordgradient exports.

Marginal: aggregate24count/class +classcontributor16, Gaussianmoment lớp;
first vàhai profile độc lập givenclass,13atom conditionalfirst0.
Joint32: first×receiverprofile4×senderprofile4; aggregateN/contributors/13atom mỗiô +classcontributors;
serverbroadcasteligibility trước phase2. N≥60nonfirst/47first,≥5contributors, chung cảcontrol/candidate.
Không gửicontinuousmoment riêng của ô khôngđạt. Ô hiếm: classmoment +mask chínhô ấy.
Own: mọimoment ởlocal; cùngNcellguard, không≥5client vì khôngexport. Ôhiếm dùngownclass,
absentownclass: standardGaussiandeclared, khôngpopulationmoment. Commonprior vẫnFL.

Covariancedimension: full66plain66; support66=59, G59=52, G61=54 vì bỏ7binary/profile.
First13constant vẫn nằm trongpacked uppertriangle đểsamewirelayout; byte redundancy côngbố,
khôngclaim covarianceactive-minimal hay tốiưu compression. Chỉcell eligible cómoment extra.

## Sampler và giới hạn fidelity

Freshdraw từngbatch; fullcovariance rồi overlay13atom,first/profileexact; khôngconditional-nonatomfit,
khôngheadingrenormalization/distanceprojection/session-rateprojection.
G59derivevelocity4+headingdot+bearing2. Bearing inverse(dist>0) đúngextractor,
otherwise0; khôngclampdenominator. G61giữbearing,derive5. Đầu vào cuối luôn66feature.
2independentdraw cùnglabel/state cho mọiarm; shuffle thay blockderived của draw1 bằngdraw2,
matchedRNG/compute; khôngrawrecordkhácxe. Khôngshuffleprototype/fullmodel.
Finite+fp16representability,binary/profile/first13 vàdecoderformulas làfidelitygate;
fullgeometrysupport không phảiclaim. Negative/nonpositive distance,bearinguniterror,
near0,maxbearing,fallbackrow báodiagnostic; không dùng chọnHP.
`distance_abs_lt_1e6` trongartifact là count `abs(rawdistance)<1e-6`, tênkey giữsourceđãtest.

NominalG51/G53FAILmissingtoy; giữ8derived⇒cùngG59. Không chạyhaiarmtrùng.
Toy47case/94rows và boundedreal10.240row math khôngchứngminhF1/fulltraintwoGPU.
Smoke driver dùng1024seededrow/client sauP.Runrandom3000,holdout256,3round/aux64;
không lấysmokeF1 làm pilot. Productiongiữfullfixture20round/aux256.

## Cổng khóa

Từngdimension59/61, từngmaskM/J, từngseed; r20gp:

- Quality: FL−max(full66plain,full66support cùngmask)≥0,015.
- FL−own≥0,010; FL−shuffle≥0,010.
- FL−local≥0,03;oldgp>local;benignF1≥local−0,03.
- Codec/source/weights/state/prediction/steps/resume/finite PASS.
- Compression riêng: FL−full66support≥−0,005 vàbootstrapbyte hai chiều≤0,65×control cùngmask.
  Khônghạ ngưỡng dùestimatefullcov khôngđạt; ghiactualgate riêng.
- Jointcontribution: J−M≥0,005 cảhai; báo interaction `(G_J−G_M)−(full66_J−full66_M)`,
  khônggọi joint giúp nếucontrast chỉdoGaussian/control thayđổi.

Khôngbest-round/head/seedselect/officialtesttuning. Partial/numericalFAIL giữartifact/chẩnđoán,
khôngrerun y nguyên đểcứu. ChỉqualityPASS mới bàn contact. R2 hiện chỉdesign.

## Vận hành

Local boundedmath/driver/strict+sampleinfer/exactresume/tamper trướcpush;
Kaggle2×T4 pinnedTorch2.10+cu128,4worker/2GPU. W&Bonline/key/auth trướctrain,
private-inline notebook/key0600 theoquyềnđãchốt; publicsecret-free. Key nằmprivateversionhistory.
Workermax1,8h; parentkhôngnhậnarm mới sau3h, dừng/finalize trước4,5h đểgiữquota.
Quotasnapshot/owners/PUSHED/attempt tại `papers/v6/rebuild.md`.
NoGit/DVCpush/cleanup. Bulkcollectruns/source; rawexecutednotebook vàoignoredprivate_push,
auditredactedcopy. Đủoutput thìkhôngrerun.
