# V5-G/J — K3 noise-init gradient/latent condensation động

2026-10-12 UTC+7, khóa trước Kaggle. Chủ dự án chốt IPC1/4, bookrefresh mỗi5round,
**80step tổng/book luânphiên16class (5/class)**, trần2×baseline. Không record thật rời xe.
Mọi encoder/classifier DAGSNet; không gộpgradient từ cácθ_i khác nhau. V3/V4source giữnguyên.

## Bằngchứng và phép thử

S-a mean-latent/randomfrozen/staticbank đãFAIL; chưa côlập bank/encoder/objective.
Đợtnày4cells **meanlatent/gradient × frozeninit/currentprivate encoder**, tấtcả bookrefresh5round,
giúp đoobjective vàencoder khácnhau. Khôngclaim táisinhbook riêng cólợi vìcả4cell đềuđộng;
khôngclaim táihiệnS-a(hp/calibration/schedule khác).
[Gradientmatching nguồn](https://arxiv.org/abs/2006.05929), [officialcode](https://github.com/VICO-UoE/DatasetCondensation)
làdatasetcondensation, khôngpaperFL/DPguarantee. Localcondensation/peerbooks ởđây làadaptation.

## Local synthesis và FL

Tại r1/6/11/16, producer copyDAGSNetinit hoặcprivateweights r−1, recalibrateBN bằng
≤4096localtrainrow rồi eval/dropoutoff. PrivateBN khôngbịcalib vìencoder làcopy.
FP32 synthesis khôngAMP. Targetmỗi lớp từ≤32localrecord lấyseededkhôngreplacement, cachedchobook;
không record/meanlatent/gradienttarget được export. Localclassvắnggiữnoise vàghi skippedclass slots.

Book `u_y∈R^(IPC×66)` từnoise, `s_y=decoded_global_mu_y+decoded_global_sd_y*u_y`.
Meanlatent objective normalizedMSE `||mean phi(s)−mean phi(real)||²/max(||target||²,eps)`.
Gradient objective: CEclassgradienttheoencoderparameters, khớp **matrix/tensorweights ndim≥2**
bằngmean cosine-distance(layersweights); norm/bias1D khôngmatched. Khônggọifullvector matching mọi parameter.
Regularizer.01*mean_IPC(u)², Adam riêngmỗilớp lr.03; 80slotround-robin, không momentumdrift lớpchưachọn.
Khôngupdateencoderweights, khôngsharedθgradient. Lossfitgiảm khôngchứngminh F1.
Encoderparameterhash/canonicalBNhash/fit/actualtargetbatchrows lưuđểaudit localstate, khôngwireknowledge.

Encodebookfp16 thật, decode trướcupload/use; **exact duplicate guard** so decodedbook với localrecord
theofixturefp16 trướcupload. Exactmatchfail; khôngraw-init/nearest-recordcopy. KhôngbảođảmDP
hoặc khôngapproximateleakage từcondensation, ngoại lệsyntheticnoise hiệnhành đượcápdụng.
RSU đọcindividualsyntheticbook đượcchophép; môhình vẫnDAGSNet.

Mỗi recipient nhận4otherproducerbooks, offsets1..4/5..8/9..12/13..16 theo4phase;
drawbalancedlabels256/privatebatch, randompeer/point. Own chỉownbook, cùngGaussian/priorbootstrap.
Noisebookcontrol gửinoisechưaopt, cùngrefresh/codec/peerdiversity; continuousdiag strongcontrol sinhfresh mỗibatch.
PrivateCEreal+CEbookλ1, impacttoànbackbone, khôngteacherKD trongbookfamily.

## Arm/protocol/cổng

G=IPC1,J=IPC4, mỗi7arm: `diag`,`noise`,`mean_frozen`,`mean_trained`,`grad_frozen`,`grad_trained`,`own`.
Own dùnggradient/currentencoder, cùngIPC/synthesis, khôngtransferbook; commonGaussian/prior vẫnFL.
20clients,alpha.5,2,5%fixture,66features,seed42/43,20round×1privateepoch;
batch512/aux256,AdamWwd0reset,cosine1e-3→1e-5,clip1,AMPprivate/deterministic/backend nhưV2.
Evalr5/10/15/20 old/group,head/gp,10metric/perclass;group/private/gp headline,r20 khóa.

TừngIPC/hai seed: gradient-trained−noisebank≥0,015;−own≥0,010;−continuousdiag≥0,015;
groupgp−localV2≥0,03,oldgp>localV2,benignF1≥localV2−0,03; cost≤2×freshdiag.
Gradienttrained−meantrained≥0,005 đểclaimobjective; gradienttrained−gradientfrozen≥0,005 đểclaimcurrentencoder.
Báo2×2interaction, khôngsuycausal cảhai từmộtcontrast. Thiếuthànhphầncontrastkhông tựphủđịnhutility,
nhưng khôngclaimcomponent. Numeric/cost/qualityFAIL dừng, khôngtăngstep/round/capđểcứu.
Hướngmeanlatent thắngriêng khôngtự mởcontactvìmaincổngkhóaởgradient-trained; proposal mới cầnmanifest riêng.

## Costgate Kaggle

MỗiIPC/seed kernelđo fresh`diag` **cùngnoisebank trênGPU0 (2worker/GPU)** trướccondensationcases;
khôngdùnglocalRTX hoặcoldCtimingthaybaseline. GPU1 idletrongbaselinephase đểkhôngfakeconcurrency.
Sau2baselinecasesđủartifact, lấydiag active minutes từresult.json; launch5condensationcases 4slot2T4.
Mỗiarm `--max_hours=2*baseline_seconds/3600`; stockdriver dựphònground+15%+60s trướcadmitround.
Parenttheodõiwallworker vàterminate khivượtbudget(bao gồmauth làconservative), preservingcommittedcheckpoints;
coststop làFAILlockedmethod, khôngOOM/rerun. Book synthesis/duplicates/calibration/evalđều nằmtrongactive driver timer.
Preflight80step/IPCcả2GPU vàactualtiming trướcbaseline; projectedoverhead>2×baseline dựbáo thìkhôngadmitcase;
sauactualbaseline dùngboundthật, khôngtuneHP từscore. Queue/servicewait khônggọitrainingtime.

## Byte và smoke

Gaussian/priorbootstrap13008B/xe. Bookpacket `32+16*IPC*66*2` =2144/8480B.
MỗiphaseFL/noise:1upload+4download ⇒10720/42400B/xe;4phase ⇒42880/169600B,
cộngbootstrap ⇒55888/182608B/xe Ustage trướcACK/schema/frame. Own khôngbookwire.
IPC1 4peerdownkhôngfitB0; contactstage mới cóthểkiểm1peer, khôngclaimB0từU.
Khôngweights/encoder/gradientvectorupload. Chưafinitecontact/cohort/cryptographic/DP/R2mixedarchitectureproof.

Local**smoke** dùng16step/book,refresh2round đểkiểmphaseboundary trong3round/256row/client;
khôngF1feasibility, khôngthayproduction80/5. Actualmixed-derivative IPC1/4local4stepPASS;
full80booknumeric/timing trênT4 trướcpilot. Exactresumequa refresh, bookpackets/peermap/encoderancestry,
savedweights/pred→10metric/sampledinference, duplicateguard/source/W&Bprivatekeyboundary bắtbuộc.
