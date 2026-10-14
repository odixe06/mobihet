# V8 — Gaussian × DAGSNet mutual learning

Chủ dự án chốt pilot 2×2 và own controls, seed42/43,20round. Khóa trước train.
7arm: h_local,h_gaussian,h_dml,h_hybrid,h_gaussian_own,h_dml_own,h_hybrid_own.
20client/fixture2,5%; headline group/private/gp r20; old/head/perclass diagnostic.
AdamWreset/wd0,batch512,aux256,cosine1e−3→1e−5,clip1 từngmodel,AMPfp16.

## Cơ chế và đối chứng

Mọi learned model DAGSNet. Private init seed, proxy init seed+1000 như P1.
DML real-only KL hai chiều, temperature1, weight=1/stopgrad(CEprivate+CEproxy).
Hybrid thêm CEsynthetic weight1 vào cả private/proxy; KD chỉreal;
Gaussianfull input stats FP32client→FP64cohortsum, factor FP16decode thật nhưV7.
Own Gaussian chỉmoment của xe, không lấy moment population khi thiếu lớp.
Proxy FL trung bình **không trọng số** 20state cảparameters/BN; integercounter=max.
FP32snapshot serialize/decode cócontext/version/CRC; gồmBN/counter, tínhbyte thật.
Coldstart gửi coherentproxy snapshot cóphí; stats/prior/bootstrap/fanout tínhriêng.
Trusted aggregate emulator chưasecurecrypto/DP; khôngrecord thật/perturbed rời xe.

Đủhai DAGSNet train trongmọiarm đểmatchnetwork/optimizerbudget. ArmkhôngKD proxyCElocal
khôngaggregate, không ảnhhưởngprivate. ArmcóGaussian forward real+synthetic trộn đểBN thấybatchtrộn.
ArmkhôngGaussian thêmdummy standard-noise forward/backwardzero, BN/dropout eval khi dummy;
không đưaGaussianpopulation vào no-Gaussian control. Cùngsốinputrow/step, khônghứa walltime/fusedkernel giốnghệt.
OwnDML vẫnmutuallearninghai model tạixe nhưngkhôngproxyFL; lợiown chưalàlợiFL.

## Cổng cuối mỗi seed, không đổi sau kết quả

Hybrid gp−max(Gaussian,DML)≥0,010; −hybrid_own≥0,010;
gp−local≥0,030; oldgp>local; benignF1≥local−0,030.
Interaction=(hybrid−DML)−(Gaussian−local) báo riêng, khôngthaycổng.
H1/H2main-effect vàFL−own báo riêng từngseed. Head tốt khôngcứugpFAIL.
Source/cohort/codec/BN/finite/weights/resume/pred→10metrics/sampledinference phảiPASS.
ChỉutilityPASS mớicontact/fullhybrid; applicationbyte chưadeadline/ACK/R1/R2proof.

W&B online/key trướctrain; lossreal/auxKD+synthetic, LR/applied/skipped/time/10metrics mỗieval.
Local1024row/client×3round, full-vs-resume, strictpayload/inference trướcpush.
Kaggle2×T4/pinnedimage, first1–5roundhealth; lỗiđangchạy báochủdựándừng.
