# Hybrid H1+H2 — xác nhận full dài đã chốt 2026-10-13

Điều kiện mở:V8pairedutilityPASS theo cổng đãkhóa, source/weights/predictionaudit đủhai seed.
Owner chọn20client,100%train80%=43.045.415rows,50round,seed42-only;
đủ7arm2×2+own:`h_local,h_gaussian,h_dml,h_hybrid,h_gaussian_own,h_dml_own,h_hybrid_own`.
Khôngreuse H1/H2single-model controls làmcompute-matchedV8controls. Đợt H1/H2 cũ vẫnnối đúngsource.

## Cơ chế giữ nguyên V8

Mọiarm trainhaiDAGSNet, private/proxy cùngkhởi tạo vàcùngbatch/SGD/inputrowcount.
Private/proxyGaussian:CEreal+CEsynthetic(weight1) trênmixedBN;
KD hai chiều chỉtrênreal, T1, weight1/stopgrad(CEprivate+CEproxy), clip1 từngmodel.
Nongaussian dummyGaussian-noise/evalBN/dropoutoff/zero-backward nhưpilot;
khôngcoi đây làsamewalltime/fusedkernel vớiGaussianarm.
ProxyFL unweighted20clientfull-state mean/BNincluded/intcountermax, FP32bothways/CRC/version,
global prior+Gaussianstats+bootstrap trảbyte. OwnGaussian khôngpopulationfallback;
ownproxy khôngaggregate. Model/encoder/teacher đềuDAGSNet; khôngraw/perturbedrecord rờixe.

Batch512,aux256,AdamWwd0/resetmỗiclient-round,cosine1e−3→1e−5 trên50round từr1.
Input66featuregốc,trainđãzscore, rawtest10.761.343row scaler.json áp đúngmộtlần.
Fulltest từngcompletedround,10metricmeanclient/perclass/benign, primaryprivategp r50;
head/proxy diagnostic, khôngđổiheadline hoặcchọnbestround/HP từtest.
Mộtseed khôngCI hoặcpairedconfirmationfull; đây protocolcốđịnh, khôngreceiver-disjointstudy mới.
StageU/full-data utility, chưaR1/contact/deadline/ACK hoặcR2khácarchitecture/DP/securecrypto.

## Thực thi và cổng kỹ thuật

Sourceisolated `papers/hybrid-full/proj/`, MH/H1-H2source đãPUSHED giữnguyên.
Hai reusabletrainingtemplates+evaltemplate, residentdata, compiledforward, tail eager;
actualT4/dropoutoffboth/restoreBN+RNG: maxlogiterror≤0,005 vàdecisivemargin0,02 argmaxequal.
Local chỉ7boundedarm/5exactresume/inference, khôngfulltrain/compilebenchmark.
W&Bonline/key trướctrain, fullroundmetrics/CE/KD-aux/LR/applied/skipped/time,
clientheartbeat1024step. Stable thìngừnghealthpoll; artifactterminal vẫnpull/audit.

Đo3roundđầu với50roundschedule đểcalibrateT4/wholebudget; checkpoint tiếp tới50,
khôngđổiLR/cấuhình theoearlytest. Hếtbudget giữcompleteboundary, source/config/datahash khôngđổi.
Owner đãchốt dùngquota hiệncó vànốitiếp sau refresh17/10 khi cần.
Budget cả7arm+tấtcảsession/setup/artifactI/O+dư15%; khôngsuy T4cost từlocal hoặcpilot2,5%.
