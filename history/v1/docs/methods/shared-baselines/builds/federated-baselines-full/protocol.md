# Full H1/H2 — xác nhận dài đã chốt

2026-10-12:20client,100%train80%=43.045.415row,50round,seed42-only.
Chủ dự án chọn7arm:local/Gaussianfull/diag/lr4/own/DML/DMLown.
Mọi learned model DAGSNet; khôngrecord thật/perturbed rời xe; source mới trong
`papers/full-validation/proj/`, source/artifact pilot đã chạy giữ nguyên.

## Protocol

Không cắt internalholdout: mọirowtrain80% train; rawtest20%=10.761.343row
áp scaler.json đúngmộtlần. Fulltest mỗi completedround, báo10metric meanclient,
privatehead/gp vàF1perclass/benign; headlineprivategp r50. Test chỉdescriptive,
không chọnHP/cổng/round/seed bằngtest. Đây fullprotocol cốđịnh, không receiver-disjointstudy mới.
Batch512,aux256,AdamWwd0resetmỗiclient/round,clip1,cosine1e−3→1e−5 trên**50round từr1**.
Gaussiansynthetic CEweight1, real+syntheticBNmixed nhưP1; FP32stats/FP16factor thật.
DML haiCEreal+KL hai chiều, T1 vàweight1/stopgrad(CEprivate+CEproxy); proxyseed+1000,
unweighted20clientfullstate mean,BNbuffersincluded/intcountermax. OwnDML khôngmodelaggregate.
OwnGaussian chỉmomentxe, thiếuownclass mean0/identitydeclared, khôngpopulationfallback.
Prior/stats/coldstart/proxyparams+BN+counters serialized thật, tínhhai chiều.
StageU; chưadeadline/contact/ACK/R1 hoặcmultiarchitectureR2proof.

## Backend, calibration và resume

Mộtarm/GPU,2×T4 mỗi notebook; khôngDDP đồngbộgradient giữaclient.
Residentdata vàhai trainingtemplate; private/proxyweightsloadstrict mỗiclient;
torch.compile forward only, stable fullbatch; tail eager; cudagraph_mark_step_begin mỗi step.
Trướccompiledtrain: actualrow/dropout-off cảeager/compiled/restoreBN+RNG,
maxlogiterror≤0,005, decisive-margin0,02 argmaxequal. Khôngsilentbackendfallback.
Evaltemplate cùngstate/BN, batch16384, tail khôngdrop; mọirowtest dùngđúngmộtlần.
Local boundedfixture chỉeager/math/resume/report, khôngfulltrain/compilebenchmark.

Đo3roundđầu trênT4với `rounds=50,stop_after=3`; checkpoint tiếp tục đúnglịch tới50.
Không đổiLR/cấuhình từtestcalibration. Chi phí=(setup+worst_round×remaining)×1,15
cộngartifact-I/O; budget whole7arm, không suy từRTX3050/historicalT4 nhưmeasurement mới.
Nếu quota khôngđủ, giữcheckpoint vàbáo phầnthiếu trướclaunchlong-chain.
Checkpoint weights-only/BN, separate perroundresume+algorithm+prior/bytes; markerlast.
Source/full-data-manifest fingerprints giữqua phiên; account/device bitwise khôngmặcđịnh.
PrivatekeyW&Bonline bắtbuộc trướcprepare/train; progressheartbeat1024step,
round loss/CE/aux/KD/LR/applied/skipped/time +10metric mỗi round.
Giámsátfirst1–5round; bấtổn đangchạy báoowner dừng, khôngagentcancel.

## Kết luận được phép

Báo r50 privateFL−own/local vàGaussianformcomparisons; r1…r50 làlearningcurve.
Mộtseed khôngCI/hai-seedconfirmation. Longr50 lợi chưachứngminhfinitecontact/R2.
Khônggiữbestround/thayhead/lịch/HP đểcứur50. Cácarmtrượtpilot cũ khôngrerun y nguyên.
