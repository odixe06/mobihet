# L0–L11 — kết quả và chẩn đoán cơ chế, 31/10/2026

**Hệ số KD cố định1 cải thiện F1 so adaptive ở cảhai seed; softtarget có ích khi kiểm đúng scale.**
L6/L7 (softfixed+realgate) cao nhất pooled; lợi riêng củarealgate nhỏ, chưa đủ chốt methodtổng hoặc50round.
Thiết kế khóa trước push: [screening-l10](screening-l10.md); [kế hoạch nguyên nhân](root-cause-experiments.md).

## Audit và phạm vi

12kernel version1 COMPLETE. Watcher20phút kếtthúc16:50:53, collect/audit nền kếtthúc16:59:20; cảhai completionqueue đãđượcconsumed.
12pull exit0,660file, khôngthiếu28artifact đích/run cộngprovenance/log; tổng4,650,598,356B = 4.331GiB.
Selectivepull gồmsource/config/10testCM/4trainCM/2rankprobe/10smallsnapshots/run. Khôngpull fullprivatebank hoặc executednotebook.
`scripts/audit_l_diagnostics.py` PASS: base+overlay3file vàknowledgehash đúng, config/HP/LR/5343ID/support đúng,
2×T4/compilegate cảhai rank,120roundtestCM→10metric,48trainCM→10metric,1632clientprobe record/teacherlag và120snapshot hữu hạn/Qchuẩnhóa.
Test9.347.679 dòng vàtrain37.361.273 dòng mỗi fullpass. Chi tiết `runs/l-screening/{audit,diagnostic-audit,collection,analysis}.json`.
Chưa prediction→CM/fullprivatebank reconstruction, exactresume, temporalheldout độc lập, kiếntrúctrộn hoặccontactdrop. Không coi snapshot nhỏ làcheckpointprivate đầyđủ.
Datasetcontentfingerprint chưa lưu; dataaudit hiện đối chiếufeatureorder/labelmap/fullsupport, không phải contenthash toàn dataset.

## Utility r10 và cổng đã khóa

Cổng: ΔF1≥.010,p10Δ≥−.010 vsG4 cùngseed. Nền42=.873187514625593/.5737619047619047; nền43=.8745079727196574/.5687566137566138.
Khôngbest-round/HPsearch hoặc nớicổng. K4/K5 augmentation làcontroltoánhọcnoQ đãkiểmCPUbitexact, xemgiớihạnwarmup bên dưới.

| cell | F1 seed42/43 | p10 seed42/43 | cổng42/43 |
|---|---|---|---|
| K4/K5 hardfixed/noReal, augmentation | .883348/.884334 | .619494/.607795 | PASS/FAIL |
| L0/L1 softfixed/noReal | 0.886315/0.886861 | 0.616754/0.600000 | PASS/PASS |
| L2/L3 hardadaptive/noReal | 0.875490/0.876207 | 0.603059/0.593629 | FAIL/FAIL |
| L4/L5 hardfixed/realgate | 0.884547/0.884579 | 0.624465/0.617117 | PASS/PASS |
| L6/L7 softfixed/realgate | 0.887030/0.887433 | 0.620423/0.606002 | PASS/PASS |
| L8/L9 softadaptive/noReal | 0.882464/0.883425 | 0.601849/0.590443 | FAIL/FAIL |
| L10/L11 softadaptive/realgate | 0.884016/0.884720 | 0.608968/0.596747 | PASS/PASS |

L6/L7 gainvsG4 +.013842/+.012925; gainvsaugmentation +.003682/+.003099. p10vsaugmentation +.000929/−.001792; khôngphảimọiclientđược lợi.
L4/L5 p10 cao nhất nhómcellL, nhưng pooledthấphơn L6/L7; seed43 chỉquacổnggain khoảng.000071.

## Đối chứng cơ chế

| phépso F1 r10 | seed42 | seed43 |
|---|---:|---:|
| soft−hard, fixed/noReal | +0.002967 | +0.002527 |
| soft−hard, adaptive/noReal | +0.006973 | +0.007218 |
| adaptive−fixed, hard/noReal | -0.007858 | -0.008127 |
| adaptive−fixed, soft/noReal | -0.003852 | -0.003436 |
| interaction target×scale | +0.004006 | +0.004691 |
| +realgate, hardfixed | +0.001199 | +0.000244 |
| +realgate, softfixed | +0.000715 | +0.000572 |
| adaptive−fixed, soft+realgate | -0.003014 | -0.002713 |
| codecnew−legacy, combo | +0.000083 | +0.000397 |
| codecnew−legacy, gate | +0.000081 | +0.000136 |

**Được hỗ trợ bởi can thiệp ở cấu hình10round này:**

- Adaptive kémfixed ở hard/noReal, soft/noReal vàsoft/realgate, cùngdấu hai seed. K4 thắngcomboK chưa đủ kếtluận bỏteacher: L naytáchscale vàtarget.
- Soft hơnhard ở fixed vàadaptive/noReal. Không loại toàn bộlearnedclassQ dựa trênK; cũng không đổi mọi synthetictarget thànhhard vì thấyteacher lớp13sai.
- Lợi riêngrealgate trênsoftfixed chỉ+.000715/+.000572. GiữnoRealsoftfixed làmcontrolmạnh khi xácnhận, chưa khẳngđịnhrealgatecầnthiết.
- Codeccontrol thayđổiF1nhỏ(+.000081–+.000397), p10cóthểgiảm; khônggọicodecbitexact/zeroeffect. CácđốichứnglossL dùngcùngcodec.

## Probe gradient: giải thích có giới hạn

Probe34client cốđịnh theo traincount×scenario/run, cùngminibatch/syntheticseed. Postepochweight, teacherQr−1, fp32/trainBN/dropout0 trêncloneweights, khôngactualAMPstep.
Cosineparameter vàlogitgradient làhaiphépđo khácnhau. `shadow_exceeds_clip` không phải actualtrainingclippingfraction.

| arm | usedweight median r10 | synthetic/real gradientnorm median | cosine real/synthetic median | cosineâm real/synthetic |
|---|---:|---:|---:|---:|
| L0 | 1.000 | 1.283 | 0.027 | 26.5% |
| L1 | 1.000 | 1.513 | 0.034 | 17.6% |
| L2 | 3.181 | 4.158 | 0.021 | 38.2% |
| L3 | 3.358 | 4.891 | 0.029 | 23.5% |
| L4 | 1.000 | 1.927 | 0.015 | 35.3% |
| L5 | 1.000 | 2.056 | 0.030 | 32.4% |
| L6 | 1.000 | 1.328 | 0.032 | 35.3% |
| L7 | 1.000 | 1.525 | 0.015 | 38.2% |
| L8 | 3.736 | 3.595 | 0.032 | 38.2% |
| L9 | 3.880 | 4.777 | 0.031 | 23.5% |
| L10 | 2.381 | 2.352 | 0.005 | 41.2% |
| L11 | 2.474 | 2.675 | 0.026 | 38.2% |

Adaptive r10median2,38–3,88; maxhardadaptive9,48. Weightedsyntheticgradientmedian2,35–4,89×realCE, trongsoftfixednoReal~1,28–1,51×.
Bằngchứngphùhợphypothesisauxiliarylossquámạnh, khôngchứngminh đây là phânphốigradient toànmạng hay cơchếclippingthực. Không gán nguyênnhân duy nhất từcosineâm.
Real-KD củaL6/L7 cócosineparameter vớiCE median.657/.427, âm6/34client mỗi seed; có cảregularizationthuận vànegative-transfer cục bộ.
L0/L1 tạiteacherQr9 đãargmaxđúngtimeDelay(Q13,13=.529937/.527923), nhưng55/93 và60/93logitgradientrowclass13 vẫnngượchướngCE trongprobe.
Argmaxđúng khôngbảođảm teacherhữuích trênmọimẫu. L6/L7Qr1–r9chỉreject13; Qr9[13,13]=.423557/.420162 vsbenign=.529751/.533198.
KênhsyntheticL6/L7 vìthế vẫn dùngteacher thiênbenign trênlớp13; syntheticfallback làgiảthuyếttiếp cầnđốichứng, chưaphảicảitiếnđãđo.

## Client/lớp và fit train

| arm | meanF1/xe | benign F1 | timeDelay F1 | train F1 | meanΔF1 xe<256 vsaugmentation | clienttốthơn/kémhơnaugmentation |
|---|---:|---:|---:|---:|---:|---|
| L0 | 0.800487 | 0.851940 | 0.283217 | 0.963288 | -0.008560 | 2861/2267 |
| L1 | 0.794859 | 0.852647 | 0.277779 | 0.963263 | -0.017263 | 2862/2269 |
| L4 | 0.802816 | 0.849619 | 0.214232 | 0.949005 | +0.005721 | 2859/2287 |
| L5 | 0.798481 | 0.850169 | 0.204784 | 0.948397 | +0.012172 | 2878/2259 |
| L6 | 0.803594 | 0.860406 | 0.195625 | 0.958402 | -0.007724 | 3015/2157 |
| L7 | 0.798740 | 0.860646 | 0.188502 | 0.957961 | -0.007348 | 3064/2104 |

L6/L7 pooled vàmean/xe tốt hơnaugmentation, nhưng meanΔnhóm<256 vẫn−.007724/−.007348. L4/L5nhómnày+.005721/+.012172; chưa có mộtcellthắngmọimục tiêu.
L6/L7 benign→timeDelay29.978/27.750, timeDelay→benign408.148/410.335; augmentation42 tươngứng180.017/329.660. Đánhđổi hai chiều phải báo rõ.
SoftfixednoReal train/testtimeDelayF1=.694046/.283217 và.691316/.277779. Trainđãfit, test là20%cuốitimeline từngxe; gap khôngtự chứngminhoverfit hayfeaturekhôngquan sátđược.
Fixedsoft tăng cảtrainF1 vàtestF1 sovớiadaptive; khôngphùhợp giải thích toàn bộ vấnđề làmodel quáfit dohọcquámạnh.
Fullcount: timeDelay5,151%train+val→5,162%test gầnnhưgiữnguyên. Khôngthể quy riênglỗitimeDelay choclassprior toàncục.
Sybil8,800%train+val→17,674%test; số8,0%trongghi chúcũkhôngđúngsplitFLhiệntại, đãsửa [dataset](../../knowledge/dataset.md).

## Thời gian và byte

MọicellL:1088B/xe/round hai chiều +3311744Bbootstrap =3322624B/xe/10round. Augmentation3311744B/xe,0learnedB/round sau bootstrap.
KênhclassQ cófp16encode/decode thực trongharness; bootstrap/transport/ACK/secureaggregation chưaend-to-endproof.

| arm | trainwall s/r r2–10 | evalwall s/r r2–10 | diagnostic s tổng10r | maxskip |
|---|---:|---:|---:|---:|
| L0 | 647.0 | 19.1 | 326.7 | 0.00e+00 |
| L1 | 621.5 | 16.6 | 291.8 | 0.00e+00 |
| L2 | 659.2 | 19.8 | 346.9 | 2.90e-05 |
| L3 | 611.4 | 16.4 | 290.2 | 2.56e-05 |
| L4 | 664.0 | 20.6 | 351.5 | 0.00e+00 |
| L5 | 624.3 | 16.9 | 297.8 | 0.00e+00 |
| L6 | 621.1 | 16.7 | 295.1 | 0.00e+00 |
| L7 | 625.4 | 17.0 | 298.3 | 0.00e+00 |
| L8 | 624.5 | 16.9 | 295.7 | 8.53e-06 |
| L9 | 659.8 | 19.3 | 338.8 | 8.53e-06 |
| L10 | 647.6 | 18.3 | 327.7 | 0.00e+00 |
| L11 | 613.8 | 16.3 | 286.3 | 0.00e+00 |

Giây làwalltimejob2×T4, khônglatencyxe. Diagnosticgồm4fulltrainpasses/probe, táchtrain/eval; khônggọi chênhthờigian giữaaccount làcausal speedup.

## Mười metric test r10

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.910027 | 0.897038 | 0.910027 | 0.901052 | 0.885652 | 0.910027 | 0.910027 | 0.886315 | 0.910027 | 0.900878 |
| L1 | 0.910483 | 0.899102 | 0.910483 | 0.901589 | 0.885538 | 0.910483 | 0.910483 | 0.886861 | 0.910483 | 0.900991 |
| L2 | 0.895400 | 0.873057 | 0.895400 | 0.887820 | 0.880483 | 0.895400 | 0.895400 | 0.875490 | 0.895400 | 0.890620 |
| L3 | 0.895767 | 0.874836 | 0.895767 | 0.887867 | 0.880186 | 0.895767 | 0.895767 | 0.876207 | 0.895767 | 0.890755 |
| L4 | 0.909049 | 0.895684 | 0.909049 | 0.896988 | 0.883747 | 0.909049 | 0.909049 | 0.884547 | 0.909049 | 0.898094 |
| L5 | 0.909209 | 0.896675 | 0.909209 | 0.896820 | 0.883448 | 0.909209 | 0.909209 | 0.884579 | 0.909209 | 0.897812 |
| L6 | 0.914653 | 0.914318 | 0.914653 | 0.908044 | 0.882964 | 0.914653 | 0.914653 | 0.887030 | 0.914653 | 0.900822 |
| L7 | 0.914814 | 0.915988 | 0.914814 | 0.908474 | 0.883270 | 0.914814 | 0.914814 | 0.887433 | 0.914814 | 0.900650 |
| L8 | 0.908804 | 0.896343 | 0.908804 | 0.899909 | 0.881968 | 0.908804 | 0.908804 | 0.882464 | 0.908804 | 0.898307 |
| L9 | 0.909256 | 0.898883 | 0.909256 | 0.900329 | 0.882336 | 0.909256 | 0.909256 | 0.883425 | 0.909256 | 0.898347 |
| L10 | 0.914945 | 0.924740 | 0.914945 | 0.912186 | 0.877758 | 0.914945 | 0.914945 | 0.884016 | 0.914945 | 0.898564 |
| L11 | 0.915079 | 0.926008 | 0.915079 | 0.912429 | 0.878667 | 0.915079 | 0.915079 | 0.884720 | 0.915079 | 0.898432 |

Đủ10metric/round,perclass,clientmean/p10/median/min tại audit.json; train10metric tại diagnostic-audit.json. CSV `runs/l-screening/metrics-r10.csv`.

## Độ tin cậy và bước tiếp

- Seed42: r1cùngcôngthứcwarmup nhưngF1từ0.819857454 tới0.820373446, spread0.000516.
- Seed43: r1cùngcôngthứcwarmup nhưngF1từ0.817460944 tới0.818048863, spread0.000588.

Đây làchênhthực giữasession, chưaisolatenguyênnhân, khôngphảiCI/noisebound vàkhôngdùng đểhiệuchỉnhF1. Nominalseedgiống chưa phải bitexact2T4proof; lợirealgate nhỏ cầnlặpđốichứng.
Chưa50round, chỉhaiseed, officialtestđãđượcđọc thíchnghi qua nhiều batch. Khôngtuyênbốsignificance hoặc tốiưu toàncục.

Đềxuấttiếp, **chưalaunch**:

1. Giữsofttarget+fixed1 làmnền; giữsoftfixed/noReal vàaugmentation làmcontrol. Lặpđúngcấuhìnhgatefixed cùngseed đểđo độnhạy sessiontrước khi chọnlợigatenhỏ.
2. Trênnềnđó kiểm syntheticfallback: argmaxQ[y]sai→hardlabel riêngsynthetic, giữrealgate/codec/lossweight/generator. Khônghardcodeclass13.
3. Đốichứngteacheronline vspostepoch eval, giữschema/codec/weight; báotănglocalforwardtime. KhôngthêmrouterB8/anchorensembleđãFAIL đểdòtìmwin.
4. Nhánhdiagnosticđộclập fitmới foldtrain, dùngfoldvaltheotimeline chưa học/statfit đểphânbiệtshift/teacher/featureoverlap; khônggọitrainF1củaL làvalidation.
5. Xácnhận50round vàkiếntrúctrộn khi ownerchốtphạmvi; khônggộpđổi generator/codec/loss cùnglúc. Chưacóviệctaybắtbuộc.
