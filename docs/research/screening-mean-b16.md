# B16 — KL trên mean prediction cùng lớp, khóa 11/11/2026

Owner yêu cầu kiểm kỹ lịch sử rồi thử hướng có căn cứ; [audit v4](../../archive/docs-20261113/research/history-audit-20261111.md),
[audit legacy](../../archive/docs-20261113/research/history-audit-legacy-20261111.md) phân biệt target mean/P30 inputfit với objective dưới.
SAM b15 đã FAIL hai seed, không ghép SAM. CBB vẫn phương pháp chính.

## Evidence → mechanism → falsifier

ClassQ từ realrows là meanprob theo trueclass; loss hiện tại ép cùng Q lên từng syntheticrow.
P20/P29 từng dùng target mean nhưng vẫn per-rowKL. Chưa tìm thấy KL-of-mean **student** từng train
trong nguồn local được rà. B13 divergence/b14 pooling không đạt utility, không mở lại những nhánh đó.
Mean-KD thay mức ràng buộc, không bổ sung tri thức hoặc sửa fidelity/split của synthetic.

Với mỗi client riêng, 64slot có labels `[0..15]` lặp4lần:

- **C** `mean_kd=row`: `CEreal(z+LA) + mean_c mean_i KL(Q[c] || p_ci)`; delegate CBB exact.
- **MK** `mean_kd=mean`: `CEreal(z+LA) + mean_c KL(Q[c] || mean_i p_ci)`.
- **HK** `mean_kd=half`: `CEreal(z+LA) + .5 mean_c mean_i KL(Q[c] || p_ci)`.

Coefficient .5 là dose comparator cố định, không tuning/giá trị tối ưu. HK chỉ khác softKD từr2;
**r1 cả3 giữ nguyên hardCEsynthetic hệ số1**. Không mean logits, không mean giữa client, không thêm temperature.
Q targetclamp1e-6 như CBB (không silently renormalize), raw synthetic logits; Qdetach, r+1lag giữ nguyên.
Tính `log(mean p)` bằng logsumexp của4logprob, không exp→clamp khiến gradient cực trị biến mất.

Jensen `Lmean≤Lrow`: bỏ penalty phân tán cùngclass. Mean4sample nhiễu, real/syntheticdistribution khác;
không gọi đây là sửa objective đúng duy nhất. Lỗi cá thể có thể bù nhau, xe ítmẫu mất regularization.
Nếu MK chỉ giảm loss hoặc đổi benign lấy TD/p10 thì bác bỏ. Nếu MK không hơn HK,
không claim lợi riêng meanconstraint; HK không loại mọi giải thích dose/gradient khác.
Hai seed không chứng minh significance; không nhận novelty từ việc chưa tìm thấy lịch sử local.

## Cấu hình và budget khóa trước chạy

**6kernel C/MK/HK × seed42/43**, sim200 panelseed7, foldtrain1.476.064→val211.118, cosine20/r20.
Mỗi round một epoch toàn realrows, batch64,64synthetic/floor16, KM16w0.5+lite6+Q8;
AdamW/reset/dropout/AMP/clip/LA chung, DAGSNetW1 395.408params, pinned2T4. Không officialtest.
Chỉ losssoftKD thay; chung forward/BN/Qcollection, no extra model/backward, không gửi model/gradient/BN.
Schema probability/count độc lập kiến trúc; batch này không chứng minh heterogeneousfull-scale.

Cap **2,75h/kernel**, dự trù16,5kernelh+15%margin trên quota sống/slot trước push;
baseline ~3100strain20r, thêm logsumexp/counter nhỏ nhưng chưa có elapsedT4 thực.
Counter trên GPU, mộtallreduce/round, không hostsync mỗi step.
Không thêm field communication; byte nén có thể đổi do Qtrajectory. Báo bootstrap+hai chiều+total50round;
bootstrap sim≈1.196.032B/xe, không dùng simbyte làm fullscaleclaim.

## Gate giữ nguyên B13/B15

Audit trước utility: source/config/panel/compile2rank, tất cả20CM→10metric, cosineLR,
skip≤1%, mọi clientapplied, teacherlag, thực realrows/activesteps và Jensen counter.
Endpoint20 **mỗi seed**: Δmacro≥+.002; p10≥−.006; benignF1≥−.0005; TDF1≥−.0035;
absent≥−.01 chỉ khi arm/control đều≥2000dòng. Sim absent191dòng, low2569xe báo mô tả,
kèm low256/rare/classP/R/F1/FPbenign/TB16–20/compute/byte. Không best-round hoặc HP qua val/test.

Từng arm PASS cả2seed→fullpilot20cos50s42, so CBB cùnground/seed.
Pilot endpointmacro≥+.002; guardTB16–20 p10≥−.006,low256≥−.003,TD≥−.0035,benign≥−.0005,absent≥−.01.
PilotPASS→resume đúng2rankstate tới50 và freshseed43r50 đồngthời.
R50từngseed endpointmacro≥+.002, guardTB46–50 nhưpilot; PASS cả2seed mới promote.
Nhiều armPASS lên riêng, không tự combo. FAIL dừng, không tune coefficient/groupsize/extend.

## Localgate / audit thực

1. Reference softmaxmean FP64 và gradient; Jensen, repeatedpredictions, hard/soft/zeroQ,
   extreme logprob hữu hạn và teacherstopgrad, grouping không trộnclient/class.
2. Rowmode hai round exact CBB weights/BN/Q/loss/byte/RNG; MK/HK r1exactCBB;
   tail/floor/padded masks, realQcount và resume state+RNG.
3. Guarded200xe W1/AMP/compile2round MK và HK, có softQ consumptionr2;
   actualCMsupport/10metric/statefinite/mọi clientapplied/Jensen/rows/steps PASS.
4. Freeze code/tests/design trước localgate; quota mới→generate/hash private2T4→push6run→collect1200s→audit/report.
   Sourcefreeze và actualcounter không thay replay mọi privategradient/prediction; auditor ghi rõ scope.

Artifact dự kiến `runs/sim-kaggle-b16/`, `artifacts/sim-kaggle-b16/`.
Job/sourceownership thực ở [continue](../operations/continue.md). Chưa utility khi khóa tài liệu này.

## Đóng 11/11/2026 09:25 — MK/HK FAIL cả hai seed

Job `mobihet-b16-20261111` succeeded/exit0 lúc09:25:04, đủ6kernel version1/20round.
Đã tái chạy source/config/panel/support audit, actualcounter audit và all20CM→10metric/LR analysis.
Compile2T4/cả2rank PASS, skip0, mọi clientapplied; 23.692activeclientsteps và1.476.064realrows/round đúngcount.
Jensen violations0; r1KDsteps0, r2–20KDsteps23.692, teacherlag đúngr−1.
13file banđầu kiểmSHA và copy `runs/sim-kaggle-b16/frozen-source/` trước cập nhật tài liệu;
`closure-source.json` giữ source-lock và exactrefs, không ghi đè source-lock gốc.

| delta vs CBB cùngseed | MK s42 | MK s43 | HK s42 | HK s43 |
|---|---:|---:|---:|---:|
| macro r20 | −.001855 | −.000390 | +.000008 | +.000365 |
| macro TB16–20, mô tả | −.001685 | −.000386 | +.000232 | +.000234 |
| p10 | −.026956 | −.017892 | −.019163 | −.005993 |
| low256,9xe mô tả | −.032800 | −.038208 | −.021592 | −.010362 |
| benign F1 | −.004288 | −.004354 | −.001208 | +.000461 |
| timeDelay F1 | +.006326 | +.013556 | +.003135 | +.006329 |
| absent,191dòng mô tả | −.036649 | −.031414 | −.031414 | −.026178 |

MK FAIL macro+p10+benign cả2seed; HK FAIL macro cả2seed, thêmp10+benign seed42.
**Không pilot, không tune coefficient/group size hoặc kéo dài. CBB giữ nguyên.**
Controlmacro .914605/.913798; MK .912750/.913407; HK .914613/.914163.
Trainseconds20r C/MK/HK: seed42 3095,7/3075,2/3214,7; seed43 3098,9/3236,7/3025,5.
Elapsedratio MK .993/1.044, HK1.038/.976; nhiễu runtime không là FLOPs proof.
Bootstrap giữ1.196.032,46B/xe; hai chiều/round C≈305,65/305,19B,
MK283,72/282,53B vàHK292,66/291,58B. Không thêm packetfield; nén khác do nội dungQ.
Không lấy simbyte làm full50roundclaim hoặc coi giảmbyte đủ cho utilityPASS.

MK benign→TD thêm694/636dòng, benign recall−.00995/−.00994; FPbenign→attack ngoàiTD giảm180/123.
TD recall+.02217/+.02694 nhưng precision−.04318/−.03174. Đây là tradeoff, chưa giải gap benign tương lai.
SuddenConstantSpeed F1−.02048/−.00548; không chỉ một lớp/cohort quyết định toàn bộ verdict.
MK sốxe meanF1 tăng70/90, giảm115/95; nhóm≥4096dòng cũng giảm−.00352/−.00091.
HK giảmnhẹconstraint chưa đủgain; không suy mọi coefficient nhỏ hơn1 đều vôích.

Trong MK r20, rowKL≈.3507/.3511 vàmeanKL≈.1046/.1043 trên cùngforward/Q,
ratio≈.298/.297. Lossgiảm đúng Jensen nhưng utilitykhông tăng. HKcũngFAIL và MK kémHK;
không có gain cần gán riêng cho meanconstraint. Không claim đã đo privategradient variance hoặc chứng minh nguyên nhân nhân quả.

**Giới hạn pairing:** CPUrowmode/r1exact proof không chuyển thành T4wholetrajectoryexact.
Actualr1CM MK/HK kháccontrol khi chưa dùngsoftQ; Δmacro MK−.000818/−.002051, HK+.001451/−.002614.
R1CM L1 là thay đổi sốđếm CM, không per-rowpredictiondisagreement. Nguyênnhân numerical/RNG/runtime chưa được isolate.
Giữ verdict thực đãkhóa; không gán mọi chênh về loss, không rerun nhánhFAIL để tìmwin.
Hai seed/panelval đã đọc lặp không là significance hoặc freshconfirmation.

[10metric/byte](../../runs/sim-kaggle-b16/report-r20.md), [verdict](../../runs/sim-kaggle-b16/verdict-r20.json),
[CM/cohort](../../runs/sim-kaggle-b16/diagnostics.md), [Jensen/r1](../../runs/sim-kaggle-b16/reception-diagnostics.md),
[actualcounter](../../runs/sim-kaggle-b16/mean-audit.json).
Phân tích bổ sung tái tạo bằng `scripts/analyse_b14_diagnostics.py b16` và `scripts/analyse_mean_diagnostics.py`.
