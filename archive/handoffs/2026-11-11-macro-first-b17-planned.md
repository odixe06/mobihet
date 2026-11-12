Snapshot; bàn giao sống: `docs/operations/continue.md`.

# Bàn giao sống — kiểm 11/11/2026 13:21 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](../../docs/operations/lessons.md) trước.
Đây là **bàn giao hiện hành duy nhất**. Artifact/state/process thật ưu tiên tóm tắt.

## Đã đóng — CBB vẫn phương pháp chính

- B13 audit12run/20round PASS; SQ/RK/SK/JS/LS FAIL cả2seed. [Results](../../docs/research/screening-module-b13.md).
- B14 audit8run/20round + actualwire PASS; MX/GP/LG FAIL cả2seed. [Results](../../docs/research/screening-uplink-b14.md).
  Recovery watcher đã collect/audit xong04:21; controllerstate cũ `running` stale sau reboot, không restart.
  Metadata chi tiết ở [snapshot B14](../../archive/handoffs/2026-11-11-b14-closed.md).
- **B15 SAM đóng07:28:38**, 4/4kernel version1 đủ20round, source/config/compile2T4/CM→10metric/SAMcounter PASS.
  Đã tái chạy auditor/analyzer artifact trong phiên completion, không train lại.
  Macro+.000531/+.001196, p10−.007145/−.010200, TD−.006887/−.001443; **FAIL cả2seed**, không pilot/tune/extend.
  BenignF1+.005452/+.005052 nhưng xeítmẫu giảm; trainelapsed≈1,58×. Báo riêngCBB+SAM vì fairness.
  [Results](../../docs/research/screening-sam-b15.md), [10metric/byte](../../runs/sim-kaggle-b15/report-r20.md),
  [diagnostics](../../runs/sim-kaggle-b15/diagnostics.md), [counteraudit](../../runs/sim-kaggle-b15/sam-audit.json).
- B15 job `mobihet-b15-20261111` **succeeded/exit0**, markerB15_AUDITED; không training/collector sống.
  State `/home/odixe/.codex/long-jobs/jobs/mobihet-b15-20261111/state.json`, log `runs/sim-kaggle-b15/pipeline.log`.
  Ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`, deliveryqueue đã nhận notification.
  Exactrefs/version ở `runs/sim-kaggle-b15/batch.json`.
  14file source-lock banđầu đã kiểmSHA và copy `runs/sim-kaggle-b15/frozen-source/`;
  `closure-source.json` giữ proof trước sửa docs và code launcher cho batchsau. Không ghi đè lock gốc.

## Audit lịch sử theo yêu cầu owner

[Audit v4](../../docs/research/history-audit-20261111.md), [audit legacy](../../docs/research/history-audit-legacy-20261111.md),
[index SHA nguồn còn local](../../runs/history-audit-20261111/evidence-index.json).
P20/P29 targetmean vẫn KL từngrow; chưa tìm thấy exact KL-of-mean **student** trong retainedsource.
P30 teachermean+output×input để fitINPUT khác output×output moment. Legacy thiếu rawcode/JSON, không gọi là tái audit.
**V20 temporalvalidation đã chạy** full5343/hai seed; chưa dùng quality ngoài fit để weight teacher.
R replay/margin/BNsplit/bank/Qregion/divergence/logpool đã có phản chứng; không mở lại bằng tên mới.
Secondmoment/recipient/OOF vẫn cần diagnostic trước launch, không đủ chỉ vì “chưa từng thử”.

## B16 đã đóng — không còn job/collector sống

[Thiết kế/kết quả](../../docs/research/screening-mean-b16.md): C/MK/HK×42/43,6kernel version1 đủ20round.
Source/config/panel/LR/all20CM→10metric/counter/compile2T4/cả2rank PASS; skip0,mọi clientapplied.
**MK/HK FAIL cả2seed**, không pilot/tune/group-size sweep/extend; CBB giữnguyên.
MKmacro−.001855/−.000390,p10−.026956/−.017892,benign−.004288/−.004354;
HKmacro+.000008/+.000365 không đủ+.002. [Report10metric/byte](../../runs/sim-kaggle-b16/report-r20.md).
MK benign→TD thêm694/636dòng dùFPngoàiTD giảm180/123; lớp11/xeítmẫu giảm.
[Jensen/r1](../../runs/sim-kaggle-b16/reception-diagnostics.md): meanKL≈30%rowKL ởMK r20 nhưng utilityFAIL;
actualr1CM kháccontrol trướcsoftQ, không causalclaim toàntrajectory. [CM/cohort](../../runs/sim-kaggle-b16/diagnostics.md).

Job `mobihet-b16-20261111` **succeeded09:25:04/exit0/B16_AUDITED**, notification đãđược xử lý.
State `/home/odixe/.codex/long-jobs/jobs/mobihet-b16-20261111/state.json`;
[log](../../runs/sim-kaggle-b16/pipeline.log). Ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`,deliveryqueue/queued.
Supervisor307339/worker307341/child307342 đãterminal, không restart từ notification muộn.
Exactrefs `runs/sim-kaggle-b16/batch.json`: C/MKseed42 `novahihi`, HK42/C43 `emtrangtri`, MK/HK43 `keocaosu`, tất cảversion1.
13source banđầu kiểmSHA/copy `runs/sim-kaggle-b16/frozen-source/` trước sửa docs;
`closure-source.json` và `closure.json` giữ proof. Source-lock gốc giữnguyên, nguồn đãhết ownershipworker.
Đã tái chạy auditor/analyzer artifact thực, khôngtrain lại.

## Temporal diagnostic đã có — xếp sau macro-first

[Feasibility cập nhật](../../docs/research/crossfit-feasibility-20261111.md) có **partitionthời gian thật**:
410parquettrain scan, chỉ200xe/foldtrain; early80%receive-time1.181.080row, heldout294.984row,
alltimestamp ties cùngphía; counts cộngkhớp cache từngxe/class. Source/sidecar/hash ở
`runs/temporal-feasibility-20261111/`. Không dùng row simval/officialtest, khôngtrainclassifier.
Ô≥8row/cảphía bao phủ98,3685%population, class11 chỉ81,8160%; balancedhalf99,8602% không chuyển sangtemporal.
PriorTVmedian .15931,p90 .38093; Sybil6,65→13,13%. Không dùngprecisionthô hoặc ôthiếusupport làmquality.
Fixedv4scaler chứa nestedheldout: chỉclaim model-out-of-fit **conditional on fixedpreprocessing**;
shadowgenerator/Q/BN/weights vẫn phải táchholdout. Chưaembargo/independenceproof qua client/featurehistory.

Hướngqualityngoàifit vẫn cần estimand/shadowcontrol/precisionreference/fallback/bytecost; chưatrain.
Steeringmacro mới bên dưới được ưu tiên trước, không tựghépquality vàoB17.

## Owner steering mới / B17 chuẩn bị chạy

**Macro-first:** p10/low256/TD/benign/absent report-only; audit source/config/round/finite/CM giữnguyên.
Giữ gainmacro≥+.002/hai seed vàthangsim2seed→pilot20cos50→r50two. Không sửa verdict lịch sử.
Owner chốt: chỉcần không gửi realrow, thống kê/xác suất được phép; đã cho thử tiếp nếu cócơsở.
Không tự biến câu hỏi privacy thành yêu cầu SecAgg/DP hoặc pause; rủirobootstrap giữ documented.
[Review số/curves](../../docs/research/macro-first-review-20261111.md), artifact `runs/macro-first-review-20261111/`:
Y50prior+.003345/+.002890 còn giữgaincuốipilot; W20realNCKD+.002461/+.002229 tươngđốico nhẹ.
LRr20cos50≈6,76e-4, khácsimcos20LR1e-5. Hai recipe cũ khácCBB, không cộng gain hoặc bảođảm r50/APFed.

[B17 thiết kế/gate](../../docs/research/screening-macro-b17.md):6kernel C/P/NR×42/43, **CBBkhôngSAM**, chỉDAGSNetW1.
Pshift synthetic+Qmeasurement đồngbộ; NRthêmNCKDtrênreal với binaryvaliditygate gốcW20, giữstream64khôngbank.
GiữKM16/lite6/Q8/real64/synth64/floor16/AdamW/LR, khôngrawdata/newmessagefield.
**34CPUtestsPASS**: referencegradient/Q/mask/BN/gate, CBB2roundexact, NRr1exact, P/NRfullRNGresume.
Chưa W1/compile smoke hoặcKaggle tại mốc ghi chuẩn bị này.

Worker: sourcefreeze→unit→guarded200xe W1/compile2round **P vàNR**→CM/state/counteraudit→freshquota
→push6privatepinned2T4→collect1200s→source/CM/coordinate/counter/10metric/reportaudit.
Cap2,75h/kernel,16,5kernelh+15%margin, khôngpushpartial nếu thiếuquota.
Command `SIM_BATCH=b17 /home/odixe/Tools/miniconda3/envs/nckh/bin/python scripts/run_b13_pipeline.py`.
Khiworkerstart, source-lock liệt kê13file phải readonly tới completion:
`methods/mobihet_{sim,refine,macro}.py`, `scripts/sim_fold_val.py`, `scripts/run_sim_kaggle.py`,
`scripts/run_b13_pipeline.py`, `scripts/analyse_b13.py`, `scripts/audit_sim_batch.py`,
`scripts/report_module_screen.py`, `scripts/audit_macro.py`, `tests/test_mobihet_{refine,macro}.py`,
`docs/research/screening-macro-b17.md`.

## Khi B17 completion

1. Đọcstate/log/artifact, đủ20round/source/config/LR/CM→10metric/compile2rank/skip/mọi clientapplied.
   `audit_macro.py`: actualrow/step, shiftP so globalcounts, NRkeepvector/gatedrowmass/Qlag.
2. `analyse_b13.py b17` chỉmacro làgate; `report_module_screen.py b17` vẫnbáo mọi guardphụ/byte/compute.
   Đọc r1differences, endpoint vàTB16–20, FPbenign/TD/lớp11/xeítmẫu, không causalclaim toàntrajectory.
3. Macro≥+.002 cả2seed→tựfullpilot20cos50s42; pilotmacro≥+.002→resume đúngstate2rank tới50 +s43r50.
   R50endpoint≥+.002 cả2seed mớipromote. Guardphụ report-only. FAILprimary thìdừng, khôngtune/extendtìmwin.
   NhiềuarmPASS lênriêng, khôngcombo. Chưa có fullpilot hoặcB17utility khi ghi đoạn này.

## Quyền / việc tay

Owner: đổi mọi module/compute, hỗ trợ khác kiến trúc; experiment chỉDAGSNetW1. Localtrước,
Kagglequotasẵn; byte giảm tối đa chưa capcứng; thang2seed→pilot20→r50two đãủyquyền.
Owner mới macro-first, guardphụ report-only, cho thửP/NR, thống kê/xác suấtđượcphép, không realrow.
Credential processscope; không globalcredential/Git/DVCpush/xóaartifact/hủyjobkhỏe.
Chưa cần thao tác tay. Không modelpolling khi chỉcòn chờ, completion đánhthức ownerthread.
