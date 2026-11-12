> **Snapshot 13/11/2026** trước khi gọn bàn giao; không phải trạng thái hiện hành — live state ở [continue](../../docs/operations/continue.md).

# Bàn giao sống — kiểm 13/11/2026 02:30 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](lessons.md) trước.
Đây là **bàn giao hiện hành duy nhất**. Artifact/state/process thật ưu tiên tóm tắt.

## Đã đóng — CBB vẫn phương pháp chính

- B13 audit12run/20round PASS; SQ/RK/SK/JS/LS FAIL cả2seed. [Results](../research/screening-module-b13.md).
- B14 audit8run/20round + actualwire PASS; MX/GP/LG FAIL cả2seed. [Results](../research/screening-uplink-b14.md).
  Recovery watcher đã collect/audit xong04:21; controllerstate cũ `running` stale sau reboot, không restart.
  Metadata chi tiết ở [snapshot B14](../../archive/handoffs/2026-11-11-b14-closed.md).
- **B15 SAM đóng07:28:38**, 4/4kernel version1 đủ20round, source/config/compile2T4/CM→10metric/SAMcounter PASS.
  Đã tái chạy auditor/analyzer artifact trong phiên completion, không train lại.
  Macro+.000531/+.001196, p10−.007145/−.010200, TD−.006887/−.001443; **FAIL cả2seed**, không pilot/tune/extend.
  BenignF1+.005452/+.005052 nhưng xeítmẫu giảm; trainelapsed≈1,58×. Báo riêngCBB+SAM vì fairness.
  [Results](../research/screening-sam-b15.md), [10metric/byte](../../runs/sim-kaggle-b15/report-r20.md),
  [diagnostics](../../runs/sim-kaggle-b15/diagnostics.md), [counteraudit](../../runs/sim-kaggle-b15/sam-audit.json).
- B15 job `mobihet-b15-20261111` **succeeded/exit0**, markerB15_AUDITED; không training/collector sống.
  State `/home/odixe/.codex/long-jobs/jobs/mobihet-b15-20261111/state.json`, log `runs/sim-kaggle-b15/pipeline.log`.
  Ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`, deliveryqueue đã nhận notification.
  Exactrefs/version ở `runs/sim-kaggle-b15/batch.json`.
  14file source-lock banđầu đã kiểmSHA và copy `runs/sim-kaggle-b15/frozen-source/`;
  `closure-source.json` giữ proof trước sửa docs và code launcher cho batchsau. Không ghi đè lock gốc.

## Audit lịch sử theo yêu cầu owner

[Audit v4](../../archive/docs-20261113/research/history-audit-20261111.md), [audit legacy](../../archive/docs-20261113/research/history-audit-legacy-20261111.md),
[index SHA nguồn còn local](../../runs/history-audit-20261111/evidence-index.json).
P20/P29 targetmean vẫn KL từngrow; chưa tìm thấy exact KL-of-mean **student** trong retainedsource.
P30 teachermean+output×input để fitINPUT khác output×output moment. Legacy thiếu rawcode/JSON, không gọi là tái audit.
**V20 temporalvalidation đã chạy** full5343/hai seed; chưa dùng quality ngoài fit để weight teacher.
R replay/margin/BNsplit/bank/Qregion/divergence/logpool đã có phản chứng; không mở lại bằng tên mới.
Secondmoment/recipient/OOF vẫn cần diagnostic trước launch, không đủ chỉ vì “chưa từng thử”.

## B16 đã đóng — không còn job/collector sống

[Thiết kế/kết quả](../research/screening-mean-b16.md): C/MK/HK×42/43,6kernel version1 đủ20round.
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

[Feasibility cập nhật](../../archive/docs-20261113/research/crossfit-feasibility-20261111.md) có **partitionthời gian thật**:
410parquettrain scan, chỉ200xe/foldtrain; early80%receive-time1.181.080row, heldout294.984row,
alltimestamp ties cùngphía; counts cộngkhớp cache từngxe/class. Source/sidecar/hash ở
`runs/temporal-feasibility-20261111/`. Không dùng row simval/officialtest, khôngtrainclassifier.
Ô≥8row/cảphía bao phủ98,3685%population, class11 chỉ81,8160%; balancedhalf99,8602% không chuyển sangtemporal.
PriorTVmedian .15931,p90 .38093; Sybil6,65→13,13%. Không dùngprecisionthô hoặc ôthiếusupport làmquality.
Fixedv4scaler chứa nestedheldout: chỉclaim model-out-of-fit **conditional on fixedpreprocessing**;
shadowgenerator/Q/BN/weights vẫn phải táchholdout. Chưaembargo/independenceproof qua client/featurehistory.

Hướngqualityngoàifit vẫn cần estimand/shadowcontrol/precisionreference/fallback/bytecost; chưatrain.
Hai nhánhmacroB17 bên dưới đãthửvàđóng; qualityngoàifit cònopen, không tựghépnhánhFAIL.

## B17 đã đóng — không còn training/collector sống

Owner **macro-first:** p10/low256/TD/benign/absent report-only; audit source/config/round/finite/CM giữnguyên.
Chỉcần không gửi realrow, thống kê/xác suất được phép; không tựyêu cầuSecAgg/DP hoặcpause.
Giữ practicalgain macro≥+.002/hai seed vàthangsim→pilot20cos50→r50two.

[B17 kết quả](../research/screening-macro-b17.md):C/P/NR×42/43,6/6kernel version1 đủ20round.
**P/NR FAILmacro cả2seed**, khôngpilot/tune/extend/combo; CBBvẫnphươngphápchính.
Pmacro−.000658/+.000192, NR−.000164/+.001630; mọiarm dưới+.002,
không dùng bất kỳguardphụ đểveto. TB16–20 P−.000858/−.000176, NR−.000318/+.001401.
Pbenign+.00834/+.00998 nhưngTD−.03485/−.02563,class11−.00998/−.00637.
BỏguardTD không bỏ lớpTD khỏi pooledmacro16class. Không lấyaccuracy/weightedF1 tăng làprimarywin.
[10metric/byte](../../runs/sim-kaggle-b17/report-r20.md), [CM/cohort](../../runs/sim-kaggle-b17/diagnostics.md),
[mechanism/r1/counter](../../runs/sim-kaggle-b17/mechanism-summary.json).

Đã tái chạy source/config/panel/CM/LR/all20CM→10metric + `audit_macro.py`;
2T4/2rankcompile PASS,skip0,mọi clientapplied,23.692activesteps/1.476.064realrows mỗi round.
Pshift khớp globalcounts; NRgate loạiTD r2–7, r8trởđi16class giữ; lossr20≈.00373/.00374.
NRr1T4 kháccontroltrướcQ dùCPUexact; không gán mọi laterdelta cho receptionloss.
Không replayprivategradient hoặc prediction→CM toànmạng.

Job **`mobihet-b17-20261111` succeeded14:57:31/exit0/B17_AUDITED**, notification đã xử lý.
Supervisor340614/worker340619/child340620 khôngcòn PID, khôngrestart từnotificationmuộn.
State `/home/odixe/.codex/long-jobs/jobs/mobihet-b17-20261111/state.json`;
[log](../../runs/sim-kaggle-b17/pipeline.log). Ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`,deliveryqueue/queued.
Exactrefs `runs/sim-kaggle-b17/batch.json`:C/P42 `thuyanh316`,NR42/C43 `minhtran0601`,P/NR43 `odixe0502`, tấtcảversion1.
13source banđầu kiểmSHA/copy `runs/sim-kaggle-b17/frozen-source/` trước sửa docs;
`closure-source.json` và `closure.json` giữproof, source-lock gốc giữnguyên. Sourceownership worker đãhết.

## B18 đã đóng — H FAIL macro, không còn training/collector

Recoveryb18r1 **2/2kernelCOMPLETE**,8phase S/C/E/H×2seed đủ20round.
Tái audit actualsource/config/panel/partition/count/CM→10metric mọi round,2T4/2rankcompile,
source/fallback/mixedframes/globalQ/hash/lag/bootstrap/byte PASS.
H−C macro−.00106570/−.00113233, H−E−.00150351/−.00063613: **FAILcả2seed**, guardphụreport-only.
Khôngpilot/tune/extend/combo; CBBgiữnguyên. BenignF1+.00814/.00817 nhưngTD−.04715/−.03805.
ActualteacherH r19 Q[TD,benign].65748/.64706 >Q[TD,TD].31102/.32157;
matchedsource291220probe/side, mainr1CMexactC/E/H hai seed, sourceauditkhônglỗi.
[Kết quả/design](../research/screening-source-b18.md), [10metric/byte](../../runs/sim-kaggle-b18r1/report-r20.md),
[verdict](../../runs/sim-kaggle-b18r1/verdict-r20.json), [diagnostic](../../runs/sim-kaggle-b18r1/diagnostics.json).

**Job `mobihet-b18r1-20261111` failed02:08:59/exit1, completion đãxửlý**;
lỗi localanalysis thiếuexecutedHP `.txt` trongprovenancePython-only, không phải remoteloss/trainingfailure.
Phục hồi auditor bằng sealedpushednotebook+payloadSHA+remoteoverlaymanifest+4config/seed;
khôngfabricateexecutedcopy vàgiữstrict8PythonSHA. CorruptPython/config/notebookhash đều bịtừchối.
WriterNumPybool sửa sauCMauditPASS, khôngrefit. Controllerfailedstate/loggiữnguyên, `DONE`/closure riêng.
13sourcegốc SHA/copy `runs/sim-kaggle-b18r1/frozen-source/`, `closure-source.json` trướcsửa;
`source-lock.json`/batchmanifestgiữnguyên. Supervisor13486/worker13487 terminal, ownership đãhết.
Exactrefs `chuchimnon/mh-sim-b18r1-source-s42-r20/1`, `khanhmay0304/mh-sim-b18r1-source-s43-r20/1`.
Khôngpullprivateweights hoặc prediction→CM toànmạng; audit giớihạn ởsmallstate/probability/CM/source đãnêu.

### Lịch sử vận hành B18 — giữ để không restart nhầm

Owner mới: **“chạy thử nghiệm đi”**. [Design khóa](../research/screening-source-b18.md):
Sshadowearly→C/E/Hmainfulltrain, 2seed42/43, mộtnotebook/seed, mỗi phase process riêng.
H phải≥+.002macro soC và>Eso sánhmatchedsource cả2seed; guardphụreport-only.
Byte tính2bootstrap+shadowQinternal+probeQ+mainfallback, khôngmodel/rawrowuplink.
Source: `methods/mobihet_source.py`, `scripts/b18_partition.py`, `b18_phase.py`, `run_b18_seed.py`,
`audit_b18.py`, `run_b18_pipeline.py`, `tests/test_mobihet_source.py`.
Unit32test cuối/resumeproof PASS, actuallocal4phase2round200W1/compileAMP/sourceCMwireauditPASS.
Owner kiểmstatus22:17: hai exactkernel **ERROR**:
`chuchimnon/mh-sim-b18-source-s42-r20/1`, `khanhmay0304/mh-sim-b18-source-s43-r20/1`.
ĐãpullERRORcảhai, verifyprovenance8overlay; **0round/0checkpoint**.
Lỗi `/kaggle/temp/src/runs/sim/core_hp.txt` thiếu trongbundle;khôngloss/data/numericalfailure.
11sourcegốc SHAcheck/copy `runs/sim-kaggle-b18/frozen-source/` trước sửa;
`packaging-failure.json` giữproof. Không rerunlauncherbatchcũ.

**Job `mobihet-b18-20261111` state`running` stale sau reboot21:24:37**;
supervisor481998/worker482000/child482001 khôngcòn PID, khônglivecollector/trainingownership.
State `/home/odixe/.codex/long-jobs/jobs/mobihet-b18-20261111/state.json`;
log `/home/odixe/project/mobihet/runs/sim-kaggle-b18/pipeline.log`;
ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`, deliveryqueue(auto-queue), pending/stale.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/sim-kaggle-b18/pipeline.log`.
Refs bảnlỗi ở `runs/sim-kaggle-b18/batch.json`; operationalqueryreceipt `owner-status-check.json`.

**Fix/recoveryb18r1:** thêm `runs/sim/core_hp.txt` vàooverlay và source-lock.
`scripts/check_b18_bundle.py` cloneGitHEAD+declaredoverlay reproduceredFileNotFound/PASSsaufix;
8trainingoverlays khôngđổiSHA. `B18_RECOVERY=packaging1 scripts/run_b18_pipeline.py` dùngbatch/slugs mới,
rerununit+deploymentimport+auditactualsmokecũ+HPidentity, khôngtrainlocal vìpackaging-only.
Freshquota2owner×8.05h →2notebook7h →collect1200s. Freshround1 vìkhôngcheckpoint, giữmọi gate/HP.
**Jobrecovery `mobihet-b18r1-20261111` đãterminal**, supervisor13486, deliveryqueue/queued/completionprocessed,
ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`.
State `/home/odixe/.codex/long-jobs/jobs/mobihet-b18r1-20261111/state.json`;
log `/home/odixe/project/mobihet/runs/sim-kaggle-b18r1/pipeline.log`.
Theo dõi `tail -n 200 -F -- /home/odixe/project/mobihet/runs/sim-kaggle-b18r1/pipeline.log`.
Sourceownership đãhết, bảnkhóaoriginal giữfrozen-source. Refs ở `runs/sim-kaggle-b18r1/batch.json`;
khôngwatchertrùng hoặc rerunlauncher từnotificationmuộn.

## Khảo sát nguồn teacher / domain vừa hoàn tất

[Report nguồn sơ cấp + đềxuất matchedsource](../../archive/docs-20261113/research/teacher-source-survey-20261111.md),
[diagnostic](../../runs/recipient-domain-review-20261111/report.md),
[số/SHA](../../runs/recipient-domain-review-20261111/analysis.json).
`scripts/analyse_recipient_domains.py` kiểmactualV20train/internalvalCM, 10metric/ID/support/SHA,
hashsplitclient2647/2696 disjoint. Forecasthardoutput conditionaltrueclass, khôngteacher hoặc classifierutility.
V4/V5scenario/run8 từtrain ΔmacroBrier−.000132/−.000133; futurepeerdiagnostic+.002632/+.002612,
16/16class cùngchiều cả4arm. Futurepeer dùngfuturevalidationlabel của donor, **không đưa làm teacher train**.
Threshold8/32giốngnhau, full5343khôngfallback; maincountsupport riêng: sim cógroup3xe/0class11,
fullgroup ấy có53expert≥8row. Không sourceinputworld/sender/officialtest.
AlgebraBruteforce/single-groupidentity/missingdonorfallback PASS; forecastgainkhônglàmacroF1gain.

Primarysource đã sửa ambiguity CRAD: precision=inversevariance, khôngPPV;
semiparametricKD crossfitguarantee không chuyểnthẳng classQ→KM. Không copyqualityweightgrid.
Report do researchagent viết rồirootkiểmnguồn/cậpnhậtdiagnostic; không agent/jobtraining/collector sống.
Không sửa trainingmethod/launcher, không push mới. CBB/B13–B17verdictgiữnguyên.

## Việc tiếp sau B18 đãđóng

1. [Temporalqualitydiagnostic](../../archive/docs-20261113/research/crossfit-feasibility-20261111.md) đã tính lại actualV20train/internal-valCM:
   V2/V3augment,V4/V5mixedinput,full5343W1/hai seed,32.685.166/4.676.107row.
   `scripts/analyse_temporal_quality.py` kiểmr20support/10metric/SHA, khôngofficialtest hoặcfitweights;
   [report](../../runs/temporal-quality-review-20261111/report.md).
   Precisionone-vs-rest tại cùngglobaltrainprior chỉtáchbinaryprior, negativeclassmixture vẫn cóthểshift.
   Correlationinfit↔future yếu/nhạysupport, countcũng cótínhiệu vài lớp; chưa chứng minh qualityweight tăngutility.
2. B18 đãthửmatchedsource C/E/H trênCBB, đủcontrol/cùngshadow/count/evalmode vàFAILmacro2seed.
   Không mở lại H qua temperature/gate/tănground hoặc rerun training đểsửaauditor.
   Qualityweight hoặc personalizedaggregation làestimandkhác, chưa cótrial/design mới;
   cần evidence/control/rule/cost riêng trướccode/train, không countweightedpopulationmean làbug.
3. Rawv4metadata/timepartition cósidecar/counts đãđo; cachetrain **fileorder khôngtimeorder** (`core/data.py`),
   không lấy80%đầucache rồi gọitemporal. Kiểmrowidentity/label/SHA vàkhôngheldout vào shadowgenerator/Q/BN/weights.
   Fixedpreprocessing chứa nestedheldout nênclaim model-out-of-fit conditional onfixedpreprocessing.
4. Completion B18r1 đãxửlý, scientificclosure/DONE riêng vớicontrollerfailed02:08:59.
   Notificationmuộn không đượcrestart job/collector/launch. Không cótraining/collector B18 sống.
5. Đãkiểm rowmask/cachelabelSHA/count/temporalpartition, sourcepacket/teacherlag/fallbackaudit;
   measuredshadowprobe291220row mỗi nguồn, source/outsidefit conditionalonfixedpreprocessing.
   Giữ 2bootstrap+shadowQinternal+Qmeasurement+mainQfallback trongmọi so sánhcost.
   Nếu thửrecipient sauđó, dùngcùng nguồnQ đểisolate aggregation; globalfallback cho groupthiếuclass,
   không chọn8group từ futurepeerforecast hoặc gọiownerrorforecast là cross-recipienttransfer.

## Rà bão hòa CBB 12/11 — chờ owner quyết, chưa launch

Owner yêu cầu phân tích lần cuối quá trình học CBB, ưu tiên macro, cho tăng chút compute/communication.
Kết quả ở [sổ tay xây dựng phương pháp](../research/method-building-brief.md), số tái tạo `scripts/analyse_saturation.py`
→ `runs/saturation-review-20261112/analysis.json` (chỉ đọc rounds.jsonl codec-combo50, không train).
Kết luận: trong hợp đồng hiện tại hướng CBB gần hết dư địa; đường học phẳng do LR, TD 44% thiếu hụt macro.
Owner 12/11: O1 gộp phiên **giữ đóng**, O2 **giữ protocol**, O3 **không trao tham số** (model mỗi xe có thể khác hoàn toàn).
Owner 12/11: thử A1; model trộn/seed41 tạm chưa. **A1 FAIL diagnostic local** (`scripts/diag_gen_posterior.py`,
`runs/a1-gen-posterior/diag.json`, không train/Kaggle). Owner duyệt A2' (B19), yêu cầu tránh sai lầm cũ, **tạm bỏ lượng tử hóa** (giảm băng thông thử sau); generator hết khả thi thì
được đổi định hướng nếu giữ đa kiến trúc + băng thông hạn chế. [B19](../research/screening-boundary-b19.md) **FAIL gate local**
(`scripts/diag_boundary.py`, `runs/b19-boundary-diag/diag.json`, không train/Kaggle); `methods/mobihet_boundary.py` giữ làm bằng chứng,
không push. Owner duyệt thử N1 (proxy server) + N2 (GBDT): N1 trùng P20 C8/C9/C14 đã FAIL nên không code;
N2 trần HGB `scripts/diag_gbdt_ceiling.py` → `runs/n2-gbdt-ceiling/diag.json` macro .8065 < .924 FAIL. Không job sống.
Việc kế: owner chọn giữ CBB/KM16 và chuyển việc khác, hoặc nới một ràng buộc (sổ tay §5).

## Đang chạy — seed 41/43 baseline + CBB s41 (push 12/11)

Owner 12/11: thêm seed 41 và 43 cho 15 baseline FL của `../fl_dagsnet_veremi_v2` (bỏ MobiHet-proxy), thêm CBB seed 41; dùng quota còn lại,
lưu state như seed 42, hạn chế nối phiên khác account.
- **Baseline:** `../fl_dagsnet_veremi_v2/scripts/run_seeds.py`, plan khóa `runs/seeds/plan.json` (repo baseline), code HEAD `36ac7ff`
  (= SHA TinyProto r4 cho core/methods/run_fl), chỉ đổi `seed=`; TinyProto `mu=0.00015`. Đợt 1 = 28 kernel version 1 đã push.
  Đợt 2: `fl2-canperfl-s41/43` sau FedAvg (heykewtiie), `fl2-nilm-s4x-p2` nối NILM (odixe0502, nguyenminhtran060113).
  Trạng thái 19:25 12/11:
  - Đợt 1: 28/28 kernel đã pull xong. 24 kernel PASS 50 round (afpha, apfed, fd_ids, fedavg, fedproto, fedprox, fedtgp, lwfednids, perfedskd, pfedes, tinyproto, v2x_htfl; v2x_htfl n_params 461456 đã sửa auditor check); nilm s41/s43 PASS 38/39 round (budget stop chờ p2); scaffold s41/s43 NaN r15 (diverge). Báo cáo tại `../fl_dagsnet_veremi_v2/runs/seeds/report.md`.
  - Đợt 2: CANPerFL s41/s43 đã COMPLETE, pull và audit PASS_BOUNDED (sửa `audit_run.py` dùng `hp['rounds']` cho `cosine_lr`).
    2 kernel còn lại đang RUNNING trên Kaggle: `odixe0502/fl2-nilm-s41-p2` và `nguyenminhtran060113/fl2-nilm-s43-p2`.
- **CBB s41:** `scripts/run_cbb_s41.py`, `thuyanh316/mh-k-cbb-s41-r50/1` COMPLETE đủ 50 round. Đã pull vào `artifacts/cbb-s41/mh-k-cbb-s41-r50`, audit PASS 10/10 checks (`runs/cbb-s41/audit.json`). F1 macro s41 = .9077 (s42 = .9074, s43 = .9081; mean ± sd = .9078 ± .0004). Bảng 10 metric 3 seed ở [report](../../runs/cbb-s41/report.md).
- **Cập nhật Notion (12/11 19:25):** Đã tính toán trung bình 3 seed ± độ lệch chuẩn (ddof=1) và cập nhật định dạng `mean ± sd` (`0.xxxx ± 0.xxxx`) lên cả 2 tài liệu Notion cho 14 phương pháp (13 baseline + MobiHet KM16). NILM-FL giữ nguyên/khoan ghi chờ Wave 2 xong. SCAFFOLD giữ seed 42 (seed 41/43 phân kỳ NaN r15).
  - Page 1: `https://app.notion.com/p/minhtriet0502-note/50-round-with-new-dataset-5000-client-409730cee68d81f19259d14bea45ea61`
  - Page 2: `https://app.notion.com/p/minhtriet0502-note/Proposal-2-Ph-ng-ph-p-MobiHet-cho-IDS-V2X-40d730cee68d80f0a49ad050377772d2`
- Lệnh idempotent theo marker `.intent/.pushed/.done`; không push lại slug đã có `.intent`.
- **21:20 12/11:** owner bỏ SCAFFOLD khỏi cả 2 trang Notion (hàng bảng, ghi chú, mục 6, page con SCAFFOLD vào Trash).
  Lý do ghi lại: cả 3 seed đủ 50 round nhưng mất ổn định từ r2–5 (F1 ~.23 vs FedAvg .86), NaN r15; nghi SCAFFOLD + AdamW.
  NILM: `fl2-nilm-s43-p2` COMPLETE, pull rc0, đủ r1–50 finite, F1 .8846; `fl2-nilm-s41-p2` (odixe0502, version 2) W&B
  `finished` r50 lúc 21:15, kernel còn RUNNING khi ghi artifact → **không crash, không push lại**. Vòng collect nền
  (`run_seeds.py collect` 20 phút/lần, tối đa 9 lần) trong phiên Claude Code; sau ALL_TERMINAL chạy `run_seeds.py audit`
  rồi cập nhật hàng NILM-FL 3 seed lên 2 trang Notion.

## Đang chạy — B20 compute screen (Claude Code, push 12/11)

Owner 12/11: kiểm 3 bảng Proposal 2 Notion (đã sửa: compute CBB **6,09×F** thay 4,00, PerFed-SKD byte theo
`selected_frac` TB .2817, vài ô GB/lần làm tròn, số 1-seed cũ ở phân tích/mục 6, câu "trung vị 8 lớp" → trung vị 15, p10 8)
và thử mọi cấu hình giảm FLOPs giữ performance. [Design/luật khóa](../research/screening-compute-b20.md):
C/SH/SM/SD × seed42/43, sim 200 xe r20, non-inferiority macro (TB r16–20 ≥ −.0015, r20 ≥ −.003).
- Code: `methods/mobihet_compute.py`, `tests/test_mobihet_compute.py` (9 test PASS local), batch `b20` trong
  `scripts/run_sim_kaggle.py`, `run_b13_pipeline.py`, gate `analyse_b13.py b20`.
- Pipeline: `SIM_BATCH=b20 python scripts/run_b13_pipeline.py` chạy nền trong phiên Claude Code (Bash background,
  không qua codex-long-jobs). Log `runs/sim-kaggle-b20/pipeline.log`; theo dõi
  `tail -n 200 -F -- /home/odixe/project/mobihet/runs/sim-kaggle-b20/pipeline.log`.
  **8/8 kernel version 1 RUNNING** (push 12/11): C/SH `novahihi`, SM/SD `keocaosu` (s42); C/SH `emtrangtri`,
  SM/SD `minhtran0601` (s43); slug `mh-sim-b20-<arm>-s<seed>-r20`, refs `runs/sim-kaggle-b20/batch.json`, `launch-complete.json` có. Không rerun launcher khi `launch-complete.json` có;
  nếu process mất (đóng phiên/reboot) chỉ chạy `SIM_BATCH=b20 python scripts/run_sim_kaggle.py collect` rồi audit/analyse.
- PASS cả hai seed → thang: SH/SM pilot full-scale r20 cosine50 s42 vs CBB42; SD lên thẳng r50 hai seed.
- **Sim B20 đóng 12/11 ~22:00:** audit 8/8, SD PASS cả 2 seed, SM lên theo quỹ đạo (phụ lục), SH FAIL (absent −.21) dừng.
  [Kết quả](../research/screening-compute-b20.md). Smoke full-scale local `run_q20.py` 100 xe compile PASS cho SD/SM
  (subset 20 làm compile gate NaN vì gate cần 64 client/nhóm — lỗi smoke, CBB gốc cũng vậy).
- **Full-scale B20 (`scripts/run_b20_full.py`, push 12/11):** `novahihi/mh-k-b20-sd42-r50/1`, `keocaosu/mh-k-b20-sd43-r50/1`
  (cap 11,5 h, `synth_late_from=25`), `emtrangtri/mh-k-b20-sm42p-r20/1` (pilot `--until 20 --save-every 5`, cap 5,5 h).
  Base `be3b7d5` + overlay sim/refine/compute, hp CBB42/43 + knob. Refs `runs/b20-full/batch.json`.
  Vòng collect nền (Claude Code, 20 phút/lần) dừng khi SM42P terminal → `run_b20_full.py audit`;
  PASS/PASS_TRAJECTORY → nối SM42 r20→r50 (cùng owner, mẫu `resume_row` của `run_codec_combo50.py`) + push SM43 r50.
  SD: sau terminal → audit, luật r50 macro mỗi seed > .9048. Không rerun `launch` (batch đã khóa).
- **13/11 02:30 (Claude Code):** collector nền phiên cũ đã mất; kiểm tay: SD42/SD43 vẫn **RUNNING**,
  SM42P **COMPLETE** → `run_b20_full.py collect` pull rc0 (`pulled.ok`), `audit` 9/9 check PASS (`runs/b20-full/audit.json`,
  log `runs/b20-full/audit-20261113.log`): macro r20 .8962 vs CBB42 .9000, Δr20 −.0037, Δ TB r16–20 −.0039,
  slope Δ nửa sau +.00022 (closing) ⇒ **PASS_TRAJECTORY**. t_train 3,57 h vs CBB 3,61 h (FLOPs −24% không thành wall time).
  Bậc kế (chưa làm): nối SM42 r20→r50 trên `emtrangtri` + push SM43 r50 — runner chưa có lệnh resume, cần viết theo `resume_row`.
  NILM s41/s43 p2 đã audit PASS_BOUNDED 50 round (21:51 12/11); chưa xác minh hàng NILM-FL trên Notion đã cập nhật.
- **Phụ lục owner 12/11** ([addendum](../research/screening-compute-b20-addendum.md)): sau DONE chạy
  `python scripts/analyse_b20_trajectory.py b20 20`; FAIL nhưng Δmacro `closing` cả 2 seed và Δr20 ≥ −.005 vẫn lên bậc;
  luật r50 = macro mỗi seed > .9048 (giữ top 3). Combo SH+SD / SM+SD chỉ sau verdict từng arm (owner xác nhận 12/11: chờ, không chạy song song).

## Quyền / việc tay

Owner: đổi mọi module/compute, hỗ trợ khác kiến trúc; experiment chỉDAGSNetW1. Localtrước,
Kagglequotasẵn; byte giảm tối đa chưa capcứng; thang2seed→pilot20→r50two đãủyquyền.
Owner mới macro-first, guardphụ report-only, cho thửP/NR, thống kê/xác suấtđượcphép, không realrow.
Credential processscope; không globalcredential/Git/DVCpush/xóaartifact/hủyjobkhỏe.
Chưa cần thao tác tay. Không modelpolling khi chỉcòn chờ, completion đánhthức ownerthread.
