# Full H1/H2 — ledger

## Cập nhật 14/10 — cont01 đã terminal/audit

7 arm đã audit: DML/own r7, Gaussian full/diag/own r10, lr4 r9, local r11/50.
Cont01 đủ output, không rerun; continuation mới dùng parent cont01 và round riêng mỗi arm.
[10 metrics cuối](latest-rounds.md), [report](report.md). Target/LR50 giữ nguyên, nối theo quota/17Oct.
Raw-test-prefix/client0 inference mới64check PASS, đã gồm round cuối tới11;
[proof](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/inference-proof.json), [log](../../../../../../methods/shared-baselines/builds/federated-baselines-full/logs/inference-20261014.log), không full-inference/leakage proof.
Các ghi chú “terminal audit còn chờ”/quota156h bên dưới là lịch sử tại mốc viết,
không dùng làm trạng thái hay reservation hiện tại. [Phiên mới](../../../../../archive/operations/collection-2026-10-14.md).

## 2026-10-13 — terminal audit và continuation

Đủ4calibrationkernel/7arm r1–r3, source/data/weights/BN/payload/pred→10metric auditPASS.
Fulltest-label/orderhash và32sample-inferencechecks PASS; sample khôngfull-inferenceproof.
**Chưa r50.** [Report10metric](report.md), [sampleproof](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/inference-proof.json), [budget](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/budget-measured.json).
Grouping cònít nhất156,2065h phiên tới50, chưareload/Hybrid; livequota≈126,62h.
Owner chốt chạy theoquota, nối checkpoint sau17/10. Target50/LR50 khôngđổi.

4notebook `full1`…`full4-s42-20261013-cont01-v1` đãdựngsameowner/sameparent,
require-resume r3/source/data trướcdecode, workerbudget≤4h/parent4,8h,
W&B cùngidentity. 12controllercases vàisolatedverifierpackagePASS.
`scripts/kaggle/full_resume.py` importSHA từngfile, markerlast, từchối6corruption/missing/ambiguouscases.
Khôngsửa `mh/*.py` hoặc `proj/*.py` đãtrain.

CPU-onlypreflight4mount01 lỗiCUDAlogging; mount02 thiếu `check_v3_driver.equal` dependency.
Giữsource/log cácCPUprobe, sửa notebookpackage ngoàiscientificfingerprint;
mount03 cả4actualmount/importgate đãPASS, đủ7arm đúngr3/fingerprint.
[Proof](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/mount-proof.json). Khônghuấnluyện/GPUquota hoặcmodelnumericalfailure ởhai lỗi trên.
RegistryW&B [wandb-runs.json](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/wandb-runs.json) gắnđúngfingerprint đãaudit.
Cont01 cả4GPU notebook đãPUSHED10:04UTC+7, sameowner/parent vàsource/data/fingerprint.
API đầuRUNNING; 7/7arm đãcófreshr4finiteSGDheartbeat/applied tăng/skips trongguard.
Ngừnghealthpoll theoownerpreference; terminalnewroundaudit cònchờ.
KernelIDs/currentgate/continuationremaining: [phiên mới](../../../../../archive/operations/continuation-2026-10-13.md).

## 2026-10-12 — calibration launch

2026-10-12:7arm/100%train80%/50round/seed42, actualfulltesteveryround.
4calibration notebookversion1 PUSHED, kếhoạchlấy3roundđầu vớilịch50 đểđo budget/checkpoint.

| notebook | owner | arms | pushUTC+7 |
|---|---|---|---|
| full1-s42-20261012-cal-v1 | catbaochau | local/Gaussianfull |17:18|
| full2-s42-20261012-cal-v1 | minhtriethihi | diag/lr4 |17:18|
| full3-s42-20261012-cal-v1 | khanhmay0304 | Gaussianown/DML |17:18|
| full4-s42-20261012-cal-v1 | catbaochau | DMLown |17:27|

Public/privatekernelmetadata/source_manifest trongnotebookfolders; inputsREADY; publicdatasetsaccessPASS.
Full4 đổiowner từodixeuit sangcatbaochau trướcPUSHED vìslot/quotareserved; không đổisource/settings.
2concurrentsession/account;catbaochau19,46h đủ2×parent4,8h×1,15=11,04h.
Nhữngowner khácminhtriethihi19,40h/khanhmay030418,19h; xemquota-before-push/full4.
Plan55h chỉlậpbudgetcalibration+cácpilot, **không làmeasurement đủ50round/7arm**.
Sauactual3round cần tínhwholeremainingbudget+dư15%, Source/notebookPUSHED giữnguyên.

Local7arm×3round/4exactresume/strictBN/fullproxycodec/pred→10metric/inferencePASS,
log papers/full-driver-initial1-recheck-20261012.log,peak2759MiB.
Checkerimportmh thiếuROOTsys.path bịfail trướcdata/train; sửa checker, chạy initial1 chưa cóoutput.
Actualnotebook3-of50/deadline/privatekeyboundaryPASS: notebook-check-final-20261012.log.
Compile/eager gate vàfullcoverage chỉtrênT4 trướcactualtrain; localkhôngcompile/fulltrain.
Preparedraw/train/test arrays chỉ/kaggle/temp, outputworking chỉsource/metrics/weights/predictions.

Collector scripts/kaggle/collect_full.py, log collection-20261012.log; partial3round≠r50complete.
API17:46UTC+7 cả4RUNNING; chưapulledsource/data/compile/budgetverified.
KhôngGit/DVCadd/push/backup/cleanup. Inlinekeyprivateversionhistory theoquyềnđãchốt.

17:59UTC+7 W&B7arm đãonline/heartbeatfinite. Diag/lr4/ownGaussian committedr1,
skips≈0,03%/applied84k, đủ10metricfulltest; ngừnghealthpoll armđãổnđịnh.
Nhữngarmkhác finiteclientheartbeat; chưaclaimroundcommit/terminalproof.
Exactrunlocator/fingerprint tại [registry](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/wandb-runs.json), [snapshot](../../../../../../methods/shared-baselines/builds/federated-baselines-full/probes/live-status.json),
source/datahash/compileproof đợiterminalpull. LiveR1walltime~1900sGaussianform
chỉước sơbộ, không thayworst3round/pulledsetup/resumewholebudget.

18:19UTC+7 chỉreadexact4run chưacommitởsnapshotbanđầu; khôngpoll3armđãổnđịnh.
Local/fullGaussian đãcommitr1/full10metric, finite/applied84069/84059,skips14/24;
ngừnghealthpoll haiarm này. DML/DMLown heartbeatSGD cuối tạiclient19, finite,
chưa córoundcommit trong lầnđọc này. Sourceevalcả20private+20proxy trênfulltest,
khôngheartbeat trongeval; khoảngimlặng chưađủchứngminhfailure hoặcworkerphase.
Khôngyêucầumanualstop từAPI RUNNING/nohistory đơnlẻ; source/data/compile/terminalaudit cònpending.
