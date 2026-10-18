# Collect và kiểm dài — 2026-10-13 UTC+7

**Mốc nối sau bật lại máy:** xem [resume-after-reboot](resume-after-reboot-2026-10-13.md)
cho jobs/source/calibration mới; phần dưới giữ lịch sử trước reboot.

Owner yêu cầu kéo kernel terminal, phân tích/bài học, mở kiểm dài cho utilityPASS,
báo đủ10metric nhữnground cuối. GiữDAGSNet/no-raw-data/localbounded-only/W&Bonline;
khôngGit/DVCpush/backup/cleanup. Stable thìngừnghealthpoll.

## Chốt mới

- Chủ dự án yêu cầu dự báo **lợi hiệu năng** trước thử phương pháp; bảng dự báo/điều kiện/bác bỏ tại
  [performance-feasibility](../research/performance-feasibility-2026-10-13.md).
- Đợt cải tiến mới: BoundaryCells cả direct và teacher DAGSNet; receiver noise/profile swap chỉ local,
  cùng lớp/cùng client/khác receiver và qua fidelity gate; EMA fresh 2×2 + own controls.
  gp hiệu dụng/P2-P4/B4/edges−1/0/1/alpha1/lambda0,5 đã chốt.
  Cổng utility/cost mới đang hỏi trước pilot; probe không train làm độc lập.

- Hybrid nếupairedPASS:100%train80%/50round/seed42, đủ2×2+own7arm,
  giữtwo-DAGScomputecontrols nhưV8. Source riêng `papers/hybrid-full/proj/`;
  khôngsửaMH hoặcfullH1/H2source đãPUSHED.
- Budget:ownerchọn **chạytheoquota hiệncó, nốitiếp sau17/10**. Target50/LRschedule
  khôngđổi; mỗi phiên chỉcheckpoint continuation, khôngrestart r1.
- QuotaCLIowner-scoped snapshot08:34 khoảng126,6h/9account, refresh17/10;
  khôngKaggleMCP callable trongtoolcatalog. Globalactive vẫndinhan25, khôngswitch.
- H1/H2 summaryT4:ít nhất156,2h phiên theo groupingcalibration đểđi từ3→50,
  chưa tínhreload/setup hoặcHybrid. Mộtarm/GPU; haiarm đồngthời dùngmaxwalltime,
  khôngcộngthờigianhaiGPU rồi gọiquota. Model/source/data/pred audit vẫnlàgate trướcpush.

## Collect đang làm

API08:34:46kernel,40COMPLETE/6ERRORcũ; khôngRUNNING. KhôngrerunERROR khioutput đủ.
V7 hai seed đãAUDITED20arm×20round, report/gates mới:rank4/8 compressionPASS,
qualitypairedFAIL. Khôngfull/contact adaptation này.
V8seed43 đãAUDITED7arm×20round:utilityPASS riêngseed43; seed42đangpull/audit,
chưapairedconfirmation. V9seed42FAIL đãaudit; seed43đangpull/audit.
FullH1/H2:4calibrationkernel/7arm đềuauditDONE đủr1–r3/50.
[Report đủ10metric](../../methods/shared-baselines/builds/federated-baselines-full/results/report.md), [sampleproof](../../../methods/shared-baselines/builds/federated-baselines-full/probes/inference-proof.json).

Collectors đanghoạtđộng: `papers/v8/collection-h-20261013-range.log` (8connections),
`papers/v9/collection-a-20261013-priority.log` (4connections),
`papers/full-validation/collection-cont01-20261013.log` (chờterminal mới).
V9 cópause ngắn đểưu tiênV8, đãresume với4connections; giữmọi`.part`.
GPUauditlock dùngchung. Range-resume giữ`.part`; khôngxóaartefact.
Networkbanđầu≈1MiB/s; ownerđãđổi mạng và yêu cầu tiếp tụctải.
ReadTimeout/IncompleteRead trongpull lànetwork/artifactIO, chưatrainingfailure.

## Source/preflight đang chuẩn bị

Hybridfullcopyisolated:full_driver/full_prepare/full_verify+parent_run;
newfull_method kếthừaV8 vàfull_run dùngtwo reusablecompiledtemplates,
CEsyntheticprivate/proxy+KDreal, dummy evalnoise giữBNcontrols.
Local7arm/fiveexactresume quaGPUguard PASS, peak2913MiB;
log `papers/hybrid-full-smoke-initial1-20261013.log`. 4notebook/controller/credentialboundaryPASS.
4private source datasets `mobihet-hybrid-full-20261013-src` READY;
chưaHybrid notebookPUSHED vìchưa pairedutilitygate. Sourcecontract đãfreeze.
Owners: minhtrit06(local/DML), odixeuit(Gaussian/Hybrid), trietbackup(Gown/DMLown), odixe0502(Hybridown).

FullH1/H2require-resume/importgate đãtriểnkhai; importmarkerlast/SHA/source/data và6refusalcasesPASS.
12controllercases vàisolatedverifierimportPASS. CPUmount01 lỗiGPUlogging, mount02 thiếudependency;
mount03 cả4source/mount/importPASS, đủ7arm đúngr3/fingerprint trướcdata decode.
[Mountproof](../../../methods/shared-baselines/builds/federated-baselines-full/probes/mount-proof.json). Khôngsửa scientificmodules.

## Continuation đãpush — 10:04 UTC+7

| kernel | owner | arms | parent |
|---|---|---|---|
| mobihet-full1-s42-20261013-cont01-v1 | catbaochau | local/Gaussianfull | full1-cal r3 |
| mobihet-full2-s42-20261013-cont01-v1 | minhtriethihi | diag/lr4 | full2-cal r3 |
| mobihet-full3-s42-20261013-cont01-v1 | khanhmay0304 | Gaussianown/DML | full3-cal r3 |
| mobihet-full4-s42-20261013-cont01-v1 | catbaochau | DMLown | full4-cal r3 |

4kernelversion1 PUSHED03:04:20–03:04:47UTC; API đầuRUNNING, W&B lầnđầu vẫnoldr3,
snapshot tiếp5arm cófreshr4SGDheartbeatfinite/applied tăng/skipped≤1;
diag/lr4 vẫnoldr3, API vẫnRUNNING. Chưafullr4commit/terminalaudit.
LiveCLI `kernels logs ... --follow` (35secondstimecap, keyredacted) đãđọcđượcfull2:
W&Bauth/2×T4/import đúngr3 vàverifiedfull source/data43.045.415/10.761.343row.
ChưacóSGDheartbeat tại mốcstream này; pipdependencywarning khônglàtrainingtraceback.
Log [full2 startup redacted](../../../methods/shared-baselines/builds/federated-baselines-full/notebooks/full2-s42-20261013-cont01-v1/live-startup.redacted.txt).
Đến10:41UTC+7 diag/lr4 cũngcófreshr4SGDheartbeatfinite, client13/applied1024/2048,
skipped0. **7/7arm initialprogresshealthy; ngừnghealthpoll theoownerpreference.**
Snapshot [continuation-health](../../../methods/shared-baselines/builds/federated-baselines-full/probes/continuation-health-20261013.json).
Parentbudget4,8h/worker≤4h, stop tạicompletedboundary, fixedtarget50/LR50.
[Quota trướcpush](../../../methods/shared-baselines/builds/federated-baselines-full/probes/quota-before-cont01.json):cat15,06h,
minhtriethihi17,73h/khanhmay15,34h; reserve2/1/1sessions×4,8h×1,15 phùhợp.
W&B locators/fingerprints giữcũ: [registry](../../../methods/shared-baselines/builds/federated-baselines-full/probes/wandb-runs.json).
Inlinekey chỉignoredprivate_push0600/privateversionhistory, khôngpublicsource/chat/log.

Read-onlywatcher `scripts/watch_current_reports.py` tựrebuildreport/index khiAUDITEDset đổi;
log `docs/operations/report-watch-20261013.log`. Watcher khônglaunch/stop remotework hoặcđổi scientificgates.
Source/notebookPUSHED khôngsửa; Git/DVC/backup/cleanup vẫnkhôngthực hiện.

Mốc cuối: V8seed42≈5,438GiB/V9seed43≈4,179GiB đãtải nhưngchưaAUDITED;
collectors vẫnchạy, khôngcoi sốbyte hoặcstatusCOMPLETE làfullpull/source/modelproof.
Sessionlocal:V8collect34351,V9collect40233,fullcollect70738,reportwatch64697.
KhôngW&Bhealthpoll tiếp vì7/7arm đãqua initialcheck; terminalcollector vẫnhoạtđộng.

## Cònthiếu

**Cập nhật 15:24 UTC+7:** các pending collect V8/V9/full cont01 liệt kê dưới đây đã xong/audit.
V8 paired PASS, V9 paired FAIL. Hybrid full đã PUSHED 4 notebook và qua initial health;
BoundaryCells calibration cả hai seed đã audit/cost PASS. Trạng thái mới nhất tại
[nối phiên sau reboot](resume-after-reboot-2026-10-13.md),
[10 metrics cuối H1/H2](../../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md).

Đợt mới ngoài continuation: [dự báo và thứ tự kiểm](../research/performance-feasibility-2026-10-13.md).
BoundaryCells bounded probe xong, cảP hơnown/shuffle; P4target mềm/F1thấp nên chưa dự báo cao.
ResponseRisk toyextractor+40.960row swap probe PASS cho closure/policy; chưa physicalcausal/utilityproof.
Chưa notebook huấn luyện mới cho hai hướng hoặc EMA; cần khóa cổng trước preflight/push.
Quota survey mới ở `papers/boundary-cells/quota-20261013.txt`; không dùng sốquota này làm reservation.

1. V8seed42 fullpull/audit → pairedgate; PASS mớipushHybridfullcalibrationfirst3-of50.
2. V9seed43 fullpull/audit → pairedreport/bài học, khôngmởKD/contact từseed42FAIL.
3. Initialcontinuationhealth đủ7/7arm; ngừnghealthpoll, chỉcollect/auditterminal.
   CLIlive-stream đãhoạtđộng trongphiên; khôngmặcđịnhAPI chỉcóterminal-log.
4. Saucont01terminal:pull/auditactualnewrounds, budgetfreshquota, nextparent chínhcontinuation đãaudit.
   Khôngrestartfromcalr3 để thaylịch sử; nối sau17/10 khiquota thiếu nhưownerchốt.
   Generator hiện `cont01`/parentcalr3: đợt sau cầntag mới/parent mới vàlast-round riêng từngarm;
   armđãr50 phảigiữcomplete, khôngđòi newround hoặctrain lại. Khôngrerungenerator đểghiđènotebookPUSHED.
5. Cập nhậtLESSONS/proposal/status theoaudit mới. R50metrics chưa tồn tại; chỉbáo actualcommittedrounds.

Executed-budget notebook full3/full4 tải vàoignoredprivate_push0600,
chỉparseoutputs summary phục vụbudget, khôngpublishsource/key; khôngthay fullartifactaudit.
