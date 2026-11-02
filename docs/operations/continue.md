# Continue — bàn giao sống, 02/11/2026 — Q20 đã đóng audit/phân tích

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).

## Hiện tại và quyền

**P/Q đều đóng. R20 phase8replay đang launch/watch theo quyền owner02/11; không restart P/Q từ callback muộn.**
Owner đã ủy quyền P21seed43 recovery và16runQ20; cảhai đã đạt endpoint20/audit/phân tích.
Owner02/11 đã ủy quyền các chẩn đoán tiếp: repeat/replay trước, validation riêng sau splitgate; R preflight đãPASS, phase8replay đã giao worker launch theo gate. Không tự tănground hoặc chọn besttest. Không Git/DVCpush/cleanup ở pha này.
Giữ DAGSNetW1/full5343/20round/cosine50/train+val/seed42–43; own-test chính, F1ưu tiên, báo đủ bootstrap/two-waybyte/time.
Không rawlocalrows/features/per-rowlabels/privateweights/BN trong FLmessages; chưa formalDP/SecAgg/mixedarchitecture/contactproof.
Credential Kaggle process-scoped, không thay global login; globalCLI quan sát09:52 làheykewtiie (đổi ngoài pipeline).
Owner chạy vẫn16account đãủyquyền theo Rquota; W&B21522798-uit/mobihet-v4, runID=slug.
Key chỉ notebookprivate/ignored, remotehistory giữ vĩnh viễn theo quyền cũ; không chép vào Git/DVC/docs.

## Q20 — closure và kết luận hiện hành

[Thiết kế khóa](../research/screening-q20.md), [manifest](../research/q20-launch.json),
[kết quả/bước tiếp](../research/results-q20.md), [đủ10metric320row](../research/metrics-q20.md), [byte/time](../research/cost-q20.md).
`runs/q-study/batch.json` SHA254520203a672ab109b8801660eb251d31e575fb6bfba193e7a58bd6686cf1bf;
baseGit8b27c02 +11runtimeoverlay.16exactversion1/8cell×2seed, không trainingfailure hoặc rerun.
`closure.json`:16PASS_BOUNDED; `closure-verification.json` khóa1920artifactfile/48sourcefile mỗi run,
320testCM/320snapshot/96trainCM. Analysis đọc actualCM/support/source/SHA,18contrast và4crossbatchreferences.
`runs/q-study/analysis/{analysis.json,metrics-all-rounds.csv,costs.csv,completion-verification.json}`.

Querycontrol−stream qua tripwire nhưng fullremote không bitexact; không dùng negativecount làm causalKDfailure/noisebound.
Mixedsupport riêng chưa hơn ổn định; inputtarget nâng weakcohort; sqrtinputweight chưa giúp.
Margin nâng pooled/mean/p10 nhưng hại absent/timeDelay, thêmbyte; chưa method tổng hoặc thắng benchmark.
Xe nhỏ vẫn tăngr15→20, không plateau chung. Các con số/inference/falsifier duy nhất ở results Q và LESSONS.

Job hoàn tất `kaggle-q-v1-auditfix-collect-20m-20261102` succeeded04:47:49+07,
supervisor36858, ownerthread01a18d65-e3b8-73f1-80b6-7b66531915da, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-q-v1-auditfix-collect-20m-20261102/state.json`;
log `/home/odixe/project/mobihet/runs/q-study/audit-recovery-20261102/collect-only.log`.
Xem log đã đóng: `tail -n 200 -- /home/odixe/project/mobihet/runs/q-study/audit-recovery-20261102/collect-only.log`.
Callback801309abef28 đã xử lý: kiểm state/log/closure/source/snapshot/report, tiếp phân tích đãủyquyền.
Ack `runs/q-verification-20261101/callback-801309abef28.json`; không launcher/collector mới.

Job Q cũ `kaggle-q-v1-watch-collect-20m-20261101` failed04:19:23 ở auditor, không trainfailure.
Q5/Q6/Q9 pullrc0/COMPLETE, CPU/CUDA normalize lệch2.384e-7 làm atol2e-7 falseFAIL.
Giữ log/checker/progress/attention tại `runs/q-study/audit-recovery-20261102`.
100snapshot khớp decodeCUDA local; auditor dùng bound sumfp32/division theo từng storedfp16entry;
7regressiontests gồm targetcorruptionPASS. Tái audit5run rồi collect-only11run còn lại; source/HP/artifact không đổi.
`repair-verification.json` và callback186bd587648d đã kiểm; không chạy lại repair.py trên progress mới.

## P21 recovery và P20 gốc

`accanniez/mh-p21-hybrid-s43-recovery1/1` P21R1/logicalP21/C10seed43 đã audit20round,
fresh r1/cosine50/full5343/W1, original8overlay+guard/sameHP. [Kết quả bổ sung](../research/results-p21-recovery.md),
[10metric](../research/metrics-p21-recovery.md), [manifest](../research/p21-recovery-launch.json).
`runs/p-study/recovery-p21/complete.json` và `analysis-p21-recovery/verification.json` khóa actualsource/CM/guard/20proxystrictrebuild.
Job `kaggle-p21-recovery20-20261101` succeeded03:35:00+07/supervisor18746;
state `/home/odixe/.codex/long-jobs/jobs/kaggle-p21-recovery20-20261101/state.json`,
log `/home/odixe/project/mobihet/runs/p-study/recovery-p21/worker.log`.
Callbackc231b56a318f đã ack `runs/p-study/callbacks/recovery-p21-c231b56a318f.json`.
Hybrid hơn proxythuần pooled hai seed, vẫn dưới streamingclassQ; chưa numericalrootcausefix hoặc proxyteacher mặc định.

OriginalP0–P31 exactversion1 giữ31PASS_BOUNDED/P21r0; [audit](../research/audit-p20.md),
[kết quả](../research/results-p20.md), [10metric](../research/metrics-p20.md), [chi phí](../research/cost-p20.md).
`runs/p-study/final-audit/closure.json` và `analysis/completion-verification.json` giữ nguyên31arm/620round/33contrast,
11SHA báo cáo/source. Supplement recovery thêm20round/8contrast, combined32logicalarm/640round;
không ghi đè originalP21FAIL hoặc gọi33attempt là33logicalarm.
Originalfailure/probe: [incident](../research/p20-numerical-incident.md), `runs/p-study/failure-p21/`;
không checkpoint hoặc tái hiện NaN gốc. Guard/recoveryPASS không chứng minh rootcause đã sửa.

## Nhận diện callback muộn

Mọi job cùng ownerthread ở trên; state chuẩn `/home/odixe/.codex/long-jobs/jobs/JOB_ID/state.json`.

| job | trạng thái đóng | log trong repo |
|---|---|---|
| q20-quota-inventory-20261101 | failed trước push, CSVref rỗng; đã sửa | runs/q-study/inventory.log |
| q20-quota-inventory-fix-20261101 | succeeded23:42:59/01-11, quota đã dùng launch16 | runs/q-study/inventory-fix.log |
| q20-local-preflight-20261101 | succeeded23:38:52/01-11,2seed×3r/35test | runs/q-study/preflight.log |
| p21-recovery-preflight-20261101 | succeeded23:10:11/01-11,3r finitepathbitexact | runs/p-study/recovery-p21/preflight.log |
| p20-analysis-20261101 | succeeded22:25:57/01-11 | runs/p-study/analysis/worker.log |
| p20-final-artifact-audit-20261101 | succeeded22:21:31/01-11 | runs/p-study/final-audit/worker.log |
| p20-all-arm-audit-20261101 | succeeded22:13:31/01-11 | runs/p-study/recovery-v1/all-arm-audit.log |
| kaggle-p-reboot-collect-20261101 | failed gate25PASS/6checkerFAIL/P21r0; finalaudit đã đóng | runs/p-study/recovery-v1/reboot-worker.log |
| p21-probe-reboot-collect-20261101 | succeeded21:19:36/01-11, probe1r khôngbenchmark | runs/p-study/numerics-t4/reboot-collect.log |
| p20-numerics-probe-20261101 | succeeded16:46:21/01-11 | runs/p-study/numerics-probe/worker.log |
| kaggle-p-v1-pipeline-20m-20261101 | failed P21ERROR, đã collect đủ32 và recovery riêng | runs/p-study/pipeline.log |

Qacks ở `runs/q-verification-20261101`:b0681edf34a5/1459c45d3306/706c1eeb91c2/186bd587648d/801309abef28.
Packs ở `runs/p-study/callbacks`:7174ef2bc89b/46906843d20b/5b73f019bf38/7046ca7f9b6f/2680c561c43f/
024548a199e8/92373afca9ff/955ad63fc9c0/4b9b74d31930/c231b56a318f.
Prebootwatchers `kaggle-p-v1-recover-collect-20m-20261101` và `kaggle-p21-numerics-t4-inventoryfix-20261101`
đã mất process; runningstate cũ là stale. Không restart từ notification của chúng.
M/N/O cũng đã đóng: canonical fullO=`artifacts/o20`8run, không subsetearly6 như runmới; result/receipt active vẫn giữ.

## Bước tiếp và backup

1. Dùng kết quả Q để khóa chẩn đoán tiếp: repeatedcontrol/spreadpercar, receptionreplay cùngQ/inputhistory,
   validation train-only sạch trước qualityweight. [Chi tiết/falsifier](../research/results-q20.md),
   [pipeline đề xuất](../research/proposed-method-p20.md). Không gọi negativecount thành causalharm hoặc tune bằng officialtest.
2. Nhánh originalfoldtrain→val cần mọi control cùngsplit và bootstrap/scaler/generator fitfoldtrain;
   chưa thay headline train+val riêng một method hoặc launch D/reliability/memory/gradientprojection.
3. Giữ output/failedsource/báo cáo gốc. Không Git/DVCpush hoặc xóa ở pha phân tích hiện tại.
   N/O/P/Q/P21 mới chưa backup, không dvcadd snapshotpartial hoặc dọnlocal.

Backup cuối đãkiểm: `mobihet-dvc-backup-20261101` succeeded02:24:44/markerBACKUP_DVC_VERIFIED,
Git/bundleHEAD8b27c02,68DVCtarget; [receipt/restore](backup.md). Không bao gồm N/O/P/Q mới.
Không việc tay bắt buộc hiện tại. Khi có run mới đãủyquyền, codex-long-jobs workerlocal poll1200s,
verifyrunning một lần rồi kết thúcturn; không modelpoll/sleep chờ hoặc duplicatewatcher.

## Job hiện tại — R20 phục hồi packaging + phase8replay

Owner02/11: tiếp tục các kiểm nghiệm, cho phép nối/xếp task tự động khi task cũ hoàn tất; sau đó bổ sung tài khoản và cho phép push kernel còn thiếu.
Thiết kế [R20](../research/screening-r20.md); R0–R3repeat đãpush đủ4version1 trên tài khoản mới, receipt/SHA đãkiểm. [V20](../research/screening-v20.md) đãpreflightPASS,
phase8 đãpush đủ8kernel version1, receipt/source/notebookSHA đãkiểm; worker startup/watch/pull/audit1200s.
Priority: đủ4reception×2seed replay, rồi completion tiếp quota/gates cho các phase tiếp; không hỏi quyền đãcó lần nữa.
Nối session không reset quota tuần. Partialrun cần fullcheckpoint hai rank; smallsnapshot không đủ.

**Job sống:** `kaggle-r-input1-stagefix-pipeline-20m-20261102`, supervisor94412, running đãkiểm,
ownerthread01a18d65-e3b8-73f1-80b6-7b66531915da, deliveryauto/queue, viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-r-input1-stagefix-pipeline-20m-20261102/state.json`;
log `/home/odixe/project/mobihet/runs/r-study/input1-stagefix-pipeline.log`.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/r-study/input1-stagefix-pipeline.log`.
Preflight45+7/privateinput8/all-ownerlaunchgates đãPASS; **push đủ8** lúc09:39:23, receipt đãkiểm.
Hiện worker collect-only1200s/exactversion1 →20roundaudit; chưa scientificcompletion hoặc T4compilePASS.
Không modelpoll; jobrunning chưa chứng minh kernels đãpush/compilePASS.
`runs/r-study/launch-complete.json`, `*.push.json`, `status.json`, `startup/startup.json` là actualreceipts tiếp theo.
Manifest [Rinput1](../research/r20-input1-launch.json), batchSHAd1448f72be56c785da61e97a76674a4405f9142ecf475cbd72d78697257afb0c;
8pushreceipts khớp ref/notebookSHA/privateinputdownloadSHA. Không dùng oldmanifest đểcollectnewslugs.
**Job sống thứhai:** `kaggle-v-v1-authfix-pipeline-20m-20261102`, supervisor105477, running đãkiểm.
Ownerthread01a18d65-e3b8-73f1-80b6-7b66531915da, auto/queue, viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-v-v1-authfix-pipeline-20m-20261102/state.json`;
log `/home/odixe/project/mobihet/runs/v-study/authfix-pipeline.log`.
Lệnh: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/v-study/authfix-pipeline.log`.
Stage: freshquota8owner ngoài R/privateNotebook/sourcefreeze/all-owner gate đãPASS; push8version1 đãkiểm, worker startup/watch/pull/audit1200s.
Cycle3 lúc10:38: V0/V1ERROR đãpull56selectedfile/run;6kernelV2–V7RUNNING tại snapshot đó.
Guard`skipped_frac==0` ném lỗi sau2rankcompilePASS, chưa round1CM/commit/fullresumeverified; [incident](../research/v20-skip-incident.md).
Không biết fractionVfailed từlog; không nới gate hoặc gọi warm-up. Originalsource/manifest/batch giữ nguyên.
Receipt callbacksadfeef3c6846/56a228ce9768 đãghi; không retrynotifier hoặc restartwholeV từ callbackV1 muộn.
V0/V1CE+LA cap4.1h; V2/V3augment/V4/V5mixedinput/V6/V7margin cap4.6h; reserve15%, wholephase41.17h.
Đối chứng chung originalfoldtrain32.685M→val4.676M, không officialtest/external/qualityselection hoặc headlineprotocolchange.
Manifest [V20](../research/v20-launch.json), batchSHA446c82594819a647ff5957d4a455845d9e57d2d2acf6a672afbdad1a5abfe9b5.
`runs/v-study/launch-verification.json`: đủ8pushreceipt khớp ref/notebook/metadata/sourceSHA; chưa startup/T4 hoặc20roundauditPASS.
GlobalCLI before/afterlaunch cùng`emtrangtri`; process-scoped8owner đượcserverintrospect, không đổi global để chạy.
Không startduplicate hoặc sửa source V trong khi worker sởhữu.
ERROR/auditfailure gửi attentioncompletion riêng nhưng mainworker giữ healthy kernels. Không restart wholebatch từ callback.

**Job sống thứba:** `kaggle-r-repeat4-pipeline-20m-20261102`, supervisor125602, running đãkiểm, ownerthread01a18d65-e3b8-73f1-80b6-7b66531915da,
deliveryauto/queue, viewer none. State `/home/odixe/.codex/long-jobs/jobs/kaggle-r-repeat4-pipeline-20m-20261102/state.json`;
log `/home/odixe/project/mobihet/runs/r-repeat/pipeline.log`.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/r-repeat/pipeline.log`.
Phase4R0–R3repeatA/B×seed42/43: 20round/cosine50/full5343/W1, exactQ0/Q1HP + factory`r_kind=repeat`.
Cap5.3h/run +15%reserve, toànphase24.38h. Sáu tài khoản mới30h/owner/không activeunknown lúc10:21/02Nov:
chuchimnon/emtrangtri/heykewtiie/keocaosu/nguyetquedo/novahihi. Owner đãmở rộng giới hạn16account cũ.
Wholephasequota/identity/access/ownedslug gates trước push; không consume16owner R/V hoặc attach teacherhistory.
Batch/watch/receipt `runs/r-repeat`; [manifest](../research/r20-repeat-launch.json); artifact`artifacts/r20` theo slugriêng.
Đãpush4version1: R0chuchimnon/R1emtrangtri/R2heykewtiie/R3keocaosu, suffix`repeat-a-s42/a-s43/b-s42/b-s43`.
`runs/r-repeat/launch-verification.json` xác nhận4ref/notebook/metadata/sourceSHA; activeR/Vsource/batch không đổi.
Startup0/4READY tại vòng đầu, chưa remotecompile hoặc20roundauditPASS. Nguyetquedo/novahihi đãprobe xong và dùng cho CErecovery dướiđây.
`repeat20-new-account-inventory-20261102` supervisor124282 succeeded10:21:19; callbackdcb1b6228075 đãreceipt,
log `runs/r-repeat/inventory.log`; không inventory/launcher trùng từ callbackmuộn.
W&B online/private inlinekey tiếp tục theo quyền đãcó; user đãđược nhắc versionhistoryprivate giữ key.
Không modelpoll; worker1200s giữ watch/pull/audit20round và completion đánh thức thread.

**Job sống thứtư:** `kaggle-v20-ce-guard-recovery2-20m-20261102`, supervisor145226, running đãkiểm,
ownerthread01a18d65-e3b8-73f1-80b6-7b66531915da, deliveryauto/queue, viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-v20-ce-guard-recovery2-20m-20261102/state.json`;
log `/home/odixe/project/mobihet/runs/v-study/recovery-ce/pipeline.log`.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/v-study/recovery-ce/pipeline.log`.
Worker freshquota/slot/access→freeze/push2freshCE20round→watch1200s→pullERROR/COMPLETE→audit.
V0R/V1R logicalparents V0/V1, ownernguyetquedo/novahihi ngoài activeR/V/repeat; cap4.1h/run +15%reserve→9.43h.
Core/data/HP/loss/AMP/fixedscale4096khôngđổi; guardcommon≤1%global +mọixeapplied +finiteweights/BN/loss, telemetrymỗiround.
[Thiết kế/quyếtđịnh](../research/v20-skip-incident.md), [manifest](../research/v20-ce-recovery-launch.json).
Không checkpoint tươngthích, chạyfreshkhôngresume; giữoriginalV0/V1ERROR/batch/source, không restart6Vkhác.
Source15overlay base8b27c02; artifact`artifacts/v20-recovery`, batch/receipt`runs/v-study/recovery-ce`, auditorriêng.
Localjob`v20-ce-recovery-preflight-20261102` supervisor143570 succeeded11:14:03, callbackc6a5d1254d3c đãreceipt:
2seed×3round privateweights/BN/RNG/info bitexact, state/load/strictW1/diagnosticsPASS,14unit +5telemetryauditcasePASS.
Đãpush đủ2version1, receipt/notebook/sourceSHA đãkiểm: `nguyetquedo/mh-v0-val-real-s42-guard1/1`,
`novahihi/mh-v1-val-real-s43-guard1/1`; `runs/v-study/recovery-ce/launch-verification.json`.
Không lặp preflight/launcher từ callbackmuộn; chưa remotecompile/startup hoặc20roundauditPASS.

Job đóng `kaggle-v20-skips-probe2-20m-20261102`, supervisor135268 succeeded11:10:58, callbacke67a5141d956 đãverify:
`nguyetquedo/mh-vnum-ce-s42/1`, `novahihi/mh-vnum-ce-s43/1` source/fold/count/compile2T4/actualartifactPASS.
Seed42applied513365/skip3; seed43applied513363/skip5; đều513368step/full5343, mọi xeapplied, weights/BN/lossfinite.
Một xe193row/seed43 mất1/4step; không gọi skip vôhại, GradScalerwarm-up hoặc xácđịnh overflowrootcause.
[Probe manifest](../research/v20-numerics-launch.json); `runs/v-study/numerics-guard/audit.json` đãreaudit,
callback/verification riêng, không gọi probe20roundutility hoặc resume. Không thêm watcher/relaunch từ callback đãđóng.
Observerlocalpreflight133479 succeeded10:45:53,2seed×3round bitexact +2unit; callback3ec5f5ff140a đãreceipt.
Giữ3watchworker R/V/repeat hiệncó; originalV phải vẫn ghi2failed, recoveryaudit xong mới ghép logicalcell.

Attempt R đầu `kaggle-r-v1-replay8-20m-20261102` supervisor81834 failed09:17:45 ở SaveKernel400, **không trainingfailure**.
Recovery ownedlisting đủpage8owner xác nhận **không slug nào được tạo**, `runs/r-study/push-recovery/recovery.json`.
Giữ oldmodule/notebook/intent/log/batch/launchgate ở `runs/r-study/push-recovery/attempt1` và oldnotebookfolders ignored.
Originalmanifest [r20-launch](../research/r20-launch.json) chỉ rejected packagingattempt, không actualtrainingreceipt.
Không đổi model/HP/history/round; decoded teacherSHA108d314e8fc6e12393a1e5c8a6abe0ba9b6b20f4307f8e3ec68aa89e136db920 giữ nguyên.
Module4918846B → wrapper7368B +privateblob3684087B (decoded3993600B), nguồnQ6/Q7 train-only, không privateweights/BN/realrows.
Chưa xác định chính xác serverlimit400; inputtransportfix phải qua API/runtimegate, không suy utility.
`r20-push-packaging-recovery-20261102` supervisor86948 succeeded09:25:57; log `runs/r-study/push-recovery.log`.
Callbackde290fefe349/79f483a6c2e7 đãreceipt trong `runs/r-study/`, không retryoldlauncher.

Preparation cũ đóng: `r20-quota-inventory-20261102` supervisor76563 succeeded09:06:26;
`r20-local-preflight-20261102` supervisor79066 succeeded09:11:04; callbacks39659aa21656/626311f79119 đãreceipt.
Oldpreflight realW1/2seed×3round manualreplay/controlsbitexact; packingwrapper đã đổi nên pipeline bắt buộc chạy preflight lại.
State globaljobs/JOB_ID/state.json, cùngowner/delivery; log `runs/r-study/{inventory,preflight}.log`.

V20 split đã audit toàn410trainpart/contentSHA,32.685.166foldtrain +4.676.107foldval, mọi client train trướcval; không đọc officialtest.
`runs/v-study/{split-audit.json,split-verification.json,train_counts.npy,val_counts.npy}`.
`v20-split-audit-20261102` supervisor83318 succeeded; state globaljobs/JOB_ID/state.json, log `runs/v-study/split-audit.log`.
`v20-local-preflight-20261102` supervisor96260 succeeded09:41:36; log `runs/v-study/preflight.log`;
state globaljobs/JOB_ID/state.json. W1real4cell×2seed×3round/150train23val, traincounts exact/valsupport/finite/skip0,
state/load/strict395408rebuild, diagnostics RNG/state và source before/afterSHA PASS. Localsm86 không full5343/T4proof.
Callbacke77ae641a047 đãack `runs/v-study/callback-e77ae641a047.json`; không preflighttrùng.
Unitfoldrouting1PASS +5auditfoldcorruptiontestsPASS. Runtime13overlay trên Qbase, core/shared scientific source không đổi.
Callback7623100745a9 đãkiểm state/log/report/countSHA,32.685.166train+4.676.107val/5343xe khớp fulltraincounts;
receipt `runs/v-study/callback-7623100745a9.json`, không rerunaudit hoặc tựlaunchVtraining.
V20 cleanbranch giữ originalfoldtrain→val cho mọicontrol CE+LA/augmentation/mixedinput/margin, không đổi headline train+val.
Bootstrap/prior/copula chỉenv.train foldtrain; scaler gốc xác nhận fitfoldtrain.
Runtime đãpreflight, chưa remote training/auditPASS; báo kết quả V như diagnosticvalidation, không bảng benchmark own-test.

Khi callback hoàn tất: inspect state/log/artifact thật; tiếp analysis R, repeatedcontrol/validation preflight và fullphasequota
trong quyền đãcó. Nếu quota không đủ: ghi actualblocker, không hạ round/reserve hoặc chọn seed/test để lấy win.
Không Git/DVCpush/cleanup. Giữ tất cả N/O/P/Q/R/V chưabackup.

### R inputstage phục hồi — callback quota muộn39659aa21656

Quota callback đã kiểm lại state/log,16account và SHA khớp originalmanifest; receipt cập nhật, không inventory/launcher trùng.
Trong kiểm ownerjob phát hiện `kaggle-r-input1-pipeline-20m-20261102` supervisor88841 đãfailed09:31:11,
log gốc `runs/r-study/input1-pipeline.log`; state globaljobs/JOB_ID/state.json, callback030fca769766 đã xửlý.
Preflight mới45test+7auditcorruptionPASS,3round×2seed/control+manualreplayPASS, sourceSHA không đổi.
Lỗi private dataset readiness403 ngay sau create. Read-only fullownedlisting+metadata xác nhận6input đãtạo đúngowner/private,
2owner chưacreate/khôngintent; không kernel R hoặc version mới nào đượcpush ở attempt này.
Evidence `runs/r-study/push-recovery/input-inspection/`; oldstage/log giữ `input-stage-first-failure/`.
Sửa `stage_r20_input.py`: kiểm create-result, readinessread403/404/transient retrybounded, không recreate khi intent có.
3test delayedvisibility/exhaustion/unexpected401PASS, `runs/r-study/input-stage-tests.xml`.
Job mới dùng `--reuse-verified-preflight`, kiểm lại từngsourceSHA và45+7XML khôngfail/skip trước resume staging.
Không reruntraining/preflight nặng hoặc thayHP. Nếu tiếpfailed, lấy create-result/metadata/log mới, không blindretry.

Callback79f483a6c2e7 đãkiểm thêm originalmodule→compressedblob/rawdecodedSHA và8privateinputdownloadcopy exact;
không recreate/restage/push từ callback đónggói cũ. R/Vworker hiện tại giữ quyền file; nextcallback tiếp failingstage hoặc result audit.

### V inventoryglobalchange — callback258675e0f49b đãxửlý

`kaggle-v-v1-pipeline-20m-20261102` supervisor103000 failed09:52:30 tạiassertglobaldinhan25 sau readonlyinventory.
Actualglobalfileđổiheykewtiie09:51:20 ngoài pipeline; không restore/switchlogin. Chưa notebook/batch/pushV, không trainingfailure.
8ownerquota/status đãđọc đúng serveridentity, allactive/unknown0; preserve `runs/v-study/inventory-global-change/`.
Sửa V inventory/launcher không hardcodeglobaluser: vẫnrequire savedauthorizedowner, serverintrospection/quota và scopedkaggle_as,
ghi globalbefore/after nhưng không mutation. Owner set khóa theo16account trong Rquota, không tự nhận account mới từglobalroster.
Preflight/source/scientificHP/seed/split không đổi. Authfixworker rerunreadonlyquota rồi fullphasegates trước push.
Rwatchworker không phụthuộcglobalCLI, giữ poll1200s; không sửa/cancel/relaunch healthy R.

### R8 resume round 20 — job sống kaggle-r8-resume-watch-20m-20261102

Owner 02/11 ủy quyền chọn Phương án 1: Nối phiên R8 (`mh-r8-replay-margin-s42-input1`, margin seed 42) trên chính tài khoản `takids1909`.
Takids1909 có 2.10h GPU left, 0 active run, output gốc chứa đủ `state.pt` + `state_rank1.pt` sau budget stop r19.
Notebook `notebooks/mh-r8-replay-margin-s42-resume1` đã tạo với `kernel_sources: ["takids1909/mh-r8-replay-margin-s42-input1"]`,
`dataset_sources`: `veremi-nextgen-v4-scaled-per-vehicle` và `takids1909/mh-r20-frozen-teachers-20261102`.
Kernel đã push thành công lúc 14:51:36, initial status `RUNNING`.
**Job sống:** `kaggle-r8-resume-watch-20m-20261102`, supervisor PID 236422.
Log: `/home/odixe/project/mobihet/runs/r-study/recovery-r8/pipeline.log`.
State: `/home/odixe/.codex/long-jobs/jobs/kaggle-r8-resume-watch-20m-20261102/state.json`.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/r-study/recovery-r8/pipeline.log`.
Worker poll 1200s, khi COMPLETE sẽ pull output về `artifacts/r20/mh-r8-replay-margin-s42-input1`, audit đủ 20 round và ghi receipt. Không model poll trong lúc chờ.
