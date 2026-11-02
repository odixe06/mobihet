Snapshot; live handoff: docs/operations/continue.md. Không đọc mặc định.

# Continue — bàn giao sống, 01/11/2026 — P20 collect phục hồi và probe số học

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).

## Phục hồi sau reboot — hiện hành

**Finalaudit đã succeeded22:21:31+07,31run PASS_BOUNDED đủ20round/P21r0/0auditfailure chưa xử lý.**
[Closure](../research/audit-p20.md), `runs/p-study/final-audit/closure.json` và2reportSHA đã đối chiếu.
Callback92373afca9ff không launch/collect lại. Gate owner pull/audit đủ32 trước phân tích đã đạt.

**Job phân tích hiện hành `p20-analysis-20261101` đã start**, supervisor12491,
ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/p20-analysis-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/p-study/analysis/worker.log`.
Lệnh xem: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/p-study/analysis/worker.log`.
Script `scripts/analyse_p20.py` chỉ đọc31run đã audit: own-test/low256/rare/absent/pairedcohorts,
proxyteacher trên cùng diagnosticrows, slopes15→20 và byte/time; không training hoặc HPselection.
Output `runs/p-study/analysis/{analysis.json,metrics-all-rounds.csv}` và `docs/research/metrics-p20.md` do job sở hữu.
C10 một seed, carriertripwire theo thiết kế; phải inspect actualoutput trước viết kết luận/lessons.
Các job collect/inventory/finalaudit bên dưới đều là lịch sử, không tiếp tục watcher/launcher.

Owner yêu cầu pull/audit đủ32arm trước phân tích. **Collector đã pull đủ32/32**,31COMPLETE/P21ERROR;
job `kaggle-p-reboot-collect-20261101` FAILED22:12:27+07 do gate cuối phát hiện25PASS/6auditFAIL/P21chưa đủround,
không phải downloadfail. Giữ `runs/p-study/recovery-v1/{collection,terminal-audit}.json`; không restart collector hoặc training.
Callback2680c561c43f đã kiểm state/log và đủ32inventory/source,31bộCM/snapshotr1–20;
receipt `runs/p-study/callbacks/reboot-collector-2680c561c43f.json`. P21r0 là trainingfailure,6QcheckFAIL đang reaudit.
Job `p20-all-arm-audit-20261101` đã ghi inventory đủ32arm; kiểm source/CM/snapshot nhưng không thay fullscientificaudit.
Receipt `runs/p-study/recovery-v1/all-arm-inventory-audit.json`; callback muộn không chạy lại.
Callback024548a199e8 đã kiểm state/log/32sourceprovenance và inventory thực:31×20round/P21r0;
ack `runs/p-study/callbacks/all-arm-inventory-024548a199e8.json`. Finalaudit vẫn do job dưới sở hữu, không rerun.

6auditFAIL tại checkQ: P15/P17/P18/P19/P20/P29. CPUreduction trướcfp16 có thể chọn bin khácCUDA;
đã giữ auditor cũ `runs/p-study/recovery-v1/original-audit-p20.py`, sửa checker bằng intervalfp32→fp16→normalization,
không sửa source train/artifact.220storedQ qua interval;3regressiontests PASS, có injection saiQ/classidentity bị chặn.
Đây là boundedconsistency, không exactCUDAreplay. Chưa kết luận utility/performance.

**Job finalaudit `p20-final-artifact-audit-20261101` đã start**, supervisor11055,
ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/p20-final-artifact-audit-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/p-study/final-audit/worker.log`.
Lệnh xem: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/p-study/final-audit/worker.log`.
Reaudit đủ31complete20round + chốt evidenceP21r0; chỉ `runs/p-study/final-audit/closure.json` xác nhận gate cuối.
Marker `P20_FINAL_AUDIT_ALL_32_CLASSIFIED` không có nghĩa32run trainPASS. Chờ kiểm closure trước phân tích.

Probe collector `p21-probe-reboot-collect-20261101` succeeded21:19:36+07; callback7046ca7f9b6f đã audit.
T4 probe đủ1round/5343xe/84group finite,2compilegatePASS/teacherSHAkhớp; chưa tái hiện lỗi, không F1.
Receipt `runs/p-study/numerics-t4/verified-diagnostic.json`. Không chạy lại probe từ callback này.

Hai worker cũ đã mất process sau reboot; state running cũ là stale. Không gọi launcher.
- `kaggle-p-reboot-collect-20261101`: supervisor3157, state `/home/odixe/.codex/long-jobs/jobs/kaggle-p-reboot-collect-20261101/state.json`, log `/home/odixe/project/mobihet/runs/p-study/recovery-v1/reboot-worker.log`, ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue.
- `p21-probe-reboot-collect-20261101`: supervisor3692, state `/home/odixe/.codex/long-jobs/jobs/p21-probe-reboot-collect-20261101/state.json`, log `/home/odixe/project/mobihet/runs/p-study/numerics-t4/reboot-collect.log`, ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue.
Hai job reboot trên đã đóng; ledger cũ giữ nguyên. Probe đã audit; P20 finalaudit là job hiện hành duy nhất.

## Hiện tại trước reboot — lịch sử

Owner01/11 sau O ủy quyền P20:32run/16cell×seed42/43,20round/cosine50/full5343/DAGSNetW1.
Giữ model như `../fl_dagsnet_veremi_v2`; v1 chỉ tham khảo66feature/20–100client, không chuyển gain/cổng cũ.
[Thiết kế](../research/screening-p20.md), [audit nguồn](../research/prior-evidence-p20.md),
[17CMview benchmarkr20](../research/benchmark-r20-reference.md). Chưa có utilityP hoặc phương pháp tổng.

**Job cũ `kaggle-p-v1-pipeline-20m-20261101` đã FAILED16:40:30+07**, exit1,
supervisor cũ80320/worker80321.32/32pushreceipt exactversion1 đã xác minh; không gọi lại launcher.
ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-p-v1-pipeline-20m-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/p-study/pipeline.log`.
P21 ERROR đã pull:8sourceoverlaySHA đúng, rank0 compilegateNaN→eager, rank1PASS;
nonfinite private probabilities bị codec chặn trước aggregation/fitproxy round1.0round hợp lệ.
[Sự cố](../research/p20-numerical-incident.md), evidence `runs/p-study/failure-p21/`.
Snapshot16:40:24+07:31RUNNING/P21ERROR; chưa coi31run khỏe hoặc kết luận utilityteacher.

**Job collect-only hiện tại `kaggle-p-v1-recover-collect-20m-20261101` running đã kiểm**,
supervisor83826; ownerthread như trên, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-p-v1-recover-collect-20m-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/p-study/recovery-v1/worker.log`.
Lệnh xem: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/p-study/recovery-v1/worker.log`.
Worker giữ cadence1200s từ snapshot cũ, pull terminal4×8connections, auditserial;
ghi ledger cảERROR/auditFAIL rồi tiếp tục các kernel khác. Chỉ báo hoàn tất audit nếu mọi run PASS;
P21 đãERROR nên completion cuối cần inspect, không đồng nghĩa batch hoàn thành huấn luyện.
Offline kiểm complete/error reuse PASS; không query/pull lại P21 đã terminal, không launchkernel.

**Job local `p20-numerics-probe-20261101` succeeded16:46:21+07**, supervisor cũ83973,
ownerthread như trên, deliveryqueue(auto), viewer none. Probe train-only/eagerforward FP16vsFP32/seed42/43,
không thay source train; đã kiểm state/log/2fixtureSHA/8forward/BNfinite. Không tái hiện lỗi trên sm86/torch2.13.
Receipt `runs/p-study/numerics-probe/verification.json`; callback muộn chỉ acknowledge, không chạy lại.
Callback token46906843d20b đã nhận: kiểm lại report/sourceSHA,2fixtureSHA và đủ8case hữu hạn;
receipt `runs/p-study/callbacks/numerics-probe-46906843d20b.json`. Hai worker collect/T4 processidentity hợp lệ,
không remotequery/relaunch từ callback local này; kết luận vẫn chưa tái hiện lỗi P21.
State `/home/odixe/.codex/long-jobs/jobs/p20-numerics-probe-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/p-study/numerics-probe/worker.log`.
Lệnh xem: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/p-study/numerics-probe/worker.log`.

**Job T4 `kaggle-p21-numerics-t4-inventoryfix-20261101` running đã kiểm**, supervisor84642,
ownerthread như trên, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-p21-numerics-t4-inventoryfix-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/p-study/numerics-t4/worker-inventoryfix.log`.
Lệnh xem: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/p-study/numerics-t4/worker-inventoryfix.log`.
Privateprobe `accanniez/mh-p21-numerics-t4-20261101/1`, frozen `runs/p-study/numerics-t4/batch.json`;
source P không đổi, thêm capture số học round1, không own-test eval/private rows/weights output hoặc W&Bkey.
Worker refreshquota (planning5h+15%), kiểm owninventory/access → pushonce → poll1200s → pull cảERROR → source/rank audit.
Compute timeout7200s sau decode; planning5h không phải đo duration. Chỉ receipt `push.json` mới xác nhận đã push.
Receipt đã kiểm: exact `accanniez/mh-p21-numerics-t4-20261101/1` đãpush, notebookSHA khớp frozenbatch.
Job trước `kaggle-p21-numerics-t4-20261101` FAILED trước intent/push: CLI missingprivate slug trả permissiondenied,
không trả404. Đã sửa bằng ownlisting đủpage; testpagination/failclosedPASS, không nới source/quota/accessgate.
Giữ log cũ `runs/p-study/numerics-t4/worker.log`; callback cũ không relaunch. Một probe, không hai kernel.
Callback prepushfail token5b73f019bf38 đã kiểm state/log và thứ tự intent sau thời điểm job cũ đóng;
receipt bản sửa/notebookSHA/10overlaySHA khớp. Hai worker T4/collect processidentity hợp lệ.
Ack `runs/p-study/callbacks/t4-prepush-failed-5b73f019bf38.json`; không launch/queryremote thêm từ callback này.
Không modelpollwaiting, không gọi launcher/collector thứhai. Giữ kernel khác; không rerun từr1 khi chưa chốt nguyên nhân.
Source P/scientificdriver giữ frozen; script recover/collect/query/audit và probe do job sở hữu, không sửa nóng.
Owner đã yêu cầu: trainingjob local/Kaggle fail phải diagnose/fix/verify rồi recovery trong scope đãủyquyền.
Đã thêm vào skillglobal `codex-long-jobs/SKILL.md` mục Recover failed training; quick_validatePASS.
Callback mainfailure receipt `runs/p-study/callbacks/pipeline-failed-7174ef2bc89b.json`.
Theo dõi `runs/p-study/{batch.json,*.push.json,launch-complete.json,status.json,progress.json}`;
`docs/research/p20-launch.json` chỉ được sinh sau gate. Không suy jobrunning thành32kernel đã train.

[ManifestP đã khóa](../research/p20-launch.json), `runs/p-study/batch.json` SHA
`46b660866c9fa1fa5c8acbaf01bb74de13ec4b34816c299850ecb59298872308`;
baseGit8b27c02,8overlay.32account/slug/version ở manifest, không sửa/bổ sung phiên bản cũ.
Mỗiaccount2kernel/cap7h,16account; livequota và access được launcher kiểm lại trướcpush, globalCLI dinhan25 giữ nguyên.
Inlinekey chỉ notebookprivate/ignored, versionhistory remote giữ vĩnh viễn theo quyền cũ; không Git/DVC notebook/draft.

83tests PASS44,09s trên source mới; receipt `runs/p-study/tests-current.json` có SHA/XML và guard.
Preflight **`p20-local-preflight-bankguard-20261101` succeeded/exit0**, supervisor79960/worker79962,
started15:53:11+07; state `/home/odixe/.codex/long-jobs/jobs/p20-local-preflight-bankguard-20261101/state.json`,
log `/home/odixe/project/mobihet/runs/p-study/preflight-bankguard.log`; artifact16cell/SHA đã kiểm độc lập trướcpush.
Fixture2xe/173train/46test/W1/compiledAMP3r×16cell, productionserver200steps/1024queryrow;
guard sourceSHA start/end, carrierlogits/model3r bitexact, diagnostic không mutate state/RNG, strictrebuild/codec/CM.
Crossrank bank/teacher SHA là gate runtime2T4, local1GPU không thay proof đó.
Callback bankguard token452818cd556f đã nhận muộn: state/log exit0 và16cell×3r PASS;
kiểm lại16stateSHA/finite/strictrebuild cả2xe,6sourceSHA và reportSHA khớp manifest32run.
Receipt `runs/p-study/callbacks/bankguard-452818cd556f.json`; pipeline3process identity hợp lệ lúc callback đó,
không queryremote/launch/watch thêm. Chưa utilityP hoặc runtime2T4proof từ callback local này.

Các job localtrước đã đóng: quota-inventory exit0; preflight-initial exit0 nhưng source đổi sauimport nên loại launchgate,
giữ `runs/p-study/preflight-initial/invalidity.json`; preflight-final exit0/guardPASS, sau đó thêm crossrankguard.
Callback quota-inventory tokenbb0eeeed888e đã nhận muộn và kiểm state/log16account, marker/exit0;
receipt `runs/p-study/callbacks/quota-inventory-bb0eeeed888e.json`. `quota.json` đã được pipeline refresh15:58,
không gán file mới cho lượt inventory15:34.32pushreceipt hiện có đã đối chiếu; không restart/launch thêm từ callback này.
Callback preflight-initial tokene26c1a60bb4a đã nhận muộn: kiểm16fixturestateSHA/finite/strictW1rebuild và trainCM173row;
vẫn **không dùng launchgate** vì source đổi trong lúc job chạy. Receipt `runs/p-study/callbacks/preflight-initial-e26c1a60bb4a.json`.
Finalpreflight SHA khớp manifest32run; processidentity pipeline hợp lệ lúc callback cũ. Không rerun/launch từ callback cũ.
Callback preflight-final token09f93c2892f4 đã kiểm state/log:16cell PASS, source ổn định, carrier3r bitexact.
Report trước crossrankguard giữ trong draft;10fixtureSHA còn khớp,6fixture đã được lượt bankguard thay thế.
Report bankguard hiện hành khớp manifest launch và đủ16fixtureSHA; pipeline processidentity hợp lệ lúc callback cũ.
Receipt `runs/p-study/callbacks/preflight-final-09f93c2892f4.json`; không launch/watch/query thêm từ callback này.
Draft32notebook trước crossrankguard **chưa push hoặc có intent**, giữ ignored `runs/p-study/draft-before-bank-guard/`;
chứa inlinekey như mọi private notebook, không Git/DVC. Callback muộn của các job này không được launch lại.

P lưu artifact ở `artifacts/p20`, source/config/CM/teacherchain/aggregate/replica audit bằng `audit_p20.py`.
Không fullprivatebank/rawpredictionaudit; không DP/contact/mixedarchitectureproof. Giữ fullN/O/P local vì chưa backup.

**O0–O7 exactversion1 đã collect/audit đủ20round; kiểm độc lập và phân tích đủ hai seed hoàn tất.**
[Kết quả O](../research/results-o20.md), [10metric](../research/metrics-o20.md), [thiết kế khóa](../research/screening-o20.md).
Không còn worker/kernel O đang chạy. Không chốt phương pháp tổng hoặc tự launch thêm/50round.

O: realKD dưới cùngteacher tăng pooled/other15 nhưng vẫn hại timeDelay, xe yếu chưa hưởng đều;
fullQ hơn uniformoffdiagonal/permutation pooled cả2seed; teachertrajectory cũng quan trọng so Nlive.
R1 vẫn học r15→20. Số đo/giới hạn chỉ ở kết quả; [registry bước tiếp](../research/diagnostic-next.md).

N0–N7 và M0–M15 đã đóng, [N20](../research/results-n20.md), [M20](../research/results-m20.md).
Backup/archive/cleanup trước N đã xác minh; **N/O/staging mới chưa DVCbackup, code/docs mới chưa commit/push**.
Giữ local, không xóa payload hoặc ghi đè manifest backup cũ bằng workspace partial.

## Hai job O đã đóng — không restart/poll/collect lại

Cùng ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue(auto), viewer none.
Callback có thể đến muộn: cảhai state/log/artifact đã xử lý trong phiên completion early, không retry delivery hoặc rerun.

- Main **kaggle-o-v1-watch-collect-20m-20261101**: succeeded14:43:55+07,exit0,
  marker `O20_LAUNCH_COLLECT_AUDIT_DONE`; supervisor cũ47305/worker47307. Startup8/8PASS10:58:51,
  exactterminal8/8COMPLETE14:33:55, sau đó fullaudit PASS. Status không thay audit, đã kiểm cảhai.
- State `/home/odixe/.codex/long-jobs/jobs/kaggle-o-v1-watch-collect-20m-20261101/state.json`;
  log `/home/odixe/project/mobihet/runs/o-study/watch-collect.log`.
- Early **kaggle-o-v1-early-pull-audit-20261101**: succeeded14:42:59+07,exit0,
  marker `O20_EARLY_PULL_AUDIT_DONE`; supervisor cũ69420/worker69424. SubsetO0/O1/O3/O4/O5/O7 đủ20round.
- State `/home/odixe/.codex/long-jobs/jobs/kaggle-o-v1-early-pull-audit-20261101/state.json`;
  log `/home/odixe/project/mobihet/runs/o-study/early-collect-20261101/worker.log`.

## Artifact và khả năng tái tạo O

- `artifacts/o20/SLUG/SLUG`:816file/6189021370B cho8run duy nhất;160snapshot/160testCM/48trainCM,
 1632shadowprobe và48externalCMarchive. Provenance ở parent. Không fullprivatebank.
- `runs/o-study/collection/{collection,audit,diagnostic-audit,replay-audit}.json`, startup/watchterminal receipts.
- `runs/o-study/analysis/{analysis.json,metrics-all-rounds.csv,completion-verification.json}`:
  kiểm lại160snapshotSHA, mọi manifestsrcSHA, scientificclosure so N,160CM→10metric/support,48trainCM,
  rawQsource và NumPy transform/teacherlag. R0tripwire cảhai seedPASS, không significance/bitexact giữaT4sessions.
- Tái tạo: `python scripts/analyse_o20.py`, rồi `python scripts/report_o20.py`, từ root/envnckh.
  `--early` chỉ đọc subset6run. Phân tích không train, không sửa payload.
- `artifacts/o20-early-20261101`:612file/4641881084Blogical cho6run trùng trong8, không thêm experiment/seed.
  Receipt/analysis ở `runs/o-study/early-collect-20261101`. Các endpoint khớp canonical độc lập.
 4run publish hardlink atomicno-clobber, log inode riêng;2run đích main đã có nên không ghi đè.
  [Báo cáo snapshot sớm](../research/results-o20-early.md) giữ scope lúc đó, fullO là kết quả hiện tại.
- [Manifest khóa](../research/o20-launch.json), batch `runs/o-study/batch.json`, SHA
  `aaad9baddc1b91d919be0a6eb119a52aa19ce18d37ef8a26a2921f0aa6beaad0`; baseGit8b27c02,
 7overlay gồm5trainingfile byte-identical N + subclass/input. DecodedinputSHA
 `79e26c53646fb7b5d7cc45b2d60690939d3a3cde5e885ad0696fff4e1d35acc8`.
- 61tests PASS34,57s và realfixture2xe/3round/4cell compiledAMP/manualinjection/strictrebuild PASS trướclaunch.
  Sau collect: source/SHA/CM/replay/pairedcontrast/tablecoverage/link/pycompile/diff checks; không sửa trainingcode.
- Packet-equivalent3333504B/xe/20round; offlinepreload3402795B gồm bootstrap+uplinks+80171Binputmodule;
  không tính hai lần cached downlink. Replay chưa liveFL/contactcostproof.
- GlobalCLI dinhan25 không đổi; W&B `21522798-uit/mobihet-v4`, runID=slug. Key chỉ notebookprivate/ignored,
  tồn tại trong remoteversionhistory theo quyền có sẵn. Không Git/DVC notebook hoặc draft private.

## M/N và backup trước N

- EvidenceM nhẹ `evidence/{m20,m20-summary,research-data}` đã backup, DVC hardlink không ghi đè.
  FullM đã cleanup, restore chọn lọc `artifacts/m20/*.dvc` nếu cần. N ở `artifacts/n20`, analysis/receipt `runs/n-study/analysis`.
- Nwatch `kaggle-n-v1-watch-collect-20m-20261101` succeeded07:41:16; marker `N20_LAUNCH_COLLECT_AUDIT_DONE`.
  Earlypreflight fail do2xe/4slotgate đã sửa, fixjob PASS; notificationmuộn không chạy lại.
- Mrecovery `kaggle-m-v1-recover-collect-audit-20261101` đã đóng đủ16×20. WatcherM trước reboot
  `kaggle-m-v1-watch-collect-20m-20261031` state stale/PID đã mất; không killPIDtáichế hoặc restarttraining. L cũng đã đóng.
- Backupjob `mobihet-dvc-backup-20261101` succeeded02:24:44,marker `BACKUP_DVC_VERIFIED`; [receipt/restore](backup.md).
  Git/bundleHEAD8b27c02,68DVCtargets đã kiểm trước cleanup65payload/1913cacheobject. Không upload lại hoặc xóaVFS/cloudobjects.
  Cloudmountcopy đã kiểm, cloudsync chưa chứng minh. Backup đó không bao phủ N/O mới.

## Bước tiếp

1. Dùng kết quả fullO, không tiếp tục coi realKD luôn hại pooled hoặc offdiagonal vôích. Chưa có winner đều cho xe yếu.
2. Khóa phép tiếp nhận KD có chọn lọc/teacherfeedback, giữ cấu trúc Q đã có utility. Gate liên tục cần controlstrength,
   postepoch M đã không hơn control; không mở lại timing hoặc đổi generator chỉ từ replaygain.
3. Thiết kế validation theo thời gian, prior/bootstrap/generator chỉ fit foldtrain; chưa đổi protocol headline.
4. Giữ20round/cosine50, seed42/43, full5343/W1; không chọn HP/best-round theo officialtest hoặc tự launch thêm.

Không việc tay bắt buộc. Không rawlocaldata/privateweights trong FLmessages; chưa DP/mixedarchitecture/contactdrop proof.
