Snapshot — trạng thái sống chỉ ở docs/operations/continue.md.

# Continue — bàn giao sống, 06/11/2026 20:51 — Y50 wave1 đóng, wave2 nối phiên

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước: [06/11 lúc launch Y50](../../archive/handoffs/2026-11-06-y50-launch.md), [05/11](../../archive/handoffs/2026-11-05-before-archive-cleanup.md).

## Quyền owner đang có

- 06/11 tối, bổ sung: được nối phiên bằng tài khoản khác khi cần; dùng checkpoint dataset private, verify state2rank/cùnground/source/config/hash; giữ cache và compile. Không mount private kernel chéo account. Worker hiện tại đang collect; chưa chuyển account.

- 06/11: thử 3 ứng viên đầu của [pipeline-review](../research/pipeline-review.md), chỉ cải thiện Core, được nối phiên quota ngắn. Chưa làm đa mô hình hay hướng mới.
- Mục tiêu đề xuất: **MobiHET Core**. Step floor hợp lệ khi bước thêm chỉ synthetic; 1 epoch dữ liệu local giữ như repo benchmark.
- Chuẩn FL: không chia sẻ dữ liệu local; server chỉ dùng tổng toàn mạng ([proposal](../research/proposal.md#quyền-riêng-tư)).
- Không đổi protocol/HP, không chọn round/HP bằng official test, không thêm run ngoài batch đã khóa, không nới gate sau khi xem số.
- Credential Kaggle process-scoped, không đổi global login. W&B online, key chỉ trong notebook private. Commit không có trailer `Co-Authored-By`.

## Job sống

Kiểm live 06/11 20:43: cả 7 kernel session1 COMPLETE. Y1/Y2 đã pull + audit r20, prior FAIL luật khóa, không nối.
Y0 đủ r50, pull + audit PASS; wave1 closure đã kiểm. Y3/Y4/Y5 audit PASS r23/r22/r18, đã push -c2 cùng owner. Y6 audit PASS r37, đang xử lý nối tiếp.
Không kết luận đủ r50 từ COMPLETE. Full state 2 rank giữ remote, không pull mặc định.

Owner phiên này cho tận dụng quota ngắn. `run_y50.py` dùng cap=min(phần còn lại, gpu_left/1,15), chỉ push khi cap ≥1,21h
(1h setup/import/I/O +0,21h/round); tối đa6session, cùng owner, giữ đích/lịch cosine50.
Quota live: novahihi1,71h, nguyetquedo1,60h, accanniez1,45h; outlookcheck3,01h dùng chung Y0/Y6, recheck trước từng push.
Cache npy public + compile giữ nguyên; Inductor tar được import từ parent, xuất lại cuối session.

Worker cũ PID91331/91332 mất sau tắt máy. Hai worker mới qua codex-long-jobs, poll1200s:

| job ID | supervisor PID | log | state |
|---|---|---|---|
| y50-wave1-recover-20261106 (succeeded20:51, artifact auditPASS) | 32373 | `runs/y-study/wave1-recover.log` | `/home/odixe/.codex/long-jobs/jobs/y50-wave1-recover-20261106/state.json` |
| y50-wave2-shortquota-20261106 | 32390 | `runs/y-study/wave2-recover.log` | `/home/odixe/.codex/long-jobs/jobs/y50-wave2-shortquota-20261106/state.json` |

Owner: Codex thread hiện tại (thread ID trong controller state); delivery `queue` về thread gốc.
Worker sở hữu `runs/y-study/wave{1,2}`, artifact Y50, notebook continuation; không chạy collector trùng.
Theo dõi: `tail -n 200 -F runs/y-study/wave2-recover.log`.
Worker local không sống qua tắt máy; kernel remote vẫn chạy. Không tự hủy job khỏe.

## Bước tiếp

1. Y50 đóng (`runs/y-study/wave{1,2}/closure.json`): chạy `python scripts/analyse_y50.py` (`keep_y50.py` tự chạy nếu đã cài), ghi `docs/research/results-y50.md`, cập nhật proposal, LESSONS, pipeline-review.
2. Cập nhật số thời gian warm start T4 của `-c2` vào skill federated-training-speed (hiện mới đo sm86).
3. Sau Y50, theo luật: Core−Q `removable` thì Core thành one-shot; ứng viên 5, 6 của review chỉ mở theo điều kiện đã ghi.
4. Chờ owner: commit, `dvc add`/push `artifacts/f50/mh-f3-core-s43-r50` và `artifacts/y50`.

## Kết quả mới nhất

[F50](../research/results-f50.md) đóng 4/4: bank trượt luật cả hai seed, phương pháp = Core. [L50](../research/results-l50.md): lossless CDF PASS, 691 KB/xe/50 round. [C10](../research/results-c10.md): boot64 PASS, Q8/combo FAIL.
Y50 oracle audit r50 PASS, macro .917581, generator_worth_it; prior FAIL r20, xem [results-y50](../research/results-y50.md).

## Việc tay cho owner

- Giữ máy bật để worker local collect/nối phiên. Kernel Kaggle vẫn chạy nếu máy tắt; supervisor local không sống qua reboot.
- Không chạy `keep_y50.py` hoặc cài cron song song worker controller hiện tại. Sau reboot, kiểm state/PID/lock rồi phục hồi qua controller, tránh collector trùng.

## Kiểm chứng phiên 06/11 buổi tối

- `tests/test_y50.py`: 6 PASS, gồm budget quota ngắn.
- `python scripts/check_docs.py`: active links PASS, archive excluded.
- Checker skill mặc định quét cả archive báo broken historical links; không sửa archive. Dùng checker active của repo như AGENTS quy định.
- Đã sửa orchestrator/auditor partial/tests; tài liệu CONTEXT, LESSONS, operations lessons/continue, screening-y50/results-y50. Không commit/push.
- Wave1 controller succeeded20:51, artifact auditPASS50; wave2 running. Chưa có bằng chứng startup continuation T4 tại lúc bàn giao. Khi worker push tiếp, kiểm log resume/cache/compile cả2rank và W&B đúng runID trước kết luận khỏe.

## Phân tích generator owner yêu cầu sau oracle

Đã tách class/cohort từ CM oracle-vs-Core r50: `scripts/analyse_generator_oracle.py`, `runs/generator-study/oracle-gap.json`. [Đề xuất](../research/generator-after-oracle.md): đo mismatch nonlinear joint rồi cân nhắc copula+pairhist calibration; chưa triển khai/launch. Không sửa worker wave2 đang sống.

## Thử generator mới — owner duyệt 06/11

Owner cho thử [pair-generator](../research/screening-pair-generator.md). Stage1 train→val local đãxong, feasibilityFAIL; chưa FLlaunch. Stage2 đối chứng2seed chỉ sau feasibilityPASS; không tự nới gate.
Job `pair-generator-feasibility-20261106`, supervisorPID48883, ownerthread `01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue.
State `/home/odixe/.codex/long-jobs/jobs/pair-generator-feasibility-20261106/state.json`; log `runs/pair-generator/probe.log`; follow `tail -n 200 -F runs/pair-generator/probe.log`.
Worker sởhữu `runs/pair-generator`, script probe đọcdata train/val, không sửa Y50. Saucompletion đọc result/gates, không coi exit0 là feasibilityPASS.

Pairprobe đãaudit artifact: [results](../research/results-pair-generator.md), TV .199→.751/ESS1–8, không TSTR hoặcFL theo gate. Owner hỏi phươngánthêm: [conditional trực tiếp](../research/generator-direct-conditional.md), đềxuất, chưa launch.

## Generator optimization — quyền tiếp tục owner06/11

Owner yêu cầu cứFAIL thìphântích vàthửcơchếmới trongcontract hiện tại; khônghỏi lại mỗiprobe. [Khóa directprobe](../research/screening-direct-generator.md). PairreweightFAIL giữevidence; directforest+conditionalGaussian đang chạy.
Job `direct-generator-feasibility-20261106`, supervisorPID54046, ownerthread `01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue.
State `/home/odixe/.codex/long-jobs/jobs/direct-generator-feasibility-20261106/state.json`; log `runs/direct-generator/probe.log`; follow `tail -n 200 -F runs/direct-generator/probe.log`.
Worker sởhữu `runs/direct-generator`, `scripts/direct_generator.py`, `scripts/probe_direct_generator.py` read-onlyđếnkhiend. Saucompletionauditresult; FAIL→diagnoseclass/S/R/outsidepairs rồicơchếmới; PASS→GPU/preflight/FL2seed trongquyềnowner. KhôngchạmY50worker.

## Tiếp sau directprobe

Directjob `direct-generator-feasibility-20261106` đãsucceeded, artifactFAIL: [results](../research/results-direct-generator.md). Pair cải thiện nhưngRsingleton xấu; khôngTSTR/FL.
Đangchạy [marginaltransport](../research/screening-marginal-direct-generator.md), job`marginal-direct-generator-20261106`, supervisorPID55799, ownerthread`01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue.
State `/home/odixe/.codex/long-jobs/jobs/marginal-direct-generator-20261106/state.json`; log `runs/marginal-direct-generator/probe.log`; follow `tail -n 200 -F runs/marginal-direct-generator/probe.log`.
Worker sởhữu code/probe/variantplan và `runs/marginal-direct-generator`; khôngsửakhiđangsống. Completion→verifyresult+CM/gates, tiếp tụcphântích/thửtrongquyềnowner. KhôngchạmY50worker.

## Recovery marginal và attribution tiếp

Marginaldirect jobfailed doJSONnumpybool saufitxong. `scripts/recover_marginal_probe.py` verifyplan/source/sampleSHA/CMsupport/metricslog, [artifact feasibilityPASS](../research/results-marginal-direct-generator.md); khôngreruntraining hoặcđổistatefailedthànhsuccess.
Tiếp [marginal-only attribution](../research/screening-marginal-copula.md), job`marginal-copula-attribution-20261106`, supervisorPID57769, ownerthread`01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue.
State `/home/odixe/.codex/long-jobs/jobs/marginal-copula-attribution-20261106/state.json`; log `runs/marginal-copula/probe.log`; follow `tail -n 200 -F runs/marginal-copula/probe.log`.
Worker sởhữu `runs/marginal-copula`, source `scripts/{probe_marginal_copula,marginal_copula_generator}.py`. Completionaudit→táchgainCDFvsjoint rồiGPU/FLtestnếuphùhợp, đãcóquyền. KhôngchạmY50worker.

## Standard copula — bước tối ưu tiếp

Marginalonlyattribution đãxong: [kếtquả](../research/results-marginal-copula.md), TSTR .853304>hybrid .831997 nhưngpairgateFAIL; khôngpromoteverdictcũ.
Đangchạy [standardlatent](../research/screening-standard-copula.md), job`standard-copula-feasibility-20261106`, supervisorPID58311, ownerthread`01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue.
State `/home/odixe/.codex/long-jobs/jobs/standard-copula-feasibility-20261106/state.json`; log `runs/standard-copula/probe.log`; follow `tail -n 200 -F runs/standard-copula/probe.log`.
Worker sởhữu `runs/standard-copula`, `scripts/{probe_standard_copula,standard_copula_generator}.py`. Saucompletionverifyplan/source/CMgates; PASS→GPUpreflight/FL2seed đãđượcownercho phép; FAIL→phântíchcơchếtiếp, khônghỏi lại mỗifail. KhôngchạmY50worker.

## Chẩn đoán marginal đang chạy

Standardcopula đãxong, [localFAIL20%valTV](../research/results-standard-copula.md), TSTR.852840 nhưngkhôngpromotegateFAIL.
Job`copula-marginal-diagnosis-20261106`, supervisorPID58745, ownerthread`01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue; confirmedrunning.
State `/home/odixe/.codex/long-jobs/jobs/copula-marginal-diagnosis-20261106/state.json`; log `runs/copula-marginal-diagnosis/probe.log`; follow `tail -n 200 -F runs/copula-marginal-diagnosis/probe.log`.
Worker sởhữu `runs/copula-marginal-diagnosis`, `scripts/diagnose_copula_marginals.py`. Đotrain-vs-val vàgenerator-vs-train riêngiữa16×77feature, khôngMLrefit/officialtest. Completionverify→phântíchshift-vs-fittingerror rồithửmechanismnext trongquyềnowner. HybridlocalPASS vẫncandidateđượcphépGPUpreflight/FL2seed, chưaimplementGPU. Marginal-only/standardFAIL khôngđượctựnhậnbằnghạgate.

## Runtime preflight PyTorch standardcopula

Copulamarginaldiagnosis đãxong: [train-vs-val/fitting decomposition](../research/copula-marginal-diagnosis.md), generator→trainTV giảm80,3%; valTVgồmtemporalshift. GiữFAILcũ, khôngđổi utilitygate.
MethodPyTorch mới`mobihet_copula_standard` đãviết; unit2PASS. Runtime/crash-resume preflight2seedđangchạy (khôngFLlaunch, khôngutilityPASS).
Job`standard-copula-production-preflight-20261106`, supervisorPID59254, ownerthread`01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue.
State `/home/odixe/.codex/long-jobs/jobs/standard-copula-production-preflight-20261106/state.json`; log `runs/standard-copula-production/preflight.log`; follow `tail -n 200 -F runs/standard-copula-production/preflight.log`.
Worker sởhữu `runs/standard-copula-production`, source `methods/mobihet_copula_standard.py`, `scripts/preflight_standard_copula.py` vàimportsrecordedread-only. Completionauditcompile/backward/resume/finite/byte; FAIL→fixcausegiữmethod; PASS→T4gate/khóaFLdiagnosticnếucầntheoquyềnowner. Khônghỏi lại mỗiprobe, khôngsửaY50worker.

## StandardT4 runtime — khóa trước launch

Productionpreflight job đãsucceeded, sourceSHA+2seedreports auditPASS. [GateT4](../research/screening-standard-t4.md): 2seed×2roundfull5343, cosine50; cap1,25h/account, ưu tiênquota ngắn idle. Runner`run_standard_t4.py` sẽinventory/push/watch1200/collect/audit, khôngautoextend50 hoặcutilitypromotion. LocalFAILcũ giữnguyên.
Nguồnartifactpreflight: `runs/standard-copula-production/preflight.json`; productionmodule/runtime2rankrecordbytekhớpCore.

JobT4 mới `standard-copula-t4-runtime-20261106`, supervisorPID64332, ownerthread `01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue; controller confirmedrunning, đanginventory/chuẩnbịpush, chưaxácnhậnremoteT4startup.
State `/home/odixe/.codex/long-jobs/jobs/standard-copula-t4-runtime-20261106/state.json`; log `runs/standard-t4/pipeline.log`; follow `tail -n 200 -F runs/standard-t4/pipeline.log`.
Worker sởhữu `runs/standard-t4`, `artifacts/standard-t4`, `scripts/run_standard_t4.py` vànotebookZ0/Z1. Khôngchạycollectortrùng. Completionđọcbatch/status/closurevàactualaudit; failurephânloạilaunch/runtime/pull/audit, khôngrestarttrainingvìcollectorerror.
