> Snapshot rà cơ chế tri thức. Live: [continue](../../docs/operations/continue.md).

# Bàn giao sống — kiểm 11/11/2026 07:03 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](../../docs/operations/lessons.md) trước.
Đây là **bàn giao hiện hành duy nhất**. Artifact/state/process thật ưu tiên hơn tóm tắt.

## Vừa xong

- **B13 đóng:** 12/12run đủr20/source+CM audit PASS, SQ/RK/SK/JS/LS **FAIL utility cả2seed**.
  Không pilot/tune; [kết quả](../../docs/research/screening-module-b13.md), [10metric/byte](../../runs/sim-kaggle-b13/report-r20.md).
- **B14 đóng:** 8/8run đủr20, source/config/panel/LR/CM→10metric PASS; compile2T4/2rank PASS,
  logskip0, mọi client applied. GP/LG actualframe→aggregate→Q/hash/byte đủ20round/2rank PASS.
  Đã chạy lại auditor/analyzer trên artifact thật trong phiên phân tích, không train lại.
  **MX/GP/LG FAIL utility cả2seed**; Δmacro s42/s43:
  MX −.015809/−.015881; GP −.000307/+.000881; LG +.000834/−.000199.
  Không pilot, không ghép/tune/kéo dài nhánhFAIL.
  [Thiết kế/kết quả](../../docs/research/screening-uplink-b14.md), [10metric/byte](../../runs/sim-kaggle-b14/report-r20.md),
  [diagnostics](../../runs/sim-kaggle-b14/diagnostics.md), [phân tích lại](../../docs/research/b14-analysis-20261111.md).
- MX hại class11 precision−.232/−.244 và cả xe lớn; LG Qr19 cực sắc (diagonal≈.994,239/256ôzero)
  vẫn không đủmacro. Low256 chỉ9xe, absent191dòng; không lấy hai nhóm này đơn lẻ kết luận thắng.
- **CBB vẫn phương pháp chính:** KM16w0.5 +lite6+QQ8, F1r50TB2seed .9078.
  Nguồntrain b13/b14 không sửa trong phiên phân tích; thêm scriptreadonly `analyse_b14_diagnostics.py`.
  Lessons/catalog/proposal/diagnosis/hub đã cập nhật.

## Job/collector — không còn job b13/b14 sống

- B13 controller `mobihet-b13-20261110` succeeded00:08; metadata cũ ở snapshotb13.
- B14 đầu `mobihet-b14-20261111` failed00:34 trướcquota/push do audit bound LGhybrid;
  sửa checker, reuse đúng smoke cũ,30testsPASS; failure/source/diagnosis giữ
  `runs/sim-kaggle-b14/recovery-20261111/`.
- Recovery controller `mobihet-b14-recovery-20261111`, supervisor690627, ownerthread
  `01a1c60e-5f5e-7311-999f-7338ab28bb59`, deliveryqueue/pending. Đãpush8/8version1 trước reboot01:53.
  **State vẫn ghi running nhưng PID690627/690628/690631 không tồn tại**; không coi là jobkhỏe hoặc restart train.
  State `/home/odixe/.codex/long-jobs/jobs/mobihet-b14-recovery-20261111/state.json`,
  log `runs/sim-kaggle-b14/pipeline-recovery.log` giữnguyên.
- Collector phục hồi `runs/sim-kaggle-b14/watch-recovery.sh` (PID114907) đã kết thúc:
  `watch-recovery.log` ghi8/8COMPLETE/pulled,source-lock14filesPASS,allauditPASS vàB14_AUDITED;
  [DONE](../../runs/sim-kaggle-b14/DONE) lúc04:21:35. Không launch/collect trùng từ notification muộn.
  Exactowner/slug/version ở `runs/sim-kaggle-b14/batch.json`, artifact ở `artifacts/sim-kaggle-b14/`.
  Originalsource-lock vàsource-lock-recovery giữnguyên; không tự sửa controllerstate lịch sử.
- Không có pilot tự động vì không ứng viên PASS. Không có batch mới push trong phiên phân tích.

## B15 đang chạy — owner đã đồng ý thử mới11/11

- [Thiết kế/gate](../../docs/research/screening-sam-b15.md): C/SAM×seed42/43 =4kernel, rho.05 cốđịnh,
  cap3,75h/kernel+15%margin; CBBgenerator/loss/AdamW/LR/batch64/synthetic64/floor16 giữnguyên.
  SAM2gradient/pass cùngbatch/synthetic/dropout, chỉpass1Q/BNcommit, 0byte schema mới.
- **29CPUtests PASS**: rho0hai roundexactCBB, explicitnonzero2passgradient (hard/softQ, tail/floor/padded,
  firstoverflow), state+RNGresume. Code/checkdocs/diffwhitespacePASS. Không sửa sharedcore/protocol.
- Job **`mobihet-b15-20261111` running**, xácnhận1lần05:29:23, stage`cuda-replay.log` sauunitPASS.
  **Đã push4/4version1**, receipt `runs/sim-kaggle-b15/launch-complete.json`; localCUDA/dropout replay + guarded200xeW1/compile2round + SAM/CM/stateaudit PASS. Worker đã qua
  quota sống→generate/push4kernel, đang collect1200s→source/CM/counter/10metric/reportaudit.
  Poll05:46:57–05:47:02UTC+7 ngày11/11: **4/4RUNNING**. C/SAMs42: `anhtruong1909`;
  C/SAMs43: `nguyenminhtran060113`; exactrefs trong`batch.json`. Chưa có utility kết thúc.
  Khôngmodelpolling khi chỉcòn chờ; completionqueue đánhthức cùngthread.
- SupervisorPID**187268**, worker187269, child187270; ownerthread `01a1c60e-5f5e-7311-999f-7338ab28bb59`;
  delivery**queue/pending**, start05:29:12UTC+7.
  State `/home/odixe/.codex/long-jobs/jobs/mobihet-b15-20261111/state.json`.
  [Logpipeline](../../runs/sim-kaggle-b15/pipeline.log);
  theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/sim-kaggle-b15/pipeline.log`.
  Command: `SIM_BATCH=b15 /home/odixe/Tools/miniconda3/envs/nckh/bin/python scripts/run_b13_pipeline.py`.
- `source-lock.json` khóa14file. **Readonly tới completion**:
  `methods/mobihet_{sim,refine,sam}.py`, `scripts/sim_fold_val.py`, `scripts/run_sim_kaggle.py`,
  `scripts/run_b13_pipeline.py`, `scripts/analyse_b13.py`, `scripts/audit_sim_batch.py`,
  `scripts/report_module_screen.py`, `scripts/check_sam_cuda.py`, `scripts/audit_sam.py`,
  `tests/test_mobihet_{refine,sam}.py`, `docs/research/screening-sam-b15.md`.
  Không bật lại launcher/watcher trùng. Nếu reboot: kiểm PID+intent+artifact rồi recover đúngstage, không restarttraining.

## Tiếp khi completion

Owner steering mới: **vẫn cho SAM hiện tại chạy thử**, lo fairness với baseline. Báo riêngCBB+SAM/compute;
không gán gain optimizer cho cơ chế FL. Job187268/187269/187270 còn sống lúc07:03; không sửa nguồn14fileđãkhóa.
[Rà cơ chế mới](../../docs/research/knowledge-transfer-review-20261111.md) + mathcounterexample đãghi,
chưa utility/newbatch: ưu tiên classmeanKD cùngforward/byte trên CBBkhôngSAM;
moment2uplink/personalizedexpert/nguồn đo temporal còn cần diagnostic/thiếtkế/gate.
Sampler/data/state cũng là vịtrí pipeline, nhưng sampler/optimizer/pretrain generic hạ ưu tiên vì fairness.
Cross-fit rowdisjoint chưa là time/eventindependent; coverage countsplit không chuyển sang blocksplit nếu chưa đo lại.

1. Đọcstate/log/artifactB15, audit source/config/LR/20CM/10metric/compile2rank+SAMcounters trướcutility.
   CUDA/localsmoke lỗi là lỗi gatevận hành, không utilityFAIL; sửa đúngstage, giữnguyên evidence.
2. PASS hai seed theo luậtb15→tự fullpilot20cos50s42; pilotPASS→resume đúngstate2rank tới50 +seed43r50.
   FAIL dừng, không tune rho/extend/ghépcombo.
3. [Cross-fit feasibility](../../docs/research/crossfit-feasibility-20261111.md) đã đo count, chưa train:
   eligible n≥16 bao phủ99,860%sim/99,798%fullrows; shadowsteps/main≈1,065/1,074× trướcheldoutinference.
   3modelthườngtrú vượtT4full; dùng hai phaseDAGSNetshadow độc lập rồi main đọcaggregate theo round.
   Cần chốt rowpartition/fallback/codec/sourceaudit/cost trướccode/launch; không dùng entropytrain/postepoch/preupdate
   làmquality, khôngholdout→shadowgenerator/Q, không coi qualityshadow làqualitymain.
   Đây là việc tiếp đãđượcownerđồngýkhảosát; không cần xin quyền lại cho triểnkhai/gate trong scope hiệnhành.
4. Không mở lại Qregion/bank/BNsplit/generator/floor đãđóng qua tênmới. ChỉDAGSNetW1, bytebáođủbootstrap+hai chiều.

## Quyền hiện hành

- Owner10/11: đổi mọi module/compute để tăngperformance, **phải hỗ trợ khác kiến trúc**,
  experiment chỉDAGSNetW1. Localtrước, Kagglequotasẵn; thang2seed→pilot20→r50two.
- Byte giảm tối đa **chưa capcứng**; báo bootstrap+hai chiều+tổng+compute. Capcompute cũ đã thay cho nghiên cứu mới.
- Credential processscope; global login giữnguyên. Không Git/DVCpush,xóaartifact hoặc đổi globalcredential.
  Không hủy jobkhỏe nếu owner chưa yêu cầu. Backup10/11 trướcb13 là mốc gần nhất; không tựbackup.

## Việc tay

Chưa cần. B15 worker đang thực hiện cácgate đãủyquyền; chưa utility để quyếtđịnhscale.
