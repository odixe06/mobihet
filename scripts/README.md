# Công cụ đang dùng

Chạy từ repo root, env conda `nckh`. Script trong archive không thuộc workflow hiện tại.

| nhiệm vụ | công cụ |
|---|---|
| FL theo protocol, artifact mỗi round | `run_fl.py` |
| Notebook private, pinned source/overlay | `gen_notebook.py` |
| Status/collect Kaggle nền, không model polling | `watch_kaggle_batch.py`, `watch_collect_screening.py` + skill codex-long-jobs |
| CLI/network có timeout | `kaggle_cli_bounded.py`, `run_bounded_python.py` |
| Audit M20, giữ exact version | `audit_m20.py`, `audit_screening.py` |
| Phân tích M20 và thống kê dùng chung | `analyse_m20.py`, `research_metrics.py`, `report_m20.py`, `diagnose_m20.py` |
| Phân tích N20 factorial/teacher và đủ10metric/figure | `analyse_n20.py`, `report_n20.py` |
| Local probe có memory guard/cache | `run_local_guarded.py`, `cache_subset.py` |
| Đo grouped/runtime và width gate | `bench_grouped.py`, `run_central.py` |
| Bảng metric từ CM | `report.py` |
| Kiểm link active | `check_docs.py` |

Mọi run cũ dùng source/config đã đóng băng trong provenance/DVC, không lấy code HEAD mới thay ngầm.
Sau cleanup, auditor snapshot cần restore DVC trước; phân tích thường dùng evidence nhẹ. Xem backup record.

M20: `python scripts/analyse_m20.py` đọc evidence nhẹ; `python scripts/report_m20.py` tái tạo báo cáo/figure.
`python scripts/audit_m20.py` cần restore đủ `artifacts/m20/*.dvc`; không đọc archive.

`python scripts/diagnose_m20.py`: phân tích paired/cohort/Q/exposure/gap, JSON+figure ở `runs/diagnosis-m20`; không train hoặc đổi evidence.

N20: `preflight_n20.py` kiểm4ô trên realdata; `prepare_n20.py` sinh notebook sau khi source đãpush; `launch_n20.py` kiểmquota rồi push đúng1lần theo SHA/intent.

`run_n20_batch.py` nối launch→watch/collect/audit qua codex-long-jobs. `check_screening_startup.py` kiểm một lượt W&B/logs; worker tự lặp khi cần, không model polling.

N20 đã đóng. `python scripts/analyse_n20.py` kiểm SHA/source/support/CM và tính factorial từ `artifacts/n20`,
ghi `runs/n-study/analysis`; `python scripts/report_n20.py` sinh `metrics-n20.md`/`figures/n20-factorial.png`.
Không chạy lại launcher/collector để tái tạo phân tích. Payload N chưa backup; không xóa.

O20: `freeze_o20_teacher.py` khóa aggregateQ từ N0/N1 (không overwriteinput), `preflight_o20.py` kiểm replay/manualinjection
trên realdata; `prepare_o20.py` khóa8notebook sau61tests/preflight; `launch_o20.py` dùng one-shotlauncher chung.
`run_o20_batch.py` nối launch→startup/watch1200s→collect→`audit_o20.py` (source/CM/lag/target/byte).
`watch_collect_screening.py --auditor ...` cho audit riêng; mặc định vẫn audit M/N. Không gọi launcher khi chỉ cần phân tích.

`collect_o20_early.py`: one-shot subset exactversionterminal đã khóa; staging/audit riêng, publish hardlink atomicno-clobber
và copylog riêng để maincollector có thể skip file mà không tranh ghi. Không watcher/training mới; xem ownership ở bàn giao.

`analyse_o20.py` kiểm snapshotSHA/CM/source/teacher và tính pairedcontrast/cohort/tripwire; `report_o20.py` sinh kết quả/10metric/figure.
Mặc định full8run từ `artifacts/o20`; `--early` đọc subset6run đã khóa ở staging riêng. Cảhai job đã đóng; không gọi launcher để phân tích.

P20: `audit_prior_evidence.py` fingerprintnguồn/auditv4missingness; `compare_benchmark_r20.py` tính đủ10metric từ17CMview benchmark cùngr20.
`preflight_p20.py` kiểm16cell/W1/compiledAMP/realfixture/codec/carrier/diagnosticRNG; `prepare_p20.py` khóa32notebook sau mọi gate.
`run_p20_pipeline.py` chờ finalpreflight rồi prepare→`run_p20_batch.py`→launch32/status1200s/earlypull/auditserial.
Không tự gọi `launch_p20.py` khi worker còn sở hữu batch. `audit_p20.py` kiểm source/20CM/teacherchain/aggregate/replica/byte.
Trạng thái và jobID ở bàn giao; các notebook/draft private chứa key, không Git/DVC.

P20 đã đóng audit32arm:31×20round/P21r0. `finalize_p20_audit.py` khóa closure, không launchtraining.
`analyse_p20.py` kiểm snapshotSHA/CM/support/10metric, tính lockedcontrasts/cohorts/teacherpaired/slopes/byte/time;
`report_p20.py` sinh bảng endpoint, đủ10metric, chi phí và figures từ analysis đã kiểm.
Receipt `runs/p-study/analysis/completion-verification.json`; không gọi launcher/collector để tái tạo phân tích.

`diagnose_p20_clients.py` kiểm endpointCM/support và phân tích transfer từngxe soC2, mô tả sau khi đọctest,
không selector theo test. `run_p21_recovery.py` là guardentrypoint giữ Method/HP cũ;
`preflight_p21_recovery.py` kiểm finitepath3round bitexact và nhánh lỗi trên realfixture.
`prepare_p21_recovery.py` khóa riêng một notebook sau gate; `run_p21_recovery_job.py` launchonce→poll1200s→pull/audit20round.
Recovery đã collect/audit20round; không gọi lại launcher. `analyse_p21_recovery.py` tái kiểm source/CM/20proxyrebuild/guard,
ghép seed43 vào đối chứng C10 và sinh analysis/10metric/figure bổ sung riêng, giữ nguyên31run/báo cáo P20 gốc.

Q20: `q20_common.py`/`methods/mobihet_next.py` khóa8cell, `run_q20.py` dùng driver chung với compile/nonfiniteguard.
`preflight_q20.py` kiểm35test + W1realfixture2seed×3round/controls/state/RNG; `inventory_q20.py` giữ P21 riêng.
`prepare_q20.py` khóa16notebook sau quota/preflight; `launch_q20.py` gate toànbatch rồi pushonce;
`run_q20_batch.py` poll1200s/earlypull/audit bằng `audit_q20.py`. Sau failure chỉ `--collect-only` khi đã inspect,
không gọi launch lại hoặc sửa frozenbatch. Q đã đóng16×20round; không restartcollector từ callback muộn.
`analyse_q20.py` kiểm closure/source/artifactSHA, tính320CM/pairedcontrasts/cohorts/teacher/byte/time;
`report_q20.py` sinh đủ10metric320row, endpoint/costtables và standalonefigures. Không train/tune hoặc sửa P20 gốc.
`verify_q20_results.py` khóa completionreceipt: source thực khớp Gitbase/overlay, CSV/workerlog/rendered3200metric,
querytripwires, SHA report/script và P20 gốc không đổi.
`tests/test_q20_audit.py` giữ storedboundary/targetcorruption cases cho checker CPU/CUDA normalization.

R20: `freeze_r20_teacher.py`, `preflight_r20.py`, `prepare_r20.py`, `launch_r20.py`, `run_r20_batch.py`, `audit_r20.py`;
phục hồi packaging `recover_r20_push.py`, `stage_r20_input.py`, `run_r20_recovery_pipeline.py`.
V20 diagnostic-only: `audit_v20_split.py`, `run_v20.py`, `preflight_v20.py`; không đổi headline train+val.

V20 wholephase: `inventory_v20.py` → `prepare_v20.py` → `launch_v20.py`/`run_v20_batch.py`/`audit_v20.py`,
controller worker `run_v20_pipeline.py`; owner ngoài R, originalfoldtrain→val, không headline test.

R0–R3 repeated controls trên tài khoản mới: `inventory_repeat20.py` → `prepare_repeat20.py` →
`launch_repeat20.py`/`run_repeat20_batch.py`, controller `run_repeat20_pipeline.py`.
State riêng`runs/r-repeat`, manifest`r20-repeat-launch.json`, dùng lại`audit_r20.py` và artifact`artifacts/r20` với slugriêng.
Không chạy lại launcher khi có pushintent/receipt; không đổi source hoặc watcher của phase R/V đang sống.

V20 zero-skip incident: `probe_v20_skips.py` quan sát mộtepochCE, `preflight_v20_skips.py` chứng minh observer bitexact;
`run_v20_skip_pipeline.py` gate/push2probe trên ownerdựphòng/watch/pull, `audit_v20_skips.py` kiểm numericalevidence.
Đây không accepted20roundrecovery; không sửa runtime/guard của6kernel V cònchạy hoặc chốt phươngpháp từ probe.

CErecovery sauprobe: `methods/mobihet_vrecover.py`, `preflight_v20_recovery.py`, `prepare_v20_recovery.py`,
`launch_v20_recovery.py`, `run_v20_recovery_batch.py`, `audit_v20_recovery.py`; controller `run_v20_recovery_pipeline.py`.
Hai runfresh20round riêng`runs/v-study/recovery-ce`/`artifacts/v20-recovery`; chỉ sửa guard/observability, giữ core/HP và originalfailures.
