# Công cụ — chọn đúng entrypoint

Chạy từ repo root, env `nckh`. Quyền/job/ownership ở [continue](../docs/operations/continue.md);
không chạy lại launcher từ một notification muộn. Code/library cũ còn là dependency thì giữ nguyên.

| tác vụ | entrypoint | evidence / lưu ý |
|---|---|---|
| Shared FL driver | `run_q20.py` → `run_fl.py` | protocol, resume/state hai rank, provenance |
| Sinh notebook | `gen_notebook.py` | private, image/accelerator pin, overlay, npy SHA |
| Pull state4GB | `pull_checkpoint_parallel.py` | Range resumable, khoảng32 connection tổng |
| Import parent trước decode | `resume_checkpoint_cells.py` | unique parent/round/config/RNG2rank/SHA |
| Y50 (đã đóng) | `audit_y50.py`, `analyse_y50.py`, `y50_common.py`; transfer `pull_y50_checkpoints.py`, `stage_y50_checkpoint.py`, `run_y50_transfer.py` | verdict ở [results-y50](../docs/research/results-y50.md); không chạy lại launcher/watcher |
| Generator đã đóng (chỉ tái lập) | standard: `preflight_standard_copula.py`, `run_standard_t4.py`, `resume_standard_t4.py`; forest: `preflight_forest_quantile.py`, `verify_forest_sampler.py`, `run_forest_t4.py`; pilot20: `run_generator_pilot20.py`, `analyse_generator_pilot20.py`; probe CPU: `probe_{pair,direct,marginal_direct,marginal_copula,standard_copula}*.py`, `recover_marginal_probe.py`; `analyse_generator_oracle.py`, `diagnose_copula_marginals.py` | generator đóng băng ([catalog](../docs/research/generators.md)); không restart/nối |
| Sim fold-val KM (b1–b11) | `sim_fold_val.py`, `run_sim_kaggle.py` (`SIM_BATCH=bN`), `watch_sim_batch.sh`, `audit_sim_batch.py`, `analyse_sim.py`, `analyse_km_lite.py` (`KM_BATCH`), `analyse_km_lite_detail.py` (theo lớp/nhiễu b9–b10), `analyse_b11.py`, `analyse_b12.py` | [screening-sim-foldval](../docs/research/screening-sim-foldval.md), [screening-km-lite](../docs/research/screening-km-lite.md), [catalog đề xuất](../docs/research/improvement-proposals.md) |
| CBB module b13/b14 | `run_b13_pipeline.py` qua codex-long-jobs (`SIM_BATCH=b13`/`b14`) → local gates/quota/push/collect1200s/audit → `analyse_b13.py [batch]` + `report_module_screen.py <batch>`; b14 thêm `audit_uplink.py` actual frames2rank/round | [b13 kết quả](../docs/research/screening-module-b13.md), [b14 luật](../docs/research/screening-uplink-b14.md); source-lock trước smoke, worker sở hữu source tới completion; không chạy launcher lại từ notification |
| Chẩn đoán b14 | `analyse_b14_diagnostics.py` đọc20CM/arm + actualwire → `runs/sim-kaggle-b14/diagnostics.{json,md}`; không train/tune/đổi gate | [phân tích](../archive/docs-20261113/research/b14-analysis-20261111.md); class P/R/F1, cohort, confusiondelta, teacher r19 và trajectory |
| SAMb15 | `SIM_BATCH=b15 run_b13_pipeline.py` qua controller; `check_sam_cuda.py` replaycompiled/dropout thực, `audit_sam.py` kiểm actualcounter/source sauauditCM | [thiết kế/gate](../docs/research/screening-sam-b15.md); C/SAM2seed, rho.05,0byte mới, sourcefreeze trướcgate |
| MeanKDb16 | `SIM_BATCH=b16 run_b13_pipeline.py` qua controller; `audit_mean.py` kiểm actualcoverage/Jensen/teacherlag, local thêmCM/state | [thiết kế/gate](../docs/research/screening-mean-b16.md); C/MK/HK2seed, MK/HK đều W1compiledsmoke2round trướcquota; rowmodeexactCBB; post-hoc `analyse_mean_diagnostics.py` + `analyse_b14_diagnostics.py b16` kiểm Jensen/r1CM/cohort |
| Sidecar phiên sender (gộp quyết định) | `build_sessions.py full` (train/test, thứ tự loader) / `build_sessions.py sim` (panel 200 xe fold val) | `cache/sessions/`; khóa (xe, world, `sender_alias`) |
| KM codec/EM probe + preflight | `probe_km_codec.py`, `probe_km_codec2.py`, `probe_km_em.py`, `check_km_lite.py` (cả knob codec `km_*`/`km_bytes`) | `runs/km-codec-probe/`; LL chỉ so trong cùng họ nhiễu |
| KM full-scale | `run_km_pilot20.py`, `resume_km_pilot50.py`, `run_km50.py`, `analyse_km50.py`, `run_km_lite_pilot20.py` + `analyse_km_lite_pilot20.py` (lite6 vs KM50 Q42 r20), `run_codec_pilot20.py` + `analyse_codec_pilot20.py` (10 arm codec vs KM50; `analyse_codec_pilot20.py 20 g1-pilot20` cho G1), `run_g1_pilot20.py` (synthetic 48/32 dòng/bước), `run_codec_combo20.py` + `analyse_codec_combo20.py` (combo r20), `run_codec_combo50.py` + `analyse_codec_combo50.py` (combo r50 CBB/CQ8: audit + luật + bảng owner; arm chưa pull chỉ báo status/ước tính) | luật [screening-km50](../docs/research/screening-km50.md), [screening-codec](../docs/research/screening-codec.md) |
| B20 giảm compute | sim: `SIM_BATCH=b20 run_b13_pipeline.py`, `analyse_b13.py b20`, `analyse_b20_trajectory.py b20 20`; full-scale: `run_b20_full.py` (SD r50, SM pilot r20), `run_b20_sm50.py` (SM42 resume r20→r50 + SM43 r50) với `quota/launch/collect/audit`; watcher `watch_b20.py` collect+audit cả hai batch | [B20](../docs/research/screening-compute-b20.md); không rerun `launch` khi `launch-complete.json` có |
| Seed 41 CBB | `run_cbb_s41.py` | [report 3 seed](../runs/cbb-s41/report.md) |
| Common audits | `audit_screening.py`, `audit_r20.py` | CM/source/teacher/finite; khôngfullpredictionreplay |
| F50/L50 | `analyse_f50.py`, `audit_f50.py`, `audit_l50.py` | nghiên cứu đãđóng; cần restore DVC snapshot nếu thiếu |
| GPU/runtime/width/local guard | `bench_grouped.py`, `run_central.py`, `run_local_guarded.py`, `cache_subset.py` | [knowledge](../knowledge/README.md), skilltraining-speed |
| CLI/read timeout | `kaggle_cli_bounded.py`, `run_bounded_python.py` | credentialquaaccount-scopedwrapper |
| Metrics/links | `report.py`, `research_metrics.py`, `check_docs.py` | 10metric từCM; linkchecker loạiarchive |

Watcher nền: Codex qua codex-long-jobs; Claude Code qua script watcher chạy Bash background (vd `watch_b20.py`, poll 1200 s). Không bật hai watcher cùng batch.

W/X/F/L/C và các helper Q/R cũ vẫn cung cấp runtime/test/import hoặc evidence; không dời code vì dời tài liệu.
Thiết kế/kết quả W/X/C và CPUgenerator đã archive: [chỉ mục](../archive/docs-20261107/README.md).
`analyse_c10.py` thuộc study đóng, có đường xuất báo cáo cũ; không chạy nó như tác vụ mặc định sau migration.

Npy cache canonical: `runs/w-study/builder.json`, public `odixe0502/mobihet-v4npy-cache`.
Không chạy lại builder nếu cache đãverify; Inductor cache đểtemp vàđóngtar trong run đểnốiphiên.
Các study DVC đãcleanup làworkspacepartial: không dvc add lại; restore đúngmanifest theo [backup](../docs/operations/backup.md).

B17 macro-first: `SIM_BATCH=b17 run_b13_pipeline.py`; `audit_macro.py` actualcount/shift/NRgate,
`analyse_b13.py b17` chỉmacro làutilitygate, `report_module_screen.py b17` đủ10metric/byte.
[Thiết kế](../docs/research/screening-macro-b17.md). Reviewpilotcũ: `analyse_macro_first.py` chỉđọcCM/audit,
khôngtrain/đổiverdict; [curves](../runs/macro-first-review-20261111/pilot-curves.png).

Qualitydiagnostic cũV20: `analyse_temporal_quality.py` đọcactualtrain/internal-valCM, kiểm10metric/support,
báo precisionchuẩnhóabinaryprior vàcorrelations hai mức support; [scope](../archive/docs-20261113/research/crossfit-feasibility-20261111.md).
Khôngfitweights hoặcstarttraining.

Recipientdomain diagnostic: `analyse_recipient_domains.py` đọcV20train/internalvalCM và publicscenario/run,
kiểm10metric/ID/support/SHA, forecastconditionedtrueclass qua clienthashhalves disjoint;
thêm counts/effectiveexperts mainfull/sim. Futurepeers chỉdiagnostic, không làm trainteacher.
[Scope/kết quả](../archive/docs-20261113/research/teacher-source-survey-20261111.md).

B18: `run_b18_pipeline.py` qua long-jobs → unit/guardedlocal2r4phase/quota/push2seed/collect1200s.
Notebookparent `run_b18_seed.py` chạy `b18_phase.py` trong process2rank mới từngS/C/E/H;
`b18_partition.py` tạo canonicaltemporalroles; `audit_b18.py` replaypacket/teacher/CM tấtcảround.
[Design khóa](../docs/research/screening-source-b18.md). Không chạy lại launcher khi cósource-lock;
`--collect-only` chỉsau launch-complete vàsourceđúnglock, không restarttrain.

B18 đãđóng12/11: `audit_b18_resources.py` phục hồi HPresourceproof từsealednotebook+manifest+runtimeconfigs,
không giảexecutedfile. `run_b18_pipeline.analyse(batch)` tái tạo verdict/10metricMD/CSV từartifactb18r1;
`analyse_b18_diagnostics.py` đọcCM/sourceQ r19, khôngtrain. Frozenlauncher cũ không chạy lại sauanalysisfix.
