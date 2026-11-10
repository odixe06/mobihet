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
| Sidecar phiên sender (gộp quyết định) | `build_sessions.py full` (train/test, thứ tự loader) / `build_sessions.py sim` (panel 200 xe fold val) | `cache/sessions/`; khóa (xe, world, `sender_alias`) |
| KM codec/EM probe + preflight | `probe_km_codec.py`, `probe_km_codec2.py`, `probe_km_em.py`, `check_km_lite.py` (cả knob codec `km_*`/`km_bytes`) | `runs/km-codec-probe/`; LL chỉ so trong cùng họ nhiễu |
| KM full-scale | `run_km_pilot20.py`, `resume_km_pilot50.py`, `run_km50.py`, `analyse_km50.py`, `run_km_lite_pilot20.py` + `analyse_km_lite_pilot20.py` (lite6 vs KM50 Q42 r20), `run_codec_pilot20.py` + `analyse_codec_pilot20.py` (10 arm codec vs KM50; `analyse_codec_pilot20.py 20 g1-pilot20` cho G1), `run_g1_pilot20.py` (synthetic 48/32 dòng/bước), `run_codec_combo20.py` + `analyse_codec_combo20.py` (combo r20), `run_codec_combo50.py` + `analyse_codec_combo50.py` (combo r50 CBB/CQ8: audit + luật + bảng owner; arm chưa pull chỉ báo status/ước tính) | luật [screening-km50](../docs/research/screening-km50.md), [screening-codec](../docs/research/screening-codec.md) |
| Common audits | `audit_screening.py`, `audit_r20.py` | CM/source/teacher/finite; khôngfullpredictionreplay |
| F50/L50 | `analyse_f50.py`, `audit_f50.py`, `audit_l50.py` | nghiên cứu đãđóng; cần restore DVC snapshot nếu thiếu |
| GPU/runtime/width/local guard | `bench_grouped.py`, `run_central.py`, `run_local_guarded.py`, `cache_subset.py` | [knowledge](../knowledge/README.md), skilltraining-speed |
| CLI/read timeout | `kaggle_cli_bounded.py`, `run_bounded_python.py` | credentialquaaccount-scopedwrapper |
| Metrics/links | `report.py`, `research_metrics.py`, `check_docs.py` | 10metric từCM; linkchecker loạiarchive |

Watcher nền: Codex qua codex-long-jobs; Claude Code qua `runs/<batch>/watch.sh` (bash nền, poll 1200 s). Không bật hai watcher cùng batch.

W/X/F/L/C và các helper Q/R cũ vẫn cung cấp runtime/test/import hoặc evidence; không dời code vì dời tài liệu.
Thiết kế/kết quả W/X/C và CPUgenerator đã archive: [chỉ mục](../archive/docs-20261107/README.md).
`analyse_c10.py` thuộc study đóng, có đường xuất báo cáo cũ; không chạy nó như tác vụ mặc định sau migration.

Npy cache canonical: `runs/w-study/builder.json`, public `odixe0502/mobihet-v4npy-cache`.
Không chạy lại builder nếu cache đãverify; Inductor cache đểtemp vàđóngtar trong run đểnốiphiên.
Các study DVC đãcleanup làworkspacepartial: không dvc add lại; restore đúngmanifest theo [backup](../docs/operations/backup.md).
