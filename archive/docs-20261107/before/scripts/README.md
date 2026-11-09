> Snapshot 07/11/2026; tài liệu hiện hành: [scripts/README.md](../../../../scripts/README.md). Không đọc mặc định.

# Công cụ đang dùng

Chạy từ repo root, env conda `nckh`. Script trong archive không thuộc workflow hiện tại.

| nhiệm vụ | công cụ |
|---|---|
| FL theo protocol, artifact mỗi round | `run_fl.py` |
| Notebook private, pinned source/overlay | `gen_notebook.py` |
| Status/collect Kaggle nền, không model polling | `watch_kaggle_batch.py`, `watch_collect_screening.py` + skill codex-long-jobs |
| CLI/network có timeout | `kaggle_cli_bounded.py`, `run_bounded_python.py` |
| Audit M20, giữ exact version | `audit_m20.py`, `audit_screening.py` |
| Phân tích M20 và thống kê dùng chung | `analyse_m20.py`, `research_metrics.py` |
| Local probe có memory guard/cache | `run_local_guarded.py`, `cache_subset.py` |
| Đo grouped/runtime và width gate | `bench_grouped.py`, `run_central.py` |
| Bảng metric từ CM | `report.py` |
| Y50: ba ứng viên Core | `run_y50.py`, `keep_y50.py` (giữ worker sống, nối phiên), `y50_common.py`, `preflight_y50.py`, `audit_y50.py`, `analyse_y50.py` |
| Kiểm link active | `check_docs.py` |

Mọi run cũ dùng source/config đã đóng băng trong provenance/DVC, không lấy code HEAD mới thay ngầm.
Sau cleanup, auditor snapshot cần restore DVC trước; phân tích thường dùng evidence nhẹ. Xem backup record.

M20: `python scripts/analyse_m20.py` đọc evidence nhẹ.
`python scripts/audit_m20.py` cần restore đủ `artifacts/m20/*.dvc`; không đọc archive.

Code riêng batch N→V đã đóng ở `archive/` (05/11): README cũ [ở đây](../../../scripts/README-before-20261105.md), danh sách `archive/move-manifest.json`.
Còn active vì F50/L50 hoặc test dùng: `run_q20.py`, `q20_common.py`, `p20_common.py`, `run_p21_recovery.py`, `preflight_q20.py`, `preflight_r20.py`, `audit_r20.py`, `stage_r20_input.py`, `launch_repeat20.py` (đọc `docs/research/r20-repeat-launch.json`), `analyse_r20_v20.py`.

W20 (`methods/mobihet_wstudy.py`, ô ở `w20_common.py`): `preflight_w20.py` (local, knob mặc định bit-exact Live/Replay) →
`run_w20.py` một worker: builder cache npy public → quota → notebook → gates/push → watch 1200 s/pull/`audit_w20.py`.
Mỗi stage có receipt trong `runs/w-study`; chạy lại tiếp từ stage thiếu, `--collect-only` sau khi đã push.
Cache npy dùng chung cho batch mới: `gen_notebook.py --npy-kernel odixe0502/mobihet-v4npy-cache --npy-manifest-sha256 <sha>`
(sha của `runs/w-study/builder/v4npy/manifest.json`); `--build-npy` sinh kernel builder CPU public, không credential.

X20: `x20_common.py`, dùng chung `run_w20.py --study x20`.
F50: `f50_common.py`, `preflight_f50.py`, `run_f50.py`, `audit_f50.py`. C10: `probe_core_compression.py`, `run_c10.py`, `audit_c10.py`, `analyse_c10.py`.
L50: `preflight_l50.py`, `preflight_l50_transfer.py`, `stage_l50.py`, `run_l50.py`, `audit_l50.py`.
F50 kết quả: `analyse_f50.py`. Core s43: `run_f50_s43.py` (receipt `runs/f-study-s43`).
Rà pipeline sau F50: `analyse_core_gaps.py`, `analyse_core_bias.py` (từ CM), `probe_information_gap.py` (train fold → val fold, không đọc official test).

- `analyse_generator_oracle.py`: đọc CM r50 oracle/Core đã audit, xuất class precision/recall/F1 và cohort gap vào `runs/generator-study/oracle-gap.json`; không fit hoặc launch.

- `probe_pair_generator.py`: probe train→val copula/pairhist residual; plan immutable, ESS/TV rồi TSTR khi qua cổng; không launchFL.

- `direct_generator.py`, `probe_direct_generator.py`: CPU forest-count ancestral sampling + Gaussian conditional; train→val gates và TSTR, không officialtest/FLlaunch.

- `marginal_direct_generator.py`, `probe_marginal_direct_generator.py`: sửa singletonR bằng syntheticcalibration/quantiletransport, giữ control vàS; fingerprint sampler override riêng, chưaFL.

- `recover_marginal_probe.py`: phục hồiJSON từCM+source/plan/datahash/log, khôngrefit.
- `probe_marginal_copula.py`: attribution marginal-only vs foresthybrid; `probe_standard_copula.py`: latentN(0,correlation) feasibility, khôngofficialtest/FLlaunch.

- `run_standard_t4.py`: gate2seed×2round full5343/T4 cho standardizedcopula; quota ngắn idle, cache/compile, watch1200/pull/audit; khôngautoutilitypromotion.

- `pull_checkpoint_parallel.py`, `pull_y50_checkpoints.py`: exact-version/checkpoint Range32streams, resumable partials.
- `stage_y50_checkpoint.py`, `run_y50_transfer.py`: owner-private crossaccountarchive/SHA/state2rank, autochainingr50; `resume_checkpoint_cells.py` locks importchecks.
- `resume_standard_t4.py`: audit r1 vàresume2rank/cachedkernel tớiđíchr2, khôngrestart hoặcautoextend50.
### Forest + quantile trên Kaggle

`preflight_forest_quantile.py`:2seed/localcompile/crashresume. `verify_forest_sampler.py`:GPU train→valstatisticalgate,
khôngofficialtest. `run_forest_t4.py`:đợipreflightPASS, samplerPASS, loạiY50receiver, freshquota/inputs, push2seed2roundT4,
cache+W&B+collect/audit. [Gate](../../../docs-20261109/research/screening-forest-t4.md). Receipts `runs/forest-{production,t4}`.
