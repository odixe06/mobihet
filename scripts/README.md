# Công cụ còn dùng

Từ repo root, env `nckh`; quyền/job tại [continue](../docs/operations/continue.md).
Launcher/watcher/analysis của study đã đóng đã chuyển [archive16/11](../archive/operations-20261116/README.md).
Không chạy lại study từ notification cũ.

| Tác vụ | Entry point |
|---|---|
| Train FL / recovery metadata | `run_fl.py`, `run_q20.py` |
| Sinh notebook / import checkpoint | `gen_notebook.py`, `resume_checkpoint_cells.py` |
| Sim foldtrain→foldval cho hướng mới | `sim_fold_val.py` (panel200xe); gate/seed mới theo [task_brief](../task_brief.md) |
| Batch sim Lite2 (quota/launch/collect/gate) | `run_sim_lite2.py`; luật ở [screening-lite2](../docs/research/screening-lite2.md) |
| Batch sim FB (cost/quota/launch/collect/gate), estimator compute theo dòng | `run_sim_budget.py`; luật ở [screening-budget](../docs/research/screening-budget.md) |
| Pilot full-scale Lite2 (quota/launch/collect/audit) + nối r21–50 (`… r50`) | `run_lite2_full.py`; control T3P seed 42 |
| Audit CM/source/support / sim | `audit_screening.py`, `audit_sim_batch.py`, `research_metrics.py` |
| Phân tích nút thắt/cohort/learning curve | `analyse_core_bottleneck.py`, `analyse_b20_cohort.py`, `analyse_saturation.py` |
| Pull checkpoint / CLI bounded | `pull_checkpoint_parallel.py`, `kaggle_cli_bounded.py`, `run_bounded_python.py` |
| Watch exact kernel | `watch_kaggle_batch.py` qua codex-long-jobs,1200s |
| Local guard/fixture/benchmark/central ceiling | `run_local_guarded.py`, `cache_subset.py`, `bench_grouped.py`, `run_central.py` |
| Báo cáo / link | `report.py`, `check_docs.py` |

## Helper frozen còn là dependency

Tên study cũ dưới đây vẫn được driver/test/helper đang giữ import hoặc đọc. Giữ nguyên byte và đường dẫn;
**không coi hàm `main`/launcher bên trong là quyền chạy lại study**. Không tách/refactor chỉ để đổi tên.

`analyse_f50.py`, `analyse_m20.py`, `audit_b18.py`, `audit_f50.py`, `audit_m20.py`, `audit_r20.py`, `audit_uplink.py`, `audit_w20.py`, `audit_y50.py`, `b18_partition.py`, `build_sessions.py`, `check_screening_startup.py`, `direct_generator.py`, `f50_common.py`, `launch_repeat20.py`, `marginal_copula_generator.py`, `marginal_direct_generator.py`, `measure_bootstrap_codec.py`, `p20_common.py`, `preflight_f50.py`, `preflight_q20.py`, `preflight_r20.py`, `preflight_w20.py`, `probe_attack_twins.py`, `probe_pair_generator.py`, `q20_common.py`, `run_f50.py`, `run_p21_recovery.py`, `run_w20.py`, `run_y50.py`, `stage_r20_input.py`, `standard_copula_generator.py`, `w20_common.py`, `x20_common.py`, `y50_common.py`.

[Migration manifest](../archive/operations-20261116/migration.json) ghi mọi script dời/giữ và hash;
`core/`, `methods/`, `tests/`, notebook, runs/artifacts và DVC metadata giữ nguyên.
Npy cache canonical theo `runs/w-study/builder.json`; workspace DVC partial xem [backup](../docs/operations/backup.md).
