# Scripts đang dùng chung

Chạy từ repository root; Python/Kaggle trong env `nckh`. Train nặng trên Kaggle.
Helper riêng nằm trong build/scripts; công cụ cũ nằm trong [archive/tools](../archive/tools/README.md).
CrossClientRank probe/utility/operational recovery tại [build scripts](../methods/cross-client-ranking/builds/rank-utility/README.md).

| công cụ | nhiệm vụ |
|---|---|
| [update_file_index.py](update_file_index.py) | Index hiện tại + index archive, route artifact |
| [project_paths.py](project_paths.py) | Chuyển đường dẫn lịch sử sang canonical hiện tại |
| [check_package_layout.py](check_package_layout.py) | Import, SHA, runtime copy, source parent |
| [check_project_layout.py](check_project_layout.py) | Source frozen, Git/DVC, links, credentials |
| [check_working_helpers.py](check_working_helpers.py) | Import/path reads các helper đang dùng |
| [restore_compat_aliases.py](restore_compat_aliases.py) | Dựng alias/import facade sau checkout |
| [restore_dvc_metadata.py](restore_dvc_metadata.py) | Restore chọn lọc, kiểm .dir/object |
| [build_pilot_fixture.py](build_pilot_fixture.py) | Fixture cho thí nghiệm ngắn |
| [run_local_guarded.py](run_local_guarded.py) | Memory guard cho local smoke/audit |
| [check_pull_range.py](check_pull_range.py) | Kiểm pull Range resume |
| [check_resume_batch.py](check_resume_batch.py) | Actual emitted controller/continuation checks |
| [gen_full_resume_batch.py](kaggle/gen_full_resume_batch.py) | Nối H1/H2/Hybrid đúng audited parent |
| [read_resume_inputs.py](kaggle/read_resume_inputs.py) | Owner-scoped input access |
| [launch_prepared_batch.py](kaggle/launch_prepared_batch.py) | Push batch được cho phép, kiểm quota/refused reply |
| [check_project_status.py](kaggle/check_project_status.py) | Status exact versions, chưa là scientific evidence |
| [push_nb.py](kaggle/push_nb.py) | Owner-scoped immutable notebook push |
| [inspect_resume_startup.py](kaggle/inspect_resume_startup.py) | Startup health; ổn định thì dừng polling |
| [prepare_cross_account_resume.py](kaggle/prepare_cross_account_resume.py) | Private checkpoint bridge từ audited parents, không đổi scientific source |
| [verify_checkpoint_dataset.py](kaggle/verify_checkpoint_dataset.py) | All-page names/sizes, exact hoặc một prefix bị strip; CPU import kiểm SHA |
| [audit_mount_probe.py](kaggle/audit_mount_probe.py) | CPU production-import proof, exact rounds/state/markers trước GPU push |
| [verify_source_dataset.py](kaggle/verify_source_dataset.py) | Names/sizes của flat source/fixture dataset |
| [inspect_new_pilots.py](kaggle/inspect_new_pilots.py) | Exact ClassTail/repair W&B identities; operational snapshot |

Namespace `scripts` tìm helper cũ trong `.compat/scripts`; alias cũ không nằm trong thư mục làm việc.
Path reads dùng current_path; script frozen của Kaggle vẫn dựng runtime nguyên bản.
[Hồ sơ đang làm](../docs/methods/README.md) · [Sổ lỗi](../docs/operations/notebook-errors.md) · [Index](../docs/file-index.md).
