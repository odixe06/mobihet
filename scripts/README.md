# Công cụ đang dùng

Chạy từ repo root, env conda `nckh`. Script trong archive không thuộc workflow hiện tại.

| nhiệm vụ | công cụ |
|---|---|
| FL theo protocol, artifact mỗi round | `run_fl.py` |
| Notebook private, pinned source/overlay | `gen_notebook.py` |
| Status Kaggle nền, không model polling | `watch_kaggle_batch.py` + skill codex-long-jobs |
| CLI/network có timeout | `kaggle_cli_bounded.py`, `run_bounded_python.py` |
| Audit M20, giữ exact version | `audit_m20.py`, `audit_screening.py` |
| Phân tích M20 và thống kê dùng chung | `analyse_m20.py`, `research_metrics.py`, `report_m20.py` |
| Local probe có memory guard/cache | `run_local_guarded.py`, `cache_subset.py` |
| Đo grouped/runtime và width gate | `bench_grouped.py`, `run_central.py` |
| Bảng metric từ CM | `report.py` |
| Kiểm link active | `check_docs.py` |

Mọi run cũ dùng source/config đã đóng băng trong provenance/DVC, không lấy code HEAD mới thay ngầm.
Sau cleanup, auditor snapshot cần restore DVC trước; phân tích thường dùng evidence nhẹ. Xem backup record.

M20: `python scripts/analyse_m20.py` đọc evidence nhẹ; `python scripts/report_m20.py` tái tạo báo cáo/figure.
`python scripts/audit_m20.py` cần restore đủ `artifacts/m20/*.dvc`; không đọc archive.
