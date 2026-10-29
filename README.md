# MobiHet

Federated IDS cho V2X: mỗi xe giữ model private (kiến trúc tùy ý), chỉ trao đổi một proxy nén (Q3 + LZMA2) với băng thông hạn chế.
Phiên bản 2 trên VeReMi NextGen v4 — **5.343 client = 5.343 xe nhận**, 16 lớp, DAGSNet, cùng protocol với 15 baseline của
[`odixe06/fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2). Lịch sử v1: [history/v1](history/v1/README.md),
repo [`odixe06/mobihet-archive-veremi-v1`](https://github.com/odixe06/mobihet-archive-veremi-v1).

- Dataset: Kaggle [`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`](https://www.kaggle.com/datasets/odixe0502/veremi-nextgen-v4-scaled-per-vehicle) — [mô tả](knowledge/dataset.md).
- Bắt đầu: [CONTEXT.md](CONTEXT.md). Env: conda `nckh` ([environment.yml](environment.yml)).

```
core/      DAGSNet tham chiếu + bản grouped (nhiều client/bước), dữ liệu thường trú GPU, AdamW theo client, eval, metric
methods/   mobihet.py (phương pháp), fedavg.py (đối chứng)
scripts/   run_fl.py, run_central.py, gen_notebook.py, cost.py, report.py, benchmark
tests/     test tương đương / đơn vị
knowledge/ dữ kiện bất biến (máy, dataset, DAGSNet)
docs/      protocol, kế hoạch cải tiến, chi phí, vận hành
history/v1 toàn bộ tài liệu + lesson MobiHet v1 (chỉ đọc)
```
