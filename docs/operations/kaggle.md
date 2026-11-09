# Kaggle — quy trình hiện hành

Job/account/round/quota sống chỉ tại [continue](continue.md). Đọc [error register](notebook-errors.md) trước tạo notebook;
áp dụng skill `kaggle-training-notebook`, `federated-training-speed`, `wandb-training-monitor` và `codex-long-jobs` theo tác vụ.
Lịch sử account/API trước07/11 ở [snapshot](../../archive/docs-20261107/before/docs/operations/kaggle.md), không dùng làm quota hiện tại.

## Identity và cấp quota

Env `nckh`, CLI/runtime pin trong `environment.yml` và [knowledge](../../knowledge/README.md).
Saved credentials nằm ngoài repo trong `~/.kaggle/accounts/`; không đưa token/email vào docs/log.
Dùng process-scoped wrapper, giữ global login hiện có; không yêu cầu MCP/reconnect cho CLI.

```bash
conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py quota
conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/kaggle_as.py OWNER -- kaggle kernels status OWNER/SLUG/VERSION
```

- Kiểm saved-owner introspection, quota còn lại/reserved và session slots trước push; tài khoản có quota chưa chắc được cấp GPU.
- Inventory authorized owners của workflow ở `scripts/run_w20.py::inventory`; quota snapshot không là reservation.
  Hai slot GPU/account và margin1,15 là planner hiện tại; push vẫn phải qua phản hồi service.
- Ưu tiên resume theo quyền mới nhất trong continue, có thể dùng quota ngắn. Reset dự kiến07:00UTC+7 không thay fresh quota check.
- Được cross-account continuation; receiver checkpoint dataset private + verify thật trước GPU. Không mount private kernel chéo account.

## Notebook và resume

- Notebook lẫn metadata đều khai báo2T4sm75, pinned image digest; fail trước train nếu sai hardware.
- Nguồn base SHA + overlay, metadata/notebook hash, frozen scientific config và input access phải khớp gate.
- Npy cache public: `odixe0502/mobihet-v4npy-cache`, manifest trong `runs/w-study/builder.json`; verify byte SHA trước đọc.
- Model training compile, threads1; cache Inductor ởtemp và một tar trong run. Không cắt client/batch/epoch/diagnostics để tăng tốc.
- Resume import duy nhất, config/source/seed/rounds và state/RNG cả hai rank cùng round; gate **trước decode**.
  Giữ out/W&B ID gốc sau verified continuation; slug session khác. W&B resume không thay state.
- Sim và KM full-scale hiện **tắt W&B** (không nhúng credential). Khi bật: online, notebook private; inline key vĩnh viễn trong version history.
  Notebook sinh trong ignored `notebooks/`, không Git/DVC; không in key/signed URL.

## Worker, collection và kết luận

Worker: phiên Codex dùng global codex-long-jobs controller (state `~/.codex/long-jobs/jobs/<id>/state.json`); phiên Claude Code dùng
`runs/<batch>/watch.sh` chạy bash nền (completion đánh thức phiên). Cả hai poll 1200 s, ghi PID/log vào continue. Không model polling khi chỉ chờ. Trước retry kiểm intent/receipt/lock/job đang sở hữu ledger.

Pull exact owner/slug/version, all-page listing, resumable/parallel, cả ERROR; khoảng32 connection tổng.
File4GB dùng Range chunks có Content-Range/size gate. Private dataset upload giữ tabular, verify download SHA/round thật.

`COMPLETE` chỉ vận hành: audit source/config/LR/IDs/support/CM→10metric/snapshot/teacher/finite và đủ endpoint mới kết luận.
Partial an toàn → audit partial → nối compatible state; lỗi watcher/download/audit → sửa stage hỏng, không restart training.
Nếu kernel cần dừng mà service không có cancel phù hợp, báo owner exact session và thao tác UI; không dùng push --no-run như stop.
