# Kaggle — quy trình hiện hành

Job/owner/quyền/round sống chỉ tại [continue](continue.md). Đọc skill `kaggle-training-notebook`;
tối ưu train dùng `federated-training-speed`, health dùng `wandb-training-monitor`, nền dùng `codex-long-jobs`.
[Sổ lỗi](notebook-errors.md) · [Lessons](lessons.md). Roster/quota/account lịch sử đã ra khỏi tuyến đọc.

## Credential và preflight

- Env `nckh`; runtime pin trong `environment.yml`/knowledge. Credential ngoài repo `~/.kaggle/accounts/`, process-scoped.
  Không đổi global login hoặc in token/email/signed URL. Introspect đúng owner khi cần.
- Query quota/slots live trước push, cộng budget đã reserved và margin; dự án khác có thể cùng dùng account.
- Private notebook + source/checkpoint private; giữ dữ liệu public canonical theo protocol. Notebook inlinekey không vào Git/DVC.
- Pin base SHA/overlay/HP/notebook SHA; notebook và metadata cùng khai2×T4sm75/image đãpin. Không bỏ hardware/finite guard.
- Source script active xem [scripts](../../scripts/README.md); launcher study đã đóng trong archive không dùng cho batch mới.

```bash
/home/odixe/Tools/miniconda3/envs/nckh/bin/python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py quota
/home/odixe/Tools/miniconda3/envs/nckh/bin/python .agents/skills/kaggle-training-notebook/scripts/kaggle_as.py OWNER -- kaggle kernels status OWNER/SLUG/VERSION
```

## Resume / collect / audit

- Resume đúng compatible parent/source/config/seed/schedule; import cả2rank trước decode. Giữ checkpoint/RNG mà driver yêu cầu;
  báo thiếu RNG ở baseline lịch sử. W&B resume không thay model state.
- Giữ1epoch thật, đủclient/tailbatch, endpoint khóa; tính setup/compile/eval/save vào worst-round budget.
- Poll exact `owner/slug/version`1200s bằng một worker/ledger qua controller; ghi job/PID/state/log/thread trong continue.
  Xác nhận running một lần, chỉ còn chờ thì kết thúc turn; không modelpoll, không duplicate collector.
- Pull cả COMPLETE/ERROR, exact version, hếtpage, resumable/parallel khoảng32connection tổng; kiểmsize/Range, giữpart khi lỗi.
- COMPLETE/exit0 chỉ là vận hành. Require source/HP/LR/support/CM→10metric/finite/đủround;
  state nhỏ không chứng minh fullprivatebank hoặc prediction replay. Query lỗi=UNKNOWN.
- Retry phải kiểm intent/reply/version trước; lỗi watcher/pull/checker chỉ sửa đúngstage, không training lại.
  Dừng/cancel job khỏe chỉ theo yêu cầu owner; không dùng `push --no-run` như stop.
