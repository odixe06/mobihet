# Kaggle — account và quy tắc vận hành

Lỗi kernel đã gặp và cách tránh: [sổ lỗi](notebook-errors.md) (đọc trước khi sinh notebook). Kernel đang chạy: [continue](continue.md).
Lịch sử cấp account/quota 15–17/10: [archive](../archive/operations/kaggle-accounts-history-2026-10-15-to-17.md).

## Account

16 account đã lưu credential (ngoài repo, `~/.kaggle/accounts/`, thư mục 0700, token 0600; không ghi token/email vào docs):
accanniez, anhtruong1909, annhannh331, annhtnt, catbaochau, dinhan25, khanhmay0304, minhtran0601, minhtriethihi, minhtrit06,
nguyenminhtran060113, odixe0502, odixeuit, takids1909, thuyanh316, trietbackup.

| lưu ý | account |
|---|---|
| Login global CLI giữ nguyên (không đổi khi làm việc account khác) | thuyanh316 |
| `kaggle datasets create` bị 403 (đọc/push kernel vẫn được) ⇒ dùng dataset share từ account khác | odixeuit, minhtrit06 |
| Refresh token OAuth hỏng (hook 20/10: "refresh token did not mint a token") ⇒ owner đăng nhập lại trước khi dùng | minhtrit06 |

## Quy tắc

- Mọi lệnh chạy theo account bằng `kaggle_as.py`, credential chỉ trong process:
  `conda run -n nckh python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <user> -- kaggle kernels status <user>/<kernel>`.
- Quota: được query mọi account (owner 17/10) bằng `kaggle_account.py quota`; reset hằng tuần thứ Bảy 00:00Z (gần nhất 24/10).
  Chỉ push khi quota live ≥ 1,15 × giờ session; tối đa 2 session GPU cùng lúc mỗi account.
- Dataset source/checkpoint phải PRIVATE, upload lên **account chạy kernel**, verify đủ file/SHA trước khi push.
- Notebook public (`notebooks/`) không chứa key; bản push (`private_push/`, git-ignored, 0600) mang key W&B inline theo quyền owner đã cấp
  (notebook `is_private: true`). Đăng nhập W&B và pin phiên bản: [skill wandb.md](../../.claude/skills/kaggle-training-notebook/references/wandb.md).
- `QUEUED` = chờ GPU, không phải lỗi. `RUNNING`/`COMPLETE`/W&B finished không chứng minh đã train: phải có log/heartbeat và audit artifact.
  Log live: `kaggle kernels logs --follow` (SSE) giới hạn ~35 s; `kernels logs` thường rỗng khi đang chạy.
- Dừng session chỉ làm được trên web (owner bấm **Stop session**); API không hủy được.
- Kaggle native MCP chưa cấu hình; CLI có xác thực là kênh chính.
