# Tài khoản Kaggle bổ sung — 15/10/2026

Chủ dự án cung cấp ba tài khoản để phục vụ công việc; đã đăng nhập CLI và có OAuth snapshot.
Credential nằm ngoài repository tại `~/.kaggle/accounts/`; tài liệu chỉ chứa metadata.
Phiên15/10: chủ dự án xác nhận mỗi account đủ30h tuần và yêu cầu không kiểm lại quota.
Batch continuation dùng xác nhận này, không gọi quota ba account; mỗi account2×9h, dư15% thành20,7h.

| Username | Email chủ dự án cung cấp | OAuth snapshot | Bearer |
|---|---|---|---|
| annhtnt | annhtnt@gmail.com | `annhtnt.credentials.json` | `annhtnt.mcp-token` |
| takids1909 | takids1909@gmail.com | `takids1909.credentials.json` | `takids1909.mcp-token` |
| accanniez | accanniez@gmail.com | `accanniez.credentials.json` | `accanniez.mcp-token` |
| nguyenminhtran060113 | nguyenminhtran060113@gmail.com | `nguyenminhtran060113.credentials.json` | `nguyenminhtran060113.mcp-token` |

Thư mục credential mode0700; bearer mode0600. Email là metadata do chủ dự án cung cấp,
không là identity do API xác nhận. Quota phiên15/10 dùng xác nhận chủ dự án ở trên;
GPU/session allocation phải xác nhận từ run;
việc bổ sung tài khoản không tự cho phép launch hoặc retry ngoài phạm vi đã duyệt.

Dùng helper trong skill đang đọc, `.agents` hoặc `.claude`:

```bash
conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py list
conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/kaggle_as.py annhtnt -- kaggle kernels list --mine
```

Thay username để thao tác với tài khoản khác. `kaggle_as.py` giữ credential theo process;
login global không đổi. CLI OAuth và MCP bearer độc lập; xác nhận cả hai bằng introspection.
MCP live cần native account-scoped read riêng; CLI thành công không chứng minh MCP live.

Kiểm tra 15/10/2026: cả ba OAuth snapshot introspect đúng username; cả ba bearer
`active=True`, introspect đúng username. Không đo quota, GPU allocation hoặc MCP live.

Bổ sung 15/10/2026 tối: owner đăng nhập CLI `nguyenminhtran060113` (account active hiện tại) và cấp
API token; token lưu `nguyenminhtran060113.mcp-token` mode0600. `health` = ok. Chưa đo quota/GPU;
quota account này chưa được owner xác nhận, đọc live trước khi phân việc.
Đọc live15/10 10:54Z: 30,0/30h GPU, reset 2026-10-17 00:00 UTC; owner giao cho ProxyFuse/AntiMem/QProxy.

Bổ sung 15/10/2026 ~14:50Z: owner cấp ba account mới chỉ có API token (KGAT, không OAuth):
`annhannh331`, `anhtruong1909`, `thuyanh316` (email metadata do owner cung cấp, không ghi token vào repo).
Token lưu `<user>.mcp-token` mode0600; `kaggle_as.py` dùng đường dẫn file. Đọc live: mỗi account 30,0h GPU, reset 17/10 00:00 UTC.
Account active CLI lúc đó là `thuyanh316` (owner đăng nhập), không ảnh hưởng push theo owner metadata.
Phân việc: annhannh331 + anhtruong1909 → FullDrift seed42; thuyanh316 → dự trữ full continuation.
`odixeuit`/`minhtrit06` bị 403 khi **tạo** dataset (đọc/push kernel vẫn được) — dùng dataset share từ owner gốc.

16/10/2026 ~04:00Z: **không dùng `kaggle_account.py quota`** (quét mọi account, kể cả annhtnt/takids1909/accanniez bị cấm query).
Đọc từng account bằng `kaggle_account.account_quota(user)`. Quota live và reset (17/10 00:00Z) ghi ở
[bàn giao](continuation-2026-10-15-evening.md) mục "Dừng phiên 16/10 ~04:30Z".
