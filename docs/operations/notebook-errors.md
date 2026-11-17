# Sổ lỗi còn liên quan

Đọc trước sinh notebook. [Lịch sử chi tiết](../../archive/operations-20261116/before/docs/operations/notebook-errors.md)
giữ nguyên evidence; [continue](continue.md) giữ job sống. Không sửa source training đã khóa.

| Dấu hiệu | Kiểm / xử lý |
|---|---|
| Quota `ConnectTimeout`, status `QUERY_ERROR` (17/11) | Lỗi mạng chập chờn/timeout; đọc lại có giới hạn đúng account/ref. Giữ UNKNOWN tới khi đọc được; không coi kernel FAIL/hết quota/slot trống, không push lại training. Nếu tuyến API lỗi, thử opt-in `MOBIHET_KAGGLE_API_VIA_WWW=1` qua hai wrapper bounded; retry connect-only, vẫn kiểm intent/version trước retry mutation |
| Zero GPU/sai runtime | Notebook + metadata, pinned image,2T4sm75; giữ assert, không reinstall torch |
| RUNNING chưa log/heartbeat | Exact version và bounded log; QUEUED/setup khác training lỗi; không push trùng |
| Push lỗi nhưng còn intent | Đọc reply/version/receipt và remote status trước retry; không xóa intent mù |
| Parent mount thiếu | Refused-input reply, actual import; checkpoint dataset private + SHA; require-resume trước decode |
| Rank state thiếu/lệch round | Cả2rank/config/source/RNG; snapshot nhỏ không thay full state |
| COMPLETE nhưng partial | Đếm rounds/CM/state; audit partial, nối compatible boundary, giữ endpoint |
| Pull thiếu / Range lỗi | Hếtpage, exact version,206/Content-Range/size; giữpart và retry stage tải |
| Compile backwardNaN/skip | Real data + paddedslots + từngclient applied; không hạguard/bỏclient đểqua |
| Auditor falseFAIL | Recheck stored values/source/rounding bound; sửa checker rồi audit artifact cũ |
| BERT buffer rỗng | Layout.B=0; không cat danh sách tensor rỗng |
| BERT pruned SDPA head-dim5 | Giữ fallback eager đã kiểm; không đổi model/dropout để né compiler |
| W&B stale/crashed | Đối chiếu exact Kaggle/session/artifact; không hủy từ W&B đơn lẻ |
| Resume history trùng | Rollback tới checkpoint boundary; không advance logging trong preflight |
| Watcher chết/reboot | PID/start identity/state/ledger; phục hồi collection, không restart training |
| OneDrive startup lâu/mount mất | Journal/service/timeout và mount thật; giữ process/cache VFSdirty, không lặp restart |

Guard phát hiện vấn đề; PASS một fixture không chứng minh toàn trajectory hoặc root cause đã sửa.
