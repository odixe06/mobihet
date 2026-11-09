# Bài học vận hành

Job sống: [continue](continue.md). Quy trình: [Kaggle](kaggle.md), [backup](backup.md), [error register](notebook-errors.md).
Chi tiết case cũ giữ trong [snapshot](../../archive/docs-20261107/before/docs/operations/lessons.md), không đọc mặc định.

## Account, input và push

- Credential theo process qua `kaggle_as.py`; global login có thể đổi bởi phiên khác. Introspect từng saved owner, không hardcode identity cũ.
  Token-only không đồng nghĩa auth hỏng; CLI không cần MCP JSON/reconnect.
- Fresh quota/slots, reserved budgets và exact owner/slug/version trước push. Ref rỗng từ listing phải lọc đúng owner;
  permission denied của private slug chưa tồn tại không tự chứng minh token hỏng.
- `run_w20.inventory()` trả `quota.json` đã có trong thư mục batch: `run_km50.py quota` sau batch cũ in quota **ngày cũ** và session `active` cũ.
  Quota sống = inventory trong thư mục batch mới chưa có receipt (09/11: catbaochau cũ 28 h → thật 7,8 h).
- Kaggle SaveKernel trả **400 Bad Request** cho notebook 1,47 MB (overlay npz base64); kernel không được tạo. Giữ notebook ≲ 0,6 MB
  (b12 538 KB push được): file dữ liệu lớn đi qua dataset, overlay chỉ cho file nhỏ đã nén.
- Push có intent/receipt. Mất reply hoặc400/403: inspect owned listing/status trước retry, không push/create mù.
  Query lỗi = UNKNOWN; không restart training vì watcher/network/auditor lỗi.
- `kernel_sources` private không mount chéo owner. Chuyển account bằng private checkpoint dataset của receiver;
  giữ tabular bytes, verify all-page names/sizes/download SHA và import gate round/config/state2rank trước decode.
- Dataset create có thể success trước ready/listing đọc được. Retry phần đọc có giới hạn, giữ intent/result;
  không suy403 tạm thời là auth hỏng hoặc upload lại.
- Zero GPU có thể theo account; giữ assert2T4/pinned image và thử runtime đúng trước sửa code. Không reinstall torch/bỏ assert.

## Train, collect và audit

- Notebook truyền child exit code; provenance = base SHA + overlay + executed files. Chụp hash trước/sau preflight;
  source thay khi module đã import làm proof cũ không đủ launchgate.
- Local fixture phải là data/shape driver đọc thật. Compile gate kiểm cả padded slot; fixture hai xe group_rows128 đã dùng,
  không hạ gate để che padded NaN. Seed/rank/group T4 thật vẫn cần gate riêng.
- State W1 hai rank khoảng8,5GB. Giữ một state/rank để không nhân đôi vượt working cap; snapshot nhỏ không đủ resume.
  Import dùng copyfile để không mang read-only permission. W&B resume không nạp state.
- Budget cả setup/import/compile/eval/diagnostic/finalsave theo worst round, cộng margin. Cap ngắn có thể chỉ commit r1;
  audit partial rồi nối checkpoint, không đổi endpoint khoa học hoặc gọi utility FAIL.
- Pull COMPLETE **và ERROR**, exact version, hết page. Khoảng32 kết nối tổng; ít file4GB cần parallel Range trong từng file,
  kiểm206/Content-Range/size trước assemble, giữ partial/retry; không in signed URL. Audit nặng nối tiếp theo RAM.
- Stored packet/source/CM/tensor mới chứng minh run. Auditor rounding lỗi thì sửa/check artifact cũ;
  giữ failure và recovery record, không train lại. NumPy bool gate phải cast `bool`; có CM/source/sampleSHA thì recover JSON không refit.
- Smoke local `sim_fold_val.py run` phải có `--compile` (Live round assert compile gate mọi rank); 200 xe 2 round ≈5,5 phút,
  peak RSS ≈9,8 GB — kiểm `free -h` trước, chạy qua `run_local_guarded.py`.
- Owner tắt máy khi kernel đang chạy: không cần watcher; ghi lệnh `collect` + analyser vào continue, phiên sau chạy lặp collect.
- Npy cache public dùng chung + manifest SHA; private weights vẫn private. Compile cache để temp, nén một tar trong run và restore qua session.
  Tar restore thành công chưa chứng minh cachehit/speedup; benchmark trên runtime thực.

## Background và bàn giao

- Dùng codex-long-jobs (Codex) hoặc `runs/<batch>/watch.sh` nền (Claude Code); poll Kaggle 1200s bằng local worker. Xác nhận running một lần, rồi kết thúc turn khi chỉ còn chờ;
  completion về đúng thread. Không model polling hoặc hai watcher cùng ledger.
- Claude Code job nền chặn Edit/Write ở checkout chính trừ khi `.claude/settings.json` có `"worktree": {"bgIsolation": "none"}`
  (owner bật 09/11). Không dùng worktree từ `origin/main` cho docs: docs/kết quả chưa commit chỉ có ở working copy chính.
- Controller wrapper thiếu executable bit: `bash <wrapper> ...`. Exit126 trước controller chưa tạo job/kernel.
- Sau reboot, `running` trong state có thể stale: kiểm PID/start identity; giữ log/state cũ, phục hồi collector, không restart training.
- Chỉ hủy job khỏe khi owner yêu cầu hoặc handover cần thiết trong workflow đã được phép. Dùng controller hủy process group;
  notification muộn không là yêu cầu bật lại job cũ. Queue accepted không chứng minh đã xử lý completion.
- W&B heartbeat/state chỉ bổ sung evidence, không thay exact Kaggle session/artifacts. Inline key private lưu vĩnh viễn trong history,
  không đưa notebook/credential vào Git/DVC hoặc doc. Không dừng từ W&B crashed đơn lẻ.

## Backup và workspace

- Backup/restore/cleanup chỉ trong quyền owner. Kiểm mount thật, object manifest và cloud trước xóa;
  không xóa rclone/VFS dirty cache hoặc cloud objects. Mount copy không chứng minh cloud synced.
- Workspace DVC partial: giữ manifest, không `dvc add` ghi snapshot nhỏ đè bản đầy đủ. Restore đúng run cần dùng.
- Git HEAD/index/staged tree giữ nguyên khi chỉ tổ chức docs; không force push/trailer `Co-Authored-By:`.
- Account có thể dùng cho dự án khác: inventory thật trước cấp quota, không gán mọi quota giảm cho MobiHet.
