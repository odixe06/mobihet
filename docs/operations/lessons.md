# Bài học vận hành còn dùng

Đọc sau [CONTEXT](../../CONTEXT.md). Job sống: [continue](continue.md); quy trình: [Kaggle](kaggle.md),
[backup](backup.md); triệu chứng runtime: [sổ lỗi](notebook-errors.md).
Lịch sử nguyên bản trước rút gọn: [snapshot](../../archive/operations-20261116/before/docs/operations/lessons.md).

## Source, dữ liệu và notebook

- Env `nckh` dùng chung; không tự cài/gỡ package hoặc ghi đè `.pth`. Python tuyệt đối vẫn cần PATH có `nckh/bin` nếu subprocess gọi `kaggle`.
- Test đúng đường production, file/shape/seed/rank mà driver thực sự đọc. Local sm86 không chứng minh compile/AMP trên 2×T4sm75.
- Pin base SHA + overlay/HP/notebook hash trước gate/push; giữ source đã chạy. Sửa checker phải kiểm artifact cũ, không train lại để tìm PASS.
- Runtime/accelerator phải có trong cả notebook và metadata; preflight backward/finite/applied-step, không chỉ forward.
- Fingerprint JSON đúng kiểu; resource nhúng trong notebook khác file thực thi. Không tạo provenance giả để auditor PASS.
- Chuỗi `--set` đi qua shell/Python/JSON phải kiểm AST/shlex/json.loads; init random cell ID có thể đổi notebook SHA.
- Smoke sim local (`sim_fold_val.py run --n 200 --rounds 2 --compile`): v4 local chỉ là pointer DVC ⇒ đặt
  `V4_ROOT=<repo>/knowledge` (có `label_mapping.json`); dòng lấy từ `cache/train_sub200_*`. ≈ 100 s/round trên GPU local.
- Dataset nhiều file phải hết pagination. Kaggle có thể tự giải nén zip; kiểm member path và SHA, không upload lại vì tên archive tải404.

## Push, resume và audit

- Credential process-scoped; quota/slot đọc live trước push. Account có quota chưa bảo đảm cấp GPU; dự án khác có thể cùng dùng account.
- Inventory quota toàn roster (22 account) mất ≈ 4 phút: chạy nền/ghi file rồi đọc, đừng để foreground timeout cắt giữa chừng.
  **Sự cố Kaggle 17/11 là lỗi mạng chập chờn/timeout**, gây quota `ConnectTimeout` và status `QUERY_ERROR`;
  nhiều ref hỏi lại không đổi đã trả `COMPLETE`. Không ghi thành lỗi training/kernel hoặc hết quota.
  Probe sau failure13:48: DNS local/Google cùng API IPv4, TCP443 tới API IP timeout xen kẽ trong khi www đáp nhanh; www cũng có connect timeout.
  Khi cần, bật riêng tiến trình `MOBIHET_KAGGLE_API_VIA_WWW=1` với `kaggle_cli_bounded.py`/`run_bounded_python.py`: SDK dùng HTTPSwww/api/v1;
  chỉ retry connect trước gửi request (3 lần), read/status/other retry0 để không nhân đôi push có reply bị mất. Không sửa DNS, SDK cài đặt hoặc credential toàn máy.
  Endpoint quota có thể `ConnectTimeout` ≈ 145 s trong khi `kernels list` vẫn chạy 4 s: đo một account trước,
  quota lỗi = chưa rõ, không suy quota từ lần đọc cũ; `W.inventory` hỏng cả lượt nếu một account timeout → chạy nền có lặp.
- Push fail/timeout có thể để intent hoặc đã tạo kernel. Kiểm reply, exact version và marker trước retry; không push trùng từ exit code đơn lẻ.
- Resume phải import đúng parent, source/config/round và cả2rank trước decode; dùng copyfile/copytree cho file/thư mục.
  W&B resume không nạp checkpoint; snapshot nhỏ không thay private bank. Baseline cũ thiếu torch RNG phải báo giới hạn.
- Budget gồm setup/import/compile/train/eval/diagnostic/finalsave theo worst round + margin. COMPLETE thiếu đích là partial, không utility FAIL.
- Pull COMPLETE và ERROR, exact owner/slug/version, all pages, resumable/parallel khoảng32 connection tổng.
  Range phải kiểm206/Content-Range/size; giữ `.part`. Không SIGSTOP pull; audit nặng tuần tự theo RAM.
- CM/support→10metric không là prediction replay. Per-client bank/state phải có đủrank; không sửa method chỉ để snapshot/checker tiện hơn.
- QUEUED/RUNNING/W&B state/query lỗi không thay artifact audit. Query lỗi là UNKNOWN, không là slot trống hoặc kernel FAIL.

## Job nền, backup và bàn giao

- Codex dùng `codex-long-jobs`, poll Kaggle1200s bằng worker; một collector/ledger. Xác nhận running một lần rồi để completion đánh thức phiên.
- Sau reboot kiểm PID/start identity/lock và completion marker thật; state `running` có thể stale. Notification trùng không cho phép rerun launcher.
- Watcher shell phải truyền exit lỗi collect/audit/gate; chỉ ghi `DONE` sau đủ artifact và mọi kiểm tra thành công. `echo "audit rc $?"` rồi ghi `DONE` vô điều kiện có thể che audit FAIL.
- Chỉ hủy job khỏe khi owner yêu cầu. Dùng controller hủy process group; không `pkill -f` pattern trùng shell.
- DVC add/status/push cùng repo chạy nối tiếp; không phá lock job khỏe. Workspace partial không add đè manifest đầy đủ.
- OneDrive phải là mount thật trước ghi. DVC mount-sync khác cloud upload; VFS dirty còn chiếm đĩa và phải giữ cache/local artifact.
- Không xóa dữ liệu chỉ từ quyền backup. Trước cleanup cần đúng phạm vi owner + object/size/manifest/recovery verification.
- Git chỉ stage thay đổi được giao; không credential/notebook inlinekey, không `Co-Authored-By:`. HEAD/index của phiên khác giữ nguyên.
- Link checker active: `python scripts/check_docs.py`, loại archive/runs/artifacts. Snapshot lịch sử không là nguồn trạng thái sống.

- Checker lịch sử phải giữ đúng inventory ban đầu; glob mọi method sau khi thêm method mới có thể báo FAIL ngoài scope.
  Pytest dùng `python -m pytest -q tests`, không quét frozen fixtures trong runs/archive.
