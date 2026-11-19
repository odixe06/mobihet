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
  BERT (`../fl_veremi_bert`, `run_t3.py --subset 100 --cache cache --compile`): cùng mẹo V4_ROOT, cache `sub100`; thiếu `--compile` thì
  method mobihet_* dừng ở "Q20 requires all compile gates PASS". ≈ 170 s/round. Port check loss/grad phải `import scripts.run_t3`
  trước (thay `core.local` → `local_t3` như trên Kaggle).
- Dataset nhiều file phải hết pagination. Kaggle có thể tự giải nén zip; kiểm member path và SHA, không upload lại vì tên archive tải404.

## Push, resume và audit

- Credential process-scoped; quota/slot đọc live trước push. Account có quota chưa bảo đảm cấp GPU; dự án khác có thể cùng dùng account.
- **Sự cố Kaggle 17/11 là lỗi mạng chập chờn/timeout**, gây quota `ConnectTimeout` và status `QUERY_ERROR`;
  nhiều ref hỏi lại không đổi đã trả `COMPLETE`. Không ghi thành lỗi training/kernel hoặc hết quota.
  Probe sau failure13:48: DNS local/Google cùng API IPv4, TCP443 tới API IP timeout xen kẽ trong khi www đáp nhanh; www cũng có connect timeout.
  Khi cần, bật riêng tiến trình `MOBIHET_KAGGLE_API_VIA_WWW=1` với `kaggle_cli_bounded.py`/`run_bounded_python.py`: SDK dùng HTTPSwww/api/v1;
  chỉ retry connect trước gửi request (3 lần), read/status/other retry0 để không nhân đôi push có reply bị mất. Không sửa DNS, SDK cài đặt hoặc credential toàn máy.
  Endpoint quota có thể `ConnectTimeout` ≈ 145 s trong khi `kernels list` vẫn chạy 4 s: đo một account trước,
  quota lỗi = chưa rõ, không suy quota từ lần đọc cũ.
- **Mặc định 18/11:** quota dùng `kaggle_quota.py --out …` (tách khỏi kernel list; subprocess/account, timeout auth+API, 3 lần backoff+jitter,
  4 song song; 23 account ≈ 15 s). Exit3 → `--retry-failed` chỉ hỏi account lỗi. `UNVERIFIED`/`gpu_left:null` ≠ 0 h.
  `W.inventory` không còn chết cả lượt vì một account: lưu `quota.partial.json` (row OK ≤ 600 s), receipt chỉ ghi khi đủ.
  Nguyên nhân gốc cũ: `account_quota` in-process timeout=None + `pool.map` ném exception đầu tiên.
- Mạng chập chờn: retry cả batch gate làm lặp quota/list/input đã PASS và dễ trượt ở query khác. Recovery FB lưu từng read thành công tối đa600s, retry đúng bước lỗi; không cache lỗi hoặc mutation. Trước retry push vẫn kiểm intent/exactversion; cache không thay source/config/round audit.
- CLI wrapper bên trong `kaggle_as.py` chưa bảo vệ auth của chính helper: `GenerateAccessToken` xảy ra trước child CLI, từng timeout với timeout=None dù flag route đã bật. Dùng `kaggle_as_bounded.py` bọc **trước auth**, giữ shape argv `[python, wrapper, owner, --]` để collect prefix không hỏng. Regression phải chạm auth phase, không chỉ CLI con.
- Push fail/timeout có thể để intent hoặc đã tạo kernel. Kiểm reply, exact version và marker trước retry; không push trùng từ exit code đơn lẻ.
- Resume phải import đúng parent, source/config/round và cả2rank trước decode; dùng copyfile/copytree cho file/thư mục.
  W&B resume không nạp checkpoint; snapshot nhỏ không thay private bank. Baseline cũ thiếu torch RNG phải báo giới hạn.
- Budget gồm setup/import/compile/train/eval/diagnostic/finalsave theo worst round + margin. COMPLETE thiếu đích là partial, không utility FAIL.
- Pull COMPLETE và ERROR, exact owner/slug/version, all pages, resumable/parallel khoảng32 connection tổng.
  Range phải kiểm206/Content-Range/size; giữ `.part`. Không SIGSTOP pull; audit nặng tuần tự theo RAM.
- CM/support→10metric không là prediction replay. Per-client bank/state phải có đủrank; không sửa method chỉ để snapshot/checker tiện hơn.
- QUEUED/RUNNING/W&B state/query lỗi không thay artifact audit. Query lỗi là UNKNOWN, không là slot trống hoặc kernel FAIL.

- 19/11: inventory một account có thể hỏng vì kernel **của dự án khác** trả 404 (`dinhan25/sgs-echohack2-c4`) ⇒ trạng thái slot không rõ, không push.
  Batch chưa ghi intent/push thì mở khóa an toàn bằng cách dời `batch.json` + notebook đã sinh rồi build lại cho owner khác.
- Gate quota `gpu_left ≥ CAP×1,15` đọc quota còn lại, **không trừ kernel đang chạy** trên account: đặt kernel thứ hai lên account đang chạy
  phải tự cộng CAP của cả hai.

## Job nền, backup và bàn giao

- Codex dùng `codex-long-jobs`, poll Kaggle1200s bằng worker; một collector/ledger. Xác nhận running một lần rồi để completion đánh thức phiên.
- Sau reboot kiểm PID/start identity/lock và completion marker thật; state `running` có thể stale. Notification trùng không cho phép rerun launcher.
- Watcher shell phải truyền exit lỗi collect/audit/gate; chỉ ghi `DONE` sau đủ artifact và mọi kiểm tra thành công. `echo "audit rc $?"` rồi ghi `DONE` vô điều kiện có thể che audit FAIL.
- `setsid nohup cmd & echo $!` ghi PID của `setsid` (nó fork khi shell là group leader) ⇒ PID sai; dùng `(setsid cmd & echo $! > pid)` trong subshell
  rồi kiểm `ps -p`. Đừng dừng job bằng `pgrep -f <mẫu>` khi chính dòng lệnh chứa mẫu đó: tự giết shell (exit 144). Dừng theo PGID đã ghi.
- Chỉ hủy job khỏe khi owner yêu cầu. Dùng controller hủy process group; không `pkill -f` pattern trùng shell.
- DVC add/status/push cùng repo chạy nối tiếp; không phá lock job khỏe. Workspace partial không add đè manifest đầy đủ.
- OneDrive phải là mount thật trước ghi. DVC mount-sync khác cloud upload; VFS dirty còn chiếm đĩa và phải giữ cache/local artifact.
- Không xóa dữ liệu chỉ từ quyền backup. Trước cleanup cần đúng phạm vi owner + object/size/manifest/recovery verification.
- Git chỉ stage thay đổi được giao; không credential/notebook inlinekey, không `Co-Authored-By:`. HEAD/index của phiên khác giữ nguyên.
- Link checker active: `python scripts/check_docs.py`, loại archive/runs/artifacts. Snapshot lịch sử không là nguồn trạng thái sống.

- Checker lịch sử phải giữ đúng inventory ban đầu; glob mọi method sau khi thêm method mới có thể báo FAIL ngoài scope.
  Pytest dùng `python -m pytest -q tests`, không quét frozen fixtures trong runs/archive.

- 19/11: gate quota "một phiên đủ CAP" không phải lý do để chờ reset. Không account nào đủ CAP×1,15 ⇒ chạy session 1 với CAP = gpu_left/1,15
  trên account quota ngắn, rồi nối cross-account. `run_t3` budget stop lưu `state.pt`+`state_rank1.pt` ở round cuối; pattern collect pilot
  (`run_lite2_pilot.PULL`) **không** lấy state ⇒ hop pull riêng `^<slug>/`, dataset private `<owner mới>/<slug>-p2` (files ở gốc để
  `**/<run>/state.pt` khớp), session 2 `--run <slug>-p2`. `kernels push` báo `ConnectTimeoutError` = request chưa gửi; vẫn kiểm inventory
  (slug không có trong versions) rồi mới đổi tên intent và push lại.
- 19/11: dừng `queue.py` bằng `kill -- -<PGID queue>` **không** giết job đang chạy: `timeout` (trong queue) và `run_local_guarded.py` tách process
  group riêng ⇒ job con vẫn giữ GPU. Sau khi dừng phải kiểm `nvidia-smi --query-compute-apps` và dừng thêm các PGID con.

- 19/11 (owner): margin quota mỗi account chỉ ≈ 1 phút (CAP ≈ gpu_left − 1/60 h, gate gpu_left ≥ ΣCAP + 1/60 h), thay `×1,15`
  và `gpu_left/1,15`. Budget stop của run_t3 lưu state round cuối nên thiếu round thì nối phiên, không cần biên lớn.
- 19/11: BERT short-quota session 1 chạy nhanh hơn ước tính (compact): L2/Q2/P2 tới r20 trong CAP 4,5–4,8 h, Q4 (CAP 3,68 h) dừng r19
  (`budget stop before round 20`). Ước ≈ 0,17 h/round + ≈ 0,1 h setup, không phải 0,31 h. Hop pull `state.pt`+`state_rank1.pt` ≈ 1,2 GB.
- 19/11: collect 12 kernel song song (DAGSNet 2 luồng × `-j8`, BERT 2 tiến trình × `-j16`) xong ≈ 2 phút; output pilot nhỏ (9–27 MB/kernel).
- 19/11: **dataset và kernel cùng owner không được trùng slug.** Hop cũ đặt dataset resume `<owner>/<slug>-p2` = slug kernel session 2 ⇒
  `kernels push` trả `409 Conflict` (SaveKernel) lặp lại, kernel không được tạo (status Not found). Sửa: kernel/run `<slug>-s2`, dataset
  `<slug>-s2-state` chứa thư mục `<run>/` (CLI `--dir-mode zip` → Kaggle giải nén, glob `**/<run>/state.pt` vẫn khớp); push đầu tiên OK.
  409 trên push = xung đột tên, không phải mạng: kiểm slug kernel/dataset của owner trước retry.
- 19/11: **upload một thư mục duy nhất (`--dir-mode zip`) → Kaggle giải nén zip vào gốc dataset, mất tên thư mục.** Kernel `-s2` ERROR ở gate
  resume (`missing … <run>/state.pt`, trước decode, không train). Mount thực = `/kaggle/input/**/<dataset slug>/state.pt` ⇒ `gen_notebook.py`
  glob theo slug dataset khi `--resume-dataset` (tương thích thiết kế cũ slug dataset = run). Listing API chỉ trả basename, không chứng minh
  layout thư mục; muốn biết chắc phải xem gate resume chạy thật. Kernel thay thế `-s2b` dùng lại dataset `-s2-state` đã kiểm size.
- 19/11: khi queue đo **time** đang chạy, không chạy pytest/benchmark GPU song song (đã lỡ trùng `bert-nilm time` 19:57 ⇒ JSON đổi tên
  `*.perturbed-by-pytest-1957.json`, pass 2 đo lại). Pull/collect Kaggle chạy trước queue time nên không trùng; job count (FLOP) không bị ảnh hưởng.

- 20/11: dataset Kaggle tạo từ thư mục có hàng nghìn file nhỏ (`inductor-cache/` 5.581 file trong output phiên) upload báo thành công
  nhưng không bao giờ được tạo (status 403, `datasets list --mine` rỗng) — lặp lại trên 2 account. Chỉ đóng gói file cần resume
  (state, rounds, config, cms); thấy 403 sau "being created" quá 10 phút ⇒ kiểm số file trước khi đổi account.
