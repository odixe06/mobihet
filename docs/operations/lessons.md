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
- B18 11/11: local4phase smokePASS nhưngremote2seed chếttrướcround1 vìruntime đọc
  `runs/sim/core_hp.txt` không tracked/khôngoverlay. Đónggói cả resourcecodeđọc,
  regression cloneGitHEAD+**chỉ declaredoverlay** rồiimport actualentrypoint; localfullcheckout không chứng minh deploymentđủfile.
  Fixpackaging giữsource/HP, re-audit smoke cũ rồislugmới;0checkpoint thìfreshstart, không gọi resume hoặcutilityFAIL.
- B18r1 12/11: cả2kernelCOMPLETE/4phase20round, localanalysisfail vìauditor đòiHP `.txt`
  trongprovenancePython-only. Phânbiệt embeddedresourceproof với executedfileproof:
  sealedpushednotebookSHA+payloadSHA+overlaymanifest+effectiveconfigs cóthểphục hồi HPproof đúngscope;
  khôngcopylocalHPgiả vàoexecutedsource vàkhôngskip `.py` hash. Futuregenerator copymọioverlayresource.
  CorruptionPython/HPconfig/notebookSHA phảibịtừchối; không trainlạivì thiếuprovenancecopy/auditor/writerJSONlỗi.
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
- B14 local preflight chết tại wire auditor sau smoke đủ2round: `run_b13_pipeline.py --reuse-local-smoke` chỉ phục hồi khi
  training overlay hash + HP còn khớp và chưa có quota/batch/push intent. Re-audit CM/state/compile/wire cũ, ghi source-lock recovery
  và SHA smoke; không chạy lại train để sửa checker. Không dùng flag này cho job đã push hoặc source train đã đổi.
- Smoke local `sim_fold_val.py run` phải có `--compile` (Live round assert compile gate mọi rank); 200 xe 2 round ≈5,5 phút,
  peak RSS ≈9,8 GB — kiểm `free -h` trước, chạy qua `run_local_guarded.py`.
- Kaggle **không trả output/log khi kernel RUNNING** (listing 0 file, log rỗng): tiến độ kernel không W&B chỉ ước tính được từ thời gian;
  watcher chỉ pull → auto-audit bằng waiter local chờ marker `DONE` (không gọi Kaggle, không thành watcher thứ hai).
- Owner tắt máy khi kernel đang chạy: không cần watcher; ghi lệnh `collect` + analyser vào continue, phiên sau chạy lặp collect.
- Npy cache public dùng chung + manifest SHA; private weights vẫn private. Compile cache để temp, nén một tar trong run và restore qua session.
  Tar restore thành công chưa chứng minh cachehit/speedup; benchmark trên runtime thực.

- DVC push lớn lên OneDrive mount (rclone `vfs-cache-mode=full`): file dirty nằm trên ổ local tới khi upload xong, vượt `--vfs-cache-max-size`.
  Disk free < tổng cần push → đẩy theo lô, giữa lô chờ `grep -rl '"Dirty": true' ~/.cache/rclone/vfsMeta/onedrive` = 0 (≈1 giờ/lô 11 GB, 10/11).
  Xóa `.pt` đã backup: hardlink workspace↔cache nên xóa cả hai; kiểm cloud object tồn tại + đúng size + cùng inode trước xóa.

## Background và bàn giao

- Link checker chung của `project-context` duyệt cả archive và báo hàng nghìn link snapshot cũ; checker active của dự án
  là `python scripts/check_docs.py` (loại archive). Dùng active gate này; không đọc/sửa lịch sử để giải lỗi ngoài scope.
- Dùng codex-long-jobs (Codex) hoặc script watcher chạy Bash background (Claude Code, vd `scripts/watch_b20.py`); poll Kaggle 1200s bằng local worker. Xác nhận running một lần, rồi kết thúc turn khi chỉ còn chờ;
  completion về đúng thread. Không model polling hoặc hai watcher cùng ledger.
- Claude Code job nền chặn Edit/Write ở checkout chính trừ khi `.claude/settings.json` có `"worktree": {"bgIsolation": "none"}`
  (owner bật 09/11). Không dùng worktree từ `origin/main` cho docs: docs/kết quả chưa commit chỉ có ở working copy chính.
- 13/11: vòng collect nền Claude Code **chết cùng phiên** (12/11 đóng phiên → SM42P COMPLETE nằm chưa pull ~4 h, thang chưa lên bậc).
  Đầu phiên luôn kiểm PID watcher + status Kaggle thật của mọi kernel trong bàn giao; ghi PID/task vào continue khi bật watcher.
- Controller wrapper thiếu executable bit: `bash <wrapper> ...`. Exit126 trước controller chưa tạo job/kernel.
- Sau reboot, `running` trong state có thể stale: kiểm PID/start identity; giữ log/state cũ, phục hồi collector, không restart training.
  B14 11/11: reboot01:53 làm controller690627 mất; `watch-recovery.sh` collect/audit8run xong04:21,
  statecontroller vẫn`running`. Kiểm DONE + logwatch + CM/wire thật; không bật lại pipeline từ state/notification cũ.
- Chỉ hủy job khỏe khi owner yêu cầu hoặc handover cần thiết trong workflow đã được phép. Dùng controller hủy process group;
  notification muộn không là yêu cầu bật lại job cũ. Queue accepted không chứng minh đã xử lý completion.
- W&B heartbeat/state chỉ bổ sung evidence, không thay exact Kaggle session/artifacts. Inline key private lưu vĩnh viễn trong history,
  không đưa notebook/credential vào Git/DVC hoặc doc. Không dừng từ W&B crashed đơn lẻ.

## Backup và workspace

- Owner 13/11: mỗi thư mục run phải đủ log và file liên quan. Trước `dvc add`, rsync `runs/<batch>/` (bỏ `.pt`) vào
  `artifacts/<batch>/_receipts/` (per-run manifest: vào thư mục run). Log rời ở root repo dời về đúng thư mục run của nó.

- Backup/restore/cleanup chỉ trong quyền owner. Kiểm mount thật, object manifest và cloud trước xóa;
  không xóa rclone/VFS dirty cache hoặc cloud objects. Mount copy không chứng minh cloud synced.
- Workspace DVC partial: giữ manifest, không `dvc add` ghi snapshot nhỏ đè bản đầy đủ. Restore đúng run cần dùng.
- Git HEAD/index/staged tree giữ nguyên khi chỉ tổ chức docs; không force push/trailer `Co-Authored-By:`.
- Account có thể dùng cho dự án khác: inventory thật trước cấp quota, không gán mọi quota giảm cho MobiHet.
- 12/11: notebook resume chép output kernel trước bằng `shutil.copy` vỡ khi output có thư mục (`models/`) ⇒ dùng `copytree` cho thư mục.
  Giá trị chuỗi trong `--set` của `fl_dagsnet_veremi_v2` vẫn phải escape `\"` (lesson 30/10 bên repo baseline); plan sinh tự động
  phải qua cùng kiểm `ast.parse`/`shlex`/`json.loads` trước push. Python dự án: `conda activate nckh` (system python thiếu nbformat/kaggle).

- 13/11: state controller `running` sau reboot không chứng minh process sống; kiểm PID/identity trước phục hồi. B21 budget stop r22/r18 lưu checkpoint ở round stop, không chỉ bội5; resume preflight phải dùng round thực, kiểm cả2rank/RNG trước decode.
