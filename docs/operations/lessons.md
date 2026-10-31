# Bài học vận hành còn áp dụng

Trạng thái thực tế ở [continue](continue.md); quy trình [Kaggle](kaggle.md), [backup](backup.md).
Hồ sơ vận hành cũ đã archive, không đọc mặc định.

- CLIglobal `dinhan25` đã đăng nhập. Dùng wrapper account-scoped; token-only vẫn dùng CLI được, thiếu OAuthsnapshot
  không tự nghĩa authhỏng. Không phục hồi MCPJSON hoặc yêu cầu login lại từ ghi chú cũ.
- KaggleCLI có live `kernels logs --follow`, đọc có timeout. Timeout124 của lần đọc không phải kernelERROR.
  Parse `KernelWorkerStatus.*`; query lỗi là unknown, không grepbừa `ERROR` từstderr.
- COMPLETE/exit0 không chứng minh đủround hoặc collector đãpull. Kiểm artifact đích và endpoint; pull cảERROR để giữ bằng chứng.
- CLI listings phân trang tối đa200; outputhelper phải đi hếtpage và giữexactowner/slug/version. Pull tối đa~32connection,
  audit nặng nối tiếp theo RAM, không SIGSTOP downloadstream.
- Notebook phải truyền exitcode subprocess; sourcebaseSHA+overlaymanifest mới xác định code đãchạy. Giữ frozenbytes;
  sửa ở version/slug mới, không thay source run cũ.
- Fixture/data được fingerprint phải chính thứ driver đọc; smoke pilot không chứng minh productionpath.
  Rowcount không chứng minh contentidentity; currentdataset còn thiếu fullcontentfingerprint.
- Privatecheckpoint phải lưu từngrank, cùng round; W1bank~4,2GB/rank. Atomic2bản có thể vượt20GBworking.
  Dừng budget đồng thuận mọi rank; không dùng smallsnapshot cho fullresume.
- Poll nền1200s bằng codex-long-jobs, không modelpollwaiting. Sau tắt máy, state có thể còn `running` dù PID đã mất;
  đối chiếu processidentity/uptime. Giữ log/state cũ, tạo jobrecovery mới; không restarttraining vì watcher bị ngắt.
- W&Bstate/heartbeat không tự chứng minh Kaggleliveness; W&Bresume không phải modelresume. Inlinekey chỉ trong notebookprivate,
  không Git/DVCbackup credentials; lịch sử phiên bản remote giữ key vĩnh viễn.
- OneDrive remote phải là mount thật trước push; verify object/manifest trước xóa local. Mountcopy và cloudsynced là hai trạng thái.
  Không xóa rclone/VFSuploadcache hoặc cloudobjects trong cleanupworkspace.
- Khi DVCworkspace đã cleanup/partial, giữ manifest, không `dvc add` ghi đè snapshot đầy đủ bằng phần còn lại.
  Backup chưa đủ nếu mới push Git hoặc mới push DVC; kiểm riêng cảhai và bundle.
- Gitindex byteSHA có thể đổi mà stagedtree không đổi. Owner cấm mọi trailer `Co-Authored-By:`; không sửa lịch sử/forcepush.
- Baseline/NILM/Notion thuộc repo khác; không chép trạng thái của chúng thành nguồn thứhai trong repo này.

- Local `compile_gate` kiểm finite cả slot padded. Fixture2xe với group_rows256 tạo4slot đã báo NaN; cùng dữ liệu với128/2slot PASS. Chọn fixture đủ activeclient cho gate; không hạ gate hoặc sửa loss để che lỗi fixture.
