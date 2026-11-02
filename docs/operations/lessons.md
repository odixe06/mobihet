# Bài học vận hành còn áp dụng

Trạng thái thực tế ở [continue](continue.md); quy trình [Kaggle](kaggle.md), [backup](backup.md).
Hồ sơ vận hành cũ đã archive, không đọc mặc định.

- CLIglobal `dinhan25` đã đăng nhập. Dùng wrapper account-scoped; token-only vẫn dùng CLI được, thiếu OAuthsnapshot
  không tự nghĩa authhỏng. Không phục hồi MCPJSON hoặc yêu cầu login lại từ ghi chú cũ.
- KaggleCLI có live `kernels logs --follow`, đọc có timeout. Timeout124 của lần đọc không phải kernelERROR.
  Parse `KernelWorkerStatus.*`; query lỗi là unknown, không grepbừa `ERROR` từstderr.
- COMPLETE/exit0 không chứng minh đủround hoặc collector đãpull. Kiểm artifact đích và endpoint; pull cảERROR để giữ bằng chứng.
- Attentioncompletion V0/V1 chỉ xác nhận notifier chạy xong; hai CEkernelERROR thật đãpull, guardzero-skip dừng trướcr1commit.
  Giữ failedsource/batch/slug; không đổi source đang chạy hoặc restart cảVphase. Probeobserver khácslug, không gọi full20recovery.
- Khi probe đãaudit thì continuation dùng source/slug/manifest recoveryriêng và compatiblecheckpointgate;
  V0/V1 chưa có commit nên retryfresh, không gọi resume. Guardfix không tự chứng minh gradientoverflowcause hoặc F1gain.
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
- Bản skillcopy có thể thiếu bit executable ở wrapper `scripts/codex-long-jobs`; gọi `bash <wrapper> ...` vẫn qua đúngcontroller,
  không tự chạy detachedworker. Exit126 trướccontroller chưa tạo job/kernel; kiểm trước retry, không sửa credential hoặc yêu cầu login.
- Preflight phải chụp sourceSHA lúc bắt đầu và kiểm lại cuối; hash cuối đơn lẻ có thể gán code mới cho module cũ đã import.
  P20 bổ sung teacherchain trong lúc lượt preflight đầu còn chạy: giữ output cũ nhưng loại khỏi launchgate, chạy lại trên source cố định.
- P21: fixture2xe/seed42 PASS không thay gate fullgroup/seed43 trên từngrank. Rank0 có compilegateNaN→eager,
  sau đó probabilityupload nonfinite trước round1; audit phải giữ cả backend từngrank và finiteinference,
  không gọi source/preflightPASS là runtime/utilityPASS. Collector không được bỏ31run đang chạy khi một arm ERROR;
  recovery chỉ collect exactversion, không gọi lại launcher.
- CLI status của private slug chưa tồn tại có thể trả permissiondenied thay404; không suy authhỏng hoặc đã có kernel.
  Trước push mới, đối chiếu ownlisting đủpage + intent/receipt; lỗi inventory phải dừng trước mutation.
- P20audit: sum/index_add float32 giữa CPU/CUDA có thể vượt ranh giới làm tròn fp16; chênh decodedQ lớn hơn sai số
  reduction ban đầu. Tái tạo CPU với atol2e-7 đã falseFAIL6run. Kiểm interval theo số hạng/precision qua quantizer,
  lưu phạm vi boundedconsistency và thử injection saiQ; không nới atol đồng loạt hoặc train lại vì checkerFAIL.
- Kaggle `kernels list --mine --csv` có thể có record `ref=""` (Qinventory:2record ở minhtrit06).
  Lọc ref hợp lệ thuộc owner đã introspect trước query/status/slotgate; không suy login sai từ assertlisting,
  không nới identity/quota/accessgate hoặc push khi inventory thật chưa rõ. Giữ log/code lỗi và verify inventory sửa trước mutation.
- Q20 collector có thể exit do auditor sau khi pullCOMPLETE thành công; record `pulled=true,audited=false` không tự được audit lại
  bởi `--collect-only`. Giữ log/checker/progress cũ, sửa và kiểm auditor trên artifact đãpull, publish reportSHA/progress sau PASS,
  rồi phục hồi collect-only phần còn lại. Không đổi source train/HP, push hoặc restartkernel từ checkerfailure.

- R20 firstpush400: phải đối chiếu ownedlisting đủpage cho mọi intent trước retry; cả8slug chưa tồn tại, không trainingfailure.
  Payload replay lớn chuyển sang datasetprivate theo owner; giữ decodedSHA/targethistory, verify privacy/listingbytes/downloadSHA
  và CUDAdecode trước train. Lỗi400 chưa đủ kết luận chính xác serverlimit; không retry source cũ hoặc thay HP để sửa packaging.

- Private dataset create có thể trả thành công trước khi GetDatasetStatus đọc được: Rinputstage403 ngay saucreate,
  fullownedlisting/metadata sau đó xác nhận6input tồn tại đúngowner/private. Chỉ retrybounded phần đọc; không coi403 là credentialhỏng
  hoặc recreate mù. Kiểm response.error/status và lưu create-result; intent có chỉcho verify, không create/version lại.
  Recovery packaging/staging giữ preflight nếu sourceSHA vàXML45+7 vẫn khớp; không rerun GPUfixture hoặc training vì lỗiI/O.

- GlobalCLI có thể đổi bởi phiên khác trong lúc readonlyinventory (V09:51:20 dinhan25→heykewtiie).
  Không restore/switch để qua assert cũ. Gate bằng serverintrospection từngsavedowner và scopedkaggle_as; ghi globalbefore/after
  thayhardcodeusername. Khóa authorizedowners trước batch, không tự thêm account mới do roster/global thay đổi.
  Vpipelinefail trước notebook/push là operationalguard, không teacher/model utilityfailure hoặc mất savedcredential.

- Kernel build npy 448 s + compile ~8 phút mỗi run. Cache npy dựng một lần bởi kernel CPU public
  `odixe0502/mobihet-v4npy-cache`, run sau mount qua `kernel_sources` + kiểm SHA manifest (`gen_notebook.py --npy-kernel`).
  Kernel private không mount chéo tài khoản được; nhánh V (không đọc test) vẫn tự build cache riêng.
- Client trải nhiều file parquet (4 nhóm liên thông tới 198 file/19M dòng): thống kê theo client phải gộp qua file, không theo từng file.
- OneDrive remote còn ~67 GB (02/11) — 97% dung lượng artifact là snapshot `models/r*.pt`. Kiểm quota cloud trước dvc push lớn.
- Guard credential của notebook public phải tìm marker credential, không chuỗi "wandb" chung (lệnh mẫu có `--wandb`).
