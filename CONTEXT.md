# CONTEXT — MobiHet / federated IDS V2X

Đọc CONTEXT → [LESSONS](LESSONS.md) → [mục lục tài liệu](docs/README.md).
[Tất cả đề xuất/trạng thái](docs/proposals.md) · [Index từng file](docs/file-index.md).

## Bài toán và quyết định đang áp dụng

Xe học IDS cộng tác qua RSU: giao tiếp hữu hạn, private model có thể khác kiến trúc ở thiết kế.
Không raw/perturbed record rời xe; synthetic từ noise được phép. Giữ 66 feature, partition/protocol gốc.
Mọi learned model hiện DAGSNet; R2 đa kiến trúc chưa kiểm. Local chỉ smoke/probe/collect/audit/report;
train nặng trên Kaggle, W&B online/key trước train. Official test chỉ báo, không chọn HP/best round.

- **14/10, layout mới nhất:** core cho code chung; methods chỉ hướng đang làm; archive cho build đã kết thúc.
  Thiết kế/report lịch sử ở docs/archive; bài học dùng tiếp ở LESSONS và docs/lessons.
  Tên theo cơ chế/nhiệm vụ; V2–V10 là ID đợt, không là chuỗi version thay thế nhau.
  mh/papers chỉ facade runtime cũ; nguồn Python/notebook và hash DVC giữ nguyên.
  [Cấu trúc](docs/operations/project-layout.md), [tên cũ/mới](docs/operations/experiment-names.md),
  [ownership/map](docs/operations/workspace-layout.json).
- Không tự Git/DVC add/commit/push/backup/cleanup. Được restore chọn lọc để phân tích hoặc dựng checkpoint input.
  14/10 phiên tiếp: chủ dự án yêu cầu push Git layout mới; DVC snapshot giữ nguyên, pull bằng manifest canonical trong chỉ mục backup. Git index có thể đổi byte cache/stat;
  kiểm HEAD/nội dung staged riêng, không ghi đè index để làm khớp SHA cũ.
- **14/10, đã cho phép:** dùng quota còn lại ngay để nối H1/H2/Hybrid và BoundaryCells; nối tiếp sau refresh khi cần.
  Giữ H1/H2/Hybrid 20client, 100%train80%, 50round, seed42 và đủ7controls, lịchLR50 từr1.
- BoundaryCells direct/teacher P2/P4 đã costPASS hai seed; utility20round giữ toàn bộ controls/gp target-mass/cổng đã chốt.
  ResponseRisk chỉ local augmentation trước; EMA cả Gaussian/DML giữ thiết kế/cổng, chưa triển khai.
- Đã ổn định thì ngừng health polling; vẫn collect/audit terminal. Bất ổn training quan sát được thì báo và yêu cầu chủ dự án dừng.

## Trạng thái — 14/10/2026

26 nhóm đề xuất +1recipe EMA; 3 nhóm có pilot tích cực: Gaussian moments, DML DAGSNet, Hybrid.
Không nhóm nào đủ r50/R1/R2 proof. Adaptation FAIL V2–V7/V9/V10 không rerun y nguyên.

Full H1/H2 đã audit r7–r11; Hybrid full r2/r3, 7arm/18round. Hai Hybrid calibration ERROR do budget/controller,
checkpoint vẫn hợp lệ. Batch continuation và BoundaryCells utility đã chuẩn bị/push; trạng thái exact version,
checkpoint bridge, quota và phần còn thiếu tại [bàn giao phiên này](docs/operations/reorganization-2026-10-14.md).
PUSHED/RUNNING không thay terminal artifact audit. Mọi số đo tại [hồ sơ build](docs/proposals.md).

## Lỗi/lưu ý bắt buộc trước tạo notebook

Đọc [sổ lỗi kernel](docs/operations/notebook-errors.md), dùng skill
[Codex](.agents/skills/federated-experiment-workflow/SKILL.md) hoặc
[Claude Code](.claude/skills/federated-experiment-workflow/SKILL.md).

- Kernel ERROR có thể sau nhiều arm/round hợp lệ: pull/audit trước rerun, nối actual committed round, không r1.
- Hybrid controller cũ đòi3round dù worker budget-stop rc0: controller mới chấp nhận boundary; phân biệt target50 với sessioncomplete.
- Output listing đọc được chưa chứng minh attach được parent ERROR. Kaggle có thể refused input mà push vẫn exit0;
  dùng dataset private checkpoint có SHA/import gate; require-resume trước data decode. Mất response thì kiểm status trước retry.
- Alias không giữ `__file__.resolve()` depth. Source frozen dựng đúng runtime tree; helper dùng root discovery.
  `git check-ignore` phải dùng đường dẫn canonical `.resolve()` trước nhúng key, nếu không lỗi beyond symbolic link.
- Fingerprint hash đủ40 tên mh/*.py qua facade, gồm core và source build/archive: không sửa/thêm/xóa tên module trong exact-resume
  hoặc đổi source run đã PUSHED. Chuyển vị trí vật lý giữ byte/hash và alias; kiểm `scripts/check_package_layout.py`.
- Metadata cả notebook/JSON, pinned image, assert2T4/sm75/torch trước train; không reinstall torch hoặc bỏ guard.
- W&B preflight không tự tăng restored step; key chỉ ignored private_push0600. Private platform history vẫn chứa key;
  pull executed vào private, audit redacted; không key trong dataset/log/Git/chat.
- Upload checkpoint dataset `-r zip -t`; kiểm file/hash/size, không tabular conversion hoặc mất thư mục.
- Budget cả batch: worst round từngarm, pairmax haiGPU, preparation/import/compile/eval/I/O và dư≥15%.
  Refresh service hiện báo17/10; không suy timezone chưa được API xác nhận. Phiên quota không là GPU-worker hours.
- Quota/listing network reset: retry read-only có giới hạn; UNKNOWN nếu chưa đọc được. Không tự lặp push.
- Verifier CPU map_location; CUDA deterministic/stable ties/CUBLAS trước CUDA; lỗi checker không yêu cầu train lại.
- DVC workspace thiếu sau cleanup không được add thay manifest đầy đủ. `deleted` cloud-status có thể là thiếu localcache;
  kiểm remote .dir/object riêng. DriveFS Errno5 có thể do cacheD: đầy, không tự là artifact nguồn hỏng.

## Muốn biết → mở

| nhu cầu | tài liệu |
|---|---|
| Toàn bộ đề xuất, thành công/thất bại/chưa kiểm | [docs/proposals](docs/proposals.md) |
| Căn cứ/cơ chế/build/kết quả/artifact | [Đang làm](docs/methods/README.md), [lưu trữ](docs/archive/README.md) |
| Đọc lại 10 metrics các round cuối | [H1/H2](docs/methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md), [Hybrid](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md) |
| Kernel/quota/việc còn thiếu | [Bàn giao hiện tại](docs/operations/reorganization-2026-10-14.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Bài học khiến train chưa tốt | [LESSONS](LESSONS.md), [bài học hướng đã đóng](docs/lessons/README.md) |
| Dữ liệu/kiến trúc/scaler | [knowledge](knowledge/README.md), [protocol](docs/research/protocol.md) |
| Scripts dùng chung, cấu trúc, khôi phục | [scripts](scripts/README.md), [layout](docs/operations/project-layout.md), [backup/restore](docs/operations/backup.md) |

Cập nhật trạng thái/lỗi ở CONTEXT, bài học chất lượng train ở LESSONS; số đo chi tiết ở docs của build.
Sau thêm/chuyển file chạy index và kiểm link/hash/ignore. Không backup hoặc dọn artifact nếu chưa được yêu cầu.
