# Bàn giao — 14/10/2026

## Layout mới nhất: tên theo cơ chế, tách archive

`core/`:19 module chung với tên nhiệm vụ. `methods/`:6 hồ sơ đang làm/đã chốt và đối chứng full chung.
Build đã kết thúc ở `archive/methods/`; thiết kế/report/handoff cũ ở `docs/archive/`; 21 nhóm đóng còn bài học
tại `docs/lessons/`. V2–V10 là ID đợt kiểm nghiệm, không phải version thay thế nhau.
[Đổi tên](experiment-names.md), [layout/map](workspace-layout.json), [quy tắc](project-layout.md).

`mh/` và `papers/` là facade mỏng tới `.compat/`. Helpers cũ không còn trong scripts đang làm; import namespace
scripts tìm chúng qua `.compat/scripts`, path reads dùng current_path. Frozen Python/notebooks và DVC identities
giữ nguyên; phần tổ chức không sửa nguồn Kaggle đã push hoặc dừng/chạy kernel. Chưa có kết quả remote mới.

Migration từng vướng README log-wrapper khi rebase; link/map đã phục hồi từ manifest, 40 source SHA không đổi.
Lỗi này thuộc công cụ tổ chức, không là lỗi training/kernel. Chỉ xóa/recreate navigation symlinks, không xóa artifact.


Kiểm cuối bước archive:40 module import/SHA/copy-runtime PASS;12 parent plan/1.120 SHA khớp;
11 helper đang dùng import/path reads PASS;32 continuation cases và6 BoundaryCells utility cases PASS.
64 DVC identities/.dir manifests,433 frozen files giữ hash; native DVC65stage;0link thiếu,6remote-only
có metadata.115 private notebooks ignored0600; skill Codex/Claude parity/packaging PASS.
Git HEAD/nội dung staged không đổi; index byte khác ở metadata/cache, không ghi đè index.
4 build canonical đang làm,12build cũ +input-statistics probe đã archive; không có train/push mới trong bước tổ chức.

Tài liệu các bước tổ chức trước bên dưới là lịch sử quyết định; ownership hiện tại theo workspace-layout.


Chủ dự án chốt: chuyển vật lý, giữ alias; dùng quota còn lại ngay; probe/calibration/pilot thuộc đúng build;
proposal/design/results vào docs; logs vật lý trong build/logs; skill Codex/Claude Code; sổ lỗi ngăn lặp.

## Run đã push version1

| owner / slug sau `mobihet-` | kịch bản / parent round | budget phiên |
|---|---|---:|
| catbaochau / full1-s42-20261014-cont02-v1 | H1/H2 local r11, Gaussian r10 → lịch50 | 3,1h |
| minhtriethihi / full2-s42-20261014-cont02-v1 | diag r10, lr4 r9 → lịch50 | 5,0h |
| khanhmay0304 / full3-s42-20261014-cont02-v1 | Gaussian-own r10, DML r7 → lịch50 | 5,0h |
| catbaochau / full4-s42-20261014-cont02-v1 | DML-own r7 → lịch50 | 3,1h |
| odixeuit / hybf2-s42-20261014-cont01-v1 | Hybrid Gaussian/Hybrid r3 → lịch50 | 5,0h |
| odixe0502 / hybf4-s42-20261014-cont01-v1 | Hybrid-own r3 → lịch50 | 4,8h |
| minhtrit06 / hybf1-s42-20261014-dsresume-v1 | Hybrid control local/DML r2, dataset bridge | 5,0h |
| trietbackup / hybf3-s42-20261014-dsresume-v1 | Hybrid Gaussian-own r3/DML-own r2, dataset bridge | 4,8h |
| minhtran0601 / bc-u-s42-20261014-stage01-v1 | BoundaryCells20arm,5arm từ calibrationr2, controls fresh | 2,2h |
| dinhan25 / bc-u-s43-20261014-stage01-v1 | BoundaryCells như seed42, fixed20schedule | 2,5h |

**10 notebook hợp lệ đã push**, chưa terminal audit. Native snapshot và W&B riêng tại
[status](kernel-status-current.json), [startup](startup-batch-2026-10-14.json).
W&B initial đã thấy training mới ở H1/H2, Hybrid nhóm2/4, và BoundaryCells baseline; không nonfinite trong snapshot.
Hai dataset-resume mới chưa có heartbeat mới khi quan sát ban đầu; không dùng lịch sử calibration để gọi healthy.
Quan sát tiếp riêng hai notebook mới: trietbackup đã có hai arm tiến tới r4/r3-in-flight, không nonfinite;
minhtrit06 còn chưa có training heartbeat mới. [Snapshot riêng](startup-dataset-resume-2026-10-14.json).
Stable thì ngừng health polling; notebook mới chưa có progress giữ trạng thái chưa xác nhận khởi động.

8 notebook H1/H2/Hybrid:14arm tới50, giữ source/config/LR50. BoundaryCells đủ20arm mỗi seed nhưng quota
có thể chỉ chạy một phần; chỉ chốt utility khi controls và r20 đủ. Không đổi primarygp/HP/cổng để tìm lợi.
Các adaptation FAIL không mở lại. EMA/ResponseRisk chưa có source/pilot utility, không push skeleton chưa preflight.

## Lỗi input và cách phục hồi

`minhtrit06/mobihet-hybf1-s42-20261014-cont01-v1` đã push nhưng response refused parent ERROR;
vẫn RUNNING trong snapshot. Require-resume trước data decode chặn chạy lại r1.
Chủ dự án đã xác nhận dừng **đúng cont01 lỗi này**, giữ `dsresume-v1`; snapshot RUNNING trước đó là lịch sử.
Trường hợp trietbackup dự kiến tương tự được dùng bridge trước push, không push cont01 nguyên bản.

Checkpoint private same-owner datasets đã upload READY, source/state/predictions/metric verify trước staging:

- [Bridge1 proof](hybrid-checkpoint1-2026-10-14.json): minhtrit06,2arm×r2,635.619.531B.
- [Bridge3 proof](hybrid-checkpoint3-2026-10-14.json): trietbackup,r3/r2,802.381.519B.
- [Dataset-resume launch/preflight](hybrid-dataset-resume-launch-2026-10-14.json):8controllercases,
  owner-scoped input read; mount/import/actualT4 kiểm trong notebook trước data decode.

Dataset `-r zip -t`, private, không CSV conversion; unique fingerprint, import SHA/copyfile/marker-last.
Không sửa notebook đã PUSHED. Partial preparation khi `git check-ignore` gặp alias bị chặn trước key injection;
đã sửa guard dùng canonical `.resolve()`, source public/private hoàn chỉnh trước push.

## Budget và bảo toàn

[Batch budget](launch-batch-2026-10-14.json) giữ snapshot trước push và các subset tiếp theo;
10scenario hữu ích tổng40,5h phiên, dự phòng15%=46,575h; kernel refused-input thêm chi phí service chưa đo cuối.
Account-scoped CLI, active dinhan25 giữ nguyên; MCP Kaggle không callable trong phiên này.
Quota cho toàn bộ target50 vẫn thiếu, nối theo quota sau refresh17/10 như đã chốt.
Hybrid dự toán314,7h trước đợt này là điều kiện/worst/fixedpair, không reservation và chưa repeatedsetup.

[Migration manifest](layout-migration-2026-10-14.json):16build chính +probe input-statistics,
59proposalMD vào docs,64DVCmetadata,433protectedsource/notebook files giữ SHA; source mh/proj không sửa.
Chủ dự án sau đó yêu cầu bỏ thư mục cũ nếu đủ nội dung: đã bỏ alias `proposals/`; tài liệu thật vẫn ở docs.
`papers/` chỉ alias cho dependency lịch sử. Sau yêu cầu tách package:21 module riêng chuyển vào10 build/source/mh,
19 module chung nay giữ vật lý trong `core/`. Alias module giữ import và fingerprint đủ40 file, không đổi byte nguồn.
[Ownership](package-ownership.md), [manifest](package-layout-2026-10-14.json), [kiểm](package-verification-2026-10-14.json).
Import 40 module tại repo và trong runtime copy không symlink đều PASS. 12 plan/1.120 SHA source parent khớp;
32 continuation-controller cases PASS. Không train local hoặc thay source các run đang chạy.
Git HEAD và nội dung staged giữ nguyên; workspace move còn unstaged/untracked. Không Git/DVC add/commit/push hoặc cleanup.
Remote .dir identities giữ nguyên, localrestore chọn lọc thiếu vẫn là thiếu; cloud-status `deleted` không tự là remote mất.
[Kiểm Git/DVC/link/key](layout-verification-2026-10-14.json), [cấu trúc](project-layout.md), [sổ lỗi](notebook-errors.md).

Kiểm layout:64DVC identities/.dir MD5,433frozenfiles,0link tài liệu thiếu;6link source archive remote-only
có metadata restore. DVC native nhận65stage gồm1knowledge checkpoint. Private notebook115file distinct ignored0600;
hai bản cũ được sửa permission, không đổi byte. Skill hai host packaging/parity PASS.

## Định tuyến và phần còn thiếu

- [Mục lục tổng quan](../README.md), [26đềxuất+EMA](../proposals.md), [index từngfile](../file-index.md).
- [Skill Codex](../../.agents/skills/federated-experiment-workflow/SKILL.md),
  [Claude Code](../../.claude/skills/federated-experiment-workflow/SKILL.md); packaging/parity PASS.
- Pull/audit đúng source/version khi terminal; report10metrics/perclass, giữ checkpoint round thực tế.
  BoundaryCells collector utility cần xử lý `utility_summary.json`, khác calibration_summary; source verifier hiện có.
- Tiếp các phần r20/r50 chưa đủ với actual latest parents, không restart1. Private/datasetbridge cần provenance từ proof này.
- EMA hai nền và local augmentation ResponseRisk còn thiết kế, chưa mở shared-response weights.

Thao tác tay dừng cont01 lỗi đã hoàn tất theo xác nhận chủ dự án. Hiện không còn yêu cầu thao tác tay.
Không cần đăng nhập mới hoặc tự push Git/DVC trong phiên này.
