# Backup / restore — nguồn hiện hành

Trạng thái job và quyền thao tác tại [continue](continue.md). Đọc skill `repo-backup` trước push/restore/cleanup.
Lịch sử các lô/cleanup/hash: [snapshot](../../archive/operations-20261116/before/docs/operations/backup.md).

## Gỡ DVC — 26/11/2026

- Repo không còn dùng DVC. Đã gỡ `.dvc/`, `.dvcignore` và 310 pointer `*.dvc` khỏi nhánh `main`. Pointer cũ còn trong lịch sử Git, xem commit `c325376`.
- Đã xóa `NCKH/veremi_build/mobihet/dvc-storage` trên OneDrive (17.690 object, 196,2 GB) vào thùng rác OneDrive. Kho này dùng chung với `mobihet-archive-*`, các repo đó cũng đã gỡ DVC. Trước khi xóa đã copy 155 blob (29 MB) của hai pointer `experiment/runs/recovery-20261120/references/dagsnet-l4/mh-fbl-l4-top-s4{1,2}-r50.dvc` sang cache của `experiment`.
- Đã xóa cache local `.dvc/cache` (15 GB).
- 27/11: trong output cũ ở working tree (gitignore), đã xóa 52 checkpoint `*.pt` và 4 `inductor-cache.tar`, tổng 13,3 GB. Danh sách nằm ở [receipts/dvc-removal-20261127-deleted-files.tsv](receipts/dvc-removal-20261127-deleted-files.tsv). Kết quả còn lại gồm 44.649 file, 2,42 GB: CM `*.npz`, metrics/history `*.json(l)`, log, md, csv, source snapshot.
- Kết quả còn lại đã đóng gói thành `NCKH/veremi_build/mobihet/mobihet-dvc-results-20261127.zip` trên OneDrive (2.256.165.058 byte, SHA-256 `670ae8d126335374de36b4a168d000d380f4df562b35610fc4f74ec40ba9bbfc`). `MANIFEST.json` trong ZIP ghi SHA-256 của từng file. Giữ đường dẫn tương đối gốc repo, giải nén tại gốc repo để khôi phục.
- Kiểm kiến thức 27/11: 120 output không có bản local đều có slug được ghi trong tài liệu hoặc registry trong Git. Không có link Markdown nào trỏ tới pointer đã gỡ. 9 output được tài liệu nhắc tới vẫn còn local và đều nằm trong ZIP.
- `dvc pull` không còn khôi phục được dữ liệu. Code, docs và metadata run vẫn trong Git. Git bundle `mobihet.bundle` và `backup-receipt.json` vẫn giữ ở `NCKH/veremi_build/mobihet/`, các bundle đời trước ở `mobihet-legacy/`.
- Lý do: owner chỉ giữ DVC cho các dự án đang làm (`odixe06/experiment`, `experiment_10k_clients`). Các hướng dẫn DVC bên dưới chỉ còn giá trị lịch sử.

## Đích

- Git `https://github.com/odixe06/mobihet.git`, branch `main`.
- DVC default `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/dvc-storage`.
- Git bundle: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/mobihet.bundle`. Các bundle đời trước nằm ở thư mục `mobihet-legacy`.
- OneDrive là rclone mount; kiểm mount thật trước ghi. Google Drive không dùng làm upload fallback.

## Artifact và trạng thái đã kiểm

- Mỗi run `artifacts/<batch>/<slug>.dvc`; sim có thể một manifest/batch.
- Nhóm mới đã xác minh16/11: `artifacts/b21-chain/`, `artifacts/b21-transfer/`, `artifacts/t3-s41/`, `artifacts/maintenance-20261116/`.
- Nhiều nhóm lịch sử N/O/P/Q/C/F/L/R/V/W/X/Y đã cleanup/partial; bảng đầy đủ ở [snapshot backup](../../archive/operations-20261116/before/docs/operations/backup.md),
  inventory gốc ở [receipt](../../archive/operations-20261116/receipts/backup-inventory.json).

### Dọn checkpoint L4 có logit adjustment ngày 26/11/2026

- Chuyển 125 blob checkpoint (28,646 GiB) của ba run `artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2,3}-r50/pull.dvc` vào thùng rác OneDrive. Đối chiếu đủ 304 manifest trong kho để xác nhận các blob chỉ được ba run L4 cũ tham chiếu.
- Theo yêu cầu của owner, đã xóa cả 125 blob local khỏi cache DVC, thư mục run và thư mục khôi phục. Đã xác minh đủ 375 hardlink bị xóa, thu hồi khoảng 28,6 GiB trên SSD.
- `runs/onedrive-l4-recovery-20261126/` lưu danh sách và biên nhận `local-delete.json`. Giữ các pointer DVC và metadata run. Muốn pull đầy đủ ba run cũ cần khôi phục các blob từ thùng rác OneDrive nếu chúng còn ở đó.

24/11/2026 08:20: `dvc add` + `dvc push -j 16` sáu target: `artifacts/fbl-hop/mh-fbl-l4-top-s43-r50/pull.dvc` (9,9 GB),
`artifacts/fbl-r50/mh-fbl-l4-top-s4{1,2,3}-r50-h2.dvc`, `artifacts/fbl-r50/mh-fbl-l4-top-s43-r50.dvc`, `artifacts/sim-kaggle-la.dvc`;
343 file, exit 0, `status -c` in sync (log `runs/dvc-push-20261124.log`). Upload cloud của rclone có thể chưa xong: không xoá `~/.cache/rclone`.
Không cleanup. Pull mới của batch MobiHet nằm ở repo experiment (ORICO), chưa backup.
20/11/2026 02:58: `artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2}-r50/pull.dvc` push-j16 + status-c in sync (commit `38754dd`); thêm manifest fb-full/fbl-full/sim-kaggle-fbl/fbl-r50 session 1; `dvc push -j 16` toàn repo + bundle `main` 38754dd.
Lần xác minh filesystem remote gần nhất:16/11/2026,16:57:46VN, push-j16 + status-c in sync cho các target trong
[receipt reboot](../../archive/operations-20261116/receipts/backup-reboot-20261116.json).
Đó là snapshot, không chứng minh output mới đã backup hoặc cloud upload đã hoàn tất.
VFSdirty có thể còn trênlocal; không xóa cache/artifact hoặc restart mount khỏe để ép sync.

## Restore / cập nhật

```bash
dvc pull PATH/TO/RUN.dvc
```

Kiểmdisk và manifest trước restore. Workspace partial/thiếu model không `dvc add` đè manifest đầy đủ;
đối chiếu `.dir`/entry cũ trước mọi cập nhật. Không dời hoặc sửa byte payload/metadata DVC khi chỉ dọn docs.
DVC cùngrepo chạy nối tiếp; push xong mới status-c. Git stage/push chỉ phạm vi owner giao; không credential/notebook inlinekey,
không `Co-Authored-By:`. Gitbundle tạo `--all`, verify rồi mới thay bản trênmount. Quyền backup không cho phép cleanup.

## Git phiên16/11 cuối

Owner yêu cầu push toàn bộ thay đổi hiện tại cả MobiHet/BERT lên `origin/main`, gồm brief/context/archive và kết quả pFedES.
Xác minh HEAD/remote và Gitbundle allrefs theo receipt cuối tại `../fl_veremi_bert/runs/maintenance-20261116/final-publish.json` tính từrootrepo.
Yêu cầu này không khởi động DVCpush mới hoặc cleanupartifact/cache.

## Git khôi phục17/11

Owner yêu cầu commit/push thay đổi trên `main` cảhai repo. Phạm vi: source Lite2/FB, tests có sẵn từ Claude, brief/results/handoff, lỗi mạng Kaggle.
Receipt HEAD/remote/bundle sẽ ghi tại `runs/maintenance-20261117/publish.json` của từngrepo. Không DVCpush hoặc cleanup; artifact Lite2 mới giữ local, chưa backup vào DVCremote.
