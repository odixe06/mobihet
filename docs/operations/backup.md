# Backup / restore — nguồn hiện hành

Trạng thái job và quyền thao tác tại [continue](continue.md). Đọc skill `repo-backup` trước push/restore/cleanup.
Lịch sử các lô/cleanup/hash: [snapshot](../../archive/operations-20261116/before/docs/operations/backup.md).

## Đích

- Git `https://github.com/odixe06/mobihet.git`, branch `main`.
- DVC default `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/dvc-storage`.
- Git bundle: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/mobihet.bundle`.
- OneDrive là rclone mount; kiểm mount thật trước ghi. Google Drive không dùng làm upload fallback.

## Artifact và trạng thái đã kiểm

- Mỗi run `artifacts/<batch>/<slug>.dvc`; sim có thể một manifest/batch.
- Nhóm mới đã xác minh16/11: `artifacts/b21-chain/`, `artifacts/b21-transfer/`, `artifacts/t3-s41/`, `artifacts/maintenance-20261116/`.
- Nhiều nhóm lịch sử N/O/P/Q/C/F/L/R/V/W/X/Y đã cleanup/partial; bảng đầy đủ ở [snapshot backup](../../archive/operations-20261116/before/docs/operations/backup.md),
  inventory gốc ở [receipt](../../archive/operations-20261116/receipts/backup-inventory.json).

20/11/2026 02:58: `artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2}-r50/pull.dvc` push-j16 + status-c in sync (manifest chưa commit Git).
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
