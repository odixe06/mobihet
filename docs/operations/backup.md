# Git / DVC backup record

Updated: 01/11/2026, Asia/Ho_Chi_Minh — DVC push nền, **chưa backup xong/chưa cleanup**.
Procedure: [repo-backup](../../.agents/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git `https://github.com/odixe06/mobihet.git`, branch `main`; base `099e95e8466811881fb71dcd7d3c44268ed082d6`.
- DVC `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/dvc-storage`.
- Bundle cố định: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/mobihet.bundle`.
- Mount thật `onedrive:`/`fuse.rclone` đã kiểm. Filesystem backup và cloud upload kiểm riêng; cloud hiện unchecked.

## 2. Tracked artifacts

[Inventory đầy đủ68manifest](backup-inventory.json) ghi target/MD5/nfiles/byte/retain_local từng bản.
Tổng2540object,30.696.795.246B duy nhất; tất cả workspace hiện đầy đủ.

| Nhóm manifest | Số | Local sau cleanup dự kiến |
|---|---:|---|
| `artifacts/m20/*.dvc` |16| xóa payload, giữ manifest |
| `archive/artifacts/pulls/*.dvc` |43| xóa payload, giữ manifest |
| `archive/artifacts/batches/*.dvc` |5| xóa payload, giữ manifest |
| `archive/artifacts/local-cache.dvc` |1| xóa payload, giữ manifest |
| `evidence/*.dvc` |3| giữ evidence nhẹ và cache tương ứng |

Không re-add workspace partial/đã xóa. Archive không đọc mặc định; restore chỉ khi cần bằng chứng lịch sử.

## 3. Checks that passed (reusable while inputs are unchanged)

| Check | Scope | Result |
|---|---|---|
| Migration + research | frozen5overlay; JSON/CSV analysis/audit; active links | SHA giữ nguyên, output khớp;54tests PASS |
| Credential boundary |926publicfile +2743artifacttextfile| không chứa credential hiện dùng; notebook generated bị loại |
| DVC push/status-c |68manifest| đang chạy nền, chưa xác minh |
| Git push / bundle verify |main / all refs| chưa thực hiện |

DVC worker sẽ kiểm size mọi2540object, MD5 mọi directorymanifest và20dataobject xác định; không gọi toàn bộ data là checksum-verified.

## 4. Local cleanup

- Owner yêu cầu 01/11/2026: archive nội dung lỗi thời, xóa local để tiết kiệm ổ.
- Chỉ xóa65payload có `retain_local=false` trong inventory và cache không thuộc3evidence sau DVC/Git/bundle đã kiểm.
- Giữ evidence, `.dvc`/ignore, code/docs và log completion. Notebook generated có credential: loại khỏi backup, xóa local sau các cổng.
- Không xóa cloudobject/rcloneVFSuploadcache/credentials hoặc sharedcache ngoài repo. Chưa xóa gì; chưa đo dung lượng thu hồi.
- Việc tay: không có.

## 5. Next session

- Job/state/log và ownership tại [continue](continue.md); completion không thay audit bản sao.
- Push mới bằng target cụ thể: `dvc add <complete-dir>` rồi `dvc push <manifest>.dvc`; bỏ qua snapshot đã cleanup.
- Phân tích M dùng evidence nhẹ: `python scripts/analyse_m20.py`, `python scripts/report_m20.py`.
- Restore một run: `dvc pull artifacts/m20/mh-m0-aug-s42.dvc` (~0,78GB); toànM ~12,4GB, kiểm disk trước.
- Artifactđổi làm mất hiệu lực checksum; config/remoteđổi phải kiểm mount lại. Không cần upload lại history đã sync.
- `deleted` trong `dvc status -c` sau cleanup có thể chỉ thiếu localcache; không suy remote bị mất.
