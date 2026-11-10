# Git / DVC backup record

Updated: 2026-11-10 21:45, Asia/Ho_Chi_Minh — mọi artifact đến 10/11 (gồm codec-combo20/50) đã DVC + OneDrive (in sync); Git `main` push 10/11 tối (đỉnh: `git ls-remote`). Workspace partial.
Procedure: [repo-backup](../../.claude/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git `https://github.com/odixe06/mobihet.git`, `main`: commit 10/11 tối (combo r50 + chốt CBB + manifest codec-combo); đối chiếu đỉnh bằng `git ls-remote`. Bundle thay 10/11 tối, verify PASS.
- DVC `onedrive` (default): `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/dvc-storage` (rclone mount `onedrive:`). `dvc` ở env conda `nckh`.
- Bundle `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/mobihet.bundle`: all refs, verify mỗi lần thay.
- Cloud upload: **xong** 20:26 05/11, 0 file dirty trong VFS cache. Mount chạy `--transfers=16`, `TimeoutStartSec=3600` qua drop-in `~/.config/systemd/user/rclone-onedrive.service.d/upload-parallel.conf`. Không thêm flag `--onedrive-*` (đổi fs thành `onedrive{hash}`, mất hàng đợi cache).
- OneDrive quota: free 43,2 GiB sau upload (20:27 05/11). Google Drive không dùng.

## 2. Tracked artifacts

Manifest cũ: 68 (archive batches H–L, pulls, M20, evidence) — [inventory](backup-inventory.json), PASS 01/11, workspace đã xóa trừ evidence.

Mới 02/11, mỗi run một manifest `artifacts/<nhóm>/<slug>.dvc`; danh sách `runs/maintenance-20261102/targets.txt`.

| nhóm | manifest | files | logical bytes | local workspace |
|---|---:|---:|---:|---|
| `artifacts/n20` | 8 | 784 | 6.186.692.517 | deleted 02/11 |
| `artifacts/o20` | 8 | 816 | 6.189.021.370 | deleted 02/11 |
| `artifacts/o20-early-20261101` | 6 | 612 | 4.641.881.084 | deleted 02/11 |
| `artifacts/p20` (gồm P21 gốc lỗi + recovery1) | 33 | 3.731 | 26.129.427.838 | deleted 02/11 |
| `artifacts/q20` | 16 | 1.920 | 12.991.878.138 | **partial**: xóa `models/`, giữ CM/rounds/config/log/provenance |

**Không `dvc add` lại `artifacts/q20/*`** (workspace partial sẽ ghi đè manifest đầy đủ). Cần snapshot Q: `dvc pull artifacts/q20/<slug>.dvc`.
Mới 05/11, mỗi run một manifest; danh sách `runs/maintenance-20261105/manifests.txt`, tổng 50 manifest, 8.533 file, 49.031.911.091 B (3.068 object duy nhất sau dedup).

| nhóm | manifest | files | logical bytes | local workspace |
|---|---:|---:|---:|---|
| `artifacts/c10` | 4 | 776 | 1.614.389.947 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/f50` | 3 | 846 | 6.064.240.390 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/l50` | 1 | 290 | 10.538.449.319 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/r20` | 12 | 1.461 | 9.739.506.252 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/v20-numerics` | 2 | 114 | 1.842.984 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/v20-recovery` | 2 | 320 | 1.632.840.814 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/v20` | 8 | 822 | 4.853.923.776 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/w20` | 10 | 2.160 | 8.092.765.877 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |
| `artifacts/x20` | 8 | 1.744 | 6.493.951.732 | **partial**: xóa `.pt`, giữ CM/rounds/config/log/provenance |

**Không `dvc add` lại 50 run này** (workspace partial sẽ ghi đè manifest đầy đủ). Cần model: `dvc pull <manifest>`.
Mới 10/11 (owner yêu cầu backup toàn bộ): 43 manifest, danh sách `runs/maintenance-20261110/targets.txt`, log `dvc-add.log`/`dvc-push.log`.

| nhóm | manifest | local workspace |
|---|---:|---|
| `artifacts/y50` (7 run) | 7 | **partial**: xóa `.pt` (y0–y6), giữ CM/rounds/config/log/provenance/inductor-cache |
| `artifacts/{forest-t4,standard-t4}` | 4 | **partial**: xóa `.pt` |
| `artifacts/f50/mh-f3-core-s43-r50` | 1 | **partial**: xóa `.pt` |
| `artifacts/{km50,km-pilot20,km-pilot50,km-lite-pilot20,codec-pilot20,g1-pilot20}` | 19 | đủ (không `.pt`) |
| `artifacts/sim-kaggle`, `-b2`…`-b12` | 12 (mỗi batch 1 manifest) | đủ |

Mới 10/11 tối: `artifacts/codec-combo20` (6 run) + `artifacts/codec-combo50` (4 run) = 10 manifest, đủ (không `.pt`, 134 MB);
danh sách `runs/maintenance-20261110b/targets.txt`, log `dvc-add.log`/`dvc-push.log`.

**Không `dvc add` lại các run trên.** Chưa DVC: `runs/` (gitignored, giữ local).

## 3. Checks that passed (reusable while inputs are unchanged)

| Check | Scope | Result |
|---|---|---|
| `dvc push -j 4` 71 manifest | N/O/P/Q | exit 0, log `runs/maintenance-20261102/dvc-backup.log` |
| `dvc status -c` 71 manifest | N/O/P/Q | `Cache and remote 'onedrive' are in sync.` |
| Remote object trước xóa | 7.863 file đã xóa: object tồn tại + đúng size; 71 `.dir` có trên remote | PASS (size, không checksum toàn bộ) |
| `git push` / bundle | main / all refs | `c980871`, verify PASS |
| `dvc push -j 4` 50 manifest | C/F/L/R/V/W/X | lần 1 lỗi Errno5 (Graph timeout khi list); chạy lại exit 0, 2.990 file, log `runs/maintenance-20261105/dvc-push-retry.log` |
| `dvc status -c` 50 manifest | C/F/L/R/V/W/X | `Cache and remote 'onedrive' are in sync.` (mức mount) |
| `rclone lsf` cloud 20:27 | 3.068 object, 49.004.158.345 B | đủ, 0 thiếu, 0 lệch size (size, không checksum) |
| `dvc push -j 4` 43 manifest (10/11) | y50/forest/standard/km*/codec/g1/f3/sim-kaggle | 5 lô, exit 0; `dvc status -c` 04:14: `Cache and remote 'onedrive' are in sync.`; dirty VFS = 0 |
| `dvc push -j 4` 10 manifest (10/11 tối) | codec-combo20/50 | exit 0, 374 file; `dvc status -c`: `Cache and remote 'onedrive' are in sync.` |
| Remote object trước xóa (10/11) | 428 `.pt` (50,78 GB) | tồn tại + đúng size + cùng inode với cache object; 0 lệch (size, không checksum) |

## 4. Local cleanup

- Owner 02/11: xóa N/O/P/Q sau backup, giữ R/V. Q giữ phần nhỏ làm control cho phân tích W20 (183 MB).
- Xóa 63 workspace N/O/P + `models/` của 16 run Q + 3.131 object cache tương ứng (hardlink, xóa cả hai).
  Disk free 43.628.335.104 → 96.503.152.640 B (**+52,9 GB**). Receipt `runs/maintenance-20261102/cleanup.json`.
- Giữ `.dvc` manifest, evidence, R/V artifacts, rclone cache, cloud objects. Không việc tay bắt buộc.

- Owner 05/11: xóa 1.042 file `.pt` (48,4 GB) + object cache tương ứng, sau khi kiểm từng object trên cloud đúng size ngay trước xóa; xóa 2 bản `runs/l50-study/transfer/*/l50-resume.bin` (19 GB, trùng nhau, state cùng size với bản DVC). Free 105 → ~172 GB. Receipt `runs/maintenance-20261105/cleanup.json`.

- Owner 10/11: xóa 428 `.pt` (50,78 GB) + object cache tương ứng của nhóm mới; xóa 4 `runs/y-transfer/Y{3..6}/upload/resume.bin` (41,7 GB; tar của chính các file
  đã DVC, `rounds.jsonl`/`recovery` trong tar là tiền tố của bản DVC). Free 44,1 → 136,6 GB (**+92,5 GB**). Receipt `runs/maintenance-20261110/cleanup.json`,
  script `runs/maintenance-20261110/cleanup.py`. Giữ rclone cache (8,5 GB), `runs/sim`, `runs/y-transfer` còn lại, cache npy.

## 5. Next session

- Hết file dirty: `grep -rl '"Dirty": true' ~/.cache/rclone/vfsMeta/onedrive` rỗng (cache không về 0 vì giữ tới 10 GB file sạch).
- Restore: `dvc pull artifacts/l50/mh-l0-core-lossless-s42-r50.dvc` (~10,5 GB). Kiểm disk trước.
- Push lớn (>~40 GB) khi disk free thấp: rclone `--vfs-cache-mode=full` giữ file dirty trên local tới khi upload xong → đẩy theo lô ≤ ~11 GB, chờ dirty = 0 giữa các lô (`runs/maintenance-20261110/push.sh`; ≈1 giờ/lô 11 GB).
- Restore model y50: `dvc pull artifacts/y50/<slug>.dvc` (~11 GB/run y3–y6).
- Thêm run mới: `dvc add artifacts/<nhóm>/<slug>` từng run hoàn chỉnh → `dvc push` → `dvc status -c`.
  `.gitignore` ignore-all trong thư mục artifact làm `dvc add` lỗi "git-ignored" — xóa file đó trước.
- `deleted` trong `dvc status -c` sau cleanup chỉ là thiếu local cache, không phải remote mất.
