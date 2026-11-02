# Git / DVC backup record

Updated: 2026-11-02 17:10, Asia/Ho_Chi_Minh — N/O/P/Q đã DVC push + verify, xóa local N/O/P và snapshot Q.
Procedure: [repo-backup](../../.claude/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git `https://github.com/odixe06/mobihet.git`, `main` = `c980871b3ce0a918eaee4b6136bcd5e1eea77abf` (khớp `git ls-remote`).
- DVC `onedrive` (default): `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/dvc-storage` (rclone mount `onedrive:`).
- Bundle `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/mobihet.bundle`: all refs, `git bundle verify` PASS, main `c980871`.
- Cloud upload: **pending** — rclone VFS cache `~/.cache/rclone` giữ ~47 GB chờ upload lúc 16:50; không xóa, nó tự giảm.
- OneDrive quota cloud còn ~67 GB trước push này → sau khi upload xong còn ~15 GB. R/V/W (~25 GB) cần thêm chỗ.

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
Chưa DVC: `artifacts/r20`, `artifacts/v20*`, `artifacts/w20` (sắp có), `runs/` (gitignored, ~0,5 GB, giữ local).

## 3. Checks that passed (reusable while inputs are unchanged)

| Check | Scope | Result |
|---|---|---|
| `dvc push -j 4` 71 manifest | N/O/P/Q | exit 0, log `runs/maintenance-20261102/dvc-backup.log` |
| `dvc status -c` 71 manifest | N/O/P/Q | `Cache and remote 'onedrive' are in sync.` |
| Remote object trước xóa | 7.863 file đã xóa: object tồn tại + đúng size; 71 `.dir` có trên remote | PASS (size, không checksum toàn bộ) |
| `git push` / bundle | main / all refs | `c980871`, verify PASS |

## 4. Local cleanup

- Owner 02/11: xóa N/O/P/Q sau backup, giữ R/V. Q giữ phần nhỏ làm control cho phân tích W20 (183 MB).
- Xóa 63 workspace N/O/P + `models/` của 16 run Q + 3.131 object cache tương ứng (hardlink, xóa cả hai).
  Disk free 43.628.335.104 → 96.503.152.640 B (**+52,9 GB**). Receipt `runs/maintenance-20261102/cleanup.json`.
- Giữ `.dvc` manifest, evidence, R/V artifacts, rclone cache, cloud objects. Không việc tay bắt buộc.

## 5. Next session

- Backup R/V/W: kiểm OneDrive còn chỗ (~25 GB cần), `dvc add artifacts/r20/<slug> ...` từng run hoàn chỉnh → `dvc push` → `dvc status -c`.
  `.gitignore` ignore-all trong thư mục artifact làm `dvc add` lỗi "git-ignored" — xóa file đó trước.
- Restore: `dvc pull artifacts/p20/mh-p2-augment-s42.dvc` (~0,8 GB/run). Kiểm disk trước.
- `deleted` trong `dvc status -c` sau cleanup chỉ là thiếu local cache, không phải remote mất.
