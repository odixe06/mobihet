# Git / DVC — backup và khôi phục

## Repo này

- Git: GitHub `odixe06/mobihet` (public, tạo 30/10/2026). Chưa có DVC; artifact run nặng (state.pt, cms) để trong output Kaggle,
  khi cần giữ lâu thì `dvc init` + remote OneDrive theo skill `repo-backup` (đích `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/dvc-storage`).

## MobiHet v1 (archive 30/10/2026)

- GitHub: `odixe06/mobihet` → đổi tên **`odixe06/mobihet-archive-veremi-v1`** (public, archived). Commit cuối `5205990` (main) + nhánh
  `codex/feature-validation-arms`.
- Bundle: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/mobihet-archive-veremi-v1.bundle` (git bundle verify OK,
  SHA256 `a015a121…51b0e1`); bundle cũ `mobihet.bundle` (24/10, có ref `refs/backup/google-drive/*`) giữ nguyên.
- Thư mục ignored `.compat/` (log cũ): `mobihet-archive-veremi-v1-compat.tgz` cùng thư mục (SHA256 `169150e0…08499f`).
- DVC v1: remote `onedrive` `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/dvc-storage`; cache local đã trống khi archive
  (mọi object đã đẩy, xem `history/v1/docs/operations/backup.md`). Thư mục local `/home/odixe/project/mobihet` (v1) đã xóa theo yêu cầu owner.
- Khôi phục v1: `git clone https://github.com/odixe06/mobihet-archive-veremi-v1` (hoặc clone bundle) → `dvc pull <target>.dvc`.
