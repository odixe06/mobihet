# Git / DVC backup record

Updated: 2026-10-10 23:16 (UTC+7) — khởi tạo repo, DVC push `knowledge/ckpt_round_005.pt`, git push `main`.
Procedure: [repo-backup](../../.claude/skills/repo-backup/SKILL.md) (bản Codex: `.agents/skills/repo-backup/`).

## 1. Destinations

- Git: `git@github.com:odixe06/mobihet.git` (private, tạo 2026-10-10), nhánh `main`. Commit đầu `9065392` khớp `git ls-remote`.
- DVC remote `gdrive` = `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`, **dùng chung với archive** (object trùng hash chỉ lưu một lần).
- Bundle: **chưa tạo.** `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle` hiện là bundle của repo **cũ** (nay là archive 3).
  Phải tạo `mobihet-archive-b9-b21.bundle` cho archive và verify trước, rồi mới thay `mobihet.bundle` bằng bundle repo này.
- Cloud: file đã nằm trên G: = đã backup. Upload lên cloud: chưa kiểm.

## 2. Tracked artifacts

| Manifest | MD5 | Files | Bytes | Local |
|---|---|---:|---:|---|
| `knowledge/ckpt_round_005.pt.dvc` | `c1466d52f38b` | 1 | 4.924.379 | có |

Repo archive (đọc [index](../legacy/kien-thuc-ke-thua.md#0-vì-sao-dừng-hướng-cũ-archive-nằm-ở-đâu)) giữ manifest của Build1–21 trên cùng remote.

## 3. Checks that passed

| Check | Scope | Result |
|---|---|---|
| `dvc push knowledge/ckpt_round_005.pt.dvc` | 1 manifest | `1 file pushed` |
| `dvc status -c` | 1 manifest | `Cache and remote 'gdrive' are in sync.` |
| `git push -u origin main` + `git ls-remote` | `main` | `9065392` |
| Quét staged: `.pt`, `.env`, `kaggle.json`, token `KGAT_` thật | commit đầu | không có (chỉ token giả trong test của skill) |

## 4. Local cleanup

Không có.

## 5. Next session

- Push artifact mới: `dvc add <target>` đúng thư mục → `dvc push <target>.dvc` → `dvc status -c <target>.dvc` → commit `.dvc` + `.gitignore`.
- Lấy dữ liệu archive:
  - Clone `odixe06/mobihet-archive-b9-b21` (hoặc dùng `../mobihet_archive`), rồi `dvc pull papers/<…>.dvc` trong repo đó.
  - Không chép manifest archive sang đây.
- Quirk: G: từ chối `chmod 0444` (DEBUG `failed to protect` không phải lỗi). `status -c` báo `deleted` = không còn trong cache local, không phải mất trên Drive.
