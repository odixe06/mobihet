# Git / DVC backup record

Updated: 2026-10-11 (UTC+7, bàn giao) — DVC push pilot P1 (fixture + runs), commit + push code/tài liệu pilot, bundle mới; xóa `~/.cache/mobihet`.
Procedure: [repo-backup](../../.claude/skills/repo-backup/SKILL.md) (bản Codex: `.agents/skills/repo-backup/`).

## 1. Destinations

- Git: `git@github.com:odixe06/mobihet.git` (private, tạo 2026-10-10), nhánh `main`. Commit đầu `9065392` khớp `git ls-remote`.
- DVC remote `gdrive` = `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`, **dùng chung với archive** (object trùng hash chỉ lưu một lần).
- Bundle: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle` = repo **này** (`--all`, cập nhật mỗi đợt backup; đợt mới nhất xem §4).
  Bundle cũ cùng tên (repo trước khi đổi tên) đã được thay sau khi kiểm: mọi ref của nó nằm trong `mobihet-archive-b9-b21.bundle`.
- Cloud: file đã nằm trên G: = đã backup. Upload lên cloud: chưa kiểm.

## 2. Tracked artifacts

| Manifest | MD5 | Files | Bytes | Local |
|---|---|---:|---:|---|
| `knowledge/ckpt_round_005.pt.dvc` | `c1466d52f38b` | 1 | 4.924.379 | có |
| `papers/pilot/fixture.dvc` | `cb782c4671b1` | 67 | 159.554.510 | có (dựng lại: `scripts/build_pilot_fixture.py`) |
| `papers/pilot/runs.dvc` | `ca8715563f5d` | 32 | 1.833.638 | có |

Repo archive (đọc [index](../legacy/kien-thuc-ke-thua.md#0-vì-sao-dừng-hướng-cũ-archive-nằm-ở-đâu)) giữ manifest của Build1–21 trên cùng remote.

## 3. Checks that passed

| Check | Scope | Result |
|---|---|---|
| `dvc push knowledge/ckpt_round_005.pt.dvc` | 1 manifest | `1 file pushed` |
| `dvc status -c` | 1 manifest | `Cache and remote 'gdrive' are in sync.` |
| `git push -u origin main` + `git ls-remote` | `main` | `9065392` |
| Quét staged: `.pt`, `.env`, `kaggle.json`, token `KGAT_` thật | commit đầu | không có (chỉ token giả trong test của skill) |

## 4. Local cleanup

**2026-10-11 — đóng archive 3** (chủ dự án: dọn `.git`, đợi kernel xong, backup hết rồi xóa thư mục archive):
- `.git` archive: xóa 91 `tmp_obj_*` (11,75 GiB) + `git gc --prune=now`, `fsck` sạch trước/sau.
- Archive: B20 4/4 + B21 18/18 verified; `dvc push` hai đợt (1.557 + 1.223 file, `status -c` in sync), gồm hai thư mục trước đây chỉ ignore
  (`papers/build7_common/local`, `papers/leakage_audit/fixtures`).
- **Kiểm remote độc lập** (đọc thẳng `dvc-storage`): 148 output của mọi `.dvc` trong archive, 84.134 object — thiếu 0.
- Commit `b8aceed` nhánh `session-20261004-leakage-probes` + tag `archive-3`, push, khớp `git ls-remote`.
  Bundle `mobihet-archive-b9-b21.bundle` (16,3 MB) verify okay ở scratch và trên G:, clone thử được.
- Xóa `/home/odixe/nckh/veremi/mobihet_archive` (35 GB, gồm `.dvc/cache`): đĩa đã dùng 151,5 → 115,0 GB.
- `~/.cache/mobihet` (39 GB scratch archive: fixture B12 20 GB, preflight B19–B21, `ab`): **xóa 2026-10-11 theo lệnh chủ dự án**.
  Không có trong DVC; tái tạo bằng script archive. Đĩa đã dùng 115,0 → 74,2 GB.

**2026-10-11 bàn giao:** `dvc add` + `dvc push` `papers/pilot/{fixture,runs}` (99 file), `dvc status -c` 3 manifest in sync; quét W&B key / `.part`: không có.
  Git: commit `c6741fb` (66 file, lớn nhất 28 kB; quét staged không có W&B key/`KGAT_`) push `main`, khớp `git ls-remote`; commit hồ sơ này push tiếp.
  Bundle `mobihet.bundle` (`--all`) tạo lại sau commit hồ sơ, verify ở scratch rồi thay trên G:.
- Kaggle (2026-10-11): dataset private `<owner>/mobihet-p2-src` (code + fixture pilot, 76 file, `ready`) cho khanhmay0304, minhtriethihi, catbaochau, minhtrit06.
  Đây là input, không phải backup. Nguồn thật là git + `papers/pilot/fixture.dvc`.

## 5. Next session

- Push artifact mới: `dvc add <target>` đúng thư mục → `dvc push <target>.dvc` → `dvc status -c <target>.dvc` → commit `.dvc` + `.gitignore`.
- Lấy dữ liệu archive:
  - Clone `odixe06/mobihet-archive-b9-b21` (tag `archive-3`) hoặc bundle `mobihet-archive-b9-b21.bundle`, rồi `dvc pull papers/<…>.dvc` trong repo đó.
    Bản local `../mobihet_archive` đã xóa 2026-10-11.
  - Không chép manifest archive sang đây.
- Quirk: G: từ chối `chmod 0444` (DEBUG `failed to protect` không phải lỗi). `status -c` báo `deleted` = không còn trong cache local, không phải mất trên Drive.
