# Git / DVC — backup và khôi phục

Chỉ backup/push/dọn khi owner yêu cầu; yêu cầu backup không cho phép xóa local. Quy trình: skill
[repo-backup](../../.claude/skills/repo-backup/SKILL.md). Bảng mọi manifest và số file: [artifact-index](artifact-index.md).
Lịch sử từng đợt 14–20/10 (đã đẩy gì, đã xóa gì): [backup-history](../archive/operations/backup-history-2026-10-14-to-20.md).

## Đích

- Git: `git@github.com:odixe06/mobihet.git`, nhánh `main`. Kiểm `git ls-remote origin refs/heads/main` = `git rev-parse HEAD`.
- DVC remote `gdrive`: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage` (layout DVC 3 `files/md5`).
- Bundle: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle` (`--all`, verify trước khi thay atomic).
- Repo lưu trữ cũ: `odixe06/mobihet-archive-b9-b21`, tag `archive-3` (giữ nguyên). Cloud sync của Drive không phải điều kiện hoàn tất.

## Lần gần nhất và phần còn lệch

- **20/10:** Git `7862c7d` trên `main` (local = remote lúc đó), bundle verify. DVC fd-probe `kaggle.dvc` `23c83f48…dir` (7033 file),
  `input.dvc` `485fc49d…dir` (1494 file); `dvc status -c`: new/missing = 0.
- **Chưa backup từ sau đó:** thay đổi docs/source chưa commit (run cuối r24/r25, gọn tài liệu 20/10), output fd-probe thu từ vòng 23 trở đi
  ⇒ đợt sau: `dvc_merge_push.py` cho fd-probe `kaggle.dvc` + `input.dvc`, commit, push, thay bundle.

## Đẩy

1. Không `dvc add`: manifest ở gốc build + `artifacts/` bị git-ignore nên DVC 3 từ chối. Dùng
   `python scripts/dvc_merge_push.py <build>/<name>.dvc`: gộp `.dir` cũ theo relpath (file local ghi đè cùng relpath, giữ entry chỉ có trên remote),
   copy object tmp → MD5 → rename, ghi `.dvc` cuối cùng.
2. Kiểm `dvc status -c <manifest>`: **new/missing = 0** mới là đủ; `deleted` chỉ là thiếu cache local (đúng thiết kế sau khi dọn).
3. Git: không file > 10 MiB, quét credential trước commit; `private_push/` và runtime cache không lên Git/DVC.
   Commit không mang co-author/attribution Claude.
4. Bundle `--all` → `git bundle verify` → thay file trên Drive.

## Khôi phục

- `dvc pull <manifest>` theo [artifact-index](artifact-index.md); kiểm dung lượng đĩa trước. Không `dvc add` workspace partial sau khi pull một phần.
- Fresh checkout: lớp `.compat` theo [project-layout](project-layout.md#lớp-tương-thích-compat-và-fresh-checkout).
- Sau khi pull build cũ, tạo lại alias run canonical từ [registry/artifact-aliases.json](registry/artifact-aliases.json):

```bash
python3 - <<'PYALIAS'
from pathlib import Path
import json, os
for a in json.loads(Path('docs/operations/registry/artifact-aliases.json').read_text())['aliases']:
    p, target = Path(a['path']), Path(a['target'])
    if target.exists() and not p.exists() and not p.is_symlink():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(os.path.relpath(target, p.parent), target_is_directory=True)
PYALIAS
```

- Một số `runs.dvc` là snapshot rỗng lịch sử: pull `kaggle.dvc` cùng build rồi tạo alias, không dùng snapshot rỗng thay output Kaggle.
- Chỉ lấy metadata: `python scripts/restore_dvc_metadata.py <canonical.dvc> --proof <proof.json>` (vẫn partial, không `dvc add`).

## Dọn đĩa (chỉ khi owner yêu cầu)

Chỉ xóa phần đã có trên remote (new/missing = 0) và không cần cho round/collect tiếp theo; giữ code/docs/notebook/manifest/private_push.
Trả dung lượng về Windows (không bắt buộc): `sudo fstrim -v /`, PowerShell Administrator `wsl --shutdown`, rồi diskpart compact VHDX.
