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

- **21/10 ~17:30Z** (owner yêu cầu backup + dọn): DVC fd-probe qua `dvc_merge_push.py` có điều tiết (dừng khi ổ cache Drive D: < 15 GB trống;
  cache Drive for Desktop = `D:\ggdrive_cached`, `df` của G: phản ánh D:, không phải quota cloud; dvc-storage trên Drive 757 GB):
  `kaggle.dvc` `381c1e32…dir` (10.777 entry, đẩy mới 71,56 GB), `input.dvc` `6ea020e5…dir` (2.644, 0 GB mới — trùng weights),
  `fixture-cache.dvc` `eb9eacad…dir` (23). `dvc status -c` 3 manifest: new/missing = 0, `deleted` 4.990 = chỉ thiếu cache local.
  Không đưa vào đợt này: hai output chấm pull dở/chưa audit (`fd-e03-12-s42zk02-…-f2` còn `.part`, `fd-e18-26-s42zi-…-f1`).
- Git `bcccbe5` (manifest + code/docs) rồi commit ghi dọn; local = remote (`git ls-remote`); bundle `--all` verify rồi thay trên Drive.
- **Dọn local 21/10 ~17:37Z** (owner yêu cầu; từng file kiểm MD5 có trên remote cùng size, weights kiểm sha256 = hợp đồng chấm đã audit):
  xóa 40 stage `artifacts/input/fd-e*` + `*-parent`, 190 predictions `.npz` của chấm đã audit, 178 weights train của round đã chấm audit;
  WSL dùng 221 → 112 GB. Giữ: weights round chưa chấm, eval JSON/metrics/AUDITED, `fd-src-*`, manifest. Ổ C: (VHDX) chưa trả lại — owner
  tự `sudo fstrim -v /` + `wsl --shutdown` + compact VHDX.
- `docs/methods/full-drift/results/per-class-f1.csv` 14 MB > 10 MiB vẫn ở Git như các commit trước; chuyển DVC nếu cần đúng quy tắc.
- **Còn lại sau đợt này:** output chấm COMPLETE chưa pull (za r45–50, ze r9–28, ze r49–50, zk r3–12, zk r13–22, zi r1–8), zi r18–26 (pull đủ, chưa audit),
  train S2 zi03/zk04 và chấm sau đó ⇒ đợt sau merge-push `kaggle.dvc` + `input.dvc`.

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
