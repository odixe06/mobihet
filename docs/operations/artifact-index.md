# Chỉ mục artifact / pull lại

Chạy lệnh từ root repo; DVC remote `gdrive` trênG:. [Hồ sơ backup](backup.md).
Manifest giữ snapshot gồm cả entry remote-only khi local restore/collect một phần. Không `dvc add` target partial.
Kiểm dung lượng trước pull; `dvc pull <manifest>` phục hồi toàn snapshot trongbảng. Backup đangpush, chưacleanup đợt mới.

| Manifest | File | GiB logical | Local | Lệnh khôi phục |
|---|---:|---:|---|---|
| `knowledge/ckpt_round_005.pt.dvc` | 1 | 0.005 | đã xóa | `dvc pull knowledge/ckpt_round_005.pt.dvc` |
| `papers/new-fl/input.dvc` | 162 | 0.297 | đã xóa | `dvc pull papers/new-fl/input.dvc` |
| `papers/new-fl/kaggle.dvc` | 622 | 7.450 | đã xóa | `dvc pull papers/new-fl/kaggle.dvc` |
| `papers/new-fl/resume_reference.dvc` | 13 | 0.093 | đã xóa | `dvc pull papers/new-fl/resume_reference.dvc` |
| `papers/new-fl/runs.dvc` | 0 | 0.000 | đã xóa | `dvc pull papers/new-fl/runs.dvc` |
| `papers/new-fl/smoke.dvc` | 79 | 0.558 | đã xóa | `dvc pull papers/new-fl/smoke.dvc` |
| `papers/p2/kaggle.dvc` | 52 | 0.001 | đã xóa | `dvc pull papers/p2/kaggle.dvc` |
| `papers/p2/runs.dvc` | 44 | 0.001 | đã xóa | `dvc pull papers/p2/runs.dvc` |
| `papers/p2/source_audit.dvc` | 32 | 0.000 | đã xóa | `dvc pull papers/p2/source_audit.dvc` |
| `papers/pilot/fixture.dvc` | 67 | 0.149 | có (có thể partial) | `dvc pull papers/pilot/fixture.dvc` |
| `papers/pilot/runs.dvc` | 32 | 0.002 | đã xóa | `dvc pull papers/pilot/runs.dvc` |
| `papers/v2/input.dvc` | 170 | 0.297 | đã xóa | `dvc pull papers/v2/input.dvc` |
| `papers/v2/kaggle.dvc` | 2422 | 29.499 | có (có thể partial) | `dvc pull papers/v2/kaggle.dvc` |
| `papers/v2/resume_reference.dvc` | 108 | 1.150 | đã xóa | `dvc pull papers/v2/resume_reference.dvc` |
| `papers/v2/runs.dvc` | 0 | 0.000 | có (có thể partial) | `dvc pull papers/v2/runs.dvc` |
| `papers/v2/smoke.dvc` | 612 | 6.392 | đã xóa | `dvc pull papers/v2/smoke.dvc` |
| `papers/v3/input.dvc` | 534 | 0.892 | đã xóa | `dvc pull papers/v3/input.dvc` |
| `papers/v3/kaggle.dvc` | 4556 | 125.595 | có (có thể partial) | `dvc pull papers/v3/kaggle.dvc` |
| `papers/v3/resume_reference.dvc` | 216 | 4.434 | đã xóa | `dvc pull papers/v3/resume_reference.dvc` |
| `papers/v3/runs.dvc` | 0 | 0.000 | có (có thể partial) | `dvc pull papers/v3/runs.dvc` |
| `papers/v3/smoke.dvc` | 888 | 20.426 | đã xóa | `dvc pull papers/v3/smoke.dvc` |
| `papers/v4/input.dvc` | 376 | 0.595 | đã xóa | `dvc pull papers/v4/input.dvc` |
| `papers/v4/kaggle.dvc` | 1784 | 18.371 | có (có thể partial) | `dvc pull papers/v4/kaggle.dvc` |
| `papers/v4/resume_reference.dvc` | 89 | 0.845 | đã xóa | `dvc pull papers/v4/resume_reference.dvc` |
| `papers/v4/runs.dvc` | 0 | 0.000 | có (có thể partial) | `dvc pull papers/v4/runs.dvc` |
| `papers/v4/smoke.dvc` | 243 | 2.241 | đã xóa | `dvc pull papers/v4/smoke.dvc` |
| `papers/v5/input.dvc` | 291 | 0.446 | đã xóa | `dvc pull papers/v5/input.dvc` |
| `papers/v5/kaggle.dvc` | 3802 | 41.338 | có (có thể partial) | `dvc pull papers/v5/kaggle.dvc` |
| `papers/v5/resume_reference.dvc` | 135 | 1.253 | đã xóa | `dvc pull papers/v5/resume_reference.dvc` |
| `papers/v5/runs.dvc` | 0 | 0.000 | có (có thể partial) | `dvc pull papers/v5/runs.dvc` |
| `papers/v5/smoke.dvc` | 711 | 6.368 | đã xóa | `dvc pull papers/v5/smoke.dvc` |
| `papers/v6/input.dvc` | 396 | 0.595 | có (có thể partial) | `dvc pull papers/v6/input.dvc` |
| `papers/v6/kaggle.dvc` | 1666 | 13.091 | có (có thể partial) | `dvc pull papers/v6/kaggle.dvc` |
| `papers/v6/resume_reference.dvc` | 60 | 0.408 | có (có thể partial) | `dvc pull papers/v6/resume_reference.dvc` |
| `papers/v6/smoke.dvc` | 240 | 1.629 | có (có thể partial) | `dvc pull papers/v6/smoke.dvc` |

`papers/<method>/runs/` hiện là alias vào `kaggle/`, không phải bản artifact riêng; các `runs.dvc` rỗng cũ chỉ là lịch sử.
158alias cómapping trong [inventory](backup-delta-2026-10-12.json). Sau pull `kaggle.dvc` cầnthiết, tạo lại alias:

```bash
python3 - <<'PY'
from pathlib import Path
import json, os
mapping = json.loads(Path('docs/operations/backup-delta-2026-10-12.json').read_text())['run_aliases']
for alias, target in mapping.items():
    p, q = Path(alias), Path(target)
    if q.exists() and not p.exists() and not p.is_symlink():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(os.path.relpath(q, p.parent), target_is_directory=True)
PY
```

Input/smoke/resume cũng cómanifest riêng. Code/publicnotebook/report nằmGit.
Credential `private_push`, runtime lock, Pythoncache vàW&B SDKcache không đưa lênGit/DVC.
