"""Rebuild docs/file-index.md from non-ignored workspace files, without reading contents."""
import subprocess
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'docs/file-index.md'


def main():
    raw = subprocess.check_output(['rg', '--files', '--hidden', '-g', '!.git/**', '-0'], cwd=ROOT)
    files = {p for p in raw.decode().split('\0') if p and (ROOT / p).is_file()}
    files.add(INDEX.relative_to(ROOT).as_posix())
    groups = {}
    for path in sorted(files):
        group = path.split('/')[0] if '/' in path else 'root'
        groups.setdefault(group, []).append(path)
    lines = ['# Index file toàn dự án', '',
             'Danh mục tự sinh bằng `python3 scripts/update_file_index.py`. Định tuyến theo nhu cầu: [README](../README.md).', '',
             f'Liệt kê {len(files)} file workspace không bị ignore; dùng `rg`, không cần commit/stage. Không đọc nội dung file. '
             'Artifact/data/key bị ignore được định tuyến theo thư mục bên dưới, không liệt kê từng checkpoint.', '',
             '## Artifact và dữ liệu bị ignore', '',
             '| vị trí | nội dung / nơi tra cứu |', '|---|---|',
             '| `papers/pilot/fixture/` | Fixture; [fixture.dvc](../papers/pilot/fixture.dvc) |',
             '| `papers/pilot/runs/` | P1; [runs.dvc](../papers/pilot/runs.dvc), [kết quả](research/probes/pilot-p1.md) |',
             '| `papers/p2/kaggle/`, `papers/p2/runs/`, `papers/p2/source_audit/` | P2 pulled outputs/source; [audit](../papers/p2/audit.json) |',
             '| `papers/new-fl/kaggle/`, `papers/new-fl/runs/` | S-a; marker `VERIFIED.json`, weights/predictions; [ledger](../papers/new-fl/rebuild.md) |',
             '| `papers/new-fl/input/`, `papers/new-fl/smoke/`, `papers/new-fl/resume_reference/` | Staging và smoke; [ledger](../papers/new-fl/rebuild.md) |',
             '| `papers/v2/`, `papers/v3/`, `papers/v4/`, `papers/v5/` | Source/artifact theo đợt; [phương pháp](../proposals/README.md), [restore](operations/backup.md) |',
             '| `papers/<method>/private_push/` | Notebook private chứa key tạm; [quy định W&B](../LESSONS.md) |',
             '| `../dataset/`, DVC remote và archive | Dữ liệu gốc/phục hồi; [dataset](../knowledge/dataset.md), [backup](operations/backup.md) |', '',
             'DVC chỉ quản lý target có `.dvc`; không suy artifact mới đã backup từ sự tồn tại của thư mục. '
             'Snapshot code chạy nằm trong notebook/input/output; đối chiếu manifest trước tái sử dụng.', '']
    for group in ('root', 'proposals', 'docs', 'knowledge', 'mh', 'scripts', 'papers', '.agents', '.claude', '.dvc'):
        if group in groups:
            lines += [f'## {group}', '']
            lines += [f'- [{p}](../{quote(p, safe="/._-")})' for p in groups.pop(group)]
            lines.append('')
    for group, paths in sorted(groups.items()):
        lines += [f'## {group}', '']
        lines += [f'- [{p}](../{quote(p, safe="/._-")})' for p in paths]
        lines.append('')
    INDEX.write_text('\n'.join(lines), encoding='utf-8')
    print(f'{INDEX.relative_to(ROOT)}: {len(files)} files')


if __name__ == '__main__':
    main()
