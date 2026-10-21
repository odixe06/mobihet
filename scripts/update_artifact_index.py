"""Rewrite docs/operations/artifact-index.md from the tracked *.dvc manifests (md5, nfiles, size) and the local workspace.

    python3 scripts/update_artifact_index.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/operations/artifact-index.md'


def manifest(path):
    fields = {}
    for line in (ROOT / path).read_text().splitlines():
        key, _, value = line.strip().lstrip('- ').partition(': ')
        if key in ('md5', 'size', 'nfiles', 'path'): fields[key] = value
    return fields


def main():
    paths = subprocess.run(['git', 'ls-files', '*.dvc'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    rows = []
    for path in sorted(paths, key=lambda p: (p.startswith('archive/'), p)):
        m = manifest(path); out = (ROOT / path).parent / m['path']
        local = sum(1 for f in out.rglob('*') if f.is_file()) if out.is_dir() else int(out.is_file())
        rows.append(f"| `{path}` | `{m['path']}` | `{m['md5'][:8]}…` | {int(m.get('nfiles', 1))} | {int(m['size']) / 2**30:.3f} | "
                    f"{local if local else 'remote-only'} |")
    OUT.write_text('\n'.join([
        '# Chỉ mục artifact DVC',
        '',
        f'Sinh bởi `python3 scripts/update_artifact_index.py` từ {len(rows)} manifest `*.dvc` đang track; chạy lại sau mỗi collect/backup.',
        'Khôi phục: `dvc pull <manifest>` (môi trường `nckh`, chạy từ gốc repo) — quy tắc và alias ở [backup](backup.md).',
        '"Local" = số file đang có trong thư mục output; lệch với "Files" nghĩa là cần gộp/đẩy (`dvc_merge_push.py`) hoặc mới pull một phần.',
        '',
        '| manifest | output | md5 | Files | GiB | Local |',
        '|---|---|---|---:|---:|---|',
        *rows, '']))
    print('artifact index rows', len(rows))


if __name__ == '__main__':
    main()
