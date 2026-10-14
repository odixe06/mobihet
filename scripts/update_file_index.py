"""Canonical file index plus DVC/artifact routes; omit compatibility symlinks and secrets."""
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'docs/file-index.md'


def main():
    raw=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-0'],cwd=ROOT).decode().split('\0')
    files={p for p in raw if p and (ROOT/p).is_file() and not (ROOT/p).is_symlink()}
    files.update(('docs/file-index.md','docs/archive/file-index.md'))
    archived={p for p in files if p.startswith(('archive/','docs/archive/'))}
    working=files-archived
    groups={}
    for p in sorted(working):groups.setdefault(p.split('/')[0] if '/' in p else 'root',[]).append(p)
    lines=['# Index từng file toàn dự án','',
        '[Mục lục tổng quan](README.md) · [Đề xuất và trạng thái](proposals.md) · [Cấu trúc](operations/project-layout.md)','',
        f'{len(working)} file khu làm việc; {len(archived)} file tại [index archive](archive/file-index.md). '
        f'Tổng {len(files)} file canonical; không nhân đôi alias hoặc liệt kê key/checkpoints.','',
        'Tự sinh: `python3 scripts/update_file_index.py`. Artifact remote-only có route DVC, không coi thiếu local là chưa từng chạy.','',
        '## Artifact và các build','',
        '| build / vị trí | thiết kế, kết quả và metadata |','|---|---|']
    registry=ROOT/'docs/operations/workspace-layout.json'
    for old,b in json.loads(registry.read_text())['builds'].items():
        if b['archived']:continue
        doc=f'methods/{b["family"]}/builds/{b["build"]}/README.md'
        base=ROOT/b['path']
        metadata=', '.join(f'[{p.name}](../{quote(str(p.relative_to(ROOT)),safe="/._-")})' for p in sorted(base.glob('*.dvc')))
        lines.append(f'| `{b["path"]}/artifacts/` ({old}) | [{b["build"]}]({doc}); {metadata or "chưa có DVC metadata"} |')
    lines+=['','Private inline notebooks: `methods/<method>/builds/<build>/private_push/` ignored0600; không phải public source.',
        'Dataset gốc: [knowledge](../knowledge/README.md). [Khôi phục DVC](operations/backup.md).','']
    for group in ['root','docs','methods','core','knowledge','scripts','logs','.agents','.claude','.dvc']+sorted(groups):
        if group not in groups:continue
        lines += [f'## {group}','']
        lines += [f'- [{p}](../{quote(p,safe="/._-")})' for p in groups.pop(group)]
        lines.append('')
    INDEX.write_text('\n'.join(lines).rstrip()+'\n')
    history=['# Index archive','', '[Mục lục hiện tại](../README.md) · [Index khu làm việc](../file-index.md) · '
        '[Các build đã kết thúc](README.md)','',f'{len(archived)} file lịch sử; không dùng số phiên bản để suy phương pháp nào thay thế phương pháp nào.','']
    for p in sorted(archived):history.append(f'- [{p}]({quote(os.path.relpath(ROOT/p,ROOT/"docs/archive"),safe="/._-")})')
    (ROOT/'docs/archive/file-index.md').write_text('\n'.join(history)+'\n')
    print('Canonical index:',len(working),'working;',len(archived),'archive;',len(files),'total')


if __name__=='__main__':main()
