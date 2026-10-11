#!/usr/bin/env python
"""Push one generated notebook as its owner (replaces push_notebook.sh, 2026-10-05). Run from mobihet/, env nckh.

    python scripts/push_kaggle.py papers/build9/notebook/<rid>            # owner read from kernel-metadata.json
    DRY_RUN=1 python scripts/push_kaggle.py papers/build9/notebook/<rid>  # every step except the push

The W&B key goes ONLY into a temporary copy (the repo is on GitHub); the two temporary files are removed by exact path.
Exactly one cell (the W&B login) may differ from the repo copy, the kernel must be private and owned by the metadata owner.
No automatic retry: a push whose reply was lost may still have started a run -- check `kaggle kernels status` first.
After a push: PUSHED marker + frozen proj/ snapshot next to the notebook. Never push a folder that already has PUSHED.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / '.claude/skills/kaggle-training-notebook/scripts'


def main():
    d = Path(sys.argv[1]).resolve()
    if (d / 'PUSHED').exists(): raise SystemExit(f'{d} already pushed: {(d / "PUSHED").read_text().strip()}')
    nb_src = next(d.glob('*.ipynb')); meta = json.loads((d / 'kernel-metadata.json').read_text())
    owner, slug = meta['id'].split('/')
    assert meta['is_private'] is True and meta['code_file'] == nb_src.name, meta
    for parent in meta.get('kernel_sources', []):
        check = subprocess.run([sys.executable, str(H / 'kaggle_as.py'), owner, '--',
            'kaggle', 'kernels', 'status', parent], capture_output=True, text=True, timeout=90)
        if check.returncode or 'KernelWorkerStatus.COMPLETE' not in check.stdout:
            raise SystemExit(f'parent {parent} is not an accessible COMPLETE kernel source; use a verified private resume dataset')
    tmp = Path(tempfile.mkdtemp(prefix='mobihet_push_')); nb, md = tmp / nb_src.name, tmp / 'kernel-metadata.json'
    try:
        shutil.copyfile(nb_src, nb); shutil.copyfile(d / 'kernel-metadata.json', md)
        subprocess.run([sys.executable, str(H / 'embed_wandb_key.py'), str(nb), '--metadata', str(md)], check=True)
        a, b = json.loads(nb_src.read_text()), json.loads(nb.read_text())
        diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if x['source'] != y['source']]
        assert len(a['cells']) == len(b['cells']) and len(diff) == 1 and 'wandb.login' in ''.join(b['cells'][diff[0]]['source']), diff
        print('ok', meta['id'], flush=True)
        if os.environ.get('DRY_RUN') == '1': print('DRY_RUN:', meta['id'], 'ready, not pushed'); return
        r = subprocess.run([sys.executable, str(H / 'kaggle_as.py'), owner, '--', 'kaggle', 'kernels', 'push', '-p', str(tmp)],
                           capture_output=True, text=True, timeout=180)
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        if r.returncode or 'error' in r.stdout.lower(): raise SystemExit(f'push failed for {meta["id"]}: check kernels status before retrying')
    finally:
        for p in (nb, md):
            if p.exists(): p.unlink()
        tmp.rmdir()
    ver = re.search(r'Kernel version (\d+) successfully pushed', r.stdout); ver = ver.group(1) if ver else '?'
    (d / 'PUSHED').write_text(f'{meta["id"]} v{ver} pushed {time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())} '
                              '(W&B key embedded in the pushed copy only)\n')
    print((d / 'PUSHED').read_text().strip())
    subprocess.run([sys.executable, str(ROOT / 'scripts/snapshot_proj.py'), str(d)], check=True)
    if 'not valid kernel sources' in r.stdout or 'could not be added' in r.stdout:
        raise SystemExit('Kernel was pushed but required input sources were refused; inspect status, use a new attempt with a verified dataset parent')


if __name__ == '__main__':
    main()
