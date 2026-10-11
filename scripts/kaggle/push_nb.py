#!/usr/bin/env python
"""Push one generated notebook folder as its metadata owner, without switching the live login (no W&B). Run from mobihet/, env nckh.

    python scripts/kaggle/push_nb.py papers/p2/notebook/<rid>

Refuses a folder that already has PUSHED (a pushed notebook is never edited; a new attempt gets a new folder).
No automatic retry: a push whose reply was lost may still have started a run -- check `kaggle kernels status` first.
"""
import json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AS = ROOT / '.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py'


def main():
    d = Path(sys.argv[1]).resolve()
    if (d / 'PUSHED').exists():
        raise SystemExit(f'{d} already pushed: {(d / "PUSHED").read_text().strip()}')
    meta = json.loads((d / 'kernel-metadata.json').read_text())
    owner = meta['id'].split('/')[0]
    assert meta['is_private'] is True and (d / meta['code_file']).exists() and len(meta['title']) <= 50, meta
    r = subprocess.run([sys.executable, str(AS), owner, '--', 'kaggle', 'kernels', 'push', '-p', str(d)],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if r.returncode or 'error' in r.stdout.lower():
        raise SystemExit(f'push failed for {meta["id"]}: check kernels status before retrying')
    ver = re.search(r'Kernel version (\d+) successfully pushed', r.stdout)
    (d / 'PUSHED').write_text(f'{meta["id"]} v{ver.group(1) if ver else "?"} pushed {time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}\n')
    print((d / 'PUSHED').read_text().strip())
    if 'not valid' in r.stdout or 'could not be added' in r.stdout:
        raise SystemExit('pushed, but an input source was refused: inspect status before anything else')


if __name__ == '__main__':
    main()
