#!/usr/bin/env python
"""Status + pull of the P2 kernels (scripts/kaggle/gen_p2.py) into papers/p2/kaggle/<rid>/, then copy every finished run to
papers/p2/runs/<arm>_s<seed>/ for scripts/report_p2.py. Read-only on Kaggle. Run from mobihet/, env nckh:

    python scripts/kaggle/collect_p2.py            # COMPLETE kernels are pulled; others print their status
"""
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
H = ROOT / '.claude/skills/kaggle-training-notebook/scripts'


def main():
    for d in sorted((ROOT / 'papers/p2/notebook').iterdir()):
        if not (d / 'PUSHED').exists():
            continue
        kid = json.loads((d / 'kernel-metadata.json').read_text())['id']; owner = kid.split('/')[0]
        status = subprocess.run([sys.executable, str(H / 'kaggle_as.py'), owner, '--', 'kaggle', 'kernels', 'status', kid],
                                capture_output=True, text=True, timeout=120)
        if status.returncode:
            raise RuntimeError(f'status failed for {kid}: {status.stderr[-600:]}')
        st = status.stdout.strip()
        print(d.name, '|', st[-160:], flush=True)
        if 'COMPLETE' not in st and 'ERROR' not in st:
            continue
        dest = ROOT / 'papers/p2/kaggle' / d.name
        r = subprocess.run([sys.executable, str(H / 'kaggle_as.py'), owner, '--', sys.executable, str(H / 'pull_kernel_output.py'),
                            kid, '-p', str(dest)], capture_output=True, text=True, timeout=3600)
        print(r.stdout[-600:], r.stderr[-600:], 'pull rc', r.returncode, flush=True)
        if r.returncode:
            raise RuntimeError(f'incomplete pull for {kid}; keep partial files for retry')
        for res in sorted(dest.rglob('result.json')):
            run = res.parent
            out = ROOT / 'papers/p2/runs' / run.name; out.mkdir(parents=True, exist_ok=True)
            for f in run.iterdir():
                shutil.copyfile(f, out / f.name)
            print('collected', run.name, flush=True)


if __name__ == '__main__':
    main()
