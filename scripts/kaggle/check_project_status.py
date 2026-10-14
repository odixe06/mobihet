"""Read latest batch status of pushed project notebooks with owner-scoped credentials."""
import concurrent.futures
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AS = ROOT / '.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py'


def main():
    notebooks = []
    for path in sorted((ROOT / 'papers').glob('*/notebook/*/kernel-metadata.json')):
        if not (path.parent / 'PUSHED').exists():
            continue
        meta = json.loads(path.read_text())
        notebooks.append(dict(kernel=meta['id'], build=path.parts[-4], folder=str(path.parent.relative_to(ROOT))))

    def read(item):
        owner = item['kernel'].split('/')[0]
        result = subprocess.run([sys.executable, str(AS), owner, '--', 'kaggle', 'kernels', 'status', item['kernel']],
                                capture_output=True, text=True, timeout=120)
        found = re.search(r'KernelWorkerStatus\.([A-Z_]+)', result.stdout)
        return dict(item, status=found[1] if result.returncode == 0 and found else 'UNKNOWN',
                    returncode=result.returncode)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(read, notebooks))
    snapshot = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    evidence='owner-scoped latest batch version status, not artifact verification or session heartbeat',
                    kernels=rows)
    (ROOT / 'docs/operations/kernel-status-current.json').write_text(json.dumps(snapshot, indent=2))
    counts = {state: sum(row['status'] == state for row in rows) for state in sorted({v['status'] for v in rows})}
    print('Project kernels:', len(rows), counts, flush=True)
    for row in rows:
        if row['status'] not in ('COMPLETE', 'ERROR'):
            print(row['kernel'], row['status'], flush=True)


if __name__ == '__main__':
    main()
