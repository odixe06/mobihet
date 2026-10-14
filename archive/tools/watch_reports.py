"""Read-only artifact watcher: rebuild reports/figures/index only when audited sets change."""
import hashlib
import subprocess
import sys
import time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
seen={}; expected={'v3':8,'v4':4,'v5':6}
while True:
    counts={}
    for build,total in expected.items():
        markers=sorted((ROOT/f'papers/{build}/kaggle').glob('*/AUDITED.json')); counts[build]=len(markers)
        signature=hashlib.sha256(b''.join(p.read_bytes() for p in markers)).hexdigest()
        if not markers or seen.get(build)==signature: continue
        subprocess.run([sys.executable,str(ROOT/f'scripts/report_{build}.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_audited.py'),'--build',build],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/update_file_index.py')],cwd=ROOT,check=True)
        seen[build]=signature
        print('Updated',build,len(markers),'/',total,'terminal kernel audits',flush=True)
    if all(counts[k]>=v for k,v in expected.items()): break
    time.sleep(55)
