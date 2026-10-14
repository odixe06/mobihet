"""Rebuild this session's reports after terminal audits; never launch or alter remote runs."""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
expected={'v8':2,'v9':2,'full-validation':8}
seen={}
while True:
    counts={}
    for build,total in expected.items():
        markers=sorted((ROOT/f'papers/{build}/kaggle').glob('*/AUDITED.json'))
        counts[build]=len(markers)
        signature=hashlib.sha256(b''.join(p.read_bytes() for p in markers)).hexdigest()
        if not markers or seen.get(build)==signature:continue
        command=[sys.executable,str(ROOT/'scripts/report_full_validation.py')] if build=='full-validation' else [sys.executable,str(ROOT/'scripts/report_recent.py'),'--build',build]
        subprocess.run(command,cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/update_file_index.py')],cwd=ROOT,check=True)
        seen[build]=signature
        print('UPDATED audited report',build,counts[build],'/',total,flush=True)
    if all(counts[k]>=n for k,n in expected.items()):break
    time.sleep(55)
