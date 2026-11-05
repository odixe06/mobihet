"""Authorized frozen N20: launch once, then read-only startup/watch/collect/audit without model polling."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
entity=json.loads((ROOT/'runs/n-study/wandb-identity.json').read_text())['entity']
subprocess.run([sys.executable,str(ROOT/'scripts/run_bounded_python.py'),str(ROOT/'scripts/launch_n20.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/watch_collect_screening.py'),'--batch',str(ROOT/'runs/n-study/batch.json'),
    '--out',str(ROOT/'runs/n-study/collection'),'--root',str(ROOT/'artifacts/n20'),'--entity',entity],check=True,cwd=ROOT)
print('N20_LAUNCH_COLLECT_AUDIT_DONE',flush=True)
