"""Recover observation of the existing T4 probe; no training launch."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/p-study/numerics-t4'
SKILL = ROOT / '.agents/skills/kaggle-training-notebook/scripts'

def main():
    spec = json.loads((OUT / 'batch.json').read_text())[0]
    assert (OUT / 'push.json').is_file()
    subprocess.run([sys.executable, str(ROOT / 'scripts/watch_kaggle_batch.py'), '--batch', str(OUT / 'batch.json'),
                    '--out', str(OUT / 'reboot-watch'), '--interval', '1200'], check=True)
    ref = f'{spec["account"]}/{spec["slug"]}/{spec["version"]}'
    subprocess.run([sys.executable, str(SKILL / 'kaggle_as.py'), spec['account'], '--', sys.executable,
                    str(SKILL / 'pull_kernel_output.py'), ref, '-p', str(OUT / 'artifacts'), '-j', '8',
                    '--pattern', r'(^provenance/|^p20-numerics-t4/.*\.jsonl?$)'], check=True)
    print('P21_PROBE_OUTPUT_COLLECTED', flush=True)

if __name__ == '__main__': main()
