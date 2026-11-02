"""After the collector exits, inventory every P20 arm, including failed/partial runs."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def main():
    state_path = Path('/home/odixe/.codex/long-jobs/jobs/kaggle-p-reboot-collect-20261101/state.json')
    while json.loads(state_path.read_text())['status'] in {'queued', 'running'}:
        time.sleep(30)
    batch = json.loads((ROOT / 'runs/p-study/batch.json').read_text())
    reports = []
    for spec in batch:
        folder = ROOT / 'artifacts/p20' / spec['slug']
        d = folder / spec['slug']
        row = {'arm': spec['arm'], 'ref': f'{spec["account"]}/{spec["slug"]}/1', 'issues': []}
        try:
            manifest = json.loads((folder / 'provenance/source_manifest.json').read_text())
            assert manifest['overlay'] == spec['overlay_sha256']
            for name, digest in manifest['files'].items():
                assert hashlib.sha256((folder / 'provenance/source' / name).read_bytes()).hexdigest() == digest, name
            row['source_verified'] = True
            rounds = [json.loads(line) for line in (d / 'rounds.jsonl').read_text().splitlines()] if (d / 'rounds.jsonl').exists() else []
            row['rounds'] = [r['round'] for r in rounds]
            row['test_cm_rounds'] = []
            for file in sorted(d.glob('cms_r*.npz')):
                with np.load(file) as z:
                    assert np.array_equal(z['ids'], np.arange(5343))
                    assert z['private'].shape == (5343, 16, 16) and (z['private'] >= 0).all()
                row['test_cm_rounds'].append(int(file.stem[-3:]))
            row['snapshot_rounds'] = []
            for file in sorted((d / 'models').glob('r*.pt')):
                state = torch.load(file, weights_only=True, map_location='cpu')
                assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)), str(file)
                row['snapshot_rounds'].append(int(file.stem[-3:]))
            row['full20_present'] = all(row[k] == list(range(1, 21)) for k in ['rounds', 'test_cm_rounds', 'snapshot_rounds'])
        except Exception as exc:
            row['issues'].append(f'{type(exc).__name__}: {exc}')
        reports.append(row)
        print('P20_INVENTORY_AUDITED', spec['arm'], flush=True)
    out = ROOT / 'runs/p-study/recovery-v1/all-arm-inventory-audit.json'
    out.write_text(json.dumps({'runs': reports, 'scope': 'Artifact inventory including failures; complete-run scientific audit remains in per-arm audit logs; no performance analysis'}, indent=2) + '\n')
    print('P20_ALL_ARM_INVENTORY_AUDITED', flush=True)

if __name__ == '__main__': main()
