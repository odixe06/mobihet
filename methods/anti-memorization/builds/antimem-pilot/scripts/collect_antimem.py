"""One-shot terminal collection + audit of AntiMem/QProxy pilot sessions (no polling loop).

Audit: executed notebook source == pushed (key redacted); runtime antimem.py/input manifest SHA; per arm config
fingerprint, marker chain, kept weights (r10/r20/latest) SHA + strict DAGSNet load + finite, every committed
predictions npz SHA and recomputed confusion -> ten metrics == marker eval. Pruned rounds are listed, not hidden.

    python methods/anti-memorization/builds/antimem-pilot/scripts/collect_antimem.py [--seed 42]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from mh.dagsnet import build_model, CFG
from mh.metrics import metrics_from_confusion

HELPER = ROOT / '.agents/skills/kaggle-training-notebook/scripts'
PATTERN = (r'^(source/|runs/antimem_summary\.json|runs/[^/]+/(complete/|config\.json|result\.json|metrics\.csv|log\.txt|'
           r'predictions/|weights/round_0(10|20)\.pt))')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_arm(run, inputs):
    cfg = json.loads((run / 'config.json').read_text())
    markers = sorted((run / 'complete').glob('round_*.json'))
    rows = [json.loads(m.read_text()) for m in markers]
    previous, checks, kept = None, 0, []
    for i, row in enumerate(rows):
        assert row['round'] == i + 1 and row['prev_sha'] == previous and row['fingerprint'] == cfg['fingerprint']
        previous = row['weights_sha']
        wp = run / 'weights' / f'round_{row["round"]:03d}.pt'
        if wp.exists():
            assert sha(wp) == row['weights_sha'], ('weights sha', run.name, row['round'])
            w = torch.load(wp, map_location='cpu', weights_only=True)
            model = build_model(dict(CFG, n_features=66, num_classes=16), latent=True)
            for sd in w['private'] + w['proxy']:
                model.load_state_dict(sd, strict=True)
                assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
            kept.append(row['round'])
        pf = run / 'predictions' / f'round_{row["round"]:03d}.npz'
        assert sha(pf) == row['predictions_sha'], ('predictions sha', run.name, row['round'])
        arrays = np.load(pf)
        for role, holds in row['eval'].items():
            for hold, rules in holds.items():
                y = arrays[hold + '_y'].astype(np.int64)
                for rule, ev in rules.items():
                    pred = arrays[f'{role}_{hold}_{rule}'].astype(np.int64)
                    cms = [np.bincount(y * 16 + p, minlength=256).reshape(16, 16) for p in pred]
                    assert all(np.array_equal(c, np.array(e)) for c, e in zip(cms, ev['confusion_clients']))
                    mm = [metrics_from_confusion(c) for c in cms]
                    for key, value in ev['mean_metrics'].items():
                        assert abs(float(np.mean([m[key] for m in mm])) - value) < 1e-12, (run.name, role, hold, rule, key)
                    checks += 1
    return dict(run=run.name, rounds=len(rows), complete=len(rows) == 20, kept_weight_rounds=kept,
                metric_checks=checks, level='marker chain, kept weights strict/finite, predictions->confusion->10 metrics')


def collect(seed, suffix='v1'):
    rid = f'am-s{seed}-20261015-{suffix}'; folder = BASE / 'notebooks' / rid
    dest = BASE / 'artifacts/kaggle' / rid; marker = dest / 'AUDITED.json'
    if not (folder / 'PUSHED').exists():
        return None
    if marker.exists():
        return json.loads(marker.read_text())
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    prefix = [sys.executable, str(HELPER / 'kaggle_as.py'), meta['id'].split('/')[0], '--']
    status = subprocess.run(prefix + ['kaggle', 'kernels', 'status', meta['id']], check=True, capture_output=True, text=True)
    print(status.stdout.strip(), flush=True)
    if not any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE', 'ERROR')):
        return None
    subprocess.run(prefix + [sys.executable, str(HELPER / 'pull_kernel_output.py'), meta['id'], '-p', str(dest), '-j', '6',
                             '--pattern', PATTERN], check=True)
    private = BASE / 'private_push' / rid; executed = private / 'executed.ipynb'
    subprocess.run(prefix + [sys.executable, str(HELPER / 'download_executed_notebook.py'), meta['id'], '--version', '1',
                             '--output', str(executed)], check=True, umask=0o077)
    executed.chmod(0o600)
    pushed = nbformat.read(private / meta['code_file'], as_version=4)
    reference = nbformat.read(folder / meta['code_file'], as_version=4)
    key = next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == '_INLINE_WANDB_KEY' for t in n.targets))
    actual = nbformat.reads(executed.read_text().replace(key, '[REDACTED]'), as_version=4)
    assert len(actual.cells) == len(reference.cells)
    for i, c in enumerate(actual.cells):
        if c.cell_type == 'code':
            assert c.source == pushed.cells[i].source.replace(key, '[REDACTED]'), ('executed source drift', i)
    actual.cells[1].source = reference.cells[1].source; actual.metadata['credential_cell_redacted'] = True
    nbformat.write(actual, folder / 'executed.redacted.ipynb')
    stage = BASE / 'artifacts/input/antimem-src-20261015'
    inputs = json.loads((stage / 'input_manifest.json').read_text())
    assert json.loads((dest / 'source/input_manifest.json').read_text()) == inputs
    assert sha(dest / 'source/antimem.py') == inputs['antimem.py'] == sha(BASE / 'source/proj/antimem.py')
    logs = ''.join(v.get('data', '') for v in json.loads((dest / f'mobihet-{rid}.log').read_text()))
    assert key not in logs
    audits = [audit_arm(run, inputs) for run in sorted((dest / 'runs').glob(f'am_*_s{seed}'))
              if (run / 'complete').exists() and list((run / 'complete').glob('round_*.json'))]
    summary = dest / 'runs/antimem_summary.json'
    record = dict(kernel=meta['id'], version=1, inputs=sha(stage / 'input_manifest.json'), audits=audits,
                  summary=json.loads(summary.read_text()) if summary.exists() else None,
                  arms_r20=sorted(a['run'] for a in audits if a['complete']),
                  limit='No independent saved-weight inference; pruned weights only for rounds outside r10/r20/latest')
    marker.write_text(json.dumps(record, indent=2) + '\n')
    print('AUDITED antimem', rid, 'arms', len(audits), 'r20', len(record['arms_r20']), flush=True)
    return record


if __name__ == '__main__':
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, action='append')
    for seed in (p.parse_args().seed or [42, 43]):
        for suffix in ('v1', 'tail-v1', 'tail-v2'):
            collect(seed, suffix)
