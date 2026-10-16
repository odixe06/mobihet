"""One-shot terminal collection + audit of FullDrift sessions (no polling loop).

Audit: executed notebook source == pushed (key redacted); runtime input manifest/fd.py SHA; data manifest = full holdout
(source/test rows, group overlap 0); per arm config fingerprint, marker chain, epoch-6 weights SHA + strict DAGSNet
load + finite; every predictions npz SHA and recomputed per-client confusion -> ten metrics == marker eval.
The shared proxy is stored as one prediction row (one model after aggregation) and scored for every client.

    python methods/full-drift/builds/fd-probe/scripts/collect_fd.py [--only 'fd-s42g-*'] [--jobs 16]
    # several terminal sessions: one process each, e.g. 2 sessions --jobs 16, 3-4 sessions --jobs 8 (~32 connections total)
"""
import argparse
import ast
import fcntl
import fnmatch
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
PATTERN = (r'^(source/|runs/fd_summary\.json|runs/[^/]+/(complete/|config\.json|result\.json|metrics\.csv|log\.txt|'
           r'predictions/|weights/round_0(06|15)\.pt))')   # last epoch of 6- and 15-epoch arms


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_arm(run):
    cfg = json.loads((run / 'config.json').read_text())
    rows = [json.loads(m.read_text()) for m in sorted((run / 'complete').glob('round_*.json'))]
    previous, checks, kept = None, 0, []
    for i, row in enumerate(rows):
        assert row['round'] == i + 1 and row['prev_sha'] == previous and row['fingerprint'] == cfg['fingerprint']
        previous = row['weights_sha']
        wp = run / 'weights' / f'round_{row["round"]:03d}.pt'
        if wp.exists():
            assert sha(wp) == row['weights_sha'], ('weights sha', run.name, row['round'])
            w = torch.load(wp, map_location='cpu', weights_only=True)
            model = build_model(dict(CFG, n_features=66, num_classes=16), latent=True)
            for sd in w['private'] + w['proxy'] + w['ema_private']:
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
                    if len(cms) == 1: cms = cms * 20          # shared proxy: one model, every client
                    assert all(np.array_equal(c, np.array(e)) for c, e in zip(cms, ev['confusion_clients']))
                    mm = [metrics_from_confusion(c) for c in cms]
                    for key, value in ev['mean_metrics'].items():
                        assert abs(float(np.mean([m[key] for m in mm])) - value) < 1e-12, (run.name, role, hold, rule, key)
                    checks += 1
    target = 15 if run.name.endswith('_e15') else 6
    return dict(run=run.name, epochs=len(rows), complete=len(rows) == target, kept_weight_rounds=kept, metric_checks=checks,
                level='marker chain, epoch-6 weights strict/finite, predictions->confusion->10 metrics')


# pull connections: skill pull-outputs.md — throughput scales with open connections, ~32 total saturates the link
JOBS = '16'


def collect(rid, stage_tag='20261015'):
    folder = BASE / 'notebooks' / rid; dest = BASE / 'artifacts/kaggle' / rid; marker = dest / 'AUDITED.json'
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
    subprocess.run(prefix + [sys.executable, str(HELPER / 'pull_kernel_output.py'), meta['id'], '-p', str(dest), '-j', JOBS,
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
    stage = BASE / f'artifacts/input/fd-src-{stage_tag}'
    inputs = json.loads((stage / 'input_manifest.json').read_text())
    assert json.loads((dest / 'source/input_manifest.json').read_text()) == inputs
    assert sha(dest / 'source/fd.py') == inputs['fd.py'] == sha(stage / 'fd.py')   # the round's frozen copy, not the live source
    data = json.loads((dest / 'source/data_manifest.json').read_text())
    assert data['source_rows'] == 43045415 and data['test_rows'] == 10761343 and data['group_overlap'] == 0
    assert data['protocol'].startswith('temporal2% whole partition') and data['train_cap'] is None
    logs = ''.join(v.get('data', '') for v in json.loads((dest / f'mobihet-{rid}.log').read_text()))
    assert key not in logs
    # pulls may run in parallel; the audit holds ~4GB (ep6 test predictions), so audits take one shared lock
    with (ROOT / 'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        audits = [audit_arm(run) for run in sorted((dest / 'runs').glob('fd_*_s4[23]*'))
                  if (run / 'complete').exists() and list((run / 'complete').glob('round_*.json'))]
    summary = dest / 'runs/fd_summary.json'
    record = dict(kernel=meta['id'], version=1, inputs=sha(stage / 'input_manifest.json'), audits=audits,
                  train_rows=sum(sum(data['counts'][f'train_{k}']) for k in range(20)),
                  summary=json.loads(summary.read_text()) if summary.exists() else None,
                  arms_complete=sorted(a['run'] for a in audits if a['complete']),
                  limit='Single seed screening; weights kept for the last epoch only in the pull; test descriptive')
    marker.write_text(json.dumps(record, indent=2) + '\n')
    print('AUDITED fulldrift', rid, 'arms', len(audits), 'complete', len(record['arms_complete']), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default='*', help='rid glob: one process per session runs pulls in parallel')
    parser.add_argument('--jobs', default=JOBS, help='pull connections for this process (split ~32 across processes)')
    args = parser.parse_args(); JOBS = args.jobs
    torch.set_num_threads(1)
    sessions = [(f'fd-s42{s}-20261015-v2', '20261015') for s in 'abc']   # v1: production gate TypeError at Run init
    sessions += [(f'fd-s42{s}-20261015-r2-v1', '20261015-r2') for s in 'def']
    sessions += [(f'fd-s42{s}-20261015-r3-v1', '20261015-r3') for s in 'gh']
    sessions += [('fd-s42i-20261015-r4-v1', '20261015-r4'), ('fd-s42j-20261015-r5-v1', '20261015-r5')]
    sessions += [(f'fd-s43{s}-20261015-r6-v1', '20261015-r6') for s in 'kl']
    sessions += [('fd-s43m-20261015-r7-v1', '20261015-r7')]
    for rid, tag in sessions:
        if fnmatch.fnmatch(rid, args.only): collect(rid, tag)
