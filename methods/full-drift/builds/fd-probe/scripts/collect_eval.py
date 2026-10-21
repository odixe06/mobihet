"""Pull and audit one detached evaluation notebook against its audited training checkpoint."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np

from collect_fd import BASE, HELPER, sha, metrics_from_confusion, METRIC_KEYS


def main():
    p = argparse.ArgumentParser(); p.add_argument('rid'); p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--partial', action='store_true', help='ERROR notebook: audit only the rounds it committed'); a = p.parse_args()
    folder = BASE / 'notebooks' / a.rid; dest = BASE / 'artifacts/kaggle' / a.rid
    assert (folder / 'PUSHED').exists(), 'missing PUSHED marker'
    assert not (dest / 'AUDITED.json').exists(), 'already audited'
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    contract = json.loads((folder / 'eval-contract.json').read_text())
    prefix = [sys.executable, str(HELPER / 'kaggle_as.py'), meta['id'].split('/')[0], '--']
    status = subprocess.run(prefix + ['kaggle', 'kernels', 'status', meta['id']], check=True, capture_output=True, text=True)
    assert any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE', 'ERROR')), status.stdout
    subprocess.run(prefix + [sys.executable, str(HELPER / 'pull_kernel_output.py'), meta['id'], '-p', str(dest), '-j', str(a.jobs),
                            '--pattern', r'^(source/|runs/[^/]+/(eval/|predictions/|metrics/|wandb_run(_r\d+)?\.json|progress_rank|round_.*\.log|worker\d\.log)|runs/eval\.log)'], check=True)
    private = BASE / 'private_push' / a.rid
    executed = private / 'executed.ipynb'
    subprocess.run(prefix + [sys.executable, str(HELPER / 'download_executed_notebook.py'), meta['id'], '--version', '1',
                            '--output', str(executed)], check=True, umask=0o077)
    executed.chmod(0o600)
    pushed = nbformat.read(private / meta['code_file'], as_version=4)
    key = next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == '_INLINE_WANDB_KEY' for t in n.targets))
    actual = nbformat.reads(executed.read_text().replace(key, '[REDACTED]'), as_version=4)
    assert len(actual.cells) == len(pushed.cells)
    for i, cell in enumerate(actual.cells):
        if cell.cell_type == 'code': assert cell.source == pushed.cells[i].source.replace(key, '[REDACTED]')
    reference = nbformat.read(folder / meta['code_file'], as_version=4)
    actual.cells[1].source = reference.cells[1].source; actual.metadata['credential_cell_redacted'] = True
    nbformat.write(actual, folder / 'executed.redacted.ipynb')
    stage = BASE / 'artifacts/input' / f'fd-src-{contract["source_tag"]}'
    inputs = json.loads((stage / 'input_manifest.json').read_text())
    assert json.loads((dest / 'source/input_manifest.json').read_text()) == inputs
    assert all(sha(dest / 'source' / name) == value for name, value in inputs.items())
    train = BASE / 'artifacts/kaggle' / contract['session']
    assert (train / 'AUDITED.json').exists()
    assert json.loads((dest / 'source/data_manifest.json').read_text()) == json.loads((train / 'source/data_manifest.json').read_text())
    run = dest / 'runs' / contract['run']
    rounds = contract.get('rounds', [contract.get('round')])   # one or several rounds per evaluation notebook
    if a.partial:
        assert 'KernelWorkerStatus.ERROR' in status.stdout
        rounds = [r for r in rounds if (run / 'eval' / f'round_{r:03d}.json').exists()]; assert rounds, 'no committed round'
    shas = contract['weights_sha'] if isinstance(contract['weights_sha'], dict) else {str(rounds[0]): contract['weights_sha']}
    checks = 0
    for r in rounds:
        row = json.loads((run / 'eval' / f'round_{r:03d}.json').read_text())
        assert row['fingerprint'] == contract['fingerprint'] and row['weights_sha'] == shas[str(r)]
        evaluator = contract.get('evaluator', 'fd_eval.py'); assert row['eval_source_sha'] == inputs[evaluator]
        if evaluator == 'fd_eval_fast.py':   # tolerance class: every model passed its gate against the reference module
            K = len(row['eval']['private']['test']['head']['confusion_clients'])
            gate = row['fast_gate']   # r29: relative bound; r27: absolute 0.25
            assert gate['models'] == 2 * K + 1 and (gate['max_rel'] <= 0.02 if 'max_rel' in gate else gate['max_error'] <= 0.25), gate
        pf = run / 'predictions' / f'round_{r:03d}.npz'; assert sha(pf) == row['predictions_sha']
        before = checks
        with np.load(pf) as data:
            y = data['test_y'].astype(np.int64); assert len(y) == 10761343
            for role, holds in row['eval'].items():
                for rule, ev in holds['test'].items():
                    assert set(ev['mean_metrics']) == set(METRIC_KEYS), 'all ten metrics required'
                    predictions = data[f'{role}_test_{rule}']
                    cms = [np.bincount(y * 16 + pred, minlength=256).reshape(16, 16) for pred in predictions]
                    if len(cms) == 1: cms *= len(ev['confusion_clients'])
                    assert len(cms) == len(ev['confusion_clients'])
                    assert all(np.array_equal(cm, saved) for cm, saved in zip(cms, ev['confusion_clients'], strict=True))
                    metrics = [metrics_from_confusion(cm) for cm in cms]
                    assert all(abs(float(np.mean([v[key] for v in metrics])) - value) < 1e-12 for key, value in ev['mean_metrics'].items())
                    checks += 1
        assert checks - before == 6, ('private head/gp/gpe, EMA head/gp, proxy head', r, checks - before)
    # complete = every round in `rounds` audited; a --partial record keeps the notebook's plan in planned_rounds
    record = dict(kernel=meta['id'], version=1, **{**contract, 'rounds': rounds}, test_rows=len(y), metric_checks=checks, complete=True,
                  **({'planned_rounds': contract['rounds']} if a.partial else {}))
    (dest / 'AUDITED.json').write_text(json.dumps(record, indent=2) + '\n')
    print('AUDITED detached evaluation', a.rid, 'rounds', rounds, flush=True)


if __name__ == '__main__': main()
