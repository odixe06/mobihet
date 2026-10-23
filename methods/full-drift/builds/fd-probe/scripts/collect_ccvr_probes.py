"""Pull exact terminal probe version, redact executed source, audit predictions and predeclared mechanism gates."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import nbformat
import numpy as np

from check_ccvr_probes import audit

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
HELPER = ROOT / '.agents/skills/kaggle-training-notebook/scripts'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def view(result, arm):
    rows = [client[arm]['group'] for client in result['clients'].values()]
    return dict(mean=statistics.mean(r['metrics']['f1_macro'] for r in rows), benign=statistics.mean(r['per_class'][0] for r in rows),
                worst=min(r['metrics']['f1_macro'] for r in rows))


def main():
    rid = 'fd-ccvr-probes-20261022-v1'; public = BASE / 'notebooks' / rid; private = BASE / 'private_push' / rid
    meta = json.loads((public / 'kernel-metadata.json').read_text()); kernel = meta['id']; owner = kernel.split('/')[0]
    dest = BASE / 'artifacts/kaggle' / rid; dest.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, str(HELPER / 'kaggle_as.py'), owner, '--']
    subprocess.run(prefix + [sys.executable, str(HELPER / 'pull_kernel_output.py'), kernel, '-p', str(dest), '-j', '16', '--pattern', r'^(source/|probes/)'], check=True)
    executed = private / 'executed.ipynb'
    subprocess.run(prefix + [sys.executable, str(HELPER / 'download_executed_notebook.py'), kernel, '--version', '1', '--output', str(executed)], check=True, umask=0o077)
    executed.chmod(0o600)
    pushed = nbformat.read(private / meta['code_file'], as_version=4)
    key = next(node.value.value for node in ast.parse(pushed.cells[1].source).body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == '_INLINE_WANDB_KEY' for target in node.targets))
    actual = nbformat.reads(executed.read_text().replace(key, '[REDACTED]'), as_version=4)
    assert len(actual.cells) == len(pushed.cells)
    for i, cell in enumerate(actual.cells):
        if cell.cell_type == 'code': assert cell.source == pushed.cells[i].source.replace(key, '[REDACTED]'), ('executed source', i)
    reference = nbformat.read(public / meta['code_file'], as_version=4)
    actual.cells[1].source = reference.cells[1].source; actual.metadata['credential_cell_redacted'] = True
    nbformat.write(actual, public / 'executed.redacted.ipynb')
    assert key not in (dest / kernel.split('/')[1]).with_suffix('.log').read_text()
    inputs = json.loads((BASE / 'artifacts/input' / rid / 'input_manifest.json').read_text())
    assert json.loads((dest / 'source/input_manifest.json').read_text()) == inputs
    for path in (dest / 'source').glob('*.py'): assert sha(path) == inputs[path.name], path.name
    completion = json.loads((dest / 'probes/COMPLETE.json').read_text()); assert completion['inputs'] == inputs
    for name, expected in completion['files'].items(): assert sha(dest / 'probes' / name) == expected
    gates = {}; summaries = {}
    for seed in (42, 43):
        teacher = json.loads((dest / f'probes/probe23-s{seed}.json').read_text())
        transfer_path = dest / f'probes/probe24-s{seed}.json'; transfer = json.loads(transfer_path.read_text())
        assert teacher['weights_sha'] == transfer['weights_sha'] == inputs[f'weights_s{seed}.pt']
        assert teacher['fixture_sha'] == transfer['fixture_sha'] == inputs['manifest.json']
        assert teacher['source_sha'] == inputs['fd_probe23.py'] and transfer['source_sha'] == inputs['fd_probe24.py']
        assert set(transfer['clients']) == set(map(str, range(20))) and transfer['args']['steps'] == 1000
        audit(transfer_path)
        for variant in [teacher['original'], *teacher['modes'].values()]:
            for split in ('old', 'group'):
                from mh.metrics import metrics_from_confusion
                for k, v in metrics_from_confusion(np.array(variant[split]['confusion'])).items():
                    assert abs(v - variant[split]['metrics'][k]) < 1e-12
        t0, t1 = teacher['original']['group'], teacher['modes']['project48']['group']
        arms = {name: view(transfer, name) for name in transfer['clients']['0']}
        base, revised, inversion = arms['original_gaussian'], arms['calibrated_gaussian'], arms['calibrated_inversion']
        preserved = lambda a, b: a['benign'] >= b['benign'] - .005 and a['worst'] >= b['worst'] - .010
        gates[seed] = dict(teacher=t1['metrics']['f1_macro'] >= t0['metrics']['f1_macro'] + .020 and t1['per_class'][0] >= t0['per_class'][0] - .005,
                           transfer=revised['mean'] >= base['mean'] + .005 and preserved(revised, base),
                           inversion=inversion['mean'] >= revised['mean'] + .005 and preserved(inversion, revised))
        summaries[seed] = dict(teacher_original=t0['metrics']['f1_macro'], teacher_project48=t1['metrics']['f1_macro'], arms=arms,
                              teacher_seconds=teacher['modes']['project48']['seconds'], transfer_seconds=transfer['seconds'])
    record = dict(kernel=kernel, version=1, completion=True, source_and_executed_verified=True, gates=gates, summaries=summaries,
                  scope='two-seed frozen-head mechanism screen; not online FL/ep15/r50 or contact/deadline utility',
                  seconds=completion['seconds'])
    (dest / 'AUDITED.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
