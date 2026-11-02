"""Close the artifact audit for all32 P20 runs; no performance comparisons or training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import atomic_json


def main():
    out = ROOT / 'runs/p-study/final-audit'
    out.mkdir(parents=True, exist_ok=True)
    batch_path = ROOT / 'runs/p-study/batch.json'
    raw = batch_path.read_bytes()
    batch = json.loads(raw)
    launch = json.loads((ROOT / 'docs/research/p20-launch.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == launch['batch_sha256'] and len(batch) == 32
    prior = json.loads((ROOT / 'runs/p-study/recovery-v1/terminal-audit.json').read_text())['runs']
    inventory_path = ROOT / 'runs/p-study/recovery-v1/all-arm-inventory-audit.json'
    inventory = {r['arm']: r for r in json.loads(inventory_path.read_text())['runs']}
    assert len(prior) == len(inventory) == 32 and all(r['pulled'] for r in prior.values())
    complete = [s for s in batch if prior[s['arm']]['status'] == 'COMPLETE']
    failed = [s for s in batch if prior[s['arm']]['status'] != 'COMPLETE']
    assert len(complete) == 31 and [s['arm'] for s in failed] == ['P21']
    assert all(inventory[s['arm']]['full20_present'] and not inventory[s['arm']]['issues'] for s in complete)
    path = out / 'complete-batch.json'
    atomic_json(path, complete)
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_p20.py'), '--batch', str(path),
                    '--root', str(ROOT / 'artifacts/p20'), '--out', str(out)], check=True, cwd=ROOT)
    common = json.loads((out / 'audit.json').read_text())
    details = json.loads((out / 'diagnostic-audit.json').read_text())
    assert len(common) == len(details) == 31
    results = []
    for spec, audited in zip(complete, details):
        assert audited['arm'] == spec['arm'] and len(audited['snapshots']) == 20
        results.append({'arm': spec['arm'], 'ref': f'{spec["account"]}/{spec["slug"]}/1',
                        'status': 'PASS_BOUNDED', 'audited_rounds': 20})
    spec = failed[0]
    inv = inventory['P21']
    assert inv['source_verified'] and not inv['issues'] and inv['rounds'] == inv['snapshot_rounds'] == inv['test_cm_rounds'] == []
    folder = ROOT / 'artifacts/p20' / spec['slug']
    text = (folder / (spec['slug'] + '.log')).read_text()
    assert 'np.isfinite(a).all()' in text and 'roundtrip(prob, 1, r)' in text
    results.append({'arm': 'P21', 'ref': f'{spec["account"]}/{spec["slug"]}/1',
                    'status': 'TRAINING_FAILED_BEFORE_R1', 'audited_rounds': 0,
                    'evidence': 'Source/config/log audited; nonfinite probability rejected before aggregation; no resume checkpoint'})
    results.sort(key=lambda r: int(r['arm'][1:]))
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ['audit.json', 'diagnostic-audit.json']}
    atomic_json(out / 'closure.json', {'batch_sha256': launch['batch_sha256'], 'runs': results,
        'full20_pass': 31, 'failed_before_r1': 1, 'unresolved_audit_failures': 0, 'report_sha256': hashes,
        'auditor_sha256': hashlib.sha256((ROOT / 'scripts/audit_p20.py').read_bytes()).hexdigest(),
        'scope': 'Permitted scientific artifacts for all32 kernels; CM-derived metrics, source/config, teacher/codec/cohorts. No rawprediction replay, full private bank, exactresume, DP or contact proof; no performance comparisons.'})
    print('P20_FINAL_AUDIT_ALL_32_CLASSIFIED', flush=True)


if __name__ == '__main__': main()
