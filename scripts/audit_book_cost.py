"""Cross-check V5 book cost markers against the actual baseline and run time."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(marker):
    record = json.loads(marker.read_text())
    if record['family'] not in ('g', 'j'):
        return
    base = marker.parent
    reference_path = base / 'source/cost_reference.json'
    observed_path = base / 'source/cost_observed.json'
    reference = json.loads(reference_path.read_text())
    observed = json.loads(observed_path.read_text())
    baseline = json.loads((base / 'runs' /
        f'{reference["baseline_arm"]}_s{record["seed"]}' / 'result.json').read_text())
    seconds = baseline['minutes'] * 60
    assert baseline['complete'] and len(baseline['history']) == 20 and seconds > 0
    assert math.isclose(reference['baseline_seconds'], seconds, rel_tol=1e-12)
    assert math.isclose(reference['max_seconds'], 2 * seconds, rel_tol=1e-12)
    rows = []
    for arm, summary in record['summary'].items():
        cost = observed[arm]
        path = base / 'runs' / f'{arm}_s{record["seed"]}' / 'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            assert math.isclose(cost['active_seconds'], result['minutes'] * 60, rel_tol=1e-12)
        limit = cost['max_seconds']
        if limit is not None:
            assert math.isclose(limit, 2 * seconds, rel_tol=1e-12)
        passed = limit is None or (cost['active_seconds'] <= limit and not cost['cost_stopped'])
        assert passed == cost['cost_pass'] == summary['cost_pass']
        assert summary['cost_stopped'] == cost['cost_stopped']
        item = next(v for v in record['audits'] if v['arm'] == arm)
        assert item['cost_pass'] == passed
        rows.append(dict(arm=arm, active_seconds=cost['active_seconds'],
                         baseline_ratio=cost['active_seconds'] / seconds, cost_pass=passed))
    proof = dict(kernel=record['kernel'], baseline_seconds=seconds, max_seconds=2*seconds,
                 rows=rows, source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (reference_path, observed_path)})
    (base / 'COST_AUDITED.json').write_text(json.dumps(proof, indent=2))
    print('COST AUDITED', record['kernel'], len(rows), 'arms')


if __name__ == '__main__':
    for marker in sorted(ROOT.glob('papers/v5/kaggle/*/AUDITED.json')):
        audit(marker)
