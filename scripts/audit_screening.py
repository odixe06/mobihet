"""Audit private-model screening artifacts; no prediction/weight reconstruction claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics
from scripts.run_fl import PROTOCOL


def audit(root, spec, support, endpoint=10):
    d = root / spec['slug'] / spec['slug']
    cfg = json.loads((d / 'config.json').read_text())
    hp = cfg['hp']
    assert cfg['clients'] == 5343 and cfg['n_params'] == 395408
    for k in ('rounds', 'batch', 'lr_max', 'lr_min', 'weight_decay', 'clip', 'amp'):
        assert hp[k] == PROTOCOL[k], (k, hp[k])
    assert hp['rounds'] == 50 and hp['batch'] == 64 and hp['arm'] == 'none'
    expected_seed = next((json.loads(s.split('=', 1)[1]) for s in spec['hp'] if s.startswith('seed=')), 42)
    assert hp['group_rows'] == 4096 and hp['seed'] == expected_seed
    for setting in spec['hp']:
        k, v = setting.split('=')
        try:
            expected = json.loads(v)
        except json.JSONDecodeError:
            expected = v
        assert hp[k] == expected, (k, hp[k], expected)
    manifest_path = root / spec['slug'] / 'provenance/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['base_sha'] == spec['base_sha']
    assert manifest['overlay']['methods/mobihet.py'] == spec['source_sha256']
    for name, sha in manifest['files'].items():
        assert hashlib.sha256((manifest_path.parent / 'source' / name).read_bytes()).hexdigest() == sha, name
    rows = [json.loads(s) for s in (d / 'rounds.jsonl').read_text().splitlines()]
    assert [r['round'] for r in rows] == list(range(1, endpoint + 1))
    result = []
    for row in rows:
        r = row['round']
        expected_lr = 1e-5 + (1e-3 - 1e-5) * (1 + np.cos(np.pi * (r - 1) / 49)) / 2
        assert np.isclose(row['lr'], expected_lr, rtol=0, atol=1e-12)
        assert np.isfinite([v for v in row.values() if isinstance(v, (float, int))]).all()
        assert 0 <= row['skipped_frac'] <= 0.01
        z = np.load(d / f'cms_r{r:03d}.npz')
        assert np.array_equal(z['ids'], np.arange(5343))
        cm = z['private']
        assert cm.shape == (5343, 16, 16) and np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
        assert np.array_equal(cm.sum(2), support)
        s = metrics.summary(cm)
        for k in metrics.TEN + ['client_f1_macro_mean', 'client_f1_macro_p10',
                                'client_f1_macro_median', 'client_f1_macro_min', 'f1_per_class']:
            assert 'private/' + k in row, (r, k)
        for k, v in s.items():
            if 'private/' + k in row:
                np.testing.assert_allclose(row['private/' + k], v, rtol=0, atol=1e-12)
        result.append({'round': r, **s, **{k: row[k] for k in ('train_loss', 'skipped_frac', 't_train', 't_eval', 'bytes_up', 'bytes_down')}})
    boot = sum(cfg['payload_bytes']['bootstrap'].values())
    return {'arm': spec['arm'], 'slug': spec['slug'], 'hp': hp, 'bootstrap_bytes': boot,
            f'total_bytes_{endpoint}': boot + sum(r['bytes_up'] + r['bytes_down'] for r in rows),
            f'train_seconds_mean_r2_{endpoint}': np.mean([r['t_train'] for r in rows[1:]]),
            'rounds': result, 'scope': 'CM/source/config audit; no predictions or full private bank reconstruction'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--batch', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    support = np.load(a.reference)['private'].sum(2)
    assert support.sum() == 9347679
    out = [audit(a.root, s, support) for s in json.loads(a.batch.read_text())]
    a.output.write_text(json.dumps(out, indent=2))
    for s in out:
        r = s['rounds'][-1]
        print(s['arm'], r['f1_macro'], r['client_f1_macro_p10'], s['total_bytes_10'])
