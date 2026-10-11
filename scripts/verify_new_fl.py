"""Verify saved new-FL weights, predictions, client confusions, 10 metrics and round chain."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mh.dagsnet import build_model, CFG
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion
from mh.new_fl import digest


def verify(out, complete=True):
    out = Path(out); identity = json.loads((out / 'config.json').read_text())
    fingerprint = identity.pop('fingerprint')
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == fingerprint
    cfg = identity['config']; result = json.loads((out / 'result.json').read_text())
    assert result['fingerprint'] == fingerprint
    fixture = ROOT / 'papers/pilot/fixture'
    assert digest(fixture / 'manifest.json') == identity['fixture']
    expected_train = [512 if cfg['smoke'] else len(np.load(fixture / f'train_{k}_y.npy')) for k in range(20)]
    assert result['train_rows'] == expected_train
    labels = {h: np.load(fixture / f'{h}_y.npy')[:512 if cfg['smoke'] else None] for h in ('old', 'group')}
    model = build_model(dict(CFG, n_features=66, num_classes=16), latent=True)
    assert sum(p.numel() for p in model.parameters()) == 395024
    knowledge_sha = digest(out / 'knowledge.pt')
    torch.load(out / 'knowledge.pt', weights_only=True, map_location='cpu')
    rows = [json.loads(f.read_text()) for f in sorted((out / 'complete').glob('round_*.json'))]
    assert rows == result['history']
    assert [r['round'] for r in rows] == list(range(1, len(rows) + 1))
    if complete:
        assert len(rows) == cfg['rounds'] and result['complete']
    expected_eval = [r for r in range(1, len(rows) + 1) if r % cfg['eval_every'] == 0 or r == cfg['rounds']]
    assert [r['round'] for r in rows if 'eval' in r] == expected_eval
    previous = None; derived = {}
    for row in rows:
        r = row['round']; w = out / 'weights' / f'round_{r:03d}.pt'
        assert row['fingerprint'] == fingerprint and row['prev_sha'] == previous
        assert row['weights_sha'] == digest(w) and row['knowledge_sha'] == knowledge_sha
        previous = row['weights_sha']
        state = torch.load(w, weights_only=True, map_location='cpu')
        assert state['round'] == r and state['fingerprint'] == fingerprint and len(state['models']) == 20
        assert state['architecture'] == dict(CFG, n_features=66, num_classes=16)
        for sd in state['models']:
            model.load_state_dict(sd, strict=True)
            assert all(torch.isfinite(t).all() for t in sd.values() if t.is_floating_point())
        assert [c['client'] for c in row['clients']] == list(range(20))
        assert [c['rows'] for c in row['clients']] == expected_train
        assert all(c['applied'] > 0 and c['applied'] + c['skipped'] == int(np.ceil(c['rows'] / cfg['batch'])) for c in row['clients'])
        lr = cfg['lr_min'] + .5 * (cfg['lr'] - cfg['lr_min']) * (1 + np.cos(np.pi * (r - 1) / max(1, cfg['rounds'] - 1)))
        assert np.isclose(row['lr'], lr, rtol=0, atol=1e-12)
        if 'eval' not in row:
            continue
        pf = out / 'predictions' / f'round_{r:03d}.npz'; assert digest(pf) == row['predictions_sha']
        with np.load(pf) as pred:
            for hold, rules in row['eval']['private'].items():
                y = pred[hold + '_y']; assert np.array_equal(y, labels[hold])
                for rule, ev in rules.items():
                    pr = pred[f'{hold}_{rule}']; assert pr.shape == (20, len(y)) and (pr < 16).all()
                    cc = np.stack([np.bincount(y.astype('int64') * 16 + p, minlength=256).reshape(16, 16) for p in pr])
                    assert np.array_equal(cc, ev['confusion_clients'])
                    mm = [metrics_from_confusion(c) for c in cc]
                    means = {k: float(np.mean([v[k] for v in mm])) for k in METRIC_KEYS}
                    assert set(ev['mean_metrics']) == set(METRIC_KEYS)
                    for k, value in means.items():
                        assert np.isfinite(value) and np.isclose(value, ev['mean_metrics'][k], rtol=0, atol=1e-12)
                    pc = np.mean([[v['f1'] for v in per_class_from_confusion(c, [str(j) for j in range(16)])] for c in cc], 0)
                    assert np.allclose(pc, ev['f1_per_class'], rtol=0, atol=1e-12)
                    assert np.isclose(pc[0], ev['benign_f1']) and np.isclose(means['f1_macro'], ev['f1_macro'])
                    derived[(r, hold, rule)] = means
    with (out / 'metrics.csv').open() as f:
        csvrows = list(csv.DictReader(f))
    assert len(csvrows) == len(derived)
    for row in csvrows:
        mm = derived.pop((int(row['round']), row['holdout'], row['rule']))
        for key, value in mm.items():
            assert np.isclose(float(row[key]), value, rtol=0, atol=1e-12)
    assert not derived
    print(f'VERIFIED {out.name}: {len(rows)}/{cfg["rounds"]} rounds; weights + predictions + 10 metrics', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('paths', nargs='+'); p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    for path in a.paths:
        verify(path, complete=not a.partial)
