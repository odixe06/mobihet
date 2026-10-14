"""Audit the evidence P2 actually retained; this is NOT a weights/prediction verifier."""
import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))
from mh.metrics import METRIC_KEYS, metrics_from_confusion


def main():
    fixture = ROOT / 'papers/pilot/fixture'
    counts = {h: np.bincount(np.load(fixture / f'{h}_y.npy'), minlength=16) for h in ('old', 'group')}
    train = [len(np.load(fixture / f'train_{k}_y.npy')) for k in range(20)]
    expected = {line.split()[0] for line in (ROOT / 'scripts/p2_queue.txt').read_text().splitlines() if line.strip()}
    seen, audit = set(), []
    for d in sorted((ROOT / 'papers/p2/notebook').iterdir()):
        meta = json.loads((d / 'kernel-metadata.json').read_text())
        owner = meta['id'].split('/')[0]
        src = ROOT / 'papers/p2/source_audit' / owner
        for name in ('pilot.py', 'p2.py', 'channel.py', 'mobility.py', 'dagsnet.py', 'metrics.py', 'archs.py'):
            assert (src / name).read_bytes() == (ROOT / 'mh' / name).read_bytes(), (owner, name, 'source mismatch')
        assert (src / 'manifest.json').read_bytes() == (fixture / 'manifest.json').read_bytes()
        pull = ROOT / 'papers/p2/kaggle' / d.name
        log = ''.join(x.get('data', '') for x in json.loads((pull / (meta['id'].split('/')[1] + '.log')).read_text()))
        assert "gpus ['Tesla T4', 'Tesla T4']" in log and 'fixture verified: 66 files' in log
        seed = int(d.name.rsplit('s', 1)[1])
        summary = json.loads((pull / 'runs' / f'summary_s{seed}.json').read_text())
        arms = {a for a in expected if a.startswith('h2_' if '-h2-' in d.name else 'dml_')}
        assert set(summary) == arms and all(x['rc'] == 0 and x['result'] for x in summary.values())
        for arm in sorted(arms):
            f = ROOT / 'papers/p2/runs' / f'{arm}_s{seed}' / 'result.json'
            assert f.read_bytes() == (pull / 'runs' / f'{arm}_s{seed}' / 'result.json').read_bytes()
            r = json.loads(f.read_text()); cfg = r['config']; hist = r['history']
            assert (cfg['arm'], cfg['seed'], cfg['rounds'], cfg['smoke']) == (arm, seed, 20, False)
            assert cfg['private'] == 'dagsnet' and r['train_rows'] == train
            assert [h['round'] for h in hist] == list(range(1, 21))
            assert [h['round'] for h in hist if 'eval' in h] == [5, 10, 15, 20]
            for h in hist:
                assert np.isfinite(h['loss']) and 0 <= h['bytes_used'] <= h['budget'] + 1e-5
                assert 0 <= h['active'] <= 20
                lr = cfg['lr_min'] + .5 * (cfg['lr'] - cfg['lr_min']) * (1 + np.cos(np.pi * (h['round'] - 1) / 19))
                assert np.isclose(h['lr'], lr, rtol=0, atol=1e-12)
                for model, holds in h.get('eval', {}).items():
                    for hold, rules in holds.items():
                        for rule, e in rules.items():
                            nmodel = 20 if model == 'private' else 1
                            cm = np.asarray(e['confusion_sum'])
                            assert cm.shape == (16, 16) and (cm >= 0).all() and (cm == cm.astype(int)).all()
                            assert np.array_equal(cm.sum(1), counts[hold] * nmodel)
                            mm = e['mean_metrics']; assert set(mm) == set(METRIC_KEYS)
                            assert all(np.isfinite(v) and 0 <= v <= 1 for v in mm.values())
                            assert len(e['f1_macro_clients']) == nmodel
                            assert np.isclose(e['f1_macro'], np.mean(e['f1_macro_clients']))
                            assert np.isclose(e['f1_macro'], mm['f1_macro'])
                            assert np.isclose(e['f1_macro'], np.mean(e['f1_per_class']))
                            assert np.isclose(e['benign_f1'], e['f1_per_class'][0])
                            assert np.isclose(mm['accuracy'], metrics_from_confusion(cm)['accuracy'])
                            for key in ('precision_micro', 'recall_micro', 'recall_weighted', 'f1_micro'):
                                assert np.isclose(mm[key], mm['accuracy'])
            seen.add((arm, seed))
            audit.append(dict(arm=arm, seed=seed, rounds=20, sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    assert seen == {(arm, seed) for arm in expected for seed in (42, 43)}
    out = dict(runs=audit, level='source and retained summary consistency only', limitations=[
        'No saved per-client confusion, predictions or weights; cannot independently recompute mean macro metrics or rerun inference.',
        'Source compared with current downloaded private input datasets; worker did not preserve an executed source hash.',
        'P2 gives evaluator a global label prior outside its contact ledger.',
        'H2 payload sizes assume fp32 upload/fp16 download, but values were not quantized in the executed simulation.',
        'DML byte estimate omits integer buffers and quantization scales; one-arrival aggregation is not a secure-aggregation cohort.'
    ])
    (ROOT / 'papers/p2/audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'PASS retained-evidence audit: {len(seen)} runs x 20 rounds; limitations in papers/p2/audit.json')


if __name__ == '__main__':
    main()
