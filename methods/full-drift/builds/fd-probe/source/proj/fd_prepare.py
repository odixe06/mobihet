"""Kaggle-only full data with the internal holdout: temporal2% + receiver-group5% excluded from training,
group/old 65,536 rows each; official raw test20% scaled once (descriptive only)."""
import json
import sys
from pathlib import Path


def prepare_fd(work, final=False):
    """final (owner 19/10, protocol §3): every train80% row trains, no internal holdout, test20% scored every round."""
    work = Path(work); sys.path.insert(0, str(work))
    from mh.prepare import prepare, prepare_test
    meta = json.loads((work / 'knowledge/meta.json').read_text())
    scaler = json.loads((work / 'knowledge/scaler.json').read_text())
    out = work / ('papers/final/fixture' if final else 'papers/pilot/fixture')
    train = [p.parent.parent for p in Path('/kaggle/input').rglob('client_id=000')
             if p.parent.name == 'train' and list(p.glob('*.parquet'))]
    test = [p for p in Path('/kaggle/input').rglob('test') if list(p.glob('part-*.parquet'))]
    assert len(train) == len(test) == 1, ('ambiguous full mounts', train, test)
    features = meta['feature_cols']; classes = meta['class_names']
    manifest = prepare(train[0], out, features, classes, train_cap=None, eval_cap=65536,
                       clients=tuple(range(20)), holdout=not final)
    assert manifest['source_rows'] == 43045415 and (final or manifest['group_overlap'] == 0)
    official = prepare_test(test[0], out, features, scaler, rows=None, sub=0)
    assert official['test_rows'] == 10761343
    manifest.update(official); manifest['files'].update(official['test_files'])
    manifest['dataset_sources'] = ['trietbackup/veremi-fl-20client', 'trietbackup/veremi-nextgen2026-test']
    manifest['evaluation'] = ('final: official test20% after every round, headline round 50' if final else
                              'decisions on internal group holdout; official test20% once after the last epoch, descriptive')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
