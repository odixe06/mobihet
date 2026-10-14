"""Kaggle-only full data cache; standardized train untouched, raw test scaled once."""
import json
import sys
from pathlib import Path


def prepare_full(work):
    work=Path(work);sys.path.insert(0,str(work))
    from mh.prepare import prepare,prepare_test
    meta=json.loads((work/'knowledge/meta.json').read_text())
    scaler=json.loads((work/'knowledge/scaler.json').read_text())
    out=work/'papers/pilot/fixture'
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest.get('test_rows')==10761343:
            assert manifest['source_rows']==43045415
            return manifest
    train=[p.parent.parent for p in Path('/kaggle/input').rglob('client_id=000')
           if p.parent.name=='train' and list(p.glob('*.parquet'))]
    test=[p for p in Path('/kaggle/input').rglob('test') if list(p.glob('part-*.parquet'))]
    assert len(train)==len(test)==1,('ambiguous full mounts',train,test)
    features=meta['feature_cols'];classes=meta['class_names']
    manifest=prepare(train[0],out,features,classes,train_cap=None,clients=tuple(range(20)),holdout=False)
    assert manifest['source_rows']==43045415
    official=prepare_test(test[0],out,features,scaler,rows=None,sub=0)
    assert official['test_rows']==10761343
    manifest.update(official);manifest['files'].update(official['test_files'])
    manifest['dataset_sources']=['trietbackup/veremi-fl-20client','trietbackup/veremi-nextgen2026-test']
    manifest['evaluation']='full official test20% every round, descriptive; no internal held-out selection'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
