"""Import the unique audited BoundaryCells parent, publish complete markers last."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_run(mount, output, name, expected, manifest, verify):
    candidates = [p.parent for p in Path(mount).rglob('config.json')
                  if p.parent.name == name and
                  json.loads(p.read_text()).get('fingerprint') == expected['fingerprint']]
    assert len(candidates) == 1, ('require-resume missing/ambiguous', name, candidates)
    source = candidates[0]
    identity = json.loads((source / 'config.json').read_text())
    assert identity['source'] == expected['source'] and identity['fixture'] == expected['fixture']
    audit = verify(source, manifest, expected['fixture'], expected['last_round'], calibration=True)
    assert audit['cost']['pass_cost'], 'cost FAIL cannot open utility'
    destination = Path(output) / name
    files = [p for p in source.rglob('*') if p.is_file() and
             'wandb' not in p.relative_to(source).parts and p.name != 'log.txt']
    for marker_pass in (False, True):
        for path in files:
            relative = path.relative_to(source)
            if (relative.parts[0] == 'complete') != marker_pass:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == sha(path), ('resume collision', relative)
                continue
            temporary = target.with_name(target.name + '.import-part')
            shutil.copyfile(path, temporary)
            assert sha(temporary) == sha(path)
            os.replace(temporary, target)
    assert verify(destination, manifest, expected['fixture'], expected['last_round'], calibration=True)['rounds'] == expected['last_round']
    return dict(run=name, last_round=expected['last_round'], fingerprint=expected['fingerprint'],
                copied_files=len(files), markers_last=True)
