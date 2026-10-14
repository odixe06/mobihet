"""Import an exact committed full run before data preparation; completion markers publish last."""
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_run(mount, output, arm, expected, verify):
    name = f'{arm}_s42'
    candidates = []
    for config in Path(mount).rglob('config.json'):
        if config.parent.name != name:
            continue
        identity = json.loads(config.read_text())
        if identity.get('fingerprint') == expected['fingerprint']:
            candidates.append(config.parent)
    assert len(candidates) == 1, ('require-resume: missing or ambiguous source', name, candidates)
    source = candidates[0]
    identity = json.loads((source / 'config.json').read_text())
    assert identity['source'] == expected['source'], 'resume source hash mismatch'
    assert identity['fixture'] == expected['fixture'], 'resume data fingerprint mismatch'
    markers = sorted((source / 'complete').glob('round_*.json'))
    rounds = [json.loads(p.read_text())['round'] for p in markers]
    assert rounds == list(range(1, expected['last_round'] + 1)), ('require-resume: wrong last round', rounds)
    # Verifier checks every stored weight/BN/state/payload/prediction; never accepts only a filename.
    verified = verify(source, complete=False, infer=False)
    assert verified['rounds'] == expected['last_round']
    destination = Path(output) / name
    destination.mkdir(parents=True, exist_ok=True)
    files = [p for p in source.rglob('*') if p.is_file() and
             'wandb' not in p.relative_to(source).parts and p.name != 'log.txt']
    for marker_pass in (False, True):
        for file in files:
            relative = file.relative_to(source)
            if (relative.parts[0] == 'complete') != marker_pass:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert digest(target) == digest(file), ('resume import collision', relative)
                continue
            temporary = target.with_name(target.name + '.import-part')
            shutil.copyfile(file, temporary)  # Never inherit read-only mount permissions.
            assert digest(temporary) == digest(file)
            os.replace(temporary, target)
    assert verify(destination, complete=False, infer=False)['rounds'] == expected['last_round']
    return dict(arm=arm, fingerprint=expected['fingerprint'], last_round=expected['last_round'],
                source_path=str(source), files=len(files), markers_published_last=True)
