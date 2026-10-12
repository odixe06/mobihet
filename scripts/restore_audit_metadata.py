"""Restore selected old audit evidence directly from immutable DVC objects.

No DVC checkout, manifest update or cache deletion. Heavy model files stay remote.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def md5(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def main():
    import configparser
    config = configparser.ConfigParser()
    config.read(ROOT / '.dvc/config')
    remote = Path(config['\'remote "gdrive"\'']['url']) / 'files/md5'
    jobs = []
    for build in ('v2', 'v3', 'v4', 'v5'):
        manifest = ROOT / f'papers/{build}/kaggle.dvc'
        directory_hash = yaml.safe_load(manifest.read_text())['outs'][0]['md5']
        directory = remote / directory_hash[:2] / directory_hash[2:]
        assert md5(directory) == directory_hash.removesuffix('.dir')
        entries = json.loads(directory.read_text())
        audited = {Path(e['relpath']).parts[0] for e in entries
                   if Path(e['relpath']).name == 'AUDITED.json'}
        for entry in entries:
            path = Path(entry['relpath'])
            if build == 'v2':
                selected = path.parts[0].startswith('v2a-') and (
                    path.name == 'AUDITED.json' or
                    len(path.parts) >= 4 and path.parts[2] in
                    ('local_s42', 'local_s43', 'a_own_s42', 'a_own_s43') and
                    path.name in ('config.json', 'result.json'))
            else:
                selected = path.parts[0] in audited and not (
                    set(path.parts) & {'weights', 'resume', 'predictions', 'wandb'})
            if selected and not path.name.endswith('.part'):
                assert not path.is_absolute() and '..' not in path.parts
                jobs.append((ROOT / f'papers/{build}/kaggle' / path, entry['md5']))

    def restore(item):
        target, hash_ = item
        source = remote / hash_[:2] / hash_[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert md5(target) == hash_, f'existing content differs: {target.relative_to(ROOT)}'
            return dict(path=str(target.relative_to(ROOT)), md5=hash_, restored=False)
        temporary = target.with_name(target.name + '.restore-part')
        with source.open('rb') as inp, temporary.open('wb') as out:
            while chunk := inp.read(1 << 20):
                out.write(chunk)
        assert md5(temporary) == hash_, f'DVC object mismatch: {target.relative_to(ROOT)}'
        os.replace(temporary, target)
        return dict(path=str(target.relative_to(ROOT)), md5=hash_, restored=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        proof = list(pool.map(restore, jobs))
    # Recreate lightweight navigation links, never duplicate checkpoints.
    for marker in ROOT.glob('papers/v[2345]/kaggle/*/AUDITED.json'):
        audit = json.loads(marker.read_text())
        base = marker.parents[2] / 'runs'
        base.mkdir(exist_ok=True)
        for item in audit['audits']:
            name = f'{item["arm"]}_s{audit["seed"]}'
            target = base / name
            run = marker.parent / 'runs' / name
            if target.is_symlink():
                assert target.resolve() == run.resolve()
            elif not target.exists():
                target.symlink_to(run.resolve(), target_is_directory=True)
    record = ROOT / 'docs/operations/restored-metadata-2026-10-12.json'
    record.write_text(json.dumps(dict(files=proof, model_files_restored=False), indent=2))
    print('DVC metadata MD5 PASS:', len(proof), 'files;',
          sum(v['restored'] for v in proof), 'restored')


if __name__ == '__main__':
    main()
