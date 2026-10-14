"""Restore lightweight evidence from an immutable DVC directory; never checkout or add."""
import argparse
import concurrent.futures
import configparser
import hashlib
import json
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def md5(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='+', help='Repository-relative .dvc files')
    parser.add_argument('--proof', required=True)
    parser.add_argument('--include', help='Explicit relative-path regex; permits required checkpoint bundles')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(ROOT / '.dvc/config')
    remote = Path(config['\'remote "gdrive"\'']['url']) / 'files/md5'
    jobs = []
    targets = []
    for name in args.targets:
        metadata = ROOT / name
        for out in yaml.safe_load(metadata.read_text())['outs']:
            directory_hash = out['md5']
            assert directory_hash.endswith('.dir')
            directory = remote / directory_hash[:2] / directory_hash[2:]
            assert md5(directory) == directory_hash.removesuffix('.dir')
            targets.append(dict(metadata=name, md5=directory_hash))
            for entry in json.loads(directory.read_text()):
                relative = Path(entry['relpath'])
                assert not relative.is_absolute() and '..' not in relative.parts
                if args.include:
                    if not re.search(args.include,relative.as_posix()):continue
                else:
                    if set(relative.parts) & {'weights', 'resume', 'predictions', 'wandb', 'private_push'}:continue
                    if relative.suffix in {'.npy', '.npz', '.pt', '.bin'}:continue
                if 'private_push' in relative.parts or relative.name.endswith('.part'):continue
                jobs.append((metadata.parent / out['path'] / relative, entry['md5']))

    def restore(item):
        target, hash_ = item
        source = remote / hash_[:2] / hash_[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert md5(target) == hash_, f'Existing file differs: {target}'
            return dict(path=str(target.relative_to(ROOT)), md5=hash_, restored=False)
        temporary = target.with_name(target.name + '.restore-part')
        with source.open('rb') as inp, temporary.open('wb') as out:
            while chunk := inp.read(1 << 20):
                out.write(chunk)
        assert md5(temporary) == hash_, f'Remote object differs: {target}'
        os.replace(temporary, target)
        return dict(path=str(target.relative_to(ROOT)), md5=hash_, restored=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(restore, jobs))
    (ROOT / args.proof).write_text(json.dumps(dict(targets=targets, files=files,
        heavy_files_restored=False), indent=2) + '\n')
    print('DVC metadata MD5 PASS:', len(files), 'files;', sum(x['restored'] for x in files), 'restored')


if __name__ == '__main__':
    main()
