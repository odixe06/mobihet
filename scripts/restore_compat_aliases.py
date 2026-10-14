"""Recreate only relative navigation aliases after a fresh checkout; no artifact checkout."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    root=args.root.resolve();record=json.loads((ROOT/'docs/operations/compatibility-aliases.json').read_text())
    created=0
    for row in record['aliases']:
        alias=root/row['alias'];target=root/row['target']
        assert root==target or root in target.parents
        # The key-bearing directory is ignored; initialize it privately when a fresh checkout omitted it.
        if target.name=='private_push' and not target.exists():target.mkdir(parents=True,mode=0o700)
        expected=os.path.relpath(target,alias.parent)
        if alias.is_symlink():
            assert os.path.normpath(str(alias.parent/alias.readlink()))==str(target), ('alias differs',row['alias'])
            continue
        assert not alias.exists(), ('refuse overwrite',row['alias'])
        alias.parent.mkdir(parents=True,exist_ok=True)
        alias.symlink_to(expected,target_is_directory=target.is_dir());created+=1
    print('Compatibility aliases:',len(record['aliases']),'checked;',created,'created; no data restored')


if __name__=='__main__':main()
