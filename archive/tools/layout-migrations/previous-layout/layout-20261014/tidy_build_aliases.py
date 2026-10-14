"""Keep compatibility aliases in papers/, leaving canonical build roots compact."""
import json
import os
from pathlib import Path
import shutil

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])


def main():
    migration=ROOT/'docs/operations/layout-migration-2026-10-14.json'
    record=json.loads(migration.read_text());assert not record.get('compact_aliases')
    for old,build in record['builds'].items():
        canonical=ROOT/build['path'];legacy=ROOT/'papers'/old
        assert legacy.is_symlink() and legacy.resolve()==canonical
        legacy.unlink();legacy.mkdir()
        for child in list(canonical.iterdir()):
            if child.is_symlink() and child.name!='logs':
                target=child.resolve(strict=False)
                child.unlink()
            else:target=child
            alias=legacy/child.name
            alias.symlink_to(os.path.relpath(target,alias.parent),target_is_directory=target.is_dir())
        for text in list(canonical.glob('*.txt')):
            target=canonical/'probes'/text.name;target.parent.mkdir(exist_ok=True)
            shutil.move(str(text),str(target))
            alias=legacy/text.name;alias.unlink();alias.symlink_to(os.path.relpath(target,alias.parent))
            record['aliases'][f'papers/{old}/{text.name}']=str(target.relative_to(ROOT))
    record['compact_aliases']=True
    migration.write_text(json.dumps(record,indent=2)+'\n')
    print('Canonical build roots compact; compatibility-only aliases remain in ignored papers/')


if __name__=='__main__':main()
