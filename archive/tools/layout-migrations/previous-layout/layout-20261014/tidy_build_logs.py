"""Owner correction: physical logs live in each build; root logs are navigation aliases."""
import json
import os
from pathlib import Path
import shutil

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])


def main():
    file=ROOT/'docs/operations/layout-migration-2026-10-14.json';record=json.loads(file.read_text())
    assert not record.get('logs_inside_build')
    for old,build in record['builds'].items():
        physical=ROOT/build['path'];target=physical/'logs'
        source=ROOT/'logs'/build['family']/build['build']
        assert target.is_symlink() and target.resolve()==source
        target.unlink();shutil.move(str(source),str(target))
        source.symlink_to(os.path.relpath(target,source.parent),target_is_directory=True)
        for alias,destination in list(record['aliases'].items()):
            prefix=str(source.relative_to(ROOT))
            if destination.startswith(prefix+'/'):record['aliases'][alias]=str(target.relative_to(ROOT))+destination[len(prefix):]
    record['logs_inside_build']=True
    file.write_text(json.dumps(record,indent=2)+'\n')
    print('Build logs moved physically inside16builds; root log aliases preserved')


if __name__=='__main__':main()
