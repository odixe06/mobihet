"""Check canonical module ownership, old imports and the complete frozen mh fingerprint."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from project_paths import current_path


def main():
    record = json.loads((ROOT / 'docs/operations/registry/package-layout-2026-10-14.json').read_text())
    names = {p.name for p in (ROOT / '.compat/mh').glob('*.py')}
    assert names == set(record['legacy_source_manifest']), 'old source-file set changed'
    sys.path.insert(0, str(ROOT))
    checked = []
    for row in record['modules']:
        canonical=current_path(row['path'])
        source = ROOT / canonical; legacy = ROOT / '.compat' / row['old_path']   # 17/10: hardlink, root mh removed
        assert source.is_file() and not source.is_symlink(), row['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256'], row['path']
        assert os.path.samefile(legacy, source), row['old_path']
        ast.parse(source.read_text(), filename=row['path'])
        name = 'mh' if row['module'] == '__init__' else f"mh.{row['module']}"
        module = importlib.import_module(name)
        assert os.path.samefile(module.__file__, source), name
        checked.append(dict(module=name,canonical=canonical,hash_preserved=True,import_pass=True))
    from mh import pilot
    assert pilot.ROOT == ROOT / '.compat', 'data/fingerprint root changed'   # 17/10: old-layout root is .compat
    plans={p.resolve() for p in (ROOT/'methods').glob('*/builds/*/notebooks/*20261014*/resume-plan.json')}
    source_checks=0
    for plan in plans:
        for identity in json.loads(plan.read_text()).values():
            for filename,expected in identity['source'].items():
                if filename.startswith('mh/'):
                    assert hashlib.sha256((ROOT/'.compat'/filename).read_bytes()).hexdigest()==expected,(plan,filename)
                    source_checks+=1
    (ROOT/'docs/operations/registry/package-resume-source-check-2026-10-14.json').write_text(json.dumps(dict(
        prepared_plan_count=len(plans),source_hash_checks=source_checks,pass_check=True,
        limit='Prepared parent source matches legacy paths; actual remote mounts/checkpoints audited separately.'),indent=2)+'\n')
    # Notebook generators copy bytes from legacy paths; verify that layout without repository symlinks.
    with tempfile.TemporaryDirectory() as directory:
        runtime=Path(directory);(runtime/'mh').mkdir();(runtime/'knowledge').mkdir()
        for filename in names:
            shutil.copyfile(ROOT/'.compat/mh'/filename,runtime/'mh'/filename)
        shutil.copyfile(ROOT/'knowledge/meta.json',runtime/'knowledge/meta.json')
        assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (runtime/'mh').glob('*.py')}==record['legacy_source_manifest']
        code=("import importlib; from pathlib import Path; "
              f"modules={list(row['module'] for row in record['modules'])!r}; "
              "[importlib.import_module('mh' if x=='__init__' else 'mh.'+x) for x in modules]; "
              "from mh import pilot; assert pilot.ROOT==Path.cwd(); "
              "assert not any(p.is_symlink() for p in Path('mh').glob('*.py')); "
              "print('copied runtime imports PASS')")
        copied=subprocess.run([sys.executable,'-c',code],cwd=runtime,check=True,capture_output=True,text=True)
        assert copied.stdout.strip()=='copied runtime imports PASS'
    result = dict(modules=checked,shared=record['shared_modules'],build_owned=record['build_owned_modules'],
        complete_legacy_fingerprint_preserved=True,pilot_root_unchanged=True,copied_runtime_imports_pass=True,
        limits='CPU import/syntax/hash checks; no local training, no GPU or remote bitwise claim.')
    (ROOT/'docs/operations/registry/package-verification-2026-10-14.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Package check:',len(checked),'imports/SHA/copy-runtime PASS;',result['shared'],'shared;',result['build_owned'],
          'build-owned;',len(plans),'parent plans;',source_checks,'SHA matches; ROOT unchanged')


if __name__ == '__main__':
    main()
