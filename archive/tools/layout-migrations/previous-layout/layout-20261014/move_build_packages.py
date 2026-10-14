"""One-time physical module relocation; preserve legacy imports and frozen source bytes."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CONTEXT.md').is_file())
OWNERS = {
    'synthetic-distribution-matching/sa-static-books': ['new_methods'],
    'contact-fragmentation/p2-deadline': ['p2'],
    'proxy-compression/v3-carrier-comparison': ['v3_dense', 'v3_math', 'v3_methods', 'v3_run'],
    'gaussian-sampling/v4-support-mala': ['v4_mala', 'v4_support', 'v4_run'],
    'synthetic-knowledge/v5-anchor-gradient-books': ['v5_anchor', 'v5_books', 'v5_run'],
    'geometry-primitive-moments/v6-g59-g61': ['v6_geometry', 'v6_schema'],
    'covariance-nystrom/v7-rank4-rank8': ['v7_run'],
    'gaussian-proxy-hybrid/v8-pilot': ['v8_hybrid', 'v8_run'],
    'anchor-function-residual/v9-teacher-residual': ['v9_anchor', 'v9_run'],
    'dagsnet-joint-energy/v10-gaussian-base-mh': ['v10_energy', 'v10_run'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'mh':
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith('mh.'):
                found.add(node.module.split('.')[1])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split('.')[1] for alias in node.names if alias.name.startswith('mh.'))
    return sorted(found)


def main():
    proof = ROOT / 'docs/operations/package-layout-2026-10-14.json'
    assert not proof.exists(), 'migration already recorded; use check_package_layout.py'
    baseline = json.loads((ROOT / 'docs/operations/layout-migration-2026-10-14.json').read_text())
    files = sorted((ROOT / 'mh').glob('*.py'))
    before = {file.name: digest(file) for file in files}
    assert len(files) == 40 and all(before[file.name] == baseline['protected_sha256'][f'mh/{file.name}'] for file in files)
    owners = {name: owner for owner, names in OWNERS.items() for name in names}
    aliases = json.loads((ROOT / 'docs/operations/compatibility-aliases.json').read_text())
    rows = []
    for file in files:
        owner = owners.get(file.stem)
        target = file
        if owner:
            method, build = owner.split('/')
            target = ROOT / 'methods' / method / 'builds' / build / 'source/mh' / file.name
        rows.append(dict(module=file.stem, old_path=f'mh/{file.name}', path=str(target.relative_to(ROOT)),
                         shared=owner is None, owner=owner, sha256=before[file.name], dependencies=dependencies(file)))
        if not owner:
            continue
        assert not file.is_symlink() and not target.exists()
        assert '__file__' not in file.read_text(), ('resolved source path requires separate adapter',file.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        file.rename(target)
        file.symlink_to(os.path.relpath(target, file.parent))
        assert digest(file) == before[file.name]
        aliases['aliases'].append(dict(alias=f'mh/{file.name}', target=str(target.relative_to(ROOT))))
    assert {file.name: digest(file) for file in sorted((ROOT / 'mh').glob('*.py'))} == before
    aliases['aliases'] = sorted(aliases['aliases'], key=lambda row: row['alias'])
    (ROOT / 'docs/operations/compatibility-aliases.json').write_text(json.dumps(aliases, indent=2)+'\n')
    proof.write_text(json.dumps(dict(modules=rows, shared_modules=sum(row['shared'] for row in rows),
        build_owned_modules=sum(not row['shared'] for row in rows),
        legacy_source_manifest=before, old_imports_preserved_by_relative_symlinks=True,
        source_bytes_changed=False, git_dvc_add_push=False,
        scope='Relocate method implementations. Reused and mixed frozen helper/driver modules remain shared; no algorithm refactor.'),indent=2)+'\n')
    print('Packages:',sum(row['shared'] for row in rows),'shared;',len(owners),'physically moved; all 40 legacy SHA preserved')


if __name__ == '__main__':
    main()
