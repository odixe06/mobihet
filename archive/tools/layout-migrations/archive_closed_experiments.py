"""One-time archive/name migration; keep immutable Python/notebook/DVC content identities."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, quote

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
ACTIVE={'proxy-mutual-distillation','gaussian-class-moments','gaussian-proxy-hybrid',
        'boundary-cell-counts','receiver-response-consistency','ema-deployment-recipe'}
WORKING=ACTIVE|{'shared-baselines'}
BUILD_NAMES={'pilot':'pilot-baselines','p2':'fragmented-snapshots','new-fl':'static-synthetic-books',
    'v2':'countsketch-subspace-ratio','v3':'proxy-carriers','v4':'support-mala',
    'v5':'anchor-synthetic-books','v6':'geometry-g59-g61','v7':'nystrom-rank4-rank8',
    'v8':'hybrid-pilot','v9':'anchor-residual','v10':'gaussian-base-energy',
    'full-validation':'federated-baselines-full','hybrid-full':'hybrid-full',
    'boundary-cells':'boundary-cell-pilot','receiver-response':'receiver-augmentation'}
OPEN_BUILDS={'full-validation','hybrid-full','boundary-cells','receiver-response'}
COMMON_NAMES={'__init__':'__init__','archs':'legacy_model_factory','channel':'rsu_contact',
    'dagsnet':'dagsnet','metrics':'classification_metrics','mobility':'vehicle_mobility',
    'model_packet':'model_codec','new_fl':'artifact_driver','pilot':'pilot_runtime','prepare':'prepare_data',
    'teacher_base':'teacher_training','teacher_driver':'teacher_driver','v2_math':'vector_codecs',
    'v2_methods':'shared_prior_and_proxy','v2_run':'federated_driver','v4_schema':'feature_support_schema',
    'v6_run':'expanded_fixture_runtime','v7_basis':'covariance_basis','v7_nystrom':'gaussian_moments'}
METHOD_NAMES={'new_methods':'distribution_matching','p2':'fragmented_snapshot_driver',
    'v3_dense':'dense_proxy_control','v3_math':'proxy_carrier_codecs','v3_methods':'proxy_carrier_methods',
    'v3_run':'proxy_carrier_driver','v4_mala':'mala_transport','v4_support':'support_conditioned_gaussian',
    'v4_run':'support_mala_driver','v5_anchor':'kernel_anchor_targets','v5_books':'gradient_latent_books',
    'v5_run':'synthetic_books_driver','v6_geometry':'geometry_moments','v6_schema':'geometry_feature_schema',
    'v7_run':'covariance_sketch_driver','v8_hybrid':'hybrid_models','v8_run':'hybrid_pilot_driver',
    'v9_anchor':'anchor_residual_teacher','v9_run':'anchor_residual_driver',
    'v10_energy':'gaussian_base_energy','v10_run':'joint_energy_driver'}
RETIRED_TOOLS=['audit_book_cost.py','budget_tiers.py','build_navigation.py','report_recent.py',
    'restore_audit_metadata.py','verify_teacher.py','watch_reports.py','watch_current_reports.py',
    'check_teacher_notebooks.py','kaggle/gen_teacher.py','kaggle/teacher_controller.py']
SEMANTIC={'v2':'sketch_subspace_ratio','v3':'proxy_carrier','v4':'support_mala','v5':'synthetic_book',
          'v6':'geometry','v7':'nystrom','v8':'hybrid_pilot','v9':'anchor_residual','v10':'joint_energy'}


def main():
    proof=ROOT/'docs/operations/workspace-layout.json'
    assert not proof.exists(), 'already migrated; use layout checks'
    old=json.loads((ROOT/'docs/operations/layout-migration-2026-10-14.json').read_text())
    packages=json.loads((ROOT/'docs/operations/package-layout-2026-10-14.json').read_text())
    catalog=json.loads((ROOT/'docs/methods/catalog.json').read_text())
    aliases=json.loads((ROOT/'docs/operations/compatibility-aliases.json').read_text())['aliases']
    raw=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-0'],cwd=ROOT).decode().split('\0')
    markdown={p:(ROOT/p).read_text() for p in raw if p and p.endswith('.md') and not (ROOT/p).is_symlink()}
    links=[]
    for row in aliases:
        alias=ROOT/row['alias']
        links.append((row['alias'],str((ROOT/row['target']).resolve().relative_to(ROOT))))
    pathmap={};steps=[];builds={}
    def plan(a,b):
        pathmap[a]=b
    def move(a,b):
        steps.append((a,b))
    for key,item in old['builds'].items():
        name=BUILD_NAMES[key];archived=key not in OPEN_BUILDS
        base=('archive/methods' if archived else 'methods')+f'/{item["family"]}/builds/{name}'
        builds[key]=dict(item,build=name,path=base,archived=archived,legacy_build=item['build'])
        plan(item['path'],base)
    # Direct file mappings take precedence over directory mappings.
    modules=[]
    for row in packages['modules']:
        if row['shared']:
            target=f'core/{COMMON_NAMES[row["module"]]}.py'
            move(row['path'],target)
        else:
            parent=str(Path(row['path']).parent)
            target_name=METHOD_NAMES[row['module']]+'.py'
            if row['module']=='v8_hybrid':
                destination='methods/gaussian-proxy-hybrid/builds/full-50rounds/source/mh/'+target_name
                target=builds['hybrid-full']['path']+'/source/mh/'+target_name
            else:
                destination=parent+'/'+target_name
                owner=next(v for v in old['builds'].values() if row['path'].startswith(v['path']+'/'))
                key=next(k for k,v in old['builds'].items() if v is owner)
                target=builds[key]['path']+'/source/mh/'+target_name
            move(row['path'],destination)
        plan(row['path'],target)
        modules.append(dict(row,path=target,name=Path(target).name,archived=target.startswith('archive/')))
    families={p.name for p in (ROOT/'methods').iterdir() if p.is_dir() and not p.is_symlink()}
    inactive=families-WORKING
    by_path={item['path']:builds[key] for key,item in old['builds'].items()}
    # Remove completed-build navigation from active families as well.
    for family in families:
        for child in (ROOT/'methods'/family/'builds').glob('*'):
            if not child.is_symlink():continue
            canonical=str(child.resolve().relative_to(ROOT))
            item=by_path.get(canonical)
            if item is None:continue
            dest=('archive/methods' if item['archived'] else 'methods')+f'/{family}/builds/{item["build"]}'
            plan(str(child.relative_to(ROOT)),dest)
    probe='methods/gaussian-class-moments/builds/input-statistics-probe'
    plan(probe,'archive/'+probe);move(probe,'archive/'+probe)
    for row in catalog:
        if row['method'] not in ACTIVE:
            file=f'docs/methods/{row["method"]}/lessons.md'
            target=f'docs/lessons/{row["method"]}.md'
            plan(file,target);move(file,target)
    for key,item in old['builds'].items():
        current=item['path'];v=builds[key]
        if v['archived'] and item['family'] in WORKING:
            move(current,v['path'])
        else:
            local=f'methods/{item["family"]}/builds/{v["build"]}'
            if current!=local:move(current,local)
        doc=f'docs/methods/{item["family"]}/builds/{item["build"]}'
        final=('docs/archive/methods' if v['archived'] else 'docs/methods')+f'/{item["family"]}/builds/{v["build"]}'
        plan(doc,final)
        if (ROOT/doc).exists():
            local=final if v['archived'] and item['family'] in WORKING else f'docs/methods/{item["family"]}/builds/{v["build"]}'
            if doc!=local:move(doc,local)
    for family in sorted(inactive):
        for top in ('methods','docs/methods'):
            a=f'{top}/{family}';b=f'archive/{a}' if top=='methods' else f'docs/archive/methods/{family}'
            plan(a,b);move(a,b)
    for top in ('legacy','overview'):
        plan(f'docs/{top}',f'docs/archive/{top}');move(f'docs/{top}',f'docs/archive/{top}')
    keep={'README.md','protocol.md','new-fl-primary-sources-2026-10-13.md','performance-feasibility-2026-10-13.md'}
    for p in sorted((ROOT/'docs/research').iterdir()):
        if p.name in keep:continue
        a=str(p.relative_to(ROOT))
        if p.name=='full-h1-h2.md':b='docs/methods/shared-baselines/builds/federated-baselines-full/protocol.md'
        elif p.name=='full-hybrid.md':b='docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/protocol.md'
        else:b='docs/archive/research/'+p.name
        plan(a,b);move(a,b)
    old_ops=['context-history-before-method-folders-2026-10-12.md','continuation-2026-10-13.md',
        'decisions-2026-10-12.md','expansion-2026-10-12.md','handoff-2026-10-11.md','live-2026-10-12.md',
        'resume-after-reboot-2026-10-13.md','kernel-status-2026-10-12.json','quota-2026-10-12.txt']
    for name in old_ops:
        a='docs/operations/'+name;b='docs/archive/operations/'+name;plan(a,b);move(a,b)
    for p in (ROOT/'docs/operations').glob('*.log'):
        a=str(p.relative_to(ROOT));b='archive/logs/operations/'+p.name;plan(a,b);move(a,b)
    plan('scripts/archive','archive/tools/layout-migrations/previous-layout');move('scripts/archive','archive/tools/layout-migrations/previous-layout')
    for name in RETIRED_TOOLS:
        a='scripts/'+name;b='archive/tools/'+name;plan(a,b);move(a,b)
        links.append(('.compat/scripts/'+name,b))
    # Rename build helpers, preserving their bytes except operational root discovery below.
    for key,item in old['builds'].items():
        if key not in SEMANTIC:continue
        for p in (ROOT/item['path']/'scripts').glob('*.py'):
            if p.is_symlink():continue
            name=re.sub(r'(?<![a-zA-Z0-9])'+key+r'(?![a-zA-Z0-9])',SEMANTIC[key],p.name)
            if name==p.name:continue
            a=str(p.relative_to(ROOT));b=builds[key]['path']+'/scripts/'+name
            plan(a,b);move(str(Path(builds[key]['path'])/'scripts'/p.name),b)
    def remap(value):
        for a in sorted(pathmap,key=len,reverse=True):
            if value==a or value.startswith(a+'/'):return pathmap[a]+value[len(a):]
        return value
    # Read Markdown link destinations before aliases or parents move.
    destinations={}
    for name,body in markdown.items():
        for match in re.finditer(r'\]\(([^)\n]+)\)',body):
            target=match.group(1).strip('<>')
            if re.match(r'^[a-zA-Z][\w+.-]*:',target) or target.startswith('#'):continue
            bare=unquote(target.partition('#')[0])
            if not bare:continue
            absolute=(ROOT/name).parent/bare
            try:destinations[(name,match.start())]=str(absolute.resolve().relative_to(ROOT))
            except ValueError:continue
    for a,b in steps:
        source=ROOT/a;destination=ROOT/b
        assert source.exists() or source.is_symlink(), ('missing planned source',a)
        assert not destination.exists() and not destination.is_symlink(), ('destination exists',b)
        destination.parent.mkdir(parents=True,exist_ok=True);source.rename(destination)
    # mh becomes a thin compatibility facade; all source files have memorable canonical names.
    (ROOT/'mh').rename(ROOT/'.compat/mh');(ROOT/'mh').symlink_to('.compat/mh',target_is_directory=True)
    links=[(('.compat/mh/'+a[3:]) if a.startswith('mh/') else a,b) for a,b in links]
    links.extend(('.compat/mh/'+row['module']+'.py',row['path']) for row in modules if row['shared'])
    links.append(('mh','.compat/mh'))
    for p in list((ROOT/'scripts').glob('*'))+list((ROOT/'scripts/kaggle').glob('*')):
        if not p.is_symlink():continue
        oldname=str(p.relative_to(ROOT));newname='.compat/'+oldname
        p.unlink()
        links=[(newname if a==oldname else a,b) for a,b in links]
    for p in list((ROOT/'logs').iterdir()):
        if p.name in {'README.md','operations'}:continue
        a=str(p.relative_to(ROOT));b='.compat/'+a
        destination=ROOT/b;destination.parent.mkdir(parents=True,exist_ok=True);p.rename(destination)
        links=[(('.compat/'+alias) if alias.startswith(a+'/') else alias,target) for alias,target in links]
    newlinks={}
    for alias,target in links:
        alias=remap(alias);target=remap(target)
        if alias.startswith('docs/methods/') and Path(alias).name in {'status-summary.md','evidence-summary.md'}:
            existing=ROOT/alias
            if existing.is_symlink():existing.unlink()
            continue
        newlinks[alias]=target
    # Remove navigation symlinks after parent moves, then recreate exactly the recorded map.
    # Do not enter committed artifacts or private credential directories.
    for top in ('.compat','methods','archive/methods','scripts','logs','docs/methods','docs/archive/methods'):
        for directory,dirs,files in os.walk(ROOT/top,followlinks=False):
            for name in list(dirs):
                file=Path(directory)/name
                if file.is_symlink():file.unlink();dirs.remove(name)
                elif name in {'artifacts','private_push','__pycache__'}:dirs.remove(name)
            for name in files:
                file=Path(directory)/name
                if file.is_symlink():file.unlink()
    for alias,target in newlinks.items():
        file=ROOT/alias;dest=ROOT/target
        if file.is_symlink():file.unlink()
        elif file.exists():raise AssertionError(('alias collides with physical file',alias))
        file.parent.mkdir(parents=True,exist_ok=True)
        file.symlink_to(os.path.relpath(dest,file.parent),target_is_directory=dest.is_dir())
    for name,body in markdown.items():
        output=ROOT/remap(name)
        def replace(match):
            value=destinations.get((name,match.start()))
            if value is None:return match.group(0)
            target=remap(value);relative=os.path.relpath(ROOT/target,output.parent)
            suffix=('#'+match.group(1).partition('#')[2]) if '#' in match.group(1) else ''
            return ']('+quote(relative,safe='/._-')+suffix+')'
        output.write_text(re.sub(r'\]\(([^)\n]+)\)',replace,body))
    for row in catalog:
        row['active']=row['method'] in ACTIVE
        row['builds']=[remap(p) for p in row['builds']]
        row['dossier']=('docs/methods/' if row['active'] else 'docs/archive/methods/')+row['method']
        row['lessons']=row['dossier']+'/lessons.md' if row['active'] else f'docs/lessons/{row["method"]}.md'
    (ROOT/'docs/methods/catalog.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'docs/operations/compatibility-aliases.json').write_text(json.dumps(dict(
        aliases=[dict(alias=a,target=b) for a,b in sorted(newlinks.items())],contains_file_copies=False),indent=2)+'\n')
    proof.write_text(json.dumps(dict(active_methods=sorted(ACTIVE),archived_methods=sorted(inactive),
        builds=builds,modules=modules,path_map=pathmap,moves=steps,
        scientific_bytes_changed=False,git_dvc_add_push=False,
        note='Labels V2-V10 identify experiment batches, not a linear replacement history. Closed evidence archived, reusable dependencies retained.'),indent=2)+'\n')
    # Only operational scripts need root discovery; scientific source is immutable.
    for name in RETIRED_TOOLS:
        file=ROOT/'archive/tools'/name
        body=file.read_text()
        body=re.sub(r'ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]',
            "ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())",body)
        file.write_text(body)
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'mh').glob('*.py')}==packages['legacy_source_manifest']
    print('Archived/named:',len(inactive),'method folders;',len(modules),'module bytes preserved;',len(newlinks),'compatibility aliases')


if __name__=='__main__':main()
