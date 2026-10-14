"""Finish metadata/link routing after the one-time migration's log-wrapper Markdown interruption."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, unquote

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
spec=importlib.util.spec_from_file_location('layout_plan',Path(__file__).with_name('archive_closed_experiments.py'))
cfg=importlib.util.module_from_spec(spec);spec.loader.exec_module(cfg)


def main():
    old=json.loads((ROOT/'docs/operations/layout-migration-2026-10-14.json').read_text())
    packages=json.loads((ROOT/'docs/operations/package-layout-2026-10-14.json').read_text())
    aliases=json.loads((ROOT/'docs/operations/compatibility-aliases.json').read_text())['aliases']
    catalog=json.loads((ROOT/'docs/methods/catalog.json').read_text())
    inactive={p.name for p in (ROOT/'archive/methods').iterdir() if p.is_dir()}-cfg.WORKING
    mapping={};builds={}
    for key,row in old['builds'].items():
        archived=key not in cfg.OPEN_BUILDS;name=cfg.BUILD_NAMES[key]
        base=('archive/methods/' if archived else 'methods/')+row['family']+'/builds/'+name
        builds[key]=dict(row,build=name,path=base,archived=archived,legacy_build=row['build'])
        mapping[row['path']]=base
        mapping[f'docs/methods/{row["family"]}/builds/{row["build"]}']=(
            ('docs/archive/methods/' if archived else 'docs/methods/')+row['family']+'/builds/'+name)
    for family in inactive:
        mapping['methods/'+family]='archive/methods/'+family
        mapping['docs/methods/'+family]='docs/archive/methods/'+family
    for row in aliases:
        match=re.match(r'methods/([^/]+)/builds/([^/]+)',row['alias'])
        if match:
            family,legacy=match.groups()
            found=next((v for v in builds.values() if v['legacy_build']==legacy),None)
            if found:mapping[match.group(0)]=('archive/methods/' if found['archived'] else 'methods/')+family+'/builds/'+found['build']
    probe='methods/gaussian-class-moments/builds/input-statistics-probe';mapping[probe]='archive/'+probe
    modules=[]
    for row in packages['modules']:
        if row['shared']:target='core/'+cfg.COMMON_NAMES[row['module']]+'.py'
        elif row['module']=='v8_hybrid':target=builds['hybrid-full']['path']+'/source/mh/hybrid_models.py'
        else:
            key=next(k for k,v in old['builds'].items() if row['path'].startswith(v['path']+'/'))
            target=builds[key]['path']+'/source/mh/'+cfg.METHOD_NAMES[row['module']]+'.py'
        mapping[row['path']]=target
        modules.append(dict(row,path=target,name=Path(target).name,archived=target.startswith('archive/')))
    for row in catalog:
        if row['method'] not in cfg.ACTIVE:mapping[f'docs/methods/{row["method"]}/lessons.md']=f'docs/lessons/{row["method"]}.md'
    for name in ('legacy','overview'):mapping['docs/'+name]='docs/archive/'+name
    for child in (ROOT/'docs/archive/research').iterdir():mapping['docs/research/'+child.name]='docs/archive/research/'+child.name
    mapping['docs/research/full-h1-h2.md']='docs/methods/shared-baselines/builds/federated-baselines-full/protocol.md'
    mapping['docs/research/full-hybrid.md']='docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/protocol.md'
    for child in (ROOT/'docs/archive/operations').iterdir():mapping['docs/operations/'+child.name]='docs/archive/operations/'+child.name
    for child in (ROOT/'archive/logs/operations').iterdir():mapping['docs/operations/'+child.name]='archive/logs/operations/'+child.name
    mapping['scripts/archive']='archive/tools/layout-migrations/previous-layout'
    for name in cfg.RETIRED_TOOLS:mapping['scripts/'+name]='archive/tools/'+name
    for key,row in builds.items():
        if key not in cfg.SEMANTIC:continue
        for p in (ROOT/row['path']/'scripts').glob('*.py'):
            oldname=p.name.replace(cfg.SEMANTIC[key],key)
            mapping[old['builds'][key]['path']+'/scripts/'+oldname]=str(p.relative_to(ROOT))
    def remap(path):
        for before in sorted(mapping,key=len,reverse=True):
            if path==before or path.startswith(before+'/'):return mapping[before]+path[len(before):]
        return path
    record={}
    for row in aliases:
        a=row['alias'];b=row['target']
        if a.startswith('mh/'):a='.compat/mh/'+a[3:]
        elif a.startswith('scripts/'):a='.compat/'+a
        elif a.startswith('logs/') and not a.startswith('logs/operations/'):a='.compat/'+a
        if a in {'docs/methods/status-summary.md','docs/methods/evidence-summary.md'}:continue
        record[remap(a)]=remap(b)
    record['mh']='.compat/mh'
    for row in modules:record['.compat/mh/'+row['module']+'.py']=row['path']
    for name in cfg.RETIRED_TOOLS:record['.compat/scripts/'+name]='archive/tools/'+name
    for a,b in record.items():
        link=ROOT/a;target=ROOT/b
        if link.is_symlink():link.unlink()
        else:assert not link.exists(),a
        link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=target.is_dir())
    (ROOT/'docs/operations/compatibility-aliases.json').write_text(json.dumps(dict(
        aliases=[dict(alias=a,target=b) for a,b in sorted(record.items())],contains_file_copies=False),indent=2)+'\n')
    proof=dict(active_methods=sorted(cfg.ACTIVE),archived_methods=sorted(inactive),builds=builds,modules=modules,
        path_map=mapping,scientific_bytes_changed=False,git_dvc_add_push=False,
        note='V2-V10 are experiment batches, not replacement versions. Log-wrapper Markdown interruption repaired; immutable source remained intact.')
    (ROOT/'docs/operations/workspace-layout.json').write_text(json.dumps(proof,indent=2)+'\n')
    # Some Markdown was already rebased before the interruption. Keep valid current links;
    # reconstruct invalid old links from the inverse relocation map.
    inverse=sorted(((new,old) for old,new in mapping.items()),key=lambda row:len(row[0]),reverse=True)
    def origin(name):
        for new,old in inverse:
            if name==new or name.startswith(new+'/'):return old+name[len(new):]
        return name
    raw=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-0'],cwd=ROOT).decode().split('\0')
    for name in raw:
        file=ROOT/name
        if not name.endswith('.md') or file.is_symlink():continue
        before=origin(name);body=file.read_text()
        def replace(match):
            value=match.group(1).strip('<>');bare,sep,anchor=value.partition('#')
            if re.match(r'^[a-zA-Z][\w+.-]*:',value) or not bare:return match.group(0)
            current=(file.parent/unquote(bare)).resolve()
            if current.exists():return match.group(0)
            original=Path(os.path.normpath(str((ROOT/before).parent/unquote(bare))))
            try:target=str(original.relative_to(ROOT))
            except ValueError:return match.group(0)
            if target.startswith('papers/'):
                alias=next((a for a in sorted(record,key=len,reverse=True) if a.startswith('.compat/papers/') and
                    ('.compat/'+target==a or ('.compat/'+target).startswith(a+'/'))),None)
                if alias:target=record[alias]+('.compat/'+target)[len(alias):]
            target=remap(target)
            relative=os.path.relpath(ROOT/target,file.parent)
            return ']('+quote(relative,safe='/._-')+('#'+anchor if sep else '')+')'
        file.write_text(re.sub(r'\]\(([^)\n]+)\)',replace,body))
    for row in catalog:
        row['active']=row['method'] in cfg.ACTIVE
        row['builds']=[remap(p) for p in row['builds']]
        row['dossier']=('docs/methods/' if row['active'] else 'docs/archive/methods/')+row['method']
        row['lessons']=row['dossier']+'/lessons.md' if row['active'] else 'docs/lessons/'+row['method']+'.md'
    (ROOT/'docs/methods/catalog.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n')
    for name in cfg.RETIRED_TOOLS:
        file=ROOT/'archive/tools'/name
        file.write_text(re.sub(r'ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]',
            "ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())",file.read_text()))
    print('Archive routing repaired:',len(inactive),'closed folders;',len(record),'aliases; metadata persisted')


if __name__=='__main__':main()
