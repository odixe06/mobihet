"""One authorized layout migration; preserve frozen bytes, DVC object identities and aliases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, quote

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
# One physical owner for a comparison build; other methods link to it.
BUILDS={
    'pilot':('shared-baselines','p1-pilot',['proxy-mutual-distillation','gaussian-class-moments','federated-head-transfer','class-prototypes','gaussian-logit-distillation']),
    'p2':('contact-fragmentation','p2-deadline',['proxy-mutual-distillation','gaussian-class-moments']),
    'new-fl':('synthetic-distribution-matching','sa-static-books',['fourier-distribution-sketch','latent-mean-synthetic-books']),
    'v2':('shared-comparisons','v2-sketch-subspace-ratio',['proxy-countsketch','proxy-seeded-subspace','gaussian-density-ratio']),
    'v3':('proxy-compression','v3-carrier-comparison',['proxy-random-coordinates','proxy-rolling-subnetwork','proxy-orthogonal-projection','proxy-powersgd']),
    'v4':('gaussian-sampling','v4-support-mala',['gaussian-mala-transport','gaussian-support-constraints']),
    'v5':('synthetic-knowledge','v5-anchor-gradient-books',['kernel-anchor-distillation','gradient-synthetic-books','latent-mean-synthetic-books']),
    'v6':('geometry-primitive-moments','v6-g59-g61',[]),
    'v7':('covariance-nystrom','v7-rank4-rank8',[]),
    'v8':('gaussian-proxy-hybrid','v8-pilot',[]),
    'v9':('anchor-function-residual','v9-teacher-residual',[]),
    'v10':('dagsnet-joint-energy','v10-gaussian-base-mh',[]),
    'full-validation':('shared-baselines','full-h1-h2-50rounds',['proxy-mutual-distillation','gaussian-class-moments']),
    'hybrid-full':('gaussian-proxy-hybrid','full-50rounds',[]),
    'boundary-cells':('boundary-cell-counts','cell-counts-v1',[]),
    'receiver-response':('receiver-response-consistency','local-augmentation-v1',[]),
}
ARTIFACT_NAMES={'kaggle','runs','input','fixture','smoke','resume_reference','diagnostics','mount-output',
                'initial-output','controller-fixture','source_audit','kaggle_src'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def link(target,alias):
    alias.parent.mkdir(parents=True,exist_ok=True)
    assert not alias.exists() and not alias.is_symlink(), ('alias collision',alias)
    alias.symlink_to(os.path.relpath(target,alias.parent),target_is_directory=target.is_dir())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    assert not (ROOT/'docs/operations/layout-migration-2026-10-14.json').exists(),'migration already applied; inspect its proof'
    original_md={str(p.relative_to(ROOT)):p.read_text() for p in (ROOT/'proposals').rglob('*.md')}
    moves={};prefixes={};scripts=[];dvc=[];protected={}
    for path in list((ROOT/'mh').glob('*.py'))+list((ROOT/'papers').glob('*/proj/*.py'))+list((ROOT/'papers').glob('*/notebook/*/*')):
        if path.is_file():protected[str(path.relative_to(ROOT))]=digest(path)
    for old,(family,build,_) in BUILDS.items():
        src=ROOT/'papers'/old;dest=ROOT/'methods'/family/'builds'/build
        prefixes[f'papers/{old}']=str(dest.relative_to(ROOT))
        for path in src.rglob('*'):
            if path.is_file() and not path.is_symlink():
                if 'private_push' not in path.parts and 'artifacts' not in path.parts:
                    moves[str(path.relative_to(ROOT))]=str((dest/path.relative_to(src)).relative_to(ROOT))
        for path in src.glob('*.dvc'):
            import yaml
            dvc.append(dict(old=str(path.relative_to(ROOT)),new=str((dest/path.name).relative_to(ROOT)),
                            sha=digest(path),outs=yaml.safe_load(path.read_text())['outs']))
    print('Layout plan:',len(BUILDS),'builds;',len(original_md),'proposal documents;',len(dvc),'DVC manifests')
    if not args.apply:return
    operations=ROOT/'docs/operations';operations.mkdir(exist_ok=True)
    old_git=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=ROOT).decode()
    index_sha=digest(ROOT/'.git/index')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Move proposal documents into docs. The old entrypoint becomes a compatibility alias.
    proposal_readme=(ROOT/'proposals/README.md').read_text()
    for path in sorted((ROOT/'proposals').iterdir()):
        if path.is_dir():
            destination=ROOT/'docs/methods'/path.name
        elif path.name=='README.md':
            destination=ROOT/'docs/proposals.md'
        else:destination=ROOT/'docs/overview'/path.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();shutil.move(str(path),str(destination))
        prefixes[str(path.relative_to(ROOT))]=str(destination.relative_to(ROOT))
    (ROOT/'proposals').rmdir();link(ROOT/'docs/methods',ROOT/'proposals')
    for name in ('status-summary.md','evidence-summary.md'):
        link(ROOT/'docs/overview'/name,ROOT/'docs/methods'/name)
    # Each build owns probes, source snapshots, notebooks, calibration and all pilot evidence.
    for old,(family,build,_) in BUILDS.items():
        src=ROOT/'papers'/old;dest=ROOT/'methods'/family/'builds'/build
        dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
        shutil.move(str(src),str(dest));link(dest,src)
        doc=ROOT/'docs/methods'/family/'builds'/build;doc.mkdir(parents=True,exist_ok=True)
        log=ROOT/'logs'/family/build;log.mkdir(parents=True,exist_ok=True)
        for item in list(dest.iterdir()):
            oldname=f'papers/{old}/{item.name}'
            if item.name in ARTIFACT_NAMES:
                target=dest/'artifacts'/item.name
            elif item.name=='proj' or item.is_dir() and ('source' in item.name):
                target=dest/'source'/item.name
            elif item.name=='notebook':target=dest/'notebooks'
            elif item.name=='private_push':continue
            elif item.name=='logs':
                for entry in list(item.iterdir()):
                    target=log/entry.name;assert not target.exists();shutil.move(str(entry),str(target))
                item.rmdir();link(log,item);continue
            elif item.suffix=='.log' or item.suffix=='.lock':target=log/item.name
            elif item.suffix in ('.md','.csv','.svg','.png'):
                target=doc/'results'/item.name
            elif item.suffix=='.json' and not item.name.endswith('.dvc'):
                target=dest/'probes'/item.name
            else:continue
            target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists(), ('target collision',target)
            shutil.move(str(item),str(target));link(target,item)
            prefixes[oldname]=str(target.relative_to(ROOT))
        if not (dest/'logs').exists():link(log,dest/'logs')
        for metadata in dest.glob('*.dvc'):
            import yaml
            value=yaml.safe_load(metadata.read_text())
            for out in value['outs']:
                if out['path'] in ARTIFACT_NAMES:out['path']='artifacts/'+out['path']
            metadata.write_text(yaml.safe_dump(value,sort_keys=False))
    # Assign loose probe bundles and logs to their actual build.
    extras={
        'papers/nystrom':ROOT/'methods/covariance-nystrom/builds/v7-rank4-rank8/probes/nystrom-preflight',
        'papers/probes':ROOT/'methods/gaussian-class-moments/builds/input-statistics-probe',
    }
    for old,dest in extras.items():
        src=ROOT/old;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dest));link(dest,src)
        prefixes[old]=str(dest.relative_to(ROOT))
    for path in list((ROOT/'papers').glob('*.log')):
        old=str(path.relative_to(ROOT));match=re.match(r'(v\d+)-',path.name)
        key=match[1] if match else 'hybrid-full' if path.name.startswith('hybrid-full') else 'full-validation' if path.name.startswith('full-driver') else None
        if key in BUILDS:
            family,build,_=BUILDS[key];target=ROOT/'logs'/family/build/path.name
        else:target=ROOT/'logs/operations'/path.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(target));link(target,path)
        prefixes[old]=str(target.relative_to(ROOT))
    # Method-specific operational scripts move; generic quota/push/index/local guard stay shared.
    for path in list((ROOT/'scripts').glob('*'))+list((ROOT/'scripts/kaggle').glob('*')):
        if not path.is_file() or path.name=='reorganize_project.py':continue
        name=path.name;key=None
        match=re.search(r'(?:^|_)v(\d+)(?:_|\.)',name)
        if match and 'v'+match[1] in BUILDS:key='v'+match[1]
        elif 'boundary' in name:key='boundary-cells'
        elif 'hybrid' in name:key='hybrid-full'
        elif 'full' in name and name not in ('gen_full_resume_batch.py','launch_prepared_batch.py'):key='full-validation'
        elif 'new_fl' in name:key='new-fl'
        elif re.search(r'(?:^|_)p2(?:_|\.)',name):key='p2'
        elif 'pilot_queue' in name or name=='report_pilot.py':key='pilot'
        elif 'nystrom' in name:key='v7'
        elif 'h2_input' in name:key='pilot'
        elif 'k5_missing' in name:key='v6'
        if key is None:continue
        family,build,_=BUILDS[key];dest=ROOT/'methods'/family/'builds'/build/'scripts'/name
        dest.parent.mkdir(parents=True,exist_ok=True);old=str(path.relative_to(ROOT));assert not dest.exists()
        shutil.move(str(path),str(dest));link(dest,path);prefixes[old]=str(dest.relative_to(ROOT))
        if dest.suffix=='.py':
            body=dest.read_text()
            # Support scripts embedded into Kaggle retain their old-depth fallback there.
            body=re.sub(r'Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]',
                lambda m:"next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents["+m[1]+'])',body)
            dest.write_text(body)
        elif dest.suffix=='.sh':
            body=dest.read_text().replace('cd "$(dirname "$0")/.."',
                'project_root=$(dirname "$(readlink -f "$0")")\nwhile [ ! -f "$project_root/CONTEXT.md" ]; do project_root=$(dirname "$project_root"); done\ncd "$project_root"')
            dest.write_text(body)
        scripts.append(dict(old=old,new=str(dest.relative_to(ROOT))))

    def mapped(name):
        if name in prefixes:return prefixes[name]
        for old in sorted(prefixes,key=len,reverse=True):
            if name.startswith(old+'/'):return prefixes[old]+name[len(old):]
        return name
    # Rewrite navigational Markdown only; immutable source/notebooks/output contents stay byte-identical.
    raw=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-0'],cwd=ROOT).decode().split('\0')
    current_files=[ROOT/p for p in raw if p and (ROOT/p).is_file() and (ROOT/p).suffix=='.md']
    reverse={mapped(old):old for old in list(prefixes)+list(moves)+list(original_md)}
    rewritten=[]
    for file in current_files:
        if file.is_symlink() or any(x in file.parts for x in ('private_push','artifacts','notebooks','source','.agents','.claude')):continue
        current=str(file.relative_to(ROOT));old=reverse.get(current,current)
        # Files nested in a moved proposal folder require reverse-prefix lookup.
        if old==current:
            for src,dest in sorted(prefixes.items(),key=lambda x:len(x[1]),reverse=True):
                if current.startswith(dest+'/'):old=src+current[len(dest):];break
        body=file.read_text()
        def replacement(match):
            target=match[1];target=target.strip('<>')
            if target.startswith(('http:','https:','mailto:','#','app:')):return match[0]
            path,sep,anchor=target.partition('#')
            resolved=os.path.normpath(os.path.join(str(Path(old).parent),unquote(path)))
            if resolved.startswith('../'):return match[0]
            destination=mapped(resolved)
            relative=os.path.relpath(ROOT/destination,file.parent)
            return ']('+quote(relative,safe='/._-')+(sep+anchor if sep else '')+')'
        changed=re.sub(r'\]\(([^)\n]+)\)',replacement,body)
        if changed!=body:file.write_text(changed);rewritten.append(current)
    for old,sha in protected.items():
        assert digest(ROOT/mapped(old))==sha, ('frozen bytes changed',old,mapped(old))
    assert digest(ROOT/'.git/index')==index_sha and subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==head
    record=dict(builds={k:dict(family=v[0],build=v[1],related=v[2],path=mapped('papers/'+k)) for k,v in BUILDS.items()},
        aliases=prefixes,scripts=scripts,dvc_before=dvc,protected_files=len(protected),
        protected_sha256=protected,markdown_rewritten=rewritten,git_head=head,git_index_sha256=index_sha,
        prior_git_status=old_git,limits='No Git/DVC add/commit/push. Remote-only files stay remote. Frozen bytes unchanged; support-script root lookup adapted.')
    (operations/'layout-migration-2026-10-14.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Moved build/proposal/script/log layout; frozen bytes PASS',len(protected),'files; Git index unchanged')


if __name__=='__main__':main()
