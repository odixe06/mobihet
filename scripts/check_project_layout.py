"""Read-only link, frozen-byte, DVC identity, Git index and credential-boundary verification."""
import configparser
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from project_paths import current_path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='docs/archive/operations/layout-verification-2026-10-14.json')
    args=parser.parse_args()
    migration=json.loads((ROOT/'docs/archive/operations/layout-migration-2026-10-14.json').read_text())
    config=configparser.ConfigParser();config.read(ROOT/'.dvc/config')
    remote=Path(config['\'remote "gdrive"\'']['url'])/'files/md5'
    dvc=[];remote_prefixes=[]
    for item in migration['dvc_before']:
        metadata=ROOT/current_path(item['new']);current=yaml.safe_load(metadata.read_text())['outs']
        assert len(current)==len(item['outs'])
        for before,after in zip(item['outs'],current):
            assert {k:v for k,v in before.items() if k!='path'}=={k:v for k,v in after.items() if k!='path'}
            object_=remote/after['md5'][:2]/after['md5'][2:]
            assert object_.exists(), ('remote manifest missing',metadata)
            if after['md5'].endswith('.dir'):
                with object_.open('rb') as stream:assert hashlib.file_digest(stream,'md5').hexdigest()==after['md5'][:-4]
            remote_prefixes.append((metadata.parent/after['path']).resolve())
            dvc.append(dict(metadata=str(metadata.relative_to(ROOT)),object=after['md5'],identity_preserved=True,remote_manifest_verified=True))
    for old,digest in migration['protected_sha256'].items():assert sha(ROOT/current_path(old))==digest, ('frozen bytes drift',old)
    packages=json.loads((ROOT/'docs/operations/package-layout-2026-10-14.json').read_text())
    assert {p.name:sha(p) for p in (ROOT/'.compat/mh').glob('*.py')}==packages['legacy_source_manifest']
    for row in packages['modules']:
        canonical=current_path(row['path'])
        assert (ROOT/row['old_path']).resolve()==(ROOT/canonical).resolve()
        assert not (ROOT/canonical).is_symlink()
        if not row['shared']:
            ignored=subprocess.run(['git','check-ignore','--no-index','-q',canonical],cwd=ROOT)
            assert ignored.returncode==1, ('canonical source ignored',row['path'])
    git_env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    index_bytes_unchanged=sha(ROOT/'.git/index')==migration['git_index_sha256']
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=git_env,text=True).strip()
    initial_rows=migration['prior_git_status'].split('\0')
    assert all(not row or row[0] in {' ','?'} for row in initial_rows), 'baseline already had staged content'
    subprocess.run(['git','diff','--cached','--quiet','HEAD','--'],cwd=ROOT,env=git_env,check=True)
    raw=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-0'],cwd=ROOT).decode().split('\0')
    files=[ROOT/p for p in raw if p and (ROOT/p).is_file() and not (ROOT/p).is_symlink()]
    missing=[];remote_links=[];checked=0
    for file in files:
        if file.suffix!='.md':continue
        body=re.sub(r'```[\s\S]*?```','',file.read_text())
        for target in re.findall(r'\]\(([^)\n]+)\)',body):
            target=target.strip('<>')
            if re.match(r'^[a-zA-Z][\w+.-]*:',target) or target.startswith('#'):continue
            path=unquote(target.partition('#')[0])
            if not path:continue
            destination=(file.parent/path).resolve();checked+=1
            if destination.exists():continue
            entry=dict(file=str(file.relative_to(ROOT)),target=target)
            if any(destination==p or p in destination.parents for p in remote_prefixes):remote_links.append(entry)
            else:missing.append(entry)
    import netrc
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    for file in files:
        if file.suffix in {'.py','.md','.json','.yaml','.yml','.ipynb','.log'}:
            assert key not in file.read_text(errors='ignore'), ('secret in public file',str(file.relative_to(ROOT)))
    private={p.resolve() for base in ('methods','archive/methods') for p in (ROOT/base).glob('*/builds/*/private_push/**/*.ipynb') if p.is_file()}
    for file in private:
        assert file.stat().st_mode & 0o777==0o600
        subprocess.run(['git','check-ignore','-q',str(file.resolve())],cwd=ROOT,check=True)
    for name in ('SKILL.md','references/build-layout.md'):
        assert (ROOT/'.agents/skills/federated-experiment-workflow'/name).read_bytes()==(
            ROOT/'.claude/skills/federated-experiment-workflow'/name).read_bytes()
    status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=ROOT,env=git_env).decode().split('\0')
    counts={}
    for row in status:
        if row:counts[row[:2]]=counts.get(row[:2],0)+1
    result=dict(dvc=dvc,frozen_files=len(migration['protected_sha256']),git_head=git_head,migration_git_head=migration['git_head'],git_head_unchanged=git_head==migration['git_head'],
        git_staged_content_unchanged=True,git_index_bytes_unchanged=index_bytes_unchanged,
        git_workspace_status_counts=counts,relative_links_checked=checked,
        missing_links=missing,remote_only_links=remote_links,private_notebooks_ignored0600=len(private),
        public_key_scan_pass=True,skill_parity=True,package_legacy_fingerprint_preserved=True,
        shared_package_modules=packages['shared_modules'],build_owned_modules=packages['build_owned_modules'],
        limits='Directory-manifest identities verified; no rehash of every heavy remote object, no Git/DVC add/push. Partial restores stay partial.')
    from dvc.repo import Repo
    with Repo(str(ROOT)) as repo:result['native_dvc_stage_count']=len(repo.index.stages)
    proof=ROOT/args.output
    proof.write_text(json.dumps(result,indent=2)+'\n')
    print('Layout:',len(dvc),'DVC identities/remote manifests;',result['frozen_files'],'frozen files;',checked,
        'links;',len(missing),'missing;',len(remote_links),'remote-only;',len(private),'private notebooks ignored0600')
    for row in missing[:35]:print('MISSING',row['file'],row['target'])
    assert not missing,'repair document links before handing back'


if __name__=='__main__':main()
