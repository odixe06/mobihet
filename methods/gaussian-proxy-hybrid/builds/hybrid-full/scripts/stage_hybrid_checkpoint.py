"""Private same-owner checkpoint dataset for ERROR outputs Kaggle refused to attach."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
BASE=ROOT/'papers/hybrid-full'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(BASE/'proj'))
import scripts  # register compatibility imports before loading the frozen verifier
import full_verify
full_verify.ROOT=ROOT


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,choices=(1,3),required=True)
    parser.add_argument('--upload',action='store_true');args=parser.parse_args()
    folder=f'hybf{args.index}-s42-20261013-cal-v1'
    parent=BASE/'kaggle'/folder;record=json.loads((parent/'AUDITED.json').read_text())
    owner=record['kernel'].split('/')[0];slug=f'mobihet-hybrid-checkpoint{args.index}-20261014'
    stage=BASE/'artifacts/checkpoint-staging'/slug
    stage.mkdir(parents=True,exist_ok=True)
    copied={};torch.set_num_threads(1)
    for audit in record['audits']:
        run=parent/'runs'/f'{audit["arm"]}_s42'
        assert full_verify.verify(run,complete=False,infer=False)['rounds']==audit['rounds']
        for path in run.rglob('*'):
            if not path.is_file() or 'wandb' in path.relative_to(run).parts or path.name=='log.txt':continue
            relative=Path('runs')/run.name/path.relative_to(run)
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==sha(path)
            else:os.link(path,target)
            copied[relative.as_posix()]=dict(sha256=sha(target),size=target.stat().st_size)
    (stage/'checkpoint-manifest.json').write_text(json.dumps(dict(parent_kernel=record['kernel'],
        parent_version=1,audited_rounds={x['arm']:x['rounds'] for x in record['audits']},files=copied),indent=2)+'\n')
    (stage/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{slug}',title=slug,
        licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    proof=ROOT/'docs/operations'/f'hybrid-checkpoint{args.index}-2026-10-14.json'
    result=dict(dataset=f'{owner}/{slug}',parent=record['kernel'],files=len(copied),
                bytes=sum(x['size'] for x in copied.values()),verified_full_parent=True,private=True,uploaded=False)
    proof.write_text(json.dumps(result,indent=2)+'\n')
    if args.upload:
        assert not (stage/'UPLOADED').exists(), 'check service before retrying a lost upload reply'
        helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py'
        command=[sys.executable,str(helper),owner,'--','kaggle','datasets','create','-p',str(stage),'-r','zip','-t']
        upload=subprocess.run(command,capture_output=True,text=True)
        assert upload.returncode==0, ('dataset upload failed',upload.stdout[-500:],upload.stderr[-500:])
        (stage/'UPLOADED').write_text(result['dataset']+'\n');result['uploaded']=True
        proof.write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
