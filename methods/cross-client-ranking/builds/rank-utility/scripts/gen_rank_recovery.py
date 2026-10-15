"""Operational recovery only: frozen source/fingerprint, verified committed r3, bytes allowlist."""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat
from gen_rank_utility import SETUP, PREFLIGHT, RUN

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]

IMPORT = r'''
compat = WORK/'rank_bytes_compat.py'
compat.write_text(COMPAT)
assert hashlib.sha256(compat.read_bytes()).hexdigest()==COMPAT_SHA
(SOURCE/'rank_bytes_compat.py').write_text(COMPAT)
torch.serialization.add_safe_globals([bytes])
matches=[p.parent for p in Path('/kaggle/input').rglob('resume_manifest.json') if json.loads(p.read_text())==RESUME]
assert len(matches)==1,('missing/ambiguous committed checkpoint',matches)
checkpoint=matches[0]
for rel,digest in RESUME.items():
    source=checkpoint/rel
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,rel
    target=OUT/Path(rel).relative_to('runs');target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
(SOURCE/'resume_manifest.json').write_text(json.dumps(RESUME,indent=2))
(SOURCE/'recovery.json').write_text(json.dumps(RECOVERY,indent=2))
from rank_verify import verify
proofs=[]
for arm in ('cr_local','cr_fl'):
    audit=verify(OUT/f'{arm}_s{SEED}',MANIFEST,hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest(),WORK/'papers/pilot/fixture')
    assert audit['logical_rounds']==3 and audit['committed_rounds']==1
    proofs.append(audit)
(SOURCE/'RECOVERY_MOUNT_VERIFIED.json').write_text(json.dumps(dict(compat_sha=COMPAT_SHA,files=len(RESUME),audits=proofs)))
print('RECOVERY SHA/strict state/wire/ten metrics PASS; frozen source, r3 preserved',flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int,choices=(42,43))
    parser.add_argument('--hours',type=float,default=2.25);args=parser.parse_args();seed=args.seed
    old=BASE/'notebooks'/f'cr-u-s{seed}-20261015-v1'
    parent=BASE/'artifacts/kaggle'/old.name;audited=json.loads((parent/'AUDITED.json').read_text())
    assert len(audited['audits'])==2 and all(a['logical_rounds']==3 for a in audited['audits'])
    rid=f'cr-u-s{seed}-20261015-v2';public=BASE/'notebooks'/rid;private=BASE/'private_push'/rid
    stage=BASE/'artifacts/input'/rid;probe=BASE/'probes/mount-probe'/rid
    assert not any(p.exists() for p in (public,private,stage,probe))
    stage.mkdir(parents=True);resume={};files={}
    for p in sorted((parent/'runs').rglob('*')):
        if not p.is_file() or p.suffix=='.log':continue
        rel=str(p.relative_to(parent));target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);resume[rel]=sha(p);files[rel]=dict(size=p.stat().st_size,sha256=sha(p))
    (stage/'resume_manifest.json').write_text(json.dumps(resume,indent=2)+'\n')
    files['resume_manifest.json']=dict(size=(stage/'resume_manifest.json').stat().st_size,sha256=sha(stage/'resume_manifest.json'))
    owner='minhtriethihi';dataset=f'{owner}/mobihet-{rid}-ckpt'
    (stage/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title=f'mobihet-{rid}-ckpt',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    meta=json.loads((old/'kernel-metadata.json').read_text());nb=nbformat.read(old/meta['code_file'],as_version=4)
    inputs=json.loads((old/'input_manifest.json').read_text());manifest=json.loads((old/'source_manifest.json').read_text())
    recovery=dict(parent=audited['kernel'],version=1,round=3,seed=seed,session_hours=args.hours,
        change='only scoped built-in bytes allowlist for PyTorch2.10; frozen source and scientific fingerprints unchanged')
    compat=(BASE/'scripts/rank_bytes_compat.py').read_text()
    header=f'SEED={seed}\nRESUME={resume!r}\nRECOVERY={recovery!r}\nCOMPAT={compat!r}\nCOMPAT_SHA={hashlib.sha256(compat.encode()).hexdigest()!r}\n'
    setup=nb.cells[2].source.removesuffix(PREFLIGHT)+header+IMPORT
    nb.cells[2].source=setup+PREFLIGHT
    run=RUN.replace("str(WORK/'papers/rank-utility/proj/boundary_logging.py')", "str(WORK/'rank_bytes_compat.py'),str(WORK/'papers/rank-utility/proj/boundary_logging.py')")
    start=run.index("active={gpu:(arm,*launch(arm,gpu,3))")
    end=run.index('from rank_verify import verify',start)
    run=run[:start]+run[end:]
    # Imported r3 is audited again; cost rows and continuation source remain identical.
    nb.cells[3].source=f'SEED={seed}\nSESSION_HOURS={args.hours}\nMETHODS={["cr_local","cr_fl","cr_own","cr_shuffle","cr_fixed","cr_classweight","cr_gaussian","cr_dml"]!r}\n'+run
    nb.cells[0].source='# CrossClientRank utility recovery\nFrozen source and audited r3; PyTorch2.10 built-in bytes compatibility only.'
    meta.update(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=rid+'.ipynb',dataset_sources=meta['dataset_sources']+[dataset])
    public.mkdir();private.mkdir(mode=0o700);probe.mkdir(parents=True)
    for name in ('source_manifest.json','input_manifest.json','warm-proof.json'):
        shutil.copyfile(old/name,public/name)
    for name,value in [('checkpoint-dataset.json',dict(dataset=dataset,files=files)),('recovery.json',recovery),('kernel-metadata.json',meta)]:
        (public/name).write_text(json.dumps(value,indent=2)+'\n')
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,rid,'exec')
    nbformat.validate(nb);nbformat.write(nb,public/meta['code_file'])
    subprocess.run(['git','check-ignore','-q',str((private/meta['code_file']).resolve())],cwd=ROOT,check=True)
    secret=netrc.netrc().authenticators('api.wandb.ai')[2];pushed=nbformat.reads(nbformat.writes(nb),as_version=4)
    pushed.cells[1].source=pushed.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(secret))
    for name,value in [(meta['code_file'],nbformat.writes(pushed)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value)
    cpu='import os,sys\n'+setup
    cpu=cpu.replace('assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))','assert torch.cuda.device_count()==0')
    cpu=cpu.replace("assert torch.__version__=='2.10.0+cu128'",'')
    cp=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cpu)])
    cp.metadata=dict(nb.metadata,kaggle=dict(accelerator='none',isGpuEnabled=False,isInternetEnabled=True,language='python',sourceType='notebook'))
    cm=dict(meta,id=meta['id']+'-cpu',title=meta['title']+'-cpu',enable_gpu=False);cm.pop('machine_shape',None)
    compile(cpu,rid,'exec');nbformat.validate(cp);nbformat.write(cp,probe/meta['code_file'])
    (probe/'kernel-metadata.json').write_text(json.dumps(cm,indent=2)+'\n')
    print(meta['id'],'prepared; original source + r3; CPU audit required')


if __name__=='__main__':main()
