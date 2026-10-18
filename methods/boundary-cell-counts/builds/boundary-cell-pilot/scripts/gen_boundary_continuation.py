"""Continue unaudited utility targets; direct P2 already fails its locked r20 gate."""
import ast
import hashlib
import json
import netrc
import os
from pathlib import Path
import sys
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[5]
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts
from scripts.kaggle.gen_boundary_utility import RUN


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seed,owner=int(sys.argv[1]),sys.argv[2]
    parent=f'bc-u-s{seed}-20261014-stage01-v1'
    source=BASE/'artifacts/kaggle'/parent
    audited=json.loads((source/'AUDITED.json').read_text())
    calibration=json.loads((BASE/'artifacts/kaggle'/f'bc-cal-s{seed}-20261013-initial2-v1/AUDITED.json').read_text())
    assert calibration['calibration_valid'] and all(v['cost']['pass_cost'] for v in calibration['audits'])
    rid=f'bc-u-s{seed}-20261015-cont02-v1'
    public,private=BASE/'notebooks'/rid,BASE/'private_push'/rid
    stage=BASE/'artifacts/checkpoint-staging'/rid
    assert not public.exists() and not private.exists() and not stage.exists()
    metadata=json.loads((BASE/'notebooks'/parent/'kernel-metadata.json').read_text())
    reference=nbf.read(BASE/'notebooks'/parent/metadata['code_file'],as_version=4)
    manifest=json.loads((BASE/'notebooks'/parent/'source_manifest.json').read_text())
    methods=[f'bc_{mode}{p}_{scope}' for mode,p in (('direct',4),('teacher',2),('teacher',4))
             for scope in ('fl','own','shuffle','marginal')]
    expected,files={},{}
    worst=0.
    for arm in methods:
        name=f'{arm}_s{seed}';item=audited['summary'][name]
        if not item['rounds']:continue
        assert any(x['run']==name and x['rounds']==item['rounds'] for x in audited['audits'])
        run=source/'runs'/name
        identity=json.loads((run/'config.json').read_text())
        expected[name]={k:identity[k] for k in ('fingerprint','source','fixture')}
        expected[name].update(last_round=item['rounds'],calibration=False,cost_calibration_pass=True)
        for p in run.rglob('*'):
            if not p.is_file() or 'wandb' in p.relative_to(run).parts or p.name=='log.txt':continue
            relative=Path('runs')/name/p.relative_to(run)
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
            files[str(relative)]=dict(sha256=sha(target),size=target.stat().st_size)
        for p in (run/'complete').glob('*.json'):worst=max(worst,json.loads(p.read_text())['sec'])
    package=stage/'package';package.mkdir()
    for name,digest in manifest.items():
        original=source/'source'/name;assert sha(original)==digest
        os.link(original,package/name)
        files['package/'+name]=dict(sha256=digest,size=original.stat().st_size)
    for p in (ROOT/'.compat/papers/pilot/fixture').iterdir():
        if not p.is_file():continue
        target=package/p.name
        assert not target.exists()
        os.link(p.resolve(),target)
        files['package/'+p.name]=dict(sha256=sha(target),size=target.stat().st_size)
    (package/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dataset=f'{owner}/mobihet-{rid}-ckpt'
    bridge=dict(dataset=dataset,files=files,parent_kernel=audited['kernel'],parent_version=1,
                verified_full_parent=True,private=True,uploaded=False)
    (stage/'checkpoint-manifest.json').write_text(json.dumps(bridge,indent=2)+'\n')
    (stage/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title=f'mobihet-{rid}-ckpt',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    setup=reference.cells[2].source
    replacements=dict(EXPECTED=expected,IMPORTER=(BASE/'scripts/boundary_resume.py').read_text(),
                      VERIFIER=(BASE/'scripts/collect_boundary_calibration.py').read_text())
    nodes=[n for n in ast.parse(setup).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in replacements for t in n.targets)]
    assert len(nodes)==3
    lines=setup.splitlines(keepends=True)
    for node in sorted(nodes,key=lambda n:n.lineno,reverse=True):
        name=node.targets[0].id
        lines[node.lineno-1:node.end_lineno]=[f'{name}={replacements[name]!r}\n']
    setup=''.join(lines)
    run=RUN.replace('assert len(summary)==20','assert len(summary)==len(METHODS)')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# BoundaryCells utility continuation\nDirect P2 failed locked r20 comparison; preserve its output and stop. Continue P4 direct and P2/P4 teachers with all own/shuffle/marginal controls. Strong references remain in audited prior output. No scientific source/HP changes.'),
        nbf.v4.new_code_cell(reference.cells[1].source),nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell(f'SEED={seed}\nSESSION_HOURS=5.0\nMETHODS={methods!r}\nWORST_SECONDS={worst!r}\n'+run)])
    nb.metadata=reference.metadata.copy();name=rid+'.ipynb'
    meta=dict(metadata,id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name,
              kernel_sources=[],dataset_sources=[dataset])
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    public.mkdir(parents=True);private.mkdir(parents=True,mode=0o700)
    nbf.write(nb,public/name)
    for filename,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest),
                           ('resume-plan.json',expected),('checkpoint-dataset.json',bridge)]:
        (public/filename).write_text(json.dumps(value,indent=2)+'\n')
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    credential=nbf.reads(nbf.writes(nb),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for filename,value in [(name,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value)
    probe=BASE/'probes/mount-probe'/rid;probe.mkdir(parents=True)
    cpu=setup.split("fixture=json.loads((SRC/'manifest.json').read_text())",1)[0]
    cpu=cpu.replace('assert torch.cuda.device_count()==2','assert torch.cuda.device_count()==0')
    cpu=cpu.replace('assert all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))','')
    cpu='import os,sys,json,time\nSESSION_T0=time.monotonic()\n'+cpu+"\n(SOURCE/'MOUNT_VERIFIED.json').write_text(json.dumps({'cpu_only':True,'imports':imports}))\n"
    assert 'get_device_capability' not in cpu
    cpu_nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(cpu)])
    cpu_nb.metadata=dict(nb.metadata,kaggle=dict(accelerator='none',isGpuEnabled=False,isInternetEnabled=True,language='python',sourceType='notebook'))
    cpu_meta=dict(meta,id=meta['id']+'-cpu',title=meta['title']+'-cpu',enable_gpu=False)
    cpu_meta.pop('machine_shape',None)
    nbf.validate(cpu_nb);compile(cpu,rid,'exec');nbf.write(cpu_nb,probe/name)
    (probe/'kernel-metadata.json').write_text(json.dumps(cpu_meta,indent=2)+'\n')
    print(meta['id'],'prepared twelve arms; actual parent rounds', {k:v['last_round'] for k,v in expected.items()})


if __name__=='__main__':main()
