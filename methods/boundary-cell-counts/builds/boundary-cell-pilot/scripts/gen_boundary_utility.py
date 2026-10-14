"""Paired utility after audited calibration; budget can stage the locked twenty-round arms."""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys

import nbformat as nbf

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
sys.path.insert(0,str(ROOT))
from scripts.project_paths import current_path
from scripts.kaggle.gen_boundary_calibration import SETUP

RUN=r'''
pending=list(METHODS);active={};summary={};seen={}
while active or pending:
    for gpu in (0,1):
        if gpu in active or not pending:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<.20+1.15*WORST_SECONDS/3600:continue
        arm=pending.pop(0);folder=OUT/f'{arm}_s{SEED}';folder.mkdir(exist_ok=True)
        previous=EXPECTED.get(folder.name,{}).get('last_round',0)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(WORK/'boundary_logging.py'),str(WORK/'papers/boundary-cells/proj/boundary_driver.py'),
            '--arm',arm,'--seed',str(SEED),'--out',str(OUT),'--rounds','20','--eval_every','1',
            '--aux_weight','.5','--temperature','1','--wandb_online','--max_hours',str(min(1.,remaining-.20))]
        if previous:command.append('--resume')
        log=open(folder/'log.txt','w');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(arm,previous,proc,log)
        print('start utility',arm,'seed',SEED,'gpu',gpu,'parent_round',previous,flush=True)
    for gpu,(arm,previous,proc,log) in list(active.items()):
        folder=OUT/f'{arm}_s{SEED}'
        lines=(folder/'log.txt').read_text().splitlines()
        for line in lines[seen.get(arm,0):]:
            if line.startswith(('setup','r','done','WANDB','Traceback','budget')) or 'Error' in line:
                print(folder.name,'|',line,flush=True)
        seen[arm]=len(lines)
        if proc.poll() is None and time.monotonic()-SESSION_T0>=SESSION_HOURS*3600:proc.terminate()
        if proc.poll() is None and time.monotonic()-SESSION_T0>=(SESSION_HOURS+.05)*3600:proc.kill()
        if proc.poll() is None:continue
        log.close();del active[gpu]
        path=folder/'result.json';result=json.loads(path.read_text()) if path.exists() else {}
        rows=result.get('history',[])
        assert len(rows)>=previous and [r['round'] for r in rows]==list(range(1,len(rows)+1))
        summary[folder.name]=dict(rc=proc.returncode,previous_round=previous,rounds=len(rows),
            new_rounds=len(rows)-previous,complete=len(rows)==20 and proc.returncode==0,target_rounds=20)
    if not active and pending and SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600<.20+1.15*WORST_SECONDS/3600:break
    if active or pending:time.sleep(2)
for arm in pending:
    name=f'{arm}_s{SEED}';previous=EXPECTED.get(name,{}).get('last_round',0)
    summary[name]=dict(previous_round=previous,rounds=previous,new_rounds=0,complete=False,
        target_rounds=20,reason='not started: session budget; preserve imported calibration if present')
(OUT/'utility_summary.json').write_text(json.dumps(summary,indent=2))
print('UTILITY BATCH',summary,flush=True)
assert len(summary)==20 and all(v.get('rc',0)==0 for v in summary.values()),'worker failure; preserve output'
print('ALL20 COMPLETE' if all(v['complete'] for v in summary.values()) else 'STAGED: utility gate remains pending',flush=True)
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(42,43),required=True)
    parser.add_argument('--hours',type=float,required=True)
    args=parser.parse_args()
    base=ROOT/'papers/boundary-cells'
    parent=f'bc-cal-s{args.seed}-20261013-initial2-v1'
    folder=base/'notebook'/parent
    metadata=json.loads((folder/'kernel-metadata.json').read_text())
    manifest=json.loads((folder/'source_manifest.json').read_text())
    record=json.loads((base/'kaggle'/parent/'AUDITED.json').read_text())
    assert record['calibration_valid'] and all(x['cost']['pass_cost'] for x in record['audits'])
    expected={};worst=0
    for item in record['audits']:
        run=base/'kaggle'/parent/'runs'/item['run'];identity=json.loads((run/'config.json').read_text())
        expected[item['run']]={k:identity[k] for k in ('fingerprint','source','fixture')}
        expected[item['run']]['last_round']=item['rounds']
        for marker in (run/'complete').glob('*.json'):worst=max(worst,json.loads(marker.read_text())['sec'])
        for name,sha in identity['source'].items():
            file=ROOT/name if name.startswith('mh/') else base/name
            assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
    reference=nbf.read(folder/metadata['code_file'],as_version=4)
    setup=reference.cells[2].source
    # Import gate precedes fixture decoding and the real-data GPU preflight.
    adapter=(ROOT/current_path('scripts/kaggle/boundary_logging.py')).read_text()
    importer=(ROOT/current_path('scripts/kaggle/boundary_resume.py')).read_text()
    verifier=(ROOT/current_path('scripts/kaggle/collect_boundary_calibration.py')).read_text()
    insert=f'EXPECTED={expected!r}\nADAPTER={adapter!r}\nIMPORTER={importer!r}\nVERIFIER={verifier!r}\n'+r'''
(WORK/'boundary_logging.py').write_text(ADAPTER)
(WORK/'boundary_resume.py').write_text(IMPORTER)
(WORK/'scripts/kaggle').mkdir(parents=True,exist_ok=True)
(WORK/'scripts/kaggle/collect_boundary_calibration.py').write_text(VERIFIER)
for name,body in [('boundary_logging.py',ADAPTER),('boundary_resume.py',IMPORTER),('boundary_verify.py',VERIFIER)]:
    (SOURCE/name).write_text(body)
sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'papers/boundary-cells/proj'))
from boundary_resume import import_run
from scripts.kaggle.collect_boundary_calibration import verify
imports=[import_run('/kaggle/input',OUT,name,value,manifest,verify) for name,value in EXPECTED.items()]
(SOURCE/'resume-import.json').write_text(json.dumps(imports,indent=2))
print('REQUIRE RESUME / COST PASS before fixture decoding',imports,flush=True)
'''
    setup=setup.replace("fixture=json.loads((SRC/'manifest.json').read_text())",insert+"\nfixture=json.loads((SRC/'manifest.json').read_text())")
    methods=['bc_single','bc_local','bc_mean','bc_gaussian']+[f'bc_{mode}{p}_{scope}'
        for mode in ('direct','teacher') for p in (2,4) for scope in ('fl','own','shuffle','marginal')]
    rid=f'bc-u-s{args.seed}-20261014-stage01-v1'
    public=base/'notebook'/rid;private=base/'private_push'/rid
    assert not public.exists() and not private.exists()
    notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# BoundaryCells — locked paired utility\n'
        '20 arms, fixed20schedule, train-only group holdouts; no official test. Five arms resume audited calibrationr2. '
        'Controls fresh. No target/HP/gate changes. Session budget can stage incomplete arms; no utility PASS before all controls/r20. '
        'Mandatory online W&B/private inline key persists in private version history.'),
        nbf.v4.new_code_cell('import time\nSESSION_T0=time.monotonic()\n'+reference.cells[1].source),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(
            f'SEED={args.seed}\nSESSION_HOURS={args.hours!r}\nMETHODS={methods!r}\nWORST_SECONDS={worst!r}\n'+RUN)])
    notebook.metadata=reference.metadata.copy()
    filename=rid+'.ipynb';owner=metadata['id'].split('/')[0]
    meta=dict(metadata,id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,kernel_sources=[record['kernel']])
    nbf.validate(notebook)
    assert len(meta['title'])<=50
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    public.mkdir(parents=True);private.mkdir(parents=True,mode=0o700)
    nbf.write(notebook,public/filename)
    for name,value in [('kernel-metadata.json',meta),('source_manifest.json',manifest),('resume-plan.json',expected)]:
        (public/name).write_text(json.dumps(value,indent=2)+'\n')
    subprocess.run(['git','check-ignore','-q',str((private/filename).resolve())],cwd=ROOT,check=True)
    key=netrc.netrc().authenticators('api.wandb.ai')[2];assert key
    credential=nbf.reads(nbf.writes(notebook),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for name,value in [(filename,nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as file:file.write(value)
    print(meta['id'],'20 arms; calibration parents verified; session_hours',args.hours,'prepared; not pushed')


if __name__=='__main__':main()
