"""Owner-approved bounded diagnostic; reuse audited inputs/CPU proof, preserve frozen probe source/gates."""
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

RUN=r'''
wb=wandb.init(project='mobihet-cross-client-rank-probe',id=f'ccr_diag_s{SEED}-{DIAGNOSTIC_SHA[:8]}',
    config=dict(seed=SEED,diagnostic_sha=DIAGNOSTIC_SHA,parent=REPLAY['parent'],no_training=True),mode='online',
    settings=wandb.Settings(save_code=False,disable_code=True))
assert not wb.settings._offline
active=[]
for shard in (0,1):
    assert time.monotonic()-SESSION_T0<SESSION_HOURS*3600-120,'budget exhausted before diagnostic'
    log=open(OUT/f'diagnostic_gpu{shard}.log','w')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),PYTHONPATH=str(WORK),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([sys.executable,str(WORK/'proj/rank_projection_diagnostic.py'),'--input',str(SRC),
        '--out',str(OUT),'--seed',str(SEED),'--shard',str(shard)],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    active.append((p,log))
while active:
    assert time.monotonic()-SESSION_T0<SESSION_HOURS*3600,'diagnostic budget exhausted'
    for p,log in list(active):
        if p.poll() is None:continue
        log.close();active.remove((p,log));assert p.returncode==0,('diagnostic worker failed',p.returncode)
    if active:time.sleep(2)
rows=[json.loads((OUT/f'diagnostic_{k:02d}.json').read_text()) for k in range(20)]
assert all(r['original_energies_reproduced'] for r in rows)
result=dict(seed=SEED,parent=REPLAY['parent'],all_original_energies_reproduced=True,models=20,
    diagnostic_sha=DIAGNOSTIC_SHA,model_training=False,holdout_read=False,gate_changed=False,
    elapsed_seconds=time.monotonic()-SESSION_T0)
(OUT/'diagnostic_result.json').write_text(json.dumps(result,indent=2))
wb.log(result);wb.finish()
print('DIAGNOSTIC COMPLETE',result,flush=True)
'''


def main():
    seed=int(sys.argv[1]);parent=f'ccr-probe-s{seed}-20261015-v1';rid=f'ccr-diag-s{seed}-20261015-v1'
    folder=BASE/'notebooks'/parent;artifact=BASE/'artifacts/kaggle'/parent
    audit=json.loads((artifact/'AUDITED.json').read_text());assert audit['result']['pass_probe']
    cpu=json.loads((folder/'CPU_MOUNT_VERIFIED.json').read_text());assert cpu['models']==20
    replay=dict(parent=audit['kernel'],audit_sha=hashlib.sha256((artifact/'AUDITED.json').read_bytes()).hexdigest(),
        cohort=(artifact/'runs/cohort.bin').read_bytes().hex(),up=[(artifact/f'runs/client_{k:02d}.bin').read_bytes().hex() for k in range(20)],
        clients=[json.loads((artifact/f'runs/client_{k:02d}_probe.json').read_text()) for k in range(20)])
    diagnostic=(BASE/'scripts/rank_projection_diagnostic.py').read_text();digest=hashlib.sha256(diagnostic.encode()).hexdigest()
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4)
    nb.cells[0].source='# CrossClientRank projection diagnostic\nOwner locked .25h/seed; same weights/sample/basis/rcond/gates. No SGD or utility. Private inline W&B key persists in private version history.'
    nb.cells[2].source+=f'\nDIAGNOSTIC={diagnostic!r}\nDIAGNOSTIC_SHA={digest!r}\nREPLAY={replay!r}\n'+r'''
assert hashlib.sha256(DIAGNOSTIC.encode()).hexdigest()==DIAGNOSTIC_SHA
(WORK/'proj/rank_projection_diagnostic.py').write_text(DIAGNOSTIC)
(SOURCE/'rank_projection_diagnostic.py').write_text(DIAGNOSTIC)
(SOURCE/'diagnostic_manifest.json').write_text(json.dumps(dict(sha256=DIAGNOSTIC_SHA,parent=REPLAY['parent'],audit_sha=REPLAY['audit_sha'])))
(OUT/'replay.json').write_text(json.dumps(REPLAY))
'''
    nb.cells[3].source=f'SEED={seed}\nSESSION_HOURS=.25\n'+RUN
    public=BASE/'notebooks'/rid;private=BASE/'private_push'/rid
    assert not public.exists() and not private.exists()
    name=rid+'.ipynb';owner=meta['id'].split('/')[0];meta.update(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=name)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,rid,'exec')
    public.mkdir();private.mkdir(mode=0o700);nbformat.write(nb,public/name)
    (public/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    for name_ in ('source_manifest.json','input_manifest.json','protocol.json','warm-proof.json','dataset-files-verified.json'):
        shutil.copyfile(folder/name_,public/name_)
    (public/'CPU_MOUNT_VERIFIED.json').write_text(json.dumps(dict(cpu,reused_for_identical_inputs=True,parent=audit['kernel']),indent=2)+'\n')
    (public/'diagnostic-manifest.json').write_text(json.dumps(dict(sha256=digest,parent=audit['kernel'],audit_sha=replay['audit_sha']),indent=2)+'\n')
    subprocess.run(['git','check-ignore','-q',str((private/name).resolve())],cwd=ROOT,check=True)
    key=netrc.netrc().authenticators('api.wandb.ai')[2];credential=nbformat.reads(nbformat.writes(nb),as_version=4)
    credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
    for filename,value in [(name,nbformat.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
        fd=os.open(private/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(value)
    print(meta['id'],'prepared; identical input dataset; no rescan/SGD; .25h')


if __name__=='__main__':main()
