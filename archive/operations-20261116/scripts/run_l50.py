"""Lossless Core seed42 r1..50; reuse F50 launch/resume/collector primitives, isolated receipts."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_f50 as F, run_w20 as W, f50_common
from scripts.watch_kaggle_batch import atomic_json, query
OUT, ART=ROOT/'runs/l50-study',ROOT/'artifacts/l50'
OVERLAY=['methods/mobihet.py','methods/mobihet_lossless.py']


def notebook(owner,kernel,run,hp,head,build,cap,resume=None,until=25):
    if (OUT/'batch.json').exists():
        frozen=json.loads((OUT/'batch.json').read_text())[0]['overlay_sha256']
        assert all(W.sha(ROOT/n)==h for n,h in frozen.items())
    args=shlex.join(['scripts/run_q20.py','--method','mobihet_lossless','--out',f'/kaggle/working/{run}',
                    '--width','1','--rounds','50','--until',str(until),'--save-every','1','--compile','--wandb','mobihet-v4',
                    '--max-hours','BUDGET','--cache','','--set',*hp])
    subprocess.run([sys.executable,'scripts/gen_notebook.py','--owner',owner,'--slug',kernel,'--run',run,
                    '--sha',head,'--max-hours',str(cap),'--args',args,'--npy-kernel',build['ref'],
                    '--npy-manifest-sha256',build['manifest_sha256'],'--overlay',*OVERLAY],check=True,capture_output=True)
    folder=ROOT/'notebooks'/kernel; nb=folder/(kernel+'.ipynb'); metadata=folder/'kernel-metadata.json'
    doc=json.loads(nb.read_text())
    for c in doc['cells']:
        if c['cell_type']=='code':c['source']=''.join(c['source']).replace('TORCHINDUCTOR_COMPILE_THREADS="2"','TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
    if resume:
        code = f"""import tarfile
checkpoint_files = list(Path('/kaggle/input').glob('**/l50-resume.bin'))
assert len(checkpoint_files) == 1, checkpoint_files
checkpoint = checkpoint_files[0]
with checkpoint.open('rb') as fh:
    assert hashlib.file_digest(fh, 'sha256').hexdigest() == {resume['archive_sha256']!r}
with tarfile.open(checkpoint, 'r') as tar:
    tar.extractall('/kaggle/working', filter='data')
for name in ('state.pt','state_rank1.pt'):
    path = Path('/kaggle/working/{run}') / name
    assert path.is_file()
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert state['round'] == {resume['round']} and 'rng_cpu' in state and 'rng_cuda' in state
    if name == 'state.pt':
        assert state['method'] == 'mobihet_lossless' and state['hp']['seed'] == 42 and state['hp']['rounds'] == 50
    del state
print('resumed from verified private dataset', {resume['ref']!r}, 'round', {resume['round']})
"""
        index=next(i for i,c in enumerate(doc['cells']) if 'NPY, =' in ''.join(c.get('source',[])))
        doc['cells'].insert(index,dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True)))
        meta=json.loads(metadata.read_text());meta['dataset_sources'].append(resume['ref'])
        metadata.write_text(json.dumps(meta,indent=1)+'\n')
    nb.write_text(json.dumps(doc,indent=1)+'\n')
    subprocess.run([sys.executable,str(W.SKILL/'embed_wandb_key.py'),str(nb),'--metadata',str(metadata)],check=True,capture_output=True)
    meta=json.loads(metadata.read_text()); runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    assert meta['is_private'] and meta['id']==f'{owner}/{kernel}'
    assert meta['kernel_sources']==[build['ref']]
    if resume: assert resume['ref'] in meta['dataset_sources']
    assert all(meta[k]==runtime[k] for k in ('machine_shape','docker_image'))
    for c in json.loads(nb.read_text())['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),str(nb),'exec')
    return W.sha(nb),W.sha(metadata)


def prepare(build):
    if (OUT/'batch.json').exists():return json.loads((OUT/'batch.json').read_text())
    smoke=json.loads((OUT/'preflight.json').read_text())
    assert smoke['source_unchanged'] and all(W.sha(ROOT/n)==h for n,h in smoke['source_sha256'].items())
    assert all(r.get('bitexact',r.get('history_cms_snapshots_state_bitexact',False)) for r in smoke['reports'])
    transfer=json.loads((OUT/'transfer-preflight.json').read_text())
    assert transfer['exact_import_cell_pass'] and transfer['tamper_rejected']
    assert all(W.sha(ROOT/n)==h for n,h in transfer['source_sha256'].items())
    quota=W.inventory()
    owners=[q for q in sorted(quota['accounts'],key=lambda q:-q.get('gpu_left',0))
            if not(q.get('inventory_error') or q['active_recent'] or q['unknown_recent']) and q.get('gpu_left',0)>=6.5*1.15]
    assert len(owners)>=2, 'Need two owners each >=7.475 GPUh including margin'
    owner=owners[0]['user'];slug='mh-l0-core-lossless-s42-r50'
    second=owners[1]['user']
    atomic_json(OUT/'transfer-plan.json',dict(first_owner=owner,second_owner=second,first_until=25,second_until=50,cap_hours=6.5,quota_margin=1.15))
    # Inspect destination slot/identity/access before any first-session mutation.
    k=W.ka();q=k.account_quota(second);assert q['gpu_left']>=6.5*1.15
    refs=F.owned_refs(W.cli(second),second)
    assert f'{second}/{slug}-c2' not in refs
    for command in ('datasets files -d odixe0502/veremi-nextgen-v4-scaled-per-vehicle', 'kernels status odixe0502/mobihet-v4npy-cache'):
        subprocess.run(['timeout','90s',*W.cli(second),*command.split()],check=True,capture_output=True)
    hp=f50_common.hp_for('core',42)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True).split()[0]
    assert head==remote
    nb,meta=notebook(owner,slug,slug,hp,head,build,6.5)
    spec=dict(arm='L0',cell='core',mode='stream',r_kind='live',account=owner,slug=slug,version=1,hp=hp,
              base_sha=head,source_sha256=W.sha(ROOT/'methods/mobihet.py'),
              overlay_sha256={n:W.sha(ROOT/n) for n in OVERLAY},notebook_sha256=nb,metadata_sha256=meta,
              cap_hours=6.5,endpoint=50,save_every=1,session_until=25,second_account=second)
    atomic_json(OUT/'batch.json',[spec])
    atomic_json(ROOT/'docs/research/l50-launch.json',dict(batch=[spec],authorization='Owner04/11: losslessCDF, full50round, sameCore80/20, seed42',preflight_sha256=W.sha(OUT/'preflight.json'),quota_sha256=W.sha(OUT/'quota.json')))
    return [spec]


def watch(batch,build):
    spec=batch[0];session_path=OUT/'sessions.json'
    sessions=json.loads(session_path.read_text()) if session_path.exists() else [dict(account=spec['account'],kernel=spec['slug'],until=25)]
    atomic_json(session_path,sessions)
    progress=OUT/'progress.json'
    results=json.loads(progress.read_text()) if progress.exists() else {}
    (OUT/'collection').mkdir(exist_ok=True)
    while True:
        current=sessions[-1];kernel=current['kernel'];owner=current['account']
        state=query(dict(spec,account=owner,slug=kernel))
        atomic_json(OUT/'status.json',state)
        print('L50_STATUS',state,flush=True)
        if state['terminal']:
            assert W.shutil.disk_usage(ROOT).free>40*2**30,'Insufficient local disk for checkpoint transfer'
            rc=F.collect(dict(spec,account=owner),kernel)
            assert rc==0,'Pull failed; preserve training output, retry collector only'
            assert state['status']=='COMPLETE','Training ERROR: outputs pulled, inspect before recovery'
            done=F.last_round(spec)
            results[kernel]=dict(state=state,last_round=done,pulled=True)
            atomic_json(progress,results)
            assert done==current['until'],f'Session ended at {done}, expected {current["until"]}; audit checkpoint before retry'
            if done==25:
                second=spec['second_account']
                ready_path=OUT/'transfer/ready.json'
                if not ready_path.exists():
                    subprocess.run([*W.cli(second)[:4],sys.executable,'scripts/stage_l50.py','--owner',second,
                                    '--run',spec['slug'],'--round','25'],check=True)
                ready=json.loads(ready_path.read_text())
                assert ready['private'] and ready['download_sha_verified'] and ready['round']==25
                kernel2=spec['slug']+'-c2'
                nb,meta=notebook(second,kernel2,spec['slug'],spec['hp'],spec['base_sha'],build,6.5,resume=ready,until=50)
                second_spec=dict(spec,account=second,slug=kernel2,notebook_sha256=nb,metadata_sha256=meta,session_until=50,wandb_run_id=spec['slug'])
                # Reuse one-shot launch gates with receipts isolated from first session.
                second_out=OUT/'resume-launch';second_out.mkdir(exist_ok=True)
                W.OUT=second_out;atomic_json(second_out/'batch.json',[second_spec])
                W.launch([second_spec]);W.OUT=OUT
                sessions.append(dict(account=second,kernel=kernel2,until=50,checkpoint_sha256=ready['archive_sha256']))
                atomic_json(session_path,sessions)
                continue
            armout=OUT/'collection'/spec['arm'];armout.mkdir(exist_ok=True)
            atomic_json(armout/'batch.json',[dict(spec,sessions=[s['kernel'] for s in sessions])])
            subprocess.run([sys.executable,'scripts/audit_l50.py','--batch',str(armout/'batch.json'),
                            '--root',str(ART),'--out',str(armout)],check=True)
            atomic_json(OUT/'closure.json',dict(full50_pass=1,sessions=sessions,audit_sha256=W.sha(armout/'audit.json')))
            print('L50_CLOSED',flush=True)
            return
        startup=OUT/('startup-'+str(len(sessions)))/'startup.json'
        if not startup.exists() or not json.loads(startup.read_text()).get('all_ready'):
            subprocess.run([sys.executable,'scripts/check_screening_startup.py','--batch',str(OUT/'batch.json') if len(sessions)==1 else str(OUT/'resume-launch/batch.json'),
                            '--out',str(startup.parent),'--entity','21522798-uit'])
        time.sleep(1200)


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    F.OUT,F.ART,F.TAG=OUT,ART,'L50'
    W.OUT,W.ART,W.TAG=OUT,ART,'L50'
    W.COLLECT=r'(^provenance/|\.log$|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|external_cms_rank[01]_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|cdf_codec_rank[01]\.json|cdf_rank[01]\.bin|state(_rank1)?\.pt|teacher_probe_rank[01]_r\d{3}\.json)$|/models/r\d{3}\.pt$|/recovery/rank[01]\.jsonl$)'
    build=json.loads((ROOT/'runs/w-study/builder.json').read_text())
    batch=prepare(build);W.launch(batch);watch(batch,build)


if __name__=='__main__':main()
