"""Terminal exact-source audit and resumable collection; stop polling once stable, collect again at terminal."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
import torch

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from rank_verify import verify
from mh.dagsnet import build_model,CFG
from mh.model_packet import Codec


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(seed,attempt='v1'):
    rid=f'cr-u-s{seed}-20261015-{attempt}';folder=BASE/'notebooks'/rid;dest=BASE/'artifacts/kaggle'/rid;marker=dest/'AUDITED.json'
    if not (folder/'PUSHED').exists():return None
    if marker.exists():return json.loads(marker.read_text())
    meta=json.loads((folder/'kernel-metadata.json').read_text());helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
    prefix=[sys.executable,str(helper/'kaggle_as.py'),meta['id'].split('/')[0],'--']
    status=subprocess.run(prefix+['kaggle','kernels','status',meta['id']],check=True,capture_output=True,text=True)
    print(status.stdout.strip(),flush=True)
    if not any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE','ERROR')):return None
    subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),meta['id'],'-p',str(dest),'-j','6',
        '--pattern',r'^(source/|runs/(?![^/]+/wandb/))'],check=True)
    private=BASE/'private_push'/rid;executed=private/'executed.ipynb'
    subprocess.run(prefix+[sys.executable,str(helper/'download_executed_notebook.py'),meta['id'],'--version','1','--output',str(executed)],check=True,umask=0o077);executed.chmod(0o600)
    reference=nbformat.read(folder/meta['code_file'],as_version=4);pushed=nbformat.read(private/meta['code_file'],as_version=4)
    key=next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
    actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4);assert len(actual.cells)==len(reference.cells)
    for i,c in enumerate(actual.cells):
        if c.cell_type=='code':assert c.source==pushed.cells[i].source.replace(key,'[REDACTED]')
    actual.cells[1].source=reference.cells[1].source;actual.metadata['credential_cell_redacted']=True;nbformat.write(actual,folder/'executed.redacted.ipynb')
    manifest=json.loads((folder/'source_manifest.json').read_text());assert manifest==json.loads((dest/'source/source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(dest/'source'/name)==digest
    for name in ('warm-proof.json','input_manifest.json'):assert json.loads((folder/name).read_text())==json.loads((dest/'source'/name).read_text())
    if (folder/'recovery.json').exists():
        assert json.loads((folder/'recovery.json').read_text())==json.loads((dest/'source/recovery.json').read_text())
        assert sha(dest/'source/rank_bytes_compat.py')==sha(BASE/'scripts/rank_bytes_compat.py')
        resume=json.loads((dest/'source/resume_manifest.json').read_text())
        assert resume==json.loads((BASE/'artifacts/input'/rid/'resume_manifest.json').read_text())
        for rel,digest in resume.items():
            # result.json/metrics.csv are rewritten by the resumed driver; committed round/config/payload files stay identical.
            if not rel.endswith(('/result.json','/metrics.csv')):assert sha(dest/rel)==digest,rel
    inputs=json.loads((folder/'input_manifest.json').read_text());warm_proof=json.loads((folder/'warm-proof.json').read_text())
    fx=ROOT/'papers/pilot/fixture';fixture=sha(fx/'manifest.json');assert sha(dest/'source/fixture_manifest.json')==fixture
    audits=[]
    with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for run in sorted((dest/'runs').glob(f'cr_*_s{seed}')):
            if not list((run/'complete').glob('round_*.json')):continue
            config=json.loads((run/'config.json').read_text())
            assert config['fork']['sha256']==inputs['warm_weights.pt'] and config['fork']['fingerprint']==warm_proof['fingerprint']
            # PyTorch 2.10 serializes built-in bytes as GLOBAL bytes; this scope
            # permits that inert type only, keeping weights_only=True and source frozen.
            with torch.serialization.safe_globals([bytes]):
                audits.append(verify(run,manifest,fixture,fx))
            model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
            codec=Codec(model,f'v8:h_dml:{seed}') if run.name==f'cr_dml_s{seed}' else None
            for file in sorted((run/'complete').glob('round_*.json')):
                row=json.loads(file.read_text());r=row['round']
                if 'payload_sha' in row:
                    payload=run/f'knowledge/round_{r:03d}.bin';assert sha(payload)==row['payload_sha']
                    if codec:
                        state,base,version=codec.decode(payload.read_bytes(),expected_base=r-1);assert version==r
                        weights=torch.load(run/f'weights/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                        assert all(all(torch.equal(v,sd[k]) for k,v in state.items()) for sd in weights['proxy'])
    cp=dest/'runs/cost_gate.json';cost=json.loads(cp.read_text()) if cp.exists() else None
    if cost:
        rows={a:json.loads((dest/f'runs/{a}_s{seed}/complete/round_003.json').read_text()) for a in ('cr_local','cr_fl')}
        ratios={k:rows['cr_fl'][k]/rows['cr_local'][k] for k in ('algorithm_seconds','sec')}
        assert ratios==cost['ratios'] and cost['pass_cost']==all(v<=2 for v in ratios.values())
    summary=dest/'runs/utility_summary.json'
    record=dict(kernel=meta['id'],version=1,source=manifest,audits=audits,cost=cost,
        summary=json.loads(summary.read_text()) if summary.exists() else None,
        utility_r20=len(audits)==8 and all(a['complete'] for a in audits),
        limit='No independent saved-weight inference, raw full-gradient recomputation or NNLS refit; inherited CE-r2 has original provenance, not newly measured gp metrics')
    marker.write_text(json.dumps(record,indent=2)+'\n');print('AUDITED rank utility',seed,'allr20',record['utility_r20'],flush=True)
    return record


if __name__=='__main__':
    torch.set_num_threads(1)
    for seed in (42,43):
        for folder in sorted((BASE/'notebooks').glob(f'cr-u-s{seed}-20261015-v*')):
            collect(seed,folder.name.rsplit('-',1)[1])
