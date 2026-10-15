"""Collect full calibration outputs; report actual committed rounds, never call3-of50 complete."""
import ast
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import nbformat
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
H=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
BASE=ROOT/'papers/hybrid-full'
sys.path.insert(0,str(ROOT))
import scripts
sys.path.insert(0,str(BASE/'proj'))
import full_verify
full_verify.ROOT=ROOT
from full_verify import verify


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    total=0
    for folder in sorted((BASE/'notebook').iterdir()):
        if not (folder/'PUSHED').exists():continue
        meta=json.loads((folder/'kernel-metadata.json').read_text());kernel=meta['id'];owner=kernel.split('/')[0]
        dest=BASE/'kaggle'/folder.name;marker=dest/'AUDITED.json'
        if marker.exists():total+=1;continue
        prefix=[sys.executable,str(H/'kaggle_as.py'),owner,'--']
        status=subprocess.run(prefix+['kaggle','kernels','status',kernel],check=True,capture_output=True,text=True)
        print(status.stdout.strip(),flush=True)
        if not any(s in status.stdout for s in ('COMPLETE','ERROR')):continue
        subprocess.run(prefix+[sys.executable,str(H/'pull_kernel_output.py'),kernel,'-p',str(dest),'-j','6',
            '--pattern',r'^(source/|runs/(?![^/]+/wandb/))'],check=True)
        private=BASE/'private_push'/folder.name;executed=private/'executed.ipynb'
        subprocess.run(prefix+[sys.executable,str(H/'download_executed_notebook.py'),kernel,'--version','1',
            '--output',str(executed)],check=True,umask=0o077)
        executed.chmod(0o600)
        pushed=nbformat.read(private/meta['code_file'],as_version=4)
        nodes=ast.parse(pushed.cells[1].source).body
        key=next(n.value.value for n in nodes if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
        reference=nbformat.read(folder/meta['code_file'],as_version=4)
        actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4)
        assert len(actual.cells)==len(reference.cells)
        for i,cell in enumerate(actual.cells):
            if cell.cell_type=='code' and i!=1:assert cell.source==reference.cells[i].source,('executed source drift',i)
        actual.cells[1].source=reference.cells[1].source;actual.metadata['credential_cell_redacted']=True
        nbformat.write(actual,folder/'executed.redacted.ipynb')
        expected=json.loads((folder/'source_manifest.json').read_text());source=dest/'source'
        audits=[]
        logpath=dest/(kernel.split('/')[1]+'.log')
        log=''.join(v.get('data','') for v in json.loads(logpath.read_text()))
        assert key not in log
        if (source/'source_manifest.json').exists():
            assert json.loads((source/'source_manifest.json').read_text())==expected
            for name,sha in expected.items():assert digest(source/name)==sha,('source mismatch',name)
        else:
            marker.write_text(json.dumps(dict(kernel=kernel,version=1,source=expected,complete=False,
                phase='failure before verified source/data',failure_tail=log[-3000:]),indent=2))
            total+=1;continue
        if not (dest/'runs/summary_s42.json').exists():
            marker.write_text(json.dumps(dict(kernel=kernel,version=1,source=expected,complete=False,
                phase='failure before driver summary',failure_tail=log[-3000:]),indent=2))
            total+=1;continue
        summary=json.loads((dest/'runs/summary_s42.json').read_text())
        assert "gpus ['Tesla T4', 'Tesla T4']" in log and 'W&B authenticated online' in log
        for arm,item in summary.items():
            run=dest/'runs'/f'{arm}_s42'
            if not (run/'config.json').exists():
                audits.append(dict(arm=arm,rounds=0,complete=False,failure_tail=(run/'log.txt').read_text()[-3000:]));continue
            identity=json.loads((run/'config.json').read_text())
            for name,sha in identity['source'].items():assert expected[name.split('/',1)[1]]==sha
            rows=list((run/'complete').glob('round_*.json'))
            if rows:
                with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);audit=verify(run,complete=False,infer=False)
                assert audit['rounds']==item['rounds']
                audit.update(arm=arm,calibration_complete=item.get('calibration_complete',False),
                             continuation_verified=item.get('continuation_verified',False))
                assert 'COMPILE PREFLIGHT PASS' in (run/'log.txt').read_text()
                audits.append(audit)
            else:audits.append(dict(arm=arm,rounds=0,complete=False,failure_tail=(run/'log.txt').read_text()[-3000:]))
        record=dict(kernel=kernel,version=1,source=expected,summary=summary,audits=audits,
            calibration_complete=all(v.get('calibration_complete',False) for v in summary.values()),
            continuation_verified=all(v.get('continuation_verified',False) for v in summary.values()),
            complete=all(v.get('complete',False) for v in summary.values()))
        marker.write_text(json.dumps(record,indent=2));total+=1
        print('AUDITED HYBRID FULL CALIBRATION',kernel,record['calibration_complete'],flush=True)
    return total


if __name__=='__main__':
    torch.set_num_threads(1)
    required=sum((p/'PUSHED').exists() for p in (BASE/'notebook').iterdir())
    while collect()<required:time.sleep(55)
