"""Watch a frozen 20-round batch every20min, pull permitted artifacts, audit. No model calls or training launch."""
import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SKILL=ROOT/'.agents/skills/kaggle-training-notebook/scripts'


def collect(spec,root,logs):
    dest=root/spec['slug'];dest.mkdir(parents=True,exist_ok=True)
    # Source is redacted .py, not executed notebook. Small server snapshots exclude private banks.
    pattern=r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|external_cms_rank[01]_r\d{3}\.npz)$|/models/r\d{3}\.pt$)'
    cmd=[sys.executable,str(SKILL/'kaggle_as.py'),spec['account'],'--',sys.executable,str(ROOT/'scripts/run_bounded_python.py'),str(SKILL/'pull_kernel_output.py'),f'{spec["account"]}/{spec["slug"]}/{spec["version"]}','-p',str(dest),'-j','8','--pattern',pattern]
    with (logs/(spec['slug']+'.collect.log')).open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    d=dest/spec['slug'];required=[d/'config.json',d/'rounds.jsonl',dest/'provenance/source_manifest.json',dest/(spec['slug']+'.log')]
    required += [d/f'cms_r{r:03}.npz' for r in range(1,21)] + [d/'models'/f'r{r:03}.pt' for r in range(1,21)]
    required += [d/f'train_cms_r{r:03}.npz' for r in [1,2,5,10,15,20]]
    required += [d/f'diagnostics_rank{rank}.jsonl' for rank in [0,1]]
    required += [d/f'external_cms_rank{rank}_r{r:03}.npz' for rank in [0,1] for r in [1,10,20]]
    missing=[str(p.relative_to(dest)) for p in required if not p.is_file()]
    return {'arm':spec['arm'],'ref':f'{spec["account"]}/{spec["slug"]}/{spec["version"]}','returncode':result.returncode,'expected_artifacts':len(required),'missing':missing,'files':sum(p.is_file() for p in dest.rglob('*'))}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--entity',default='');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/watch_kaggle_batch.py'),'--batch',str(a.batch),'--out',str(a.out/'watch'),'--interval','1200','--stop-on-error'],cwd=ROOT)
    while a.entity:
        subprocess.run([sys.executable,str(ROOT/'scripts/check_screening_startup.py'),'--batch',str(a.batch),'--out',str(a.out/'startup'),'--entity',a.entity],check=True,cwd=ROOT)
        ready=json.loads((a.out/'startup/startup.json').read_text())['all_ready']
        if ready or proc.poll() is not None:break
        try:
            proc.wait(timeout=1200)
            break
        except subprocess.TimeoutExpired:pass
    code=proc.wait()
    assert code in [0,2],f'Watcher failed {code}'
    assert shutil.disk_usage(a.root).free>12*2**30,'Need12GiB free for8runs/snapshots'
    batch=json.loads(a.batch.read_text())
    if code==2:
        terminal=json.loads((a.out/'watch/attention.json').read_text())['runs']
        refs={r['ref'] for r in terminal if r['terminal']}
        batch=[r for r in batch if f'{r["account"]}/{r["slug"]}/{r["version"]}' in refs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda s:collect(s,a.root,a.out),batch))
    (a.out/'collection.json').write_text(json.dumps(rows,indent=2)+'\n')
    if code==2:
        print('TERMINAL_KERNEL_ERROR_COLLECTED; leave other kernels running',flush=True)
        return 2
    assert all(r['returncode']==0 and not r['missing'] for r in rows),rows
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_m20.py'),'--batch',str(a.batch),'--root',str(a.root),'--out',str(a.out)],check=True,cwd=ROOT)
    print('SCREENING20_PULL_AUDIT_DONE',flush=True)


if __name__=='__main__':sys.exit(main() or 0)
