"""Owner-scoped V2 pull + immutable source audit + complete artifact verification."""
import argparse
import ast
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
import nbformat
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2]); H=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from scripts.kaggle.gen_v2 import FAMILIES
from mh.new_fl import digest
from verify_v2 import verify


def collect(family,build='v2',jobs=12):
    base=ROOT/f'papers/{build}'; families=FAMILIES; verifier=verify
    if build=='v3':
        from mh.v3_math import FAMILIES as families
        from verify_v3 import verify as verifier
    elif build=='v4':
        from mh.v4_run import FAMILIES as families
        from verify_v4 import verify as verifier
    elif build=='v5':
        from mh.v5_run import FAMILIES as families
        from verify_v5 import verify as verifier
    elif build=='v6':
        from mh.v6_run import FAMILIES as families
        from verify_v6 import verify as verifier
    elif build=='v7':
        from mh.v7_run import FAMILIES as families
        from verify_v7 import verify as verifier
    elif build=='v8':
        from mh.v8_run import FAMILIES as families
        from verify_v8 import verify as verifier
    elif build in ('v9','v10'):
        if build=='v9':
            from mh.v9_run import FAMILIES as families
        else:
            from mh.v10_run import FAMILIES as families
        from verify_teacher import verify as verifier
    completed=0
    for folder in sorted((base/'notebook').glob(f'{build}{family}-s*')):
        if not (folder/'PUSHED').exists(): continue
        meta=json.loads((folder/'kernel-metadata.json').read_text()); kernel=meta['id']; owner=kernel.split('/')[0]
        dest=base/'kaggle'/folder.name; marker=dest/'AUDITED.json'
        expected=json.loads((folder/'source_manifest.json').read_text())
        if marker.exists():
            old=json.loads(marker.read_text()); assert old['kernel']==kernel and old['source']==expected
            completed+=1; continue
        prefix=[sys.executable,str(H/'kaggle_as.py'),owner,'--']
        status=subprocess.run(prefix+['kaggle','kernels','status',kernel],check=True,capture_output=True,text=True)
        print(status.stdout.strip(),flush=True)
        if not any(s in status.stdout for s in ('COMPLETE','ERROR')): continue
        subprocess.run(prefix+[sys.executable,str(H/'pull_kernel_output.py'),kernel,'-p',str(dest),'-j',str(jobs),
                              '--pattern',r'^(runs|source)/'],check=True)
        private=base/'private_push'/folder.name
        executed=private/'executed.ipynb'  # raw executed source contains key, NEVER public audit tree
        subprocess.run(prefix+[sys.executable,str(H/'download_executed_notebook.py'),kernel,
                              '--version','1','--output',str(executed)],check=True)
        executed.chmod(0o600)
        nb=nbformat.read(executed,as_version=4)
        reference=nbformat.read(folder/meta['code_file'],as_version=4)
        pushed=nbformat.read(private/meta['code_file'],as_version=4)
        assignments=ast.parse(pushed.cells[1].source).body
        keys=[node.value.value for node in assignments if isinstance(node,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in node.targets)]
        assert len(keys)==1 and keys[0]
        cleaned=nbformat.reads(nbformat.writes(nb).replace(keys[0],'[REDACTED]'),as_version=4)
        assert len(cleaned.cells)==len(reference.cells)
        for i,cell in enumerate(cleaned.cells):
            if cell.cell_type=='code' and i!=1: assert cell.source==reference.cells[i].source,('executed source drift',i)
        cleaned.cells[1].source=reference.cells[1].source
        cleaned.metadata['credential_cell_redacted']=True
        nbformat.write(cleaned,folder/'executed.redacted.ipynb')
        source=dest/'source'; manifest=json.loads((source/'source_manifest.json').read_text())
        assert manifest==expected
        for name,sha in manifest.items(): assert digest(source/name)==sha,('source mismatch',name)
        for name,path in (('dagsnet.py',ROOT/'mh/dagsnet.py'),('metrics.py',ROOT/'mh/metrics.py'),
                          ('v2_math.py',ROOT/'mh/v2_math.py'),('meta.json',ROOT/'knowledge/meta.json')):
            assert digest(path)==expected[name],('verifier source drift',name)
        if build=='v3':
            for name in ('v3_math.py','v3_methods.py','v3_dense.py','v3_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('V3 source drift',name)
        elif build=='v4':
            for name in ('v4_mala.py','v4_support.py','v4_schema.py','v4_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('V4 source drift',name)
        elif build=='v5':
            for name in ('v5_anchor.py','v5_books.py','v5_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('V5 source drift',name)
        elif build=='v6':
            for name in ('v6_geometry.py','v6_schema.py','v6_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('V6 source drift',name)
        elif build=='v7':
            for name in ('v7_basis.py','v7_nystrom.py','v7_run.py','v6_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('V7 source drift',name)
        elif build=='v8':
            for name in ('v8_hybrid.py','v8_run.py','model_packet.py','v7_nystrom.py','v7_basis.py','v6_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('V8 source drift',name)
        elif build in ('v9','v10'):
            specific=('v9_anchor.py','v9_run.py') if build=='v9' else ('v10_energy.py','v10_run.py')
            for name in (*specific,'teacher_driver.py','teacher_base.py','model_packet.py','v6_run.py','v2_methods.py','v2_run.py'):
                assert digest(ROOT/'mh'/name)==expected[name],('teacher source drift',name)
        seed=int(folder.name.split('-')[1][1:])
        summary=json.loads((dest/'runs'/f'summary_s{seed}.json').read_text())
        assert set(summary)==set(families[family]),summary
        logpath=dest/(kernel.split('/')[1]+'.log')
        records=json.loads(logpath.read_text()); log=''.join(v.get('data','') for v in records)
        assert "gpus ['Tesla T4', 'Tesla T4']" in log and 'W&B authenticated online' in log and 'verified source and fixture' in log
        audits=[]
        for arm in families[family]:
            run=dest/'runs'/f'{arm}_s{seed}'; identity=json.loads((run/'config.json').read_text())
            assert identity['config']['arm']==arm and identity['config']['seed']==seed
            assert not identity['config']['smoke'] and identity['config']['rounds']==20
            assert all(expected[name]==sha for name,sha in identity['source'].items())
            finished=summary[arm]['rc']==0 and summary[arm]['complete']
            if list((run/'complete').glob('round_*.json')):
                # One artifact inference process at a time on local 4GB GPU.
                with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    audit=verifier(run,complete=finished,infer=True)
                    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
            else:
                assert not finished,'finished arm without committed rounds'
                audit=dict(run=run.name,rounds=0,complete=False,inference=[],
                    level='source/config audited; failure before any committed round, no model evidence')
            audit.update(arm=arm,complete=finished,returncode=summary[arm]['rc'])
            if build in ('v5','v9','v10'): audit['cost_pass']=summary[arm].get('cost_pass')
            if not finished:
                tail='\n'.join((run/'log.txt').read_text().splitlines()[-25:])
                audit.update(failure_tail=tail,failure_category='numerical' if
                    any(s in tail for s in ("'proxy loss'","'private loss'",'in finite')) else 'unknown')
                if summary[arm].get('cost_stopped') or 'COST PREFLIGHT REJECT' in tail: audit['failure_category']='cost'
                elif 'COST BASELINE MISSING' in tail: audit['failure_category']='baseline'
                elif 'kernel coverage gate failed' in tail: audit['failure_category']='coverage'
                elif 'nonfinite condensation objective' in tail: audit['failure_category']='numerical'
                elif 'synthetic book exactly equals' in tail: audit['failure_category']='privacy'
                elif 'V6 synthetic fp16 input overflow' in tail or 'linalg.cholesky' in tail: audit['failure_category']='numerical'
            audits.append(audit)
            target=base/'runs'/run.name; target.parent.mkdir(exist_ok=True)
            if target.is_symlink(): assert target.resolve()==run.resolve()
            else: assert not target.exists(); target.symlink_to(run.resolve(),target_is_directory=True)
        record=dict(kernel=kernel,version=1,seed=seed,family=family,
            runs=len(audits),source=expected,summary=summary,audits=audits,
            complete=all(v['complete'] and v['rc']==0 for v in summary.values()))
        marker.write_text(json.dumps(record,indent=2))
        if record['complete']: (dest/'VERIFIED.json').write_text(marker.read_text())
        print('AUDITED KERNEL',kernel,'complete',record['complete'],flush=True); completed+=1
    return completed


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(); p.add_argument('--family',required=True)
    p.add_argument('--build',choices=['v2','v3','v4','v5','v6','v7','v8','v9','v10'],default='v2')
    p.add_argument('--jobs',type=int,default=12,choices=range(1,13),help='download connections per kernel')
    p.add_argument('--watch',action='store_true'); a=p.parse_args()
    with (ROOT/f'papers/{a.build}/collection-{a.family}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while collect(a.family,a.build,a.jobs)<2 and a.watch: time.sleep(55)
