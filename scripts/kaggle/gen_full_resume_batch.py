"""New continuations of audited full builds, preserving every scientific source byte."""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.kaggle.gen_full_validation import PLAN as FULL_PLAN
from scripts.kaggle.gen_hybrid_full import PLAN as HYBRID_PLAN

RUN = r'''
active={};summary={}
remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
worker_hours=remaining-.30
for arm in METHODS:
    prior=json.loads((OUT/f'{arm}_s42/result.json').read_text())
    worst=max(row['sec'] for row in prior['history'])/3600
    assert worker_hours>=1.15*worst+.20,'not enough time for setup and a complete resumed round'
for gpu,arm in enumerate(METHODS):
    folder=OUT/f'{arm}_s42'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(WORK/PROJECT_PATH/'full_run.py'),
        '--arm',arm,'--seed','42','--rounds','50','--eval_every','1','--backend','compiled',
        '--wandb_online','--wandb_project',WANDB_PROJECT,'--out',str(OUT),'--resume','--max_hours',str(worker_hours)]
    log=open(folder/'log.txt','w');process=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    active[gpu]=(arm,process,log)
while active:
    for gpu,(arm,process,log) in list(active.items()):
        if process.poll() is None and time.monotonic()-SESSION_T0>=SESSION_HOURS*3600:process.terminate()
        if process.poll() is None and time.monotonic()-SESSION_T0>=(SESSION_HOURS+.05)*3600:process.kill()
        if process.poll() is None:continue
        log.close();del active[gpu]
        path=OUT/f'{arm}_s42/result.json';result=json.loads(path.read_text()) if path.exists() else {}
        rows=result.get('history',[]);last=EXPECTED[arm]['last_round']
        assert [r['round'] for r in rows]==list(range(1,len(rows)+1)), 'noncontiguous committed history'
        assert len(rows)>=last,'continuation lost imported rounds'
        new=[r for r in rows if r['round']>last]
        summary[arm]=dict(rc=process.returncode,complete=result.get('complete',False),rounds=len(rows),
            previous_round=last,new_rounds=len(new),target_rounds=50,
            continuation_verified=bool(new),valid_budget_boundary=process.returncode==0,
            worst_round_seconds=max([r['sec'] for r in rows] or [0]))
    time.sleep(3)
(OUT/'summary_s42.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
assert all(v['rc']==0 for v in summary.values()),'worker failed; preserve committed rounds'
if not all(v['continuation_verified'] for v in summary.values()):
    print('NO NEW ROUND for some arms; valid boundary is not training completion',flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', choices=('full-validation', 'hybrid-full'), required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--hours', type=float, nargs=4, required=True)
    args = parser.parse_args()
    base = ROOT / '.compat/papers' / args.build
    plan = FULL_PLAN if args.build == 'full-validation' else HYBRID_PLAN
    template_folder = ROOT / '.compat/papers/full-validation/notebook/full1-s42-20261013-cont01-v1'
    template_meta = json.loads((template_folder / 'kernel-metadata.json').read_text())
    template = nbf.read(template_folder / template_meta['code_file'], as_version=4)
    setup_body = template.cells[2].source[template.cells[2].source.index('import hashlib'):]
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key
    runtime = json.loads((ROOT / 'knowledge/runtime.json').read_text())
    for index, ((owner, arms), hours) in enumerate(zip(plan, args.hours), 1):
        candidates = []
        for marker in (base / 'kaggle').glob('*/AUDITED.json'):
            record = json.loads(marker.read_text())
            if record['kernel'].split('/')[0] == owner and set(record.get('summary', {})) == set(arms):
                candidates.append((sum(v['rounds'] for v in record['audits']), marker, record))
        assert candidates, ('missing audited parent', args.build, owner, arms)
        _, marker, audited = max(candidates, key=lambda x: x[0])
        parent = marker.parent.name
        metadata = json.loads((base / 'notebook' / parent / 'kernel-metadata.json').read_text())
        manifest = json.loads((base / 'notebook' / parent / 'source_manifest.json').read_text())
        expected = {}
        for arm in arms:
            run = marker.parent / 'runs' / f'{arm}_s42'
            identity = json.loads((run / 'config.json').read_text())
            audit = next(v for v in audited['audits'] if v['arm'] == arm)
            assert audit['rounds'] > 0 and not audit['complete']
            expected[arm] = {k: identity[k] for k in ('fingerprint', 'source', 'fixture')}
            expected[arm]['last_round'] = audit['rounds']
            for path, sha in identity['source'].items():
                file = ROOT / path if path.startswith('mh/') else base / path
                assert hashlib.sha256(file.read_bytes()).hexdigest() == sha, ('frozen source drift', path)
        prefix = 'full' if args.build == 'full-validation' else 'hybf'
        rid = f'{prefix}{index}-s42-{args.tag}-v1'
        public = base / 'notebook' / rid
        private = base / 'private_push' / rid
        assert not public.exists() and not private.exists(), 'new immutable tag required'
        code = [n for n in manifest if (ROOT / '.compat/mh' / n).exists()]
        project = [p.name for p in (base / 'proj').glob('*.py')]
        body = setup_body.replace('papers/full-validation/proj', f'papers/{args.build}/proj')
        setup = f'CODE={code!r}\nPROJ={project!r}\nEXPECTED={expected!r}\n' + body
        run = (f'METHODS={arms!r}\nSESSION_HOURS={hours!r}\n'
               f'PROJECT_PATH={"papers/" + args.build + "/proj"!r}\n'
               f'WANDB_PROJECT={"mobihet-" + args.build!r}\n' + RUN)
        notebook = nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(f'# {args.build} — audited checkpoint continuation\n'
                f'Parent `{audited["kernel"]}`; actual last rounds {expected}. '
                'Fixed50schedule/fulltrain/fulltest; no scientific changes. Budget boundary preserves checkpoints. '
                'Private W&B inline key persists in private version history. R1/R2 not proved.'),
            nbf.v4.new_code_cell(template.cells[1].source),
            nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(run)])
        notebook.metadata = template.metadata.copy()
        filename = rid + '.ipynb'
        meta = dict(metadata, id=f'{owner}/mobihet-{rid}', title=f'mobihet-{rid}',
                    code_file=filename, kernel_sources=[audited['kernel']])
        assert len(meta['title']) <= 50 and meta['is_private'] and meta['docker_image'] == runtime['docker_image']
        nbf.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code': compile(cell.source, rid, 'exec')
        public.mkdir(parents=True); private.mkdir(parents=True, mode=0o700)
        nbf.write(notebook, public / filename)
        for name, value in [('kernel-metadata.json', meta), ('source_manifest.json', manifest),
                            ('resume-plan.json', expected)]:
            (public / name).write_text(json.dumps(value, indent=2) + '\n')
        subprocess.run(['git', 'check-ignore', '-q', str((private / filename).resolve())], cwd=ROOT, check=True)
        credential = nbf.reads(nbf.writes(notebook), as_version=4)
        credential.cells[1].source = credential.cells[1].source.replace(
            '_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = ' + json.dumps(key))
        for name, value in [(filename, nbf.writes(credential)), ('kernel-metadata.json', json.dumps(meta, indent=2))]:
            fd = os.open(private / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as file: file.write(value)
        print(meta['id'], 'parent', audited['kernel'], 'rounds', {k:v['last_round'] for k,v in expected.items()},
              'session_hours', hours, 'prepared; not pushed', flush=True)


if __name__ == '__main__':
    main()
