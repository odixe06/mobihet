"""Prepare one private evaluation notebook and its flat checkpoint dataset, from audited train rounds.

No upload/push: upload the emitted checkpoint stage, verify it, then push private_push/<rid>.
    python gen_eval.py --session fd-s42za-20261015-r24-v1 --round 1 --owner dinhan25
    python gen_eval.py --session fd-s42za-20261015-r25-v1 --rounds 1-4 --owner odixe0502   # owner 20/10: several rounds per notebook
Owner 21/10 (score as fast as possible): notebooks run fd_eval_fast.py from the evaluation source tag r26 (= r25 + that file;
training stays on r25). Upload artifacts/input/fd-src-20261015-r26 as <owner>/mobihet-fulldrift-src-20261015-r26 once per account.
"""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil

import nbformat

from gen_fd import BASE, ROOT, RUNTIME, TAG, WANDB, SETUP, fixture_manifest, fixture_kernel

EVAL_TAG = f'{TAG}-r26'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_stage(train_tag):
    """Frozen evaluation source: every r25 file + fd_eval_fast.py; an evaluator changed after staging needs a new tag."""
    assert train_tag == f'{TAG}-r25', train_tag
    src = BASE / 'artifacts/input' / f'fd-src-{train_tag}'; stage = BASE / 'artifacts/input' / f'fd-src-{EVAL_TAG}'
    files = {n: src / n for n in json.loads((src / 'input_manifest.json').read_text())}
    files['fd_eval_fast.py'] = BASE / 'source/proj/fd_eval_fast.py'
    if stage.exists():
        inputs = json.loads((stage / 'input_manifest.json').read_text())
        assert inputs == {n: sha(p) for n, p in files.items()}, 'evaluator changed after staging: new tag'
        return inputs
    stage.mkdir()
    for name, path in files.items(): shutil.copyfile(path, stage / name)
    inputs = {n: sha(stage / n) for n in sorted(files)}
    (stage / 'input_manifest.json').write_text(json.dumps(inputs, indent=1) + '\n')
    return inputs


def main():
    p = argparse.ArgumentParser(); p.add_argument('--session', required=True); p.add_argument('--round', type=int)
    p.add_argument('--rounds', help='inclusive range a-b: score these rounds one after another in one notebook (fixture prepared once)')
    p.add_argument('--owner', required=True); p.add_argument('--suffix', default='f1'); a = p.parse_args()
    assert (a.round is None) != (a.rounds is None), 'give --round or --rounds'
    lo, hi = map(int, a.rounds.split('-')) if a.rounds else (a.round, a.round); rounds = list(range(lo, hi + 1)); assert rounds
    source_session = BASE / 'artifacts/kaggle' / a.session
    proof = json.loads((source_session / 'AUDITED.json').read_text())
    assert len(proof['audits']) == 1
    run = source_session / 'runs' / proof['audits'][0]['run']
    cfg = json.loads((run / 'config.json').read_text()); clients = cfg['config'].get('clients', 20)
    assert hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest() == cfg['fingerprint']
    assert cfg['config']['final'] and cfg['config']['defer_eval']
    markers = {r: json.loads((run / f'complete/round_{r:03d}.json').read_text()) for r in rounds}
    assert all(markers[r]['weights_sha'] == sha(run / f'weights/round_{r:03d}.pt') for r in rounds)
    rid = (f'fd-e{lo:02d}' + (f'-{hi:02d}' if hi != lo else '')) + f'-{a.session.removeprefix("fd-")}-{a.suffix}'
    # Session already ends in v1; the suffix identifies this evaluation attempt separately.
    public, private = BASE / 'notebooks' / rid, BASE / 'private_push' / rid
    assert not public.exists() and not private.exists()
    public.mkdir(); private.mkdir(mode=0o700)
    stage = BASE / 'artifacts/input' / rid; stage.mkdir()
    files = {'config.json': 'config.json'}
    for r in rounds: files.update({f'complete/round_{r:03d}.json': f'round_{r:03d}.json', f'weights/round_{r:03d}.pt': f'round_{r:03d}.pt'})
    checks = {}
    for rel, flat in files.items():
        shutil.copyfile(run / rel, stage / flat); checks[rel] = dict(flat=flat, sha=sha(stage / flat))
    dataset = f'{a.owner}/mh-{rid}'
    (stage / 'dataset-metadata.json').write_text(json.dumps(dict(id=dataset, title=f'mh-{rid}', licenses=[dict(name='CC0-1.0')])))
    train_notebook = BASE / 'notebooks' / a.session
    train_inputs = json.loads((train_notebook / 'kernel-metadata.json').read_text())['dataset_sources'][0]
    tag = train_inputs.split('mobihet-fulldrift-src-')[1]
    inputs = eval_stage(tag)
    assert inputs['fd.py'] == cfg['source']['proj/fd.py']
    mh = sorted(n for n in inputs if n.endswith('.py') and n not in ('fd.py', 'fd_eval.py', 'fd_eval_fast.py', 'fd_prepare.py', 'full_run.py', 'full_method.py', 'full_driver.py'))
    assert len(mh) == 40
    head = f'MH={mh!r}\nINPUTS={inputs!r}\nFINAL=True\nCLIENTS={clients}\nPARENT=None\nCHECKS={checks!r}\nFIXTURE={fixture_manifest(clients)!r}\n'
    precheck = '''
PARENT_DIR=Path('/kaggle/temp/eval-parent')
for rel,check in CHECKS.items():
    hits=[p for p in Path('/kaggle/input').rglob(check['flat']) if hashlib.sha256(p.read_bytes()).hexdigest()==check['sha']]
    assert len(hits)==1,('missing/ambiguous checkpoint',rel)
    target=PARENT_DIR/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(hits[0],target)
'''
    setup = SETUP.replace('t=time.monotonic()', precheck + '\nt=time.monotonic()')
    execute = f'''
import subprocess
target=OUT/{run.name!r}
env=dict(os.environ,PYTHONPATH=f'{{PROJ}}:{{WORK}}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('CUDA_VISIBLE_DEVICES',None)
command=[sys.executable,str(PROJ/'fd_eval_fast.py'),'--run',str(PARENT_DIR),'--out',str(target),'--rounds',{','.join(map(str, rounds))!r},
         '--scratch','/kaggle/temp/eval-parts','--wandb_online']
with (OUT/'eval.log').open('a') as log:
    proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        log.write(line);log.flush()
        if line.startswith(('evaluation round','EVALUATED')):print(line.rstrip(),'session_h',round((time.monotonic()-SESSION_T0)/3600,2),flush=True)
    assert proc.wait()==0,'fast evaluation failed; see eval.log'
print('evaluated rounds',{rounds!r},'; scientific result requires collector audit',flush=True)
'''
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(f'# Evaluate rounds {lo}-{hi}, {clients} clients\nPrivate, EMA and proxy; whole fixed test; no training. Fast evaluator (fd_eval_fast.py).'),
                                        nbformat.v4.new_code_cell(WANDB), nbformat.v4.new_code_cell(head + setup),
                                        nbformat.v4.new_code_cell(execute)])
    nb.metadata = dict(kaggle=dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True, language='python', sourceType='notebook'),
                       kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.12'))
    for cell in nb.cells:
        if cell.cell_type == 'code': compile(cell.source, rid, 'exec')
    nbformat.validate(nb)
    meta = dict(id=f'{a.owner}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=f'{rid}.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=True, enable_internet=True, machine_shape=RUNTIME['machine_shape'], docker_image=RUNTIME['docker_image'],
                dataset_sources=[f'{a.owner}/mobihet-fulldrift-src-{EVAL_TAG}', dataset,
                                 f'{ {20:"trietbackup",50:"minhtriethihi",100:"khanhmay0304"}[clients]}/veremi-fl-{clients}client',
                                 'trietbackup/veremi-nextgen2026-test'], kernel_sources=fixture_kernel(a.owner, clients), competition_sources=[])
    assert len(meta['title']) <= 50 and meta['is_private']
    nbformat.write(nb, public / meta['code_file'])
    (public / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    (public / 'eval-contract.json').write_text(json.dumps(dict(session=a.session, run=run.name, rounds=rounds,
         fingerprint=cfg['fingerprint'], weights_sha={str(r): markers[r]['weights_sha'] for r in rounds}, source_tag=EVAL_TAG,
         train_source_tag=tag, evaluator='fd_eval_fast.py',
         checkpoint_dataset=dataset), indent=2) + '\n')
    secret = netrc.netrc().authenticators('api.wandb.ai')[2]
    nb.cells[1].source = nb.cells[1].source.replace('_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = ' + json.dumps(secret))
    for name, content in [(meta['code_file'], nbformat.writes(nb)), ('kernel-metadata.json', json.dumps(meta, indent=2))]:
        fd = os.open(private / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as f: f.write(content)
    print(meta['id'], 'prepared checkpoint dataset', dataset, 'stage', stage, flush=True)


if __name__ == '__main__': main()
