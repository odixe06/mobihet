#!/usr/bin/env python
"""Build the P2 Kaggle inputs (docs/research/pilot-plan.md §5): one private dataset per owner with the code + pilot fixture,
and one notebook per (method group, seed). Each kernel runs its arms of scripts/p2_queue.txt, 2 jobs per T4 (4 at once),
and keeps only runs/<arm>_s<seed>/{result.json,log.txt}. No W&B. Run from mobihet/ (env nckh):

    python scripts/kaggle/gen_p2.py            # writes papers/p2/kaggle_src/ and papers/p2/notebook/<rid>/
Then, per owner: create the dataset, wait until ready, push the notebook (commands printed at the end).
"""
import json, shutil
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = json.loads((ROOT / 'knowledge/runtime.json').read_text())
QUEUE = [l.split(maxsplit=1) for l in (ROOT / 'scripts/p2_queue.txt').read_text().splitlines() if l.strip()]
KERNELS = [  # rid, owner, seed, arm prefix
    ('p2-h2-s42', 'khanhmay0304', 42, 'h2_'), ('p2-h2-s43', 'minhtriethihi', 43, 'h2_'),
    ('p2-dml-s42', 'catbaochau', 42, 'dml_'), ('p2-dml-s43', 'minhtrit06', 43, 'dml_')]
DS_SLUG = 'mobihet-p2-src'
CODE = ['__init__.py', 'pilot.py', 'p2.py', 'channel.py', 'mobility.py', 'dagsnet.py', 'metrics.py', 'archs.py']
BUDGET_H = 9.0       # stop launching new jobs after this; the session limit is 12 h

SETUP = r'''
import json, os, shutil, subprocess, sys, time, hashlib
from pathlib import Path
import torch
n = torch.cuda.device_count()
print('torch', torch.__version__, 'gpus', [torch.cuda.get_device_name(i) for i in range(n)], flush=True)
assert n == 2, f'expected 2 GPUs, got {n}. Set machine_shape=NvidiaTeslaT4.'
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(n))
hits = sorted(p.parent for p in Path('/kaggle/input').rglob('p2.py') if (p.parent / 'train_0_X.npy').exists())
assert len(hits) == 1, hits
SRC, WORK, OUT = hits[0], Path('/kaggle/working/mobihet'), Path('/kaggle/working/runs')
CODE = CODE_FILES
for sub in ('mh', 'knowledge', 'papers/pilot/fixture'):
    (WORK / sub).mkdir(parents=True, exist_ok=True)
for f in sorted(SRC.iterdir()):             # the dataset is flat: rebuild the repo layout the code expects
    dst = WORK / 'mh' / f.name if f.name in CODE else WORK / 'knowledge' / f.name if f.name == 'meta.json' else WORK / 'papers/pilot/fixture' / f.name
    shutil.copyfile(f, dst)
man = json.loads((WORK / 'papers/pilot/fixture/manifest.json').read_text())
for name, digest in man['files'].items():
    h = hashlib.sha256((WORK / 'papers/pilot/fixture' / name).read_bytes()).hexdigest()
    assert h == digest, ('fixture corrupted', name)
print('fixture verified:', len(man['files']), 'files', flush=True)
'''

RUN = r'''
T0, slots, running, done = time.time(), [0, 0, 1, 1], {}, {}
OUT.mkdir(parents=True, exist_ok=True)
queue = [a for a in JOBS if not (OUT / f"{a[0]}_s{SEED}" / 'result.json').exists()]
seen = {}
while queue or running:
    for slot, gpu in enumerate(slots):
        if slot not in running and queue and time.time() - T0 < BUDGET_H * 3600:
            arm, args = queue.pop(0)
            d = OUT / f'{arm}_s{SEED}'; d.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1')
            cmd = [sys.executable, '-m', 'mh.p2', '--arm', arm, '--seed', str(SEED), '--out', str(OUT)] + args.split()
            running[slot] = (arm, subprocess.Popen(cmd, cwd=WORK, env=env, stdout=open(d / 'log.txt', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%H:%M:%S'), 'start', arm, 'gpu', gpu, flush=True)
    for slot, (arm, p) in list(running.items()):
        log = OUT / f'{arm}_s{SEED}' / 'log.txt'
        lines = log.read_text().splitlines() if log.exists() else []
        for l in lines[seen.get(arm, 0):]:
            if l.startswith(('r', 'done', 'Traceback')) or 'Error' in l:
                print(arm, '|', l, flush=True)
        seen[arm] = len(lines)
        if p.poll() is not None:
            done[arm] = p.returncode; del running[slot]
            print(time.strftime('%H:%M:%S'), 'exit', arm, p.returncode, flush=True)
    if not running and queue and time.time() - T0 >= BUDGET_H * 3600:
        print('budget reached, not started:', [a for a, _ in queue], flush=True); break
    time.sleep(20)
summary = {a: dict(rc=rc, result=(OUT / f'{a}_s{SEED}' / 'result.json').exists()) for a, rc in done.items()}
(OUT / f'summary_s{SEED}.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1), flush=True)
shutil.rmtree(WORK, ignore_errors=True)   # keep the output to runs/ only
assert all(v['rc'] == 0 and v['result'] for v in summary.values()), summary
'''


def main():
    src = ROOT / 'papers/p2/kaggle_src'
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)                # flat: `kaggle datasets create` skips sub-directories by default
    for f in CODE:
        shutil.copyfile(ROOT / 'mh' / f, src / f)
    shutil.copyfile(ROOT / 'knowledge/meta.json', src / 'meta.json')
    for f in sorted((ROOT / 'papers/pilot/fixture').iterdir()):
        assert f.name not in CODE and f.name != 'meta.json', f
        shutil.copyfile(f, src / f.name)
    for rid, owner, seed, prefix in KERNELS:
        d = ROOT / 'papers/p2/notebook' / rid
        assert not (d / 'PUSHED').exists(), f'{d} already pushed'
        d.mkdir(parents=True, exist_ok=True)
        jobs = [(arm, rest) for arm, rest in QUEUE if arm.startswith(prefix)]
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(f'# MobiHet P2 `{rid}`\nArms {[a for a, _ in jobs]}, seed {seed}. Plan: docs/research/pilot-plan.md §5.'),
                    nbf.v4.new_code_cell(f'CODE_FILES = {json.dumps(CODE)}\n' + SETUP),
                    nbf.v4.new_code_cell(f'SEED = {seed}\nBUDGET_H = {BUDGET_H}\nJOBS = {json.dumps(jobs)}\n' + RUN)]
        nb.metadata.update({'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                            'language_info': {'name': 'python', 'version': '3.12'},
                            'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': False,
                                       'language': 'python', 'sourceType': 'notebook'}})
        nbf.write(nb, d / f'{rid}.ipynb')
        meta = {'id': f'{owner}/mobihet-{rid}', 'title': f'mobihet-{rid}', 'code_file': f'{rid}.ipynb', 'language': 'python',
                'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'dataset_sources': [f'{owner}/{DS_SLUG}'], 'competition_sources': [], 'kernel_sources': [],
                'machine_shape': RUNTIME['machine_shape'], 'docker_image': RUNTIME['docker_image']}
        (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
        print(rid, owner, len(jobs), 'jobs')
    print('dataset staging:', src)


if __name__ == '__main__':
    main()
