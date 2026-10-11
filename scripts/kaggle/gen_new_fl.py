"""Immutable source/fixture dataset + two seeded private Kaggle pilot notebooks."""
import hashlib, json, shutil
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['local', 'diag', 'kme_gauss', 'kme', 'dm_gauss', 'dm']
KERNELS = [('sa-s42', 'minhtran0601', 42), ('sa-s43', 'dinhan25', 43)]
SLUG = 'mobihet-sa-20261011-src'

SETUP = r'''
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import torch
n = torch.cuda.device_count()
print('torch', torch.__version__, 'gpus', [torch.cuda.get_device_name(i) for i in range(n)], flush=True)
assert n == 2 and all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(n))
hits = [p.parent for p in Path('/kaggle/input').rglob('new_fl.py') if (p.parent / 'train_0_X.npy').exists()]
assert len(hits) == 1, hits
SRC, WORK, OUT = hits[0], Path('/kaggle/working/work'), Path('/kaggle/working/runs')
SOURCE_OUT = Path('/kaggle/working/source'); SOURCE_OUT.mkdir(exist_ok=True)
for sub in ('mh', 'knowledge', 'papers/pilot/fixture'):
    (WORK / sub).mkdir(parents=True, exist_ok=True)
code_manifest = json.loads((SRC / 'source_manifest.json').read_text())
for name, sha in code_manifest.items():
    assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == sha, ('source mismatch', name)
    shutil.copyfile(SRC / name, SOURCE_OUT / name)
for f in SRC.iterdir():
    if f.name in CODE:
        dst = WORK / 'mh' / f.name
    elif f.name == 'meta.json':
        dst = WORK / 'knowledge' / f.name
    else:
        dst = WORK / 'papers/pilot/fixture' / f.name
    shutil.copyfile(f, dst)
manifest = json.loads((SRC / 'manifest.json').read_text())
for name, sha in manifest['files'].items():
    assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == sha, ('fixture mismatch', name)
shutil.copyfile(SRC / 'source_manifest.json', SOURCE_OUT / 'source_manifest.json')
shutil.copyfile(SRC / 'manifest.json', SOURCE_OUT / 'fixture_manifest.json')
print('verified source and fixture', len(code_manifest), len(manifest['files']), flush=True)
'''

RUN = r'''
T0 = time.time(); OUT.mkdir(exist_ok=True)
queue = list(METHODS); slots = [0, 0, 1, 1]; running = {}; done = {}; seen = {}
while queue or running:
    for slot, gpu in enumerate(slots):
        if slot not in running and queue and time.time() - T0 < 2 * 3600:
            method = queue.pop(0); d = OUT / f'{method}_s{SEED}'; d.mkdir(exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            cmd = [sys.executable, '-m', 'mh.new_fl', '--arm', method, '--method', method, '--seed', str(SEED), '--out', str(OUT)]
            log = open(d / 'log.txt', 'w')
            running[slot] = (method, subprocess.Popen(cmd, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT), log)
            print('start', method, 'gpu', gpu, flush=True)
    for slot, (method, proc, log) in list(running.items()):
        f = OUT / f'{method}_s{SEED}' / 'log.txt'; lines = f.read_text().splitlines()
        for line in lines[seen.get(method, 0):]:
            if line.startswith(('setup', 'r', 'done', 'Traceback')) or 'Error' in line:
                print(method, '|', line, flush=True)
        seen[method] = len(lines)
        if proc.poll() is not None:
            done[method] = proc.returncode; log.close(); del running[slot]
    if queue and not running and time.time() - T0 >= 2 * 3600:
        break
    if queue or running:
        time.sleep(20)
summary = {m: {'rc': rc, 'complete': json.loads((OUT / f'{m}_s{SEED}' / 'result.json').read_text())['complete']
              if (OUT / f'{m}_s{SEED}' / 'result.json').exists() else False} for m, rc in done.items()}
(OUT / f'summary_s{SEED}.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
shutil.rmtree(WORK)  # source retained separately; raw fixture is an input, never a run output
assert set(summary) == set(METHODS) and all(v['rc'] == 0 and v['complete'] for v in summary.values()), summary
'''


def main():
    base = ROOT / 'papers/new-fl'; runtime = json.loads((ROOT / 'knowledge/runtime.json').read_text())
    code = [f.name for f in sorted((ROOT / 'mh').glob('*.py'))]
    for rid, owner, seed in KERNELS:
        d = base / 'notebook' / rid
        assert not (d / 'PUSHED').exists(), 'never edit pushed source'
        src = base / 'input' / owner; src.mkdir(parents=True, exist_ok=True)
        for name in code:
            shutil.copyfile(ROOT / 'mh' / name, src / name)
        shutil.copyfile(ROOT / 'knowledge/meta.json', src / 'meta.json')
        for f in (ROOT / 'papers/pilot/fixture').iterdir():
            shutil.copyfile(f, src / f.name)
        manifest = {name: hashlib.sha256((src / name).read_bytes()).hexdigest() for name in code + ['meta.json']}
        (src / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
        (src / 'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{SLUG}', title=SLUG, licenses=[{'name': 'CC0-1.0'}])))
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / 'source_manifest.json', d / 'source_manifest.json')
        shutil.copyfile(src / 'manifest.json', d / 'fixture_manifest.json')
        nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet S-a: new FL screening\nPrivate DAGSNet, no raw client data exchanged. Internal holdouts only.\n'
            'Locked protocol: docs/research/pilot-new-fl.md. Unlimited-channel feasibility; does not certify finite contact.\n'
            '10 metrics from per-client confusion: accuracy, precision/recall/F1 macro/micro/weighted. '
            'Single-label micro precision/recall/F1 and weighted recall equal accuracy. Final r20; no best checkpoint.'),
            nbf.v4.new_code_cell(f'CODE = {code!r}\n' + SETUP),
            nbf.v4.new_code_cell(f'SEED = {seed}\nMETHODS = {METHODS!r}\n' + RUN)])
        nb.metadata.update(kernelspec={'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            language_info={'name': 'python', 'version': '3.12'},
            kaggle={'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': False,
                    'language': 'python', 'sourceType': 'notebook'})
        nbf.write(nb, d / f'{rid}.ipynb'); nbf.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(d), 'exec')
        meta = dict(id=f'{owner}/mobihet-{rid}-20261011', title=f'mobihet-{rid}-20261011', code_file=f'{rid}.ipynb',
            language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=False,
            machine_shape=runtime['machine_shape'], docker_image=runtime['docker_image'],
            dataset_sources=[f'{owner}/{SLUG}'], kernel_sources=[], competition_sources=[])
        (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(meta['id'], 'input', src, 'source files', len(manifest))


if __name__ == '__main__':
    main()
