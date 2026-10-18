"""Probe 19 notebook (design §Probe 19, federated 16-class logit bias, no DAGSNet training): stage = round-13 FullDrift source + fd_probe14.py, plus the
audited epoch-15 weights it diagnoses (SHA-pinned) and their audited private gp values. Nothing is pushed here.

    python methods/full-drift/builds/fd-probe/scripts/gen_probe19.py [--owner nguyenminhtran060113] [--suffix v1]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import nbformat
from gen_fd import BASE, ROOT, FULL, RUNTIME, SETUP, sha

TAG = '20261015-p19'
WEIGHTS = {'m4b_s42': 'fd-s42j-20261015-r5-v1/runs/fd_dml_m4b_s42_e15', 'frz_s42': 'fd-s42oa-20261015-r15-v1/runs/fd_dml_m4b_frz_s42_e15',
           'gkd_s42': 'fd-s42oc-20261015-r15-v1/runs/fd_dml_m4b_gpool_frz_gkd_s42_e15'}

RUN = r'''
weights={name:digest for name,digest in WEIGHTS.items()}
for name,digest in weights.items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest,name
env=dict(os.environ,PYTHONPATH=f'{PROJ}:{WORK}',OMP_NUM_THREADS='4')
t=time.monotonic()
r=subprocess.run([sys.executable,str(PROJ/'fd_probe19.py'),'--weights',str(SRC),'--out',str(OUT/'probe19.json')],
                 cwd=WORK,env=env,capture_output=True,text=True)
(OUT/'probe19.log').write_text(r.stdout+r.stderr);print(r.stdout[-8000:],r.stderr[-4000:],flush=True)
assert r.returncode==0,'probe failed'
print('probe19 done min',round((time.monotonic()-t)/60,1),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--owner', default='nguyenminhtran060113')
    parser.add_argument('--suffix', default='v1'); args = parser.parse_args()
    stage = BASE / 'artifacts/input' / f'fd-probe19-src-{TAG}'
    mh = sorted(p.name for p in (ROOT / '.compat/mh').glob('*.py')); assert len(mh) == 40
    files = {**{n: ROOT / '.compat/mh' / n for n in mh}, **{n: FULL / n for n in ('full_run.py', 'full_method.py', 'full_driver.py')},
             **{n: BASE / 'source/proj' / n for n in ('fd.py', 'fd_prepare.py', 'fd_probe14.py', 'fd_probe16.py', 'fd_probe19.py')},
             'meta.json': ROOT / 'knowledge/meta.json', 'scaler.json': ROOT / 'knowledge/scaler.json'}
    runs = BASE / 'artifacts/kaggle'
    expected = {name: json.loads((runs / path / 'result.json').read_text())['history'][14]['eval']['private']['group']['gp']['f1_macro']
                for name, path in WEIGHTS.items()}
    if stage.exists():
        inputs = json.loads((stage / 'input_manifest.json').read_text())
        assert inputs == {n: sha(p) for n, p in files.items()}, 'source changed after staging: new tag'
    else:
        stage.mkdir(parents=True)
        for name, path in files.items(): shutil.copyfile(path, stage / name)
        for name, path in WEIGHTS.items(): shutil.copyfile(runs / path / 'weights/round_015.pt', stage / f'{name}.pt')
        (stage / 'expected.json').write_text(json.dumps(expected, indent=1) + '\n')
        inputs = {n: sha(stage / n) for n in sorted(files)}
        (stage / 'input_manifest.json').write_text(json.dumps(inputs, indent=1) + '\n')
    weights = {f'{n}.pt': sha(stage / f'{n}.pt') for n in WEIGHTS}
    for n, path in WEIGHTS.items(): assert weights[f'{n}.pt'] == sha(runs / path / 'weights/round_015.pt'), n
    rid = f'fd-probe19-{TAG}-{args.suffix}'; dataset = f'{args.owner}/mobihet-fd-probe19-src-{TAG}'
    public = BASE / 'notebooks' / rid; assert not public.exists(); public.mkdir(parents=True)
    head = f'import os,sys,subprocess,time\nSESSION_T0=time.monotonic()\nMH={mh!r}\nINPUTS={inputs!r}\n'
    cells = [nbformat.v4.new_markdown_cell(f'# FullDrift probe 19 (no training)\nFederated logit bias; why private gp stalls: cross-client matrix, gap by local count, frozen-backbone refits, '
                                           'oracle bias, duplicates + kNN ceiling. Contract: docs/methods/full-drift/design.md §Probe 19'),
             nbformat.v4.new_code_cell(head + SETUP), nbformat.v4.new_code_cell(f'WEIGHTS={weights!r}\n' + RUN)]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = dict(kaggle=dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True, language='python',
                                   sourceType='notebook'),
                       kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.12'))
    for c in nb.cells:
        if c.cell_type == 'code': compile(c.source, rid, 'exec')
    nbformat.validate(nb)
    meta = dict(id=f'{args.owner}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=f'{rid}.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=True, enable_internet=True, machine_shape=RUNTIME['machine_shape'], docker_image=RUNTIME['docker_image'],
                dataset_sources=[dataset, 'trietbackup/veremi-fl-20client', 'trietbackup/veremi-nextgen2026-test'], kernel_sources=[],
                competition_sources=[])
    assert len(meta['title']) <= 50
    nbformat.write(nb, public / meta['code_file']); (public / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    names = [*inputs, 'input_manifest.json', 'expected.json', *weights]
    (public / 'checkpoint-dataset.json').write_text(json.dumps(dict(dataset=dataset, files={n: dict(size=(stage / n).stat().st_size)
                                                                                           for n in names}, uploaded=False), indent=2) + '\n')
    (public / 'source_manifest.json').write_text(json.dumps(dict(inputs=sha(stage / 'input_manifest.json'), weights=weights,
                                                                 expected=expected), indent=2) + '\n')
    print(meta['id'], 'prepared; expected', expected)


if __name__ == '__main__':
    main()
