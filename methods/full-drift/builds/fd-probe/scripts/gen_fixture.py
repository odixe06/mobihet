"""Fixture cache (owner 20/10): a CPU kernel per (account, client count) runs the final `prepare_fd` once and keeps the fixture as
its output, so GPU sessions of the same account attach it (`kernel_sources`) instead of preparing again (17-147 min measured r25).
The kernel asserts the manifest equals the audited r25 one (all file SHA); GPU notebooks re-check every file SHA before use.

    python gen_fixture.py --owner minhtriethihi --clients 20           # prepare notebook (no push)
    python gen_fixture.py --owner minhtriethihi --clients 20 --check   # after COMPLETE: write the marker gen_fd/gen_eval attach
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat

from gen_fd import BASE, ROOT, RUNTIME, TAG, fixture_manifest

HELPER = ROOT / '.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py'
SOURCE_TAG = f'{TAG}-r25'

CELL = r'''
import hashlib,json,shutil,sys,time
from pathlib import Path
T0=time.monotonic()
hits=[p.parent for p in Path('/kaggle/input').rglob('input_manifest.json') if json.loads(p.read_text())==INPUTS]
assert len(hits)==1,('missing/ambiguous inputs',hits)
SRC=hits[0];WORK=Path('/kaggle/working/cache');PROJ=WORK/'papers/full-drift/proj'
for folder in (WORK/'mh',WORK/'knowledge',PROJ):folder.mkdir(parents=True,exist_ok=True)
for name,digest in INPUTS.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest,name
    shutil.copyfile(SRC/name,WORK/'mh'/name if name in MH else PROJ/name if name.endswith('.py') else WORK/'knowledge'/name)
sys.path.insert(0,str(WORK));sys.path.insert(0,str(PROJ))
from fd_prepare import prepare_fd
data=prepare_fd(WORK,final=True,clients=CLIENTS)
fixture=WORK/'papers/final/fixture'
assert json.loads(json.dumps(data))==EXPECTED and json.loads((fixture/'manifest.json').read_text())==EXPECTED,'fixture differs from the audited r25 manifest'
assert sorted(p.name for p in fixture.glob('*.npy'))==sorted(EXPECTED['files'])
for folder in (WORK/'mh',WORK/'knowledge',WORK/'papers/full-drift'):shutil.rmtree(folder)   # output = fixture only
size=sum(p.stat().st_size for p in fixture.iterdir())
(Path('/kaggle/working')/'fixture_cache.json').write_text(json.dumps(dict(clients=CLIENTS,files=len(EXPECTED['files']),bytes=size,
    prepare_seconds=round(time.monotonic()-T0),source_rows=data['source_rows'],test_rows=data['test_rows']),indent=2))
print('fixture cache ready',CLIENTS,'clients',len(EXPECTED['files']),'files',round(size/2**30,2),'GiB','sec',round(time.monotonic()-T0),flush=True)
'''


def kernel_id(owner, clients): return f'{owner}/mobihet-fd-fixture-k{clients}-{SOURCE_TAG}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--owner', required=True); p.add_argument('--clients', type=int, choices=(20, 50, 100), required=True)
    p.add_argument('--check', action='store_true'); a = p.parse_args()
    kid = kernel_id(a.owner, a.clients); rid = kid.split('/')[1].removeprefix('mobihet-')
    if a.check:
        env = dict(os.environ, PATH=f'{Path(sys.executable).parent}:{os.environ.get("PATH", "")}')   # kaggle CLI of this interpreter's env
        status = subprocess.run([sys.executable, str(HELPER), a.owner, '--', 'kaggle', 'kernels', 'status', kid], capture_output=True, text=True, env=env)
        assert 'KernelWorkerStatus.COMPLETE' in status.stdout, status.stdout + status.stderr
        marker = BASE / 'artifacts/fixture-cache' / f'{a.owner}-k{a.clients}.json'; marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps(dict(kernel=kid, clients=a.clients, status='COMPLETE', note='kernel asserted the audited r25 manifest'), indent=2) + '\n')
        print('cache ready', kid, '->', marker); return
    stage = BASE / 'artifacts/input' / f'fd-src-{SOURCE_TAG}'
    inputs = json.loads((stage / 'input_manifest.json').read_text())
    mh = sorted(p.name for p in (ROOT / '.compat/mh').glob('*.py')); assert len(mh) == 40
    head = f'MH={mh!r}\nINPUTS={inputs!r}\nCLIENTS={a.clients}\nEXPECTED={fixture_manifest(a.clients)!r}\n'
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        f'# Final fixture cache, {a.clients} clients\nCPU only: `prepare_fd(final=True)` once; output reused by this account\'s GPU sessions.'),
        nbformat.v4.new_code_cell(head + CELL)])
    nb.metadata = dict(kaggle=dict(accelerator='none', isGpuEnabled=False, isInternetEnabled=True, language='python', sourceType='notebook'),
                       kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.12'))
    for cell in nb.cells:
        if cell.cell_type == 'code': compile(cell.source, rid, 'exec')
    nbformat.validate(nb)
    folder = BASE / 'notebooks' / f'{rid}-{a.owner}'; assert not folder.exists(); folder.mkdir()
    meta = dict(id=kid, title=kid.split('/')[1], code_file=f'{rid}.ipynb', language='python', kernel_type='notebook', is_private=True,
                enable_gpu=False, enable_internet=True, docker_image=RUNTIME['docker_image'],
                dataset_sources=[f'{a.owner}/mobihet-fulldrift-src-{SOURCE_TAG}',
                                 f'{ {20: "trietbackup", 50: "minhtriethihi", 100: "khanhmay0304"}[a.clients]}/veremi-fl-{a.clients}client',
                                 'trietbackup/veremi-nextgen2026-test'], kernel_sources=[], competition_sources=[])
    assert len(meta['title']) <= 50
    nbformat.write(nb, folder / meta['code_file'])
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('prepared', kid, folder, flush=True)


if __name__ == '__main__': main()
