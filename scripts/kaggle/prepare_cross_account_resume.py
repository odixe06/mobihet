"""Stage audited full histories and immutable continuations for new quota owners."""
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
from scripts.kaggle.gen_full_resume_batch import RUN


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(row):
    base = (ROOT / 'papers' / row['build']).resolve()
    owner, rid = row['owner'], row['rid']
    public, private = base / 'notebook' / rid, base / 'private_push' / rid
    assert not public.exists() and not private.exists(), 'immutable tag already exists'
    stage = base / 'artifacts/checkpoint-staging' / rid
    assert not (stage / 'UPLOADED').exists(), 'inspect uploaded dataset before retry'
    expected, parents, copied = {}, {}, {}
    reference = None
    manifest = None
    for arm, parent in row['parents'].items():
        folder = base / 'kaggle' / parent
        record = json.loads((folder / 'AUDITED.json').read_text())
        audit = next(x for x in record['audits'] if x['arm'] == arm)
        assert audit['rounds'] > 0 and not audit['complete']
        run = folder / 'runs' / f'{arm}_s42'
        identity = json.loads((run / 'config.json').read_text())
        expected[arm] = {k: identity[k] for k in ('fingerprint', 'source', 'fixture')}
        expected[arm]['last_round'] = audit['rounds']
        parents[arm] = dict(kernel=record['kernel'], version=record['version'], round=audit['rounds'])
        meta = json.loads((base / 'notebook' / parent / 'kernel-metadata.json').read_text())
        current = json.loads((base / 'notebook' / parent / 'source_manifest.json').read_text())
        if manifest is None:
            manifest = current
            reference = nbf.read(base / 'notebook' / parent / meta['code_file'], as_version=4)
        assert current == manifest, 'different frozen source packages'
        for name, digest in identity['source'].items():
            source = ROOT / name if name.startswith('mh/') else base / name
            assert sha(source) == digest, ('frozen source drift', name)
        for source in run.rglob('*'):
            if not source.is_file() or 'wandb' in source.relative_to(run).parts or source.name == 'log.txt':
                continue
            relative = Path('runs') / run.name / source.relative_to(run)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == sha(source), ('staging collision', relative)
            else:
                os.link(source, target)
            copied[relative.as_posix()] = dict(sha256=sha(target), size=target.stat().st_size)
    for name, digest in manifest.items():
        source = folder / 'source' / name
        assert sha(source) == digest, ('source package drift', name)
        target = stage / 'package' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == digest, ('staging source collision', name)
        else:
            os.link(source.resolve(), target)
        copied['package/' + name] = dict(sha256=digest, size=target.stat().st_size)
    (stage / 'package/source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dataset = f'{owner}/mobihet-{rid}-ckpt'
    bridge = dict(dataset=dataset, parents=parents, files=copied,
                  verified_full_parent=True, private=True, uploaded=False)
    (stage / 'checkpoint-manifest.json').write_text(json.dumps(bridge, indent=2) + '\n')
    (stage / 'dataset-metadata.json').write_text(json.dumps(dict(
        id=dataset, title=f'mobihet-{rid}-ckpt', licenses=[{'name': 'CC0-1.0'}]), indent=2) + '\n')
    setup = reference.cells[2].source
    body = setup[setup.index('import hashlib'):]
    setup = f'CODE={[n for n in manifest if (ROOT / "mh" / n).exists()]!r}\n' + \
            f'PROJ={[p.name for p in (base / "proj").glob("*.py")]!r}\nEXPECTED={expected!r}\n' + body
    run = f'METHODS={list(expected)!r}\nSESSION_HOURS={row["hours"]!r}\n' + \
          f'PROJECT_PATH={"papers/" + row["build"] + "/proj"!r}\n' + \
          f'WANDB_PROJECT={row.get("wandb_project", "mobihet-" + row["build"])!r}\n' + RUN
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# Audited cross-account continuation\nFixed50schedule; full history and frozen source. '
        'Private inline W&B key persists in private version history.'),
        nbf.v4.new_code_cell(reference.cells[1].source), nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell(run)])
    nb.metadata = reference.metadata.copy()
    filename = rid + '.ipynb'
    metadata = dict(meta, id=f'{owner}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=filename,
                    kernel_sources=[], dataset_sources=[dataset, 'trietbackup/veremi-fl-20client',
                                                        'trietbackup/veremi-nextgen2026-test'])
    assert metadata['is_private'] and len(metadata['title']) <= 50
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, rid, 'exec')
    public.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    nbf.write(nb, public / filename)
    for name, value in [('kernel-metadata.json', metadata), ('source_manifest.json', manifest),
                        ('resume-plan.json', expected), ('checkpoint-dataset.json', bridge)]:
        (public / name).write_text(json.dumps(value, indent=2) + '\n')
    subprocess.run(['git', 'check-ignore', '-q', str((private / filename).resolve())], cwd=ROOT, check=True)
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key
    credential = nbf.reads(nbf.writes(nb), as_version=4)
    credential.cells[1].source = credential.cells[1].source.replace(
        '_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = ' + json.dumps(key))
    for name, value in [(filename, nbf.writes(credential)), ('kernel-metadata.json', json.dumps(metadata, indent=2))]:
        fd = os.open(private / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as file:
            file.write(value)
    probe = base / 'mount-probe' / rid
    probe.mkdir(parents=True)
    cpu = setup.split('from full_prepare import prepare_full', 1)[0]
    cpu = cpu.replace('assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))',
                      'assert torch.cuda.device_count()==0')
    cpu = cpu.replace("print('torch',torch.__version__,'gpus',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)",
                      "print('CPU checkpoint probe',flush=True)")
    assert 'get_device_name' not in cpu and 'get_device_capability' not in cpu
    cpu = 'import os,sys,json,time\n' + cpu + \
          "\n(SOURCE/'MOUNT_VERIFIED.json').write_text(json.dumps({'imports':imports,'cpu_only':True}))\n"
    cpu_nb = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(cpu)])
    cpu_nb.metadata = dict(nb.metadata, kaggle=dict(accelerator='none', isGpuEnabled=False,
        isInternetEnabled=True, language='python', sourceType='notebook'))
    cpu_meta = dict(metadata, id=f'{owner}/mobihet-{rid}-cpu', title=f'mobihet-{rid}-cpu', enable_gpu=False,
                    dataset_sources=[dataset])
    cpu_meta.pop('machine_shape', None)
    nbf.validate(cpu_nb)
    compile(cpu, rid, 'exec')
    nbf.write(cpu_nb, probe / filename)
    (probe / 'kernel-metadata.json').write_text(json.dumps(cpu_meta, indent=2) + '\n')
    print(metadata['id'], 'prepared', {k: v['last_round'] for k, v in expected.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    for row in json.loads(args.plan.read_text())['rows']:
        prepare(row)
