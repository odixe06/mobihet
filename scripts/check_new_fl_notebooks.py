"""Validate frozen sources and run the exact generated six-job orchestration with cheap workers."""
import hashlib, json, sys, tempfile, types
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
for d in sorted((ROOT / 'papers/new-fl/notebook').iterdir()):
    meta = json.loads((d / 'kernel-metadata.json').read_text())
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['machine_shape'] == 'NvidiaTeslaT4'
    assert '@sha256:' in meta['docker_image'] and len(meta['title']) <= 50
    nb = nbformat.read(d / meta['code_file'], as_version=4); nbformat.validate(nb)
    assert nb.metadata.kaggle.isGpuEnabled and nb.metadata.kaggle.accelerator == 'nvidiaTeslaT4'
    src = ROOT / 'papers/new-fl/input' / meta['id'].split('/')[0]
    sm = json.loads((src / 'source_manifest.json').read_text())
    for name, sha in sm.items():
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == sha
        local = ROOT / ('knowledge' if name == 'meta.json' else 'mh') / name
        assert local.read_bytes() == (src / name).read_bytes()
    with tempfile.TemporaryDirectory(prefix='mobihet-notebook-') as tmp:
        out = Path(tmp) / 'runs'; work = Path(tmp) / 'work'; work.mkdir()
        started = []
        class Process:
            def __init__(self, cmd, **kwargs):
                method = cmd[cmd.index('--method') + 1]; seed = cmd[cmd.index('--seed') + 1]
                assert cmd[cmd.index('-m') + 1] == 'mh.new_fl'
                started.append((method, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                (out / f'{method}_s{seed}' / 'result.json').write_text(json.dumps({'complete': True}))
                self.returncode = 0
            def poll(self):
                return 0
        scope = dict(Path=Path, OUT=out, WORK=work, os=__import__('os'), json=json, sys=sys,
                     subprocess=types.SimpleNamespace(Popen=Process, STDOUT=-2),
                     time=types.SimpleNamespace(time=lambda: 0., sleep=lambda _: None), shutil=__import__('shutil'))
        exec(nb.cells[2].source, scope)
        assert [m for m, _ in started] == scope['METHODS'] and len(started) == 6
        assert {g for _, g in started} == {'0', '1'}
    print('VALIDATED', meta['id'], 'source hashes, metadata, exact six-job orchestration')
